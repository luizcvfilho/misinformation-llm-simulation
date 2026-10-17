"""Persist validated call responses for explicitly requested run recovery."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
from pathlib import Path
from time import sleep
from uuid import uuid4

from misinformation_simulation.llm.progress import mark_cached, progress_context

_checkpoint: ContextVar[CallCheckpoint | None] = ContextVar("call_checkpoint", default=None)
_pending: ContextVar[dict | None] = ContextVar("checkpoint_pending", default=None)


def write_json_atomic(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                sleep(0.02 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class CallCheckpoint:
    def __init__(self, directory: Path, source: Path | None = None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.rejected = set()

    def get(self, key):
        if self.source is None or key in self.rejected:
            return None
        path = self.source / f"{key}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") != 1 or not isinstance(value.get("response"), str):
            raise ValueError(f"Invalid call checkpoint: {path}")
        return value["response"]

    def put(self, key, response):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        write_json_atomic(path, {"version": 1, "response": response})


@contextmanager
def use_call_checkpoint(directory: Path, source: Path | None = None):
    token = _checkpoint.set(CallCheckpoint(directory, source))
    try:
        yield
    finally:
        _checkpoint.reset(token)


def discard_pending_responses():
    if (pending := _pending.get()) is not None:
        if checkpoint := _checkpoint.get():
            checkpoint.rejected.update(pending)
        pending.clear()


def checkpoint_transaction(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        checkpoint = _checkpoint.get()
        if checkpoint is None or _pending.get() is not None or kwargs.get("isolated"):
            return function(*args, **kwargs)
        pending = {}
        token = _pending.set(pending)
        try:
            result = function(*args, **kwargs)
            for key, response in pending.items():
                checkpoint.put(key, response)
            return result
        finally:
            _pending.reset(token)

    return wrapped


def checkpoint_response(function):
    @wraps(function)
    def wrapped(client, **kwargs):
        checkpoint = _checkpoint.get()
        if checkpoint is None:
            return function(client, **kwargs)
        location = progress_context()
        inputs = {
            "function": function.__name__,
            "endpoint": str(getattr(client, "base_url", "")),
            **{
                key: kwargs[key] for key in ("model", "prompt", "system_instruction", "temperature")
            },
            "location": {
                key: location[key]
                for key in ("news_id", "row", "node_id", "stage", "comparison", "repeat")
                if key in location
            },
        }
        key = sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()
        response = checkpoint.get(key)
        if response is None:
            response = function(client, **kwargs)
        else:
            from misinformation_simulation.llm.usage import record_cache_hit

            mark_cached()
            record_cache_hit("call", source="run_checkpoint")
        if (pending := _pending.get()) is not None:
            pending[key] = response
        else:
            checkpoint.put(key, response)
        return response

    return wrapped
