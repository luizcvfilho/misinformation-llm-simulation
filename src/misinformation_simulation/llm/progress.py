"""Structured execution events shared by the simulation and API helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("execution_sink", default=None)
_context: ContextVar[dict | None] = ContextVar("execution_context", default=None)
_operation: ContextVar[dict | None] = ContextVar("execution_operation", default=None)


def set_progress_context(**values: Any) -> None:
    _context.set(values)


def emit_event(kind: str, **values: Any) -> None:
    if sink := _sink.get():
        current = _operation.get() or {}
        sink(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "monotonic": monotonic(),
                **(_context.get() or {}),
                **{
                    key: current[key]
                    for key in ("operation_id", "stage", "model", "provider")
                    if key in current
                },
                "kind": kind,
                **values,
            }
        )


@contextmanager
def observe_events(callback: Callable[[dict], None] | None):
    token = _sink.set(callback)
    context_token = _context.set({})
    operation_token = _operation.set(None)
    try:
        yield
    finally:
        _operation.reset(operation_token)
        _context.reset(context_token)
        _sink.reset(token)


@contextmanager
def operation(stage: str, *, model: str = "", provider: str = "", **context: Any):
    record = {
        "operation_id": uuid4().hex,
        "stage": stage,
        "model": str(model),
        "provider": str(provider),
        "cache_hit": False,
        "status": "completed",
        **context,
    }
    token = _operation.set(record)
    context_token = _context.set({**(_context.get() or {}), **context})
    started = monotonic()
    emit_event("started")
    try:
        yield record
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        emit_event(
            "finished",
            duration=monotonic() - started,
            **{
                key: value
                for key, value in record.items()
                if key not in {"operation_id", "stage", "model", "provider"}
            },
        )
        _context.reset(context_token)
        _operation.reset(token)


def mark_cached() -> None:
    if current := _operation.get():
        current["cache_hit"] = True


def mark_failed(error: str) -> None:
    if current := _operation.get():
        current.update(status="failed", error=error)


def timed_operation(stage: str, *, model: str = "", provider: str = ""):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with operation(stage, model=model, provider=provider):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def retry_wait(delay: float, *, attempt: int, max_attempts: int, reason: str) -> None:
    emit_event("retry_wait", delay=delay, attempt=attempt, max_attempts=max_attempts, reason=reason)


def track_run_progress(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        callback = kwargs.get("operation_callback")
        path = None
        started = False
        if kwargs.get("persist_results", True):
            directory = Path(kwargs.get("output_dir") or "output/interaction_graph")
            path = directory / f"{kwargs.get('output_prefix', 'simulation')}_events.jsonl"

        def receive(event):
            nonlocal started
            started = started or event["kind"] == "started"
            if path is not None and started:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            if callback is not None:
                callback(event)
            elif kwargs.get("progress_callback") is not None:
                from misinformation_simulation.simulation.progress import event_message

                if message := event_message(event):
                    kwargs["progress_callback"](message)

        with observe_events(receive):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                emit_event("run_failed", error=str(exc))
                raise

    return wrapped
