"""Execution-scoped Gemini credential failover shared by concurrent news runs."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from threading import RLock
from zoneinfo import ZoneInfo

from misinformation_simulation.llm.progress import emit_event

_pool: ContextVar[GeminiKeyPool | None] = ContextVar("gemini_key_pool", default=None)


def daily_quota_exhausted(error: Exception) -> bool:
    details = json.dumps(getattr(error, "details", {}), default=str).lower()
    message = str(getattr(error, "message", error)).lower()
    return any(
        marker in details + message
        for marker in ("perday", "per_day", "per day", "daily", "permodelperday")
    ) and is_quota_error(error)


def is_quota_error(error: Exception) -> bool:
    return str(getattr(error, "code", "")) == "429" or (getattr(error, "status_code", None) == 429)


def next_daily_reset() -> float:
    pacific = ZoneInfo("America/Los_Angeles")
    now = datetime.now(UTC).astimezone(pacific)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def quota_retry_delay(error: Exception) -> float:
    details = getattr(error, "details", {})
    if isinstance(details, dict):
        body = details.get("error", details)
        for detail in body.get("details", []) if isinstance(body, dict) else []:
            if isinstance(detail, dict) and "retryDelay" in detail:
                try:
                    return max(1.0, float(str(detail["retryDelay"]).removesuffix("s")))
                except (ValueError, TypeError):
                    pass
    headers = getattr(getattr(error, "response", None), "headers", {})
    try:
        return max(1.0, float(headers.get("retry-after", 60)))
    except (ValueError, TypeError):
        return 60.0


class GeminiKeyPool:
    def __init__(self):
        self.lock = RLock()
        self.active = {}
        self.blocked = {}

    @staticmethod
    def identity(keys, model):
        return model, tuple(
            (slot, hashlib.sha256(value.encode()).hexdigest()) for slot, value in keys.items()
        )

    def select(self, identity, keys, preferred=None):
        from misinformation_simulation.llm.recovery import AttemptFailures, get_recovery

        announced = False
        while True:
            if session := get_recovery():
                session.check_cancelled()
            with self.lock:
                now = time.time()
                first = preferred or self.active.get(identity, "primary")
                order = [first] if first in keys else []
                order.extend(slot for slot in keys if slot not in order)
                failures, waits = [], []
                for slot in order:
                    blocked = self.blocked.get((identity, slot))
                    if blocked is None or blocked[0] <= now:
                        self.active[identity] = slot
                        return slot
                    until, daily, context, error = blocked
                    failures.append((context, error))
                    if not daily:
                        waits.append(until - now)
                if not waits:
                    raise AttemptFailures(failures)
                delay = min(waits)
            if not announced:
                emit_event("rate_limit_wait", provider="gemini", delay=delay)
                announced = True
            time.sleep(min(delay, 0.1))

    def failed(self, identity, slot, context, error):
        daily = daily_quota_exhausted(error)
        until = next_daily_reset() if daily else time.time() + quota_retry_delay(error)
        with self.lock:
            previous = self.blocked.get((identity, slot))
            if previous is None or previous[0] <= until:
                self.blocked[(identity, slot)] = until, daily, context, error
        emit_event(
            "credential_failed",
            provider="gemini",
            model=context.model,
            credential=slot,
            quota="daily" if daily else "temporary",
        )

    def reset(self, identity, slot):
        with self.lock:
            self.blocked.pop((identity, slot), None)


_automatic_pool = GeminiKeyPool()


@contextmanager
def use_gemini_key_pool():
    token = _pool.set(GeminiKeyPool())
    try:
        yield
    finally:
        _pool.reset(token)


@contextmanager
def _request_client(client, key):
    from misinformation_simulation.llm.clients import _create_llm_client

    replacement = None
    try:
        if key != getattr(client, "_recovery_resolved_key", None):
            _, replacement = _create_llm_client(
                provider="gemini",
                api_key=key,
                base_url=None,
                max_retries=0,
            )
        yield replacement if replacement is not None else client
    finally:
        if replacement is not None:
            replacement.close()


def gemini_call(client, function, *, model):
    from misinformation_simulation.llm.clients import refresh_llm_client
    from misinformation_simulation.llm.credentials import configured_gemini_keys
    from misinformation_simulation.llm.recovery import (
        AttemptFailures,
        RequestContext,
        get_recovery,
        manual_credential,
        record_credential_attempt,
        register_secret,
        set_request_context,
    )

    client = refresh_llm_client(client)
    configuration = getattr(client, "_recovery_configuration", {})
    # Externally supplied SDK clients retain their existing single-client behavior.
    keys = configured_gemini_keys(configuration.get("api_key")) if configuration else {}
    for value in keys.values():
        register_secret(value)
    if len(keys) < 2:
        set_request_context(RequestContext("gemini", model))
        return function(client)
    pool = _pool.get() or _automatic_pool
    identity = pool.identity(keys, model)
    preferred = manual_credential()
    if preferred in keys:
        pool.reset(identity, preferred)
    failures = []
    for _ in keys:
        try:
            slot = pool.select(identity, keys, preferred)
        except AttemptFailures as error:
            set_request_context(error.failures[-1][0])
            raise
        preferred = None
        context = RequestContext("gemini", model, slot, tuple(keys))
        set_request_context(context)
        try:
            with _request_client(client, keys[slot]) as request_client:
                if session := get_recovery():
                    session.check_cancelled()
                result = function(request_client)
        except Exception as error:
            record_credential_attempt(context, error)
            if not is_quota_error(error):
                raise
            failures.append((context, error))
            pool.failed(identity, slot, context, error)
        else:
            record_credential_attempt(context, None)
            return result
    raise AttemptFailures(failures)
