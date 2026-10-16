"""Execution-wide Flex capacity fallback with one recovery probe at a time."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from misinformation_simulation.llm.progress import emit_event

_fallback: ContextVar[FlexFallback | None] = ContextVar("flex_fallback", default=None)


@dataclass(frozen=True)
class FlexAttempt:
    tier: str
    generation: int
    probe: bool = False


class FlexFallback:
    def __init__(self, retry_seconds: float = 300):
        if not math.isfinite(retry_seconds) or retry_seconds <= 0:
            raise ValueError("Flex retry interval must be a positive finite number of seconds.")
        self.retry_seconds = float(retry_seconds)
        self.lock = Lock()
        self.retry_at = 0.0
        self.generation = 0
        self.probing = False

    def select(self) -> FlexAttempt:
        with self.lock:
            probe = bool(self.retry_at and monotonic() >= self.retry_at and not self.probing)
            if probe:
                self.probing = True
            tier = "flex" if not self.retry_at or probe else "default"
            attempt = FlexAttempt(tier, self.generation, probe)
        if probe:
            emit_event("flex_probe", service_tier="flex")
        return attempt

    def unavailable(self, attempt: FlexAttempt) -> None:
        with self.lock:
            # Late failures from calls started before the window must not extend it.
            if attempt.generation != self.generation:
                return
            self.retry_at = monotonic() + self.retry_seconds
            self.generation += 1
            self.probing = False
        emit_event("flex_fallback_started", service_tier="default", delay=self.retry_seconds)

    def finish(self, attempt: FlexAttempt, *, succeeded: bool) -> None:
        if not attempt.probe:
            return
        with self.lock:
            if attempt.generation != self.generation:
                return
            self.probing = False
            if succeeded:
                self.retry_at = 0.0
                self.generation += 1
        if succeeded:
            emit_event("flex_restored", service_tier="flex")

    def snapshot(self) -> dict:
        with self.lock:
            return dict(
                enabled=True,
                retry_seconds=self.retry_seconds,
                temporary_standard=bool(self.retry_at),
                probing=self.probing,
                retry_in_seconds=max(0.0, self.retry_at - monotonic()),
            )


def get_flex_fallback() -> FlexFallback | None:
    return _fallback.get()


def flex_fallback_settings() -> dict:
    controller = get_flex_fallback()
    return dict(
        openai_flex_fallback=controller is not None,
        openai_flex_retry_seconds=controller.retry_seconds if controller else 300.0,
    )


@contextmanager
def use_flex_fallback(enabled: bool = False, retry_seconds: float = 300):
    controller = get_flex_fallback() or (FlexFallback(retry_seconds) if enabled else None)
    token = _fallback.set(controller)
    try:
        yield controller
    finally:
        _fallback.reset(token)


def is_flex_unavailable(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    return getattr(error, "code", None) == "flex_unavailable" or (
        isinstance(body, dict) and body.get("code") == "flex_unavailable"
    )
