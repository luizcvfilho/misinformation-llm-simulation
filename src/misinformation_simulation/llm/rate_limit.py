from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition

from misinformation_simulation.llm.progress import emit_event

_shared: ContextVar[SharedRequestLimiter | None] = ContextVar(
    "shared_request_limiter", default=None
)


class SharedRequestLimiter:
    """Share independent provider RPM budgets, or a legacy global budget."""

    def __init__(self, rpm: int | Mapping[str, int | None] | None, cancel_check=lambda: False):
        limits = rpm if isinstance(rpm, Mapping) else {None: rpm}
        self.intervals = {
            provider: 60.0 / limit if limit else 0.0 for provider, limit in limits.items()
        }
        self.cancel_check = cancel_check
        self.condition = Condition()
        self.queue = deque()
        self.next_requests: dict[str | None, float] = {}
        self.cooldown_until = 0.0
        self.gates: dict[str, int] = {}

    def block(self, provider: str):
        with self.condition:
            self.gates[provider] = self.gates.get(provider, 0) + 1

    def unblock(self, provider: str):
        with self.condition:
            self.gates[provider] = max(0, self.gates.get(provider, 0) - 1)
            self.condition.notify_all()

    def cooldown(self, seconds: float):
        with self.condition:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
            self.condition.notify_all()

    def _request_delay(self, provider: str, now: float) -> float:
        next_request = self.next_requests.get(provider, self.next_requests.get(None, 0.0))
        return max(next_request, self.cooldown_until) - now

    def acquire(self, provider: str):
        from misinformation_simulation.llm.recovery import RecoveryCancelled

        ticket = object()
        with self.condition:
            self.queue.append((ticket, provider))
            announced = False
            try:
                while True:
                    from misinformation_simulation.llm.recovery import get_recovery

                    if session := get_recovery():
                        session.check_cancelled()
                    if self.cancel_check():
                        raise RecoveryCancelled("Execution cancelled while waiting for API quota.")
                    now = time.monotonic()
                    eligible = next(
                        (
                            item
                            for item in self.queue
                            if not self.gates.get(item[1])
                            and self._request_delay(item[1], now) <= 0
                        ),
                        None,
                    )
                    delay = self._request_delay(provider, now)
                    if eligible and eligible[0] is ticket and delay <= 0:
                        budget = provider if provider in self.intervals else None
                        if budget in self.intervals:
                            self.next_requests[budget] = time.monotonic() + self.intervals[budget]
                        return
                    if not announced:
                        emit_event("rate_limit_wait", delay=max(0, delay), provider=provider)
                        announced = True
                    self.condition.wait(timeout=min(max(delay, 0.01), 0.1))
            finally:
                self.queue.remove((ticket, provider))
                self.condition.notify_all()


def request_limits_from_settings(settings: Mapping) -> int | Mapping[str, int | None] | None:
    if (limits := settings.get("provider_rpm_limits")) is not None:
        return limits
    rpm = int(settings.get("max_requests_per_minute", 0)) or None
    if provider := settings.get("rpm_provider"):
        return {provider: rpm}
    return rpm


@contextmanager
def use_shared_limiter(limiter: SharedRequestLimiter):
    token = _shared.set(limiter)
    try:
        yield
    finally:
        _shared.reset(token)


def get_shared_limiter():
    return _shared.get()


def shared_request_slot(provider: str):
    if (limiter := _shared.get()) is not None and provider != "local":
        limiter.acquire(provider)


class MinuteRateLimiter:
    def __init__(self, max_requests_per_minute: int | None) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.current_minute_bucket = int(time.time() // 60)
        self.requests_in_current_minute = 0

    def acquire(self) -> None:
        if _shared.get() is not None:
            return
        if self.max_requests_per_minute is None:
            return

        now_bucket = int(time.time() // 60)
        if now_bucket != self.current_minute_bucket:
            self.current_minute_bucket = now_bucket
            self.requests_in_current_minute = 0

        if self.requests_in_current_minute >= self.max_requests_per_minute:
            seconds_until_next_minute = 60 - (time.time() % 60)
            emit_event("rate_limit_wait", delay=seconds_until_next_minute)
            time.sleep(seconds_until_next_minute)
            self.current_minute_bucket = int(time.time() // 60)
            self.requests_in_current_minute = 0

        self.requests_in_current_minute += 1
