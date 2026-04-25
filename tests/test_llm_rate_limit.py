from __future__ import annotations

from misinformation_simulation.llm import rate_limit
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter


def test_rate_limiter_noops_when_limit_is_disabled(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(rate_limit.time, "sleep", sleeps.append)

    limiter = MinuteRateLimiter(None)
    limiter.acquire()
    limiter.acquire()

    assert sleeps == []
    assert limiter.requests_in_current_minute == 0


def test_rate_limiter_sleeps_after_limit_is_reached(monkeypatch) -> None:
    times = iter([120.0, 120.0, 120.0, 120.5, 180.0])
    sleeps: list[float] = []
    monkeypatch.setattr(rate_limit.time, "time", lambda: next(times))
    monkeypatch.setattr(rate_limit.time, "sleep", sleeps.append)

    limiter = MinuteRateLimiter(1)
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [59.5]
    assert limiter.current_minute_bucket == 3
    assert limiter.requests_in_current_minute == 1


def test_rate_limiter_resets_counter_on_new_minute(monkeypatch) -> None:
    times = iter([120.0, 120.0, 181.0])
    monkeypatch.setattr(rate_limit.time, "time", lambda: next(times))

    limiter = MinuteRateLimiter(2)
    limiter.acquire()
    limiter.acquire()

    assert limiter.current_minute_bucket == 3
    assert limiter.requests_in_current_minute == 1
