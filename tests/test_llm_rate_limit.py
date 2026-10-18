from __future__ import annotations

import pytest

from misinformation_simulation.llm import rate_limit
from misinformation_simulation.llm.rate_limit import (
    MinuteRateLimiter,
    SharedRequestLimiter,
    request_limits_from_settings,
    use_shared_limiter,
)
from misinformation_simulation.llm.usage import api_request


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


@pytest.mark.parametrize("provider, evaluator", [("gemini", "chatgpt"), ("chatgpt", "gemini")])
def test_scoped_rpm_counts_retries_without_delaying_other_providers(
    monkeypatch, provider, evaluator
):
    now = 120.0
    waits = []
    limiter = SharedRequestLimiter({provider: 12})

    def advance(*, timeout):
        nonlocal now
        waits.append(timeout)
        now = round(now + timeout, 6)

    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now)
    monkeypatch.setattr(limiter.condition, "wait", advance)
    with use_shared_limiter(limiter):
        with pytest.raises(RuntimeError), api_request(provider=provider, model="rewrite"):
            raise RuntimeError("Retry this request")
        for _ in range(3):
            with api_request(provider=evaluator, model="evaluate"):
                pass
        assert waits == []
        assert limiter.next_requests == {provider: 125.0}
        with api_request(provider=provider, model="rewrite"):
            assert now == pytest.approx(125.0)
    assert sum(waits) == pytest.approx(5.0)
    assert limiter.next_requests == {provider: pytest.approx(130.0)}


@pytest.mark.parametrize("rpm", [None, 0])
def test_scoped_rpm_can_be_disabled(monkeypatch, rpm):
    limiter = SharedRequestLimiter({"gemini": rpm})
    waits = []
    monkeypatch.setattr(limiter.condition, "wait", lambda **kwargs: waits.append(kwargs))
    for provider in ["gemini", "chatgpt", "gemini", "gemini"]:
        limiter.acquire(provider)
    assert waits == []


def test_providers_use_independent_rpm_budgets(monkeypatch):
    now = 120.0
    waits = []
    limiter = SharedRequestLimiter({"gemini": 12, "chatgpt": 60, "grok": 0})

    def advance(*, timeout):
        nonlocal now
        waits.append(timeout)
        now = round(now + timeout, 6)

    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: now)
    monkeypatch.setattr(limiter.condition, "wait", advance)
    limiter.acquire("gemini")
    limiter.acquire("chatgpt")
    for provider in ["grok", "deepseek", "grok"]:
        limiter.acquire(provider)
    assert waits == []
    assert limiter.next_requests["gemini"] == 125.0
    assert limiter.next_requests["chatgpt"] == 121.0
    limiter.acquire("chatgpt")
    assert now == 121.0
    assert limiter.next_requests["gemini"] == 125.0
    assert limiter.next_requests["chatgpt"] == 122.0
    limiter.acquire("gemini")
    assert now == 125.0
    assert limiter.next_requests["gemini"] == 130.0
    assert limiter.next_requests["chatgpt"] == 122.0
    limiter.acquire("chatgpt")
    assert now == 125.0
    assert limiter.next_requests["chatgpt"] == 126.0
    assert sum(waits) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, None),
        ({"max_requests_per_minute": 12}, 12),
        ({"max_requests_per_minute": 12, "rpm_provider": "gemini"}, {"gemini": 12}),
        ({"provider_rpm_limits": {"gemini": 12, "chatgpt": 60}}, {"gemini": 12, "chatgpt": 60}),
        ({"max_requests_per_minute": 12, "provider_rpm_limits": {}}, {}),
    ],
)
def test_request_limits_restore_provider_budgets_and_legacy_settings(settings, expected):
    assert request_limits_from_settings(settings) == expected
