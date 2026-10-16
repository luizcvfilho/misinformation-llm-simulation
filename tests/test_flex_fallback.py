from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event

import httpx
import pytest
from openai import OpenAI, RateLimitError

from misinformation_simulation.llm import flex_fallback, retry
from misinformation_simulation.llm.credentials import ProviderAPIKeys, use_api_credentials
from misinformation_simulation.llm.flex_fallback import (
    FlexFallback,
    get_flex_fallback,
    use_flex_fallback,
)
from misinformation_simulation.llm.progress import observe_events
from misinformation_simulation.llm.rate_limit import SharedRequestLimiter, use_shared_limiter
from misinformation_simulation.llm.recovery import PermissionStore, RecoverySession, use_recovery
from misinformation_simulation.llm.service_tier import use_openai_service_tier
from misinformation_simulation.llm.usage import UsageTracker, collect_usage


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setenv("LLM_ERROR_RECOVERY", "off")
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "auto")


def generate(client, **kwargs):
    return retry.generate_openai_text_with_retry(
        client,
        model="gpt-6-luna",
        prompt="News",
        system_instruction="Rewrite",
        temperature=0,
        max_attempts=1,
        **kwargs,
    )


def response(tier):
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-6-luna",
            "service_tier": tier,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
    )


def unavailable(code="flex_unavailable"):
    return httpx.Response(
        429,
        headers={"retry-after": "300"},
        json={
            "error": {"code": code, "type": "resource_unavailable", "message": "Unavailable"},
        },
    )


def client_for(handle, *, max_retries=0):
    return OpenAI(
        api_key="test-key",
        max_retries=max_retries,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


def test_capacity_fallback_bypasses_shared_cooldown_and_tracks_actual_tiers(tmp_path):
    import json

    calls, events = [], []

    def handle(request):
        tier = json.loads(request.content)["service_tier"]
        calls.append((tier, request.extensions["timeout"]["read"]))
        return unavailable() if len(calls) == 1 else response(tier)

    def unexpected_recovery(request):
        pytest.fail(f"Flex capacity fallback must not request recovery: {request}")

    session = RecoverySession(unexpected_recovery, store=PermissionStore(tmp_path / "permissions"))
    limiter = SharedRequestLimiter(None)
    tracker = UsageTracker()
    with (
        client_for(handle, max_retries=2) as client,
        use_api_credentials({"chatgpt": ProviderAPIKeys("test-key")}),
        use_openai_service_tier("flex"),
        use_flex_fallback(True) as controller,
        use_recovery(session),
        use_shared_limiter(limiter),
        observe_events(events.append),
        collect_usage(tracker),
    ):
        assert generate(client) == "Done"
        assert generate(client) == "Done"
        assert controller.snapshot()["temporary_standard"] is True
    assert [tier for tier, _ in calls] == ["flex", "default", "default"]
    assert calls[0][1] == 900
    assert calls[1][1] != 900
    assert limiter.cooldown_until == 0
    assert not any(event["kind"] in {"rate_limit_wait", "retry_wait"} for event in events)
    assert sum(event["kind"] == "flex_fallback_started" for event in events) == 1
    assert [record["requested_service_tier"] for record in tracker.records] == [
        "flex",
        "default",
        "default",
    ]
    assert tracker.records[-1]["service_tier"] == "default"
    assert tracker.records[-1]["estimated_cost_usd"] == pytest.approx(0.000015)
    assert tracker.records[0]["estimated_cost_usd"] == 0
    assert tracker.summary()["estimated_cost_usd"] == pytest.approx(0.00003)


@pytest.mark.parametrize(
    "code", ["rate_limit_exceeded", "insufficient_quota", "resource_unavailable"]
)
def test_other_429_errors_do_not_enable_fallback(code):
    with (
        client_for(lambda _: unavailable(code)) as client,
        use_openai_service_tier("flex"),
        use_flex_fallback(True) as controller,
        pytest.raises(RateLimitError),
    ):
        try:
            generate(client)
        finally:
            assert controller.snapshot()["temporary_standard"] is False


def test_disabled_fallback_preserves_flex():
    with (
        client_for(lambda _: unavailable()) as client,
        use_openai_service_tier("flex"),
        use_flex_fallback(),
        pytest.raises(RateLimitError),
    ):
        generate(client)


def test_standard_failure_propagates_without_retrying_flex():
    import json

    calls = []

    def handle(request):
        calls.append(json.loads(request.content)["service_tier"])
        return unavailable() if len(calls) == 1 else unavailable("rate_limit_exceeded")

    with (
        client_for(handle) as client,
        use_openai_service_tier("flex"),
        use_flex_fallback(True),
        pytest.raises(RateLimitError) as error,
    ):
        generate(client)
    assert error.value.code == "rate_limit_exceeded"
    assert calls == ["flex", "default"]


def test_unattended_fallback_disables_sdk_retries():
    import json

    calls = []

    def handle(request):
        tier = json.loads(request.content)["service_tier"]
        calls.append(tier)
        if len(calls) == 1:
            rejected = unavailable()
            rejected.headers["retry-after"] = "0.001"
            return rejected
        return response(tier)

    with (
        client_for(handle, max_retries=2) as client,
        use_openai_service_tier("flex"),
        use_flex_fallback(True),
    ):
        assert generate(client) == "Done"
    assert calls == ["flex", "default"]


def test_single_probe_with_concurrent_requests_continuing_in_standard(monkeypatch):
    import json

    now = [100.0]
    monkeypatch.setattr(flex_fallback, "monotonic", lambda: now[0])
    probe_started, release_probe = Event(), Event()
    calls = []

    def handle(request):
        tier = json.loads(request.content)["service_tier"]
        calls.append(tier)
        if len(calls) == 1:
            return unavailable()
        if tier == "flex":
            probe_started.set()
            assert release_probe.wait(5)
        return response(tier)

    with (
        client_for(handle) as client,
        use_openai_service_tier("flex"),
        use_flex_fallback(True),
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        assert generate(client) == "Done"
        # All worker contexts inherit the same controller, as in the graph scheduler.
        now[0] += 300
        probe = pool.submit(copy_context().run, generate, client)
        try:
            assert probe_started.wait(5)
            others = [pool.submit(copy_context().run, generate, client) for _ in range(2)]
            assert [future.result(timeout=5) for future in others] == ["Done", "Done"]
            assert calls == ["flex", "default", "flex", "default", "default"]
        finally:
            release_probe.set()
        assert probe.result(timeout=5) == "Done"
        assert generate(client) == "Done"
        assert calls[-1] == "flex"


def test_failed_probe_restarts_window_and_late_responses_cannot_change_it(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(flex_fallback, "monotonic", lambda: now[0])
    controller = FlexFallback(300)
    first, late_success, late_error = (controller.select() for _ in range(3))
    controller.unavailable(first)
    now[0] += 200
    controller.unavailable(late_error)
    controller.finish(late_success, succeeded=True)
    assert controller.snapshot()["retry_in_seconds"] == 100
    now[0] += 100
    probe = controller.select()
    assert probe.probe
    controller.unavailable(probe)
    controller.finish(probe, succeeded=False)
    assert controller.snapshot()["retry_in_seconds"] == 300
    assert controller.select().tier == "default"
    now[0] += 300
    next_probe = controller.select()
    controller.finish(next_probe, succeeded=True)
    assert controller.select().tier == "flex"


def test_probe_exception_releases_probe_and_execution_context_is_restored(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(flex_fallback, "monotonic", lambda: now[0])
    with use_flex_fallback(True) as controller:
        controller.unavailable(controller.select())
        now[0] += 300
        probe = controller.select()
        controller.finish(probe, succeeded=False)
        assert controller.select().probe
    assert get_flex_fallback() is None
    with use_flex_fallback(True) as next_controller:
        assert next_controller.select().tier == "flex"


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_interval_is_rejected(seconds):
    with pytest.raises(ValueError, match="interval"):
        FlexFallback(seconds)
