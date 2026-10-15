import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest
from openai import OpenAI

from misinformation_simulation.llm import recovery, retry
from misinformation_simulation.llm.credentials import ProviderAPIKeys, use_api_credentials
from misinformation_simulation.llm.service_tier import (
    configured_openai_service_tier,
    openai_request_options,
    use_openai_service_tier,
)
from misinformation_simulation.llm.usage import UsageTracker, collect_usage


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "auto")
    monkeypatch.setenv("LLM_ERROR_RECOVERY", "off")
    monkeypatch.setattr(retry, "get_recovery", lambda: None)
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)


def completion(*, tier="flex"):
    return {
        "id": "test-response",
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
    }


def generate(client, **kwargs):
    return retry.generate_openai_text_with_retry(
        client,
        model="gpt-6-luna",
        prompt="Test input",
        system_instruction="Test instruction",
        temperature=0.0,
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["auto", "default", "flex"])
def test_real_sdk_sends_selected_tier_and_flex_timeout(mode):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    with (
        OpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client,
        use_openai_service_tier(mode),
    ):
        assert generate(client) == "Done"
    body = json.loads(requests[0].content)
    assert body.get("service_tier") == (None if mode == "auto" else mode)
    assert "temperature" not in body
    if mode == "flex":
        assert requests[0].extensions["timeout"]["read"] == 900.0


def test_flex_retries_capacity_errors_without_changing_tier_and_records_actual_cost():
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Resource unavailable",
                        "type": "resource_unavailable",
                        "code": "resource_unavailable",
                    }
                },
            )
        # Estimate from the actual response tier, even if it differs from the request.
        return httpx.Response(200, json=completion(tier="default"))

    tracker = UsageTracker()
    with (
        OpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client,
        use_openai_service_tier("flex"),
        collect_usage(tracker),
    ):
        assert generate(client, max_attempts=2) == "Done"
    assert [body["service_tier"] for body in bodies] == ["flex", "flex"]
    assert [record["status"] for record in tracker.records] == ["error", "response"]
    record = tracker.records[-1]
    assert record["requested_service_tier"] == "flex"
    assert record["service_tier"] == "default"
    assert record["estimated_cost_usd"] == pytest.approx(0.000015)


def test_credential_failover_preserves_flex(monkeypatch):
    monkeypatch.setenv("LLM_ERROR_RECOVERY", "auto")
    monkeypatch.setattr(recovery, "_automatic_session", None)
    requests = []

    def handle(request):
        requests.append((request.headers["authorization"], json.loads(request.content)))
        if len(requests) == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Quota exhausted",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
            )
        return httpx.Response(200, json=completion())

    with (
        OpenAI(
            api_key="primary",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client,
        use_api_credentials({"chatgpt": ProviderAPIKeys("primary", "secondary")}),
        use_openai_service_tier("flex"),
    ):
        assert generate(client) == "Done"
    assert [auth for auth, _ in requests] == ["Bearer primary", "Bearer secondary"]
    assert [body["service_tier"] for _, body in requests] == ["flex", "flex"]


@pytest.mark.parametrize("provider", ["openrouter", "deepseek", "grok", "local", "unknown"])
def test_non_openai_providers_do_not_receive_openai_options(provider):
    with use_openai_service_tier("flex"):
        assert openai_request_options(provider) == {}


def test_environment_override_and_context_restore_after_error(monkeypatch):
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "flex")
    with pytest.raises(RuntimeError):
        with use_openai_service_tier("default"):
            assert configured_openai_service_tier() == "default"
            raise RuntimeError("Test error")
    assert configured_openai_service_tier() == "flex"
    with pytest.raises(ValueError, match="service tier"):
        with use_openai_service_tier("invalid"):
            pytest.fail("Invalid modes must be rejected before sending a request")


def test_concurrent_runs_keep_independent_modes():
    barrier = Barrier(2)

    def work(tier):
        with use_openai_service_tier(tier):
            barrier.wait(timeout=5)
            return configured_openai_service_tier()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(work, ["default", "flex"])) == ["default", "flex"]
    assert configured_openai_service_tier() == "auto"
