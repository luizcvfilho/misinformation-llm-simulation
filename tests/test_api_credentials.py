from __future__ import annotations

from threading import Barrier, Event, Thread

import httpx
import pandas as pd
import pytest

from misinformation_simulation.apps import interaction_graph_run_job as run_job
from misinformation_simulation.llm import clients
from misinformation_simulation.llm.credentials import (
    API_KEY_NAMES,
    OPENAI_FALLBACK_NAMES,
    ProviderAPIKeys,
    configured_api_key,
    configured_openai_fallback,
    use_api_credentials,
)
from misinformation_simulation.llm.recovery import RecoverySession, safe_message, use_recovery
from misinformation_simulation.llm.retry import generate_openai_text_with_retry


@pytest.fixture(autouse=True)
def empty_environment(monkeypatch):
    for names in [*API_KEY_NAMES.values(), OPENAI_FALLBACK_NAMES]:
        for name in names:
            monkeypatch.delenv(name, raising=False)


class FakeSDK:
    def __init__(self, *, api_key, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs


@pytest.mark.parametrize("provider", ["chatgpt", "gemini", "openrouter", "deepseek", "grok"])
def test_ui_credentials_reach_each_sdk_without_changing_environment(monkeypatch, provider):
    monkeypatch.setattr(clients, "OpenAI", FakeSDK)
    monkeypatch.setattr(clients.genai, "Client", FakeSDK)
    with use_api_credentials({provider: ProviderAPIKeys(primary="session-primary")}):
        _, client = clients.create_llm_client(provider=provider)
        assert client.api_key == "session-primary"
        assert all(clients.os.getenv(name) is None for name in API_KEY_NAMES[provider])
    assert configured_api_key(provider) is None


def test_explicit_key_has_precedence_over_session_key(monkeypatch):
    monkeypatch.setattr(clients, "OpenAI", FakeSDK)
    with use_api_credentials({"chatgpt": ProviderAPIKeys("session-primary", "session-backup")}):
        _, client = clients.create_llm_client(provider="chatgpt", api_key="explicit-key")
        assert client.api_key == "explicit-key"
        assert clients.openai_credentials(client) == {
            "primary": "explicit-key",
            "secondary": "session-backup",
        }


def test_ui_fallback_combines_with_environment_primary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-primary")
    monkeypatch.setattr(clients, "OpenAI", FakeSDK)
    with use_api_credentials({"chatgpt": ProviderAPIKeys(fallback="session-backup")}):
        _, client = clients.create_llm_client(provider="chatgpt")
        assert clients.openai_credentials(client) == {
            "primary": "environment-primary",
            "secondary": "session-backup",
        }
    assert configured_openai_fallback() is None


def test_ui_openai_keys_override_environment_and_restore_after_execution(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-primary")
    monkeypatch.setenv("OPENAI_API_KEY_FALLBACK", "environment-fallback")
    monkeypatch.setattr(clients, "OpenAI", FakeSDK)
    with use_api_credentials({"chatgpt": ProviderAPIKeys("session-primary", "session-fallback")}):
        _, client = clients.create_llm_client(provider="chatgpt")
        assert client.api_key == "session-primary"
        assert clients.openai_credentials(client) == {
            "primary": "session-primary",
            "secondary": "session-fallback",
        }
        assert clients.os.getenv("OPENAI_API_KEY") == "environment-primary"
        assert clients.os.getenv("OPENAI_API_KEY_FALLBACK") == "environment-fallback"
    assert configured_api_key("chatgpt") == "environment-primary"
    assert configured_openai_fallback() == "environment-fallback"


def test_context_is_restored_even_after_an_exception(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    with use_api_credentials({"gemini": ProviderAPIKeys("outer-key")}):
        with (
            pytest.raises(ValueError, match="failure"),
            use_api_credentials(
                {
                    "gemini": ProviderAPIKeys("inner-key"),
                }
            ),
        ):
            assert configured_api_key("gemini") == "inner-key"
            raise ValueError("failure")
        assert configured_api_key("gemini") == "outer-key"
    assert configured_api_key("gemini") == "environment-key"


def test_concurrent_sessions_cannot_use_each_others_keys():
    barrier = Barrier(2)
    observed = {}

    def run(name):
        with use_api_credentials({"chatgpt": ProviderAPIKeys(name, f"{name}-backup")}):
            barrier.wait(timeout=3)
            observed[name] = (configured_api_key("chatgpt"), configured_openai_fallback())

    threads = [Thread(target=run, args=(name,)) for name in ("first-session", "second-session")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(4)
        assert not thread.is_alive()
    assert observed == {name: (name, f"{name}-backup") for name in observed}
    assert len(observed) == 2
    assert configured_api_key("chatgpt") is None


def test_job_receives_immutable_snapshot_of_credentials(monkeypatch):
    started, release = Event(), Event()
    observed = []

    def work(**kwargs):
        started.set()
        assert release.wait(3)
        observed.append((configured_api_key("gemini"), configured_openai_fallback()))

    monkeypatch.setattr(run_job, "_run_graph_queue", work)
    supplied = {
        "gemini": ProviderAPIKeys("ui-gemini"),
        "chatgpt": ProviderAPIKeys(fallback="ui-backup"),
    }
    job = run_job.start_graph_run_job(
        df=pd.DataFrame(),
        graphs=[],
        settings={},
        runner=lambda **kwargs: None,
        queue_mode=False,
        api_credentials=supplied,
    )
    try:
        assert started.wait(3)
        supplied["gemini"] = ProviderAPIKeys("changed-after-start")
        release.set()
        job.thread.join(4)
        assert not job.thread.is_alive()
        assert observed == [("ui-gemini", "ui-backup")]
        assert configured_api_key("gemini") is None
    finally:
        release.set()
        job.thread.join(4)


def test_session_credentials_are_hidden_in_representations_and_error_messages():
    keys = ProviderAPIKeys("session-secret-primary", "session-secret-backup")
    with use_api_credentials({"chatgpt": keys}):
        text = repr(keys) + safe_message("session-secret-primary / session-secret-backup")
        assert "session-secret-primary" not in text
        assert "session-secret-backup" not in text


def test_openai_sdk_uses_ui_keys_for_failover_in_both_directions(monkeypatch):
    sdk = clients.OpenAI
    seen = []

    def respond(request):
        seen.append(request.headers["authorization"])
        if len(seen) in {1, 3}:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "No funds",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def factory(**kwargs):
        return sdk(http_client=httpx.Client(transport=httpx.MockTransport(respond)), **kwargs)

    def unexpected_prompt(request):
        pytest.fail("A working backup should avoid the recovery dialog")

    monkeypatch.setattr(clients, "OpenAI", factory)
    with (
        use_api_credentials({"chatgpt": ProviderAPIKeys("ui-primary", "ui-backup")}),
        use_recovery(RecoverySession(unexpected_prompt)),
    ):
        _, client = clients.create_llm_client(provider="chatgpt")
        try:
            for _ in range(2):
                assert (
                    generate_openai_text_with_retry(
                        client,
                        model="test",
                        prompt="input",
                        system_instruction="system",
                        temperature=0,
                    )
                    == "done"
                )
        finally:
            client.close()
    assert seen == [
        "Bearer ui-primary",
        "Bearer ui-backup",
        "Bearer ui-backup",
        "Bearer ui-primary",
    ]
