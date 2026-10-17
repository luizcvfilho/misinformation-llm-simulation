from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Lock
from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError

from misinformation_simulation.llm import clients, gemini_keys
from misinformation_simulation.llm.credentials import (
    GEMINI_EXTRA_KEY_NAMES,
    ProviderAPIKeys,
    configured_gemini_keys,
    use_api_credentials,
)
from misinformation_simulation.llm.gemini_keys import (
    GeminiKeyPool,
    daily_quota_exhausted,
    next_daily_reset,
    quota_retry_delay,
)
from misinformation_simulation.llm.rate_limit import SharedRequestLimiter, use_shared_limiter
from misinformation_simulation.llm.recovery import (
    AttemptFailures,
    PermissionStore,
    RecoveryCancelled,
    RecoveryDecision,
    RecoverySession,
    safe_message,
    use_recovery,
)
from misinformation_simulation.llm.retry import generate_gemini_text_with_retry
from misinformation_simulation.llm.usage import UsageTracker, collect_usage


def quota_error(*, daily=True, delay="7s"):
    return ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": "GenerateRequestsPerDay"
                                if daily
                                else "GenerateRequestsPerMinute",
                            }
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                ],
            }
        },
    )


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for name in ["GEMINI_API_KEY", *(n for v in GEMINI_EXTRA_KEY_NAMES.values() for n in v)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_ERROR_RECOVERY", "off")


@pytest.fixture
def fake_sdk(monkeypatch):
    calls, closed = [], []

    def responses(key, kwargs):
        return SimpleNamespace(text="done")

    class FakeClient:
        def __init__(self, *, api_key, **kwargs):
            self.api_key = api_key
            self.options = kwargs
            self.models = SimpleNamespace(generate_content=self.generate)

        def generate(self, **kwargs):
            calls.append((self.api_key, kwargs))
            return responses(self.api_key, kwargs)

        def close(self):
            closed.append(self.api_key)

    def configure(function):
        nonlocal responses
        responses = function

    monkeypatch.setattr(clients.genai, "Client", FakeClient)
    return calls, closed, configure


def generate(client, **kwargs):
    return generate_gemini_text_with_retry(
        client,
        model="gemini-3.5-flash-lite",
        prompt="same message",
        system_instruction="same persona",
        temperature=0.8,
        **kwargs,
    )


def keys():
    return {
        "gemini": ProviderAPIKeys(
            primary="project-one-key",
            secondary="project-two-key",
            tertiary="project-three-key",
        )
    }


def test_three_project_daily_failover_completes_1200_rewrites_without_recovery(tmp_path, fake_sdk):
    calls, closed, configure = fake_sdk
    counts = Counter()

    def respond(key, kwargs):
        counts[key] += 1
        if counts[key] > 500:
            raise quota_error()
        return SimpleNamespace(text="done")

    configure(respond)
    usage = UsageTracker()
    session = RecoverySession(
        lambda request: pytest.fail("A remaining project quota must avoid recovery"),
        store=PermissionStore(tmp_path / "permissions.json"),
    )
    with (
        use_api_credentials(keys()),
        use_recovery(session),
        use_shared_limiter(SharedRequestLimiter(None)),
        collect_usage(usage),
    ):
        _, client = clients.create_llm_client(provider="gemini")
        for _ in range(1200):
            assert generate(client) == "done"
    assert counts == {"project-one-key": 501, "project-two-key": 501, "project-three-key": 200}
    assert len(calls) == 1202
    assert len(closed) == 701
    assert all(call[1]["contents"] == "same message" for call in calls)
    assert all(call[1]["config"].temperature == 0.8 for call in calls)
    assert len([r for r in usage.records if r.get("kind") == "request"]) == 1202


def test_environment_keys_explicit_primary_and_duplicate_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-one")
    monkeypatch.setenv("GEMINI_API_KEY_2", "environment-two")
    monkeypatch.setenv("GEMINI_API_KEY_3", "environment-three")
    with use_api_credentials({"gemini": ProviderAPIKeys(secondary="session-two")}):
        assert configured_gemini_keys("explicit-one") == {
            "primary": "explicit-one",
            "secondary": "session-two",
            "tertiary": "environment-three",
        }
        assert configured_gemini_keys("session-two") == {
            "primary": "session-two",
            "tertiary": "environment-three",
        }


def test_all_daily_quotas_open_recovery_once_with_three_choices(tmp_path, fake_sdk):
    calls, _, configure = fake_sdk
    configure(lambda key, kwargs: (_ for _ in ()).throw(quota_error()))
    prompts = []

    def recover(request):
        prompts.append(request)
        return RecoveryDecision("cancel")

    with (
        use_api_credentials(keys()),
        use_recovery(RecoverySession(recover, store=PermissionStore(tmp_path / "permissions"))),
        use_shared_limiter(SharedRequestLimiter(None)),
    ):
        _, client = clients.create_llm_client(provider="gemini")
        with pytest.raises(RecoveryCancelled):
            generate(client)
    assert len(calls) == 3
    assert len(prompts) == 1
    assert prompts[0].context.choices == ("primary", "secondary", "tertiary")
    assert all(slot in prompts[0].message for slot in prompts[0].context.choices)


def test_manual_retry_can_select_third_key_after_all_daily_failures(tmp_path, fake_sdk):
    calls, _, configure = fake_sdk
    ready = False

    def respond(key, kwargs):
        if ready and key == "project-three-key":
            return SimpleNamespace(text="recovered")
        raise quota_error()

    def recover(request):
        nonlocal ready
        ready = True
        return RecoveryDecision("retry", "tertiary")

    configure(respond)
    with (
        use_api_credentials(keys()),
        use_recovery(RecoverySession(recover, store=PermissionStore(tmp_path / "permissions"))),
    ):
        _, client = clients.create_llm_client(provider="gemini")
        assert generate(client) == "recovered"
    assert [key for key, _ in calls] == [
        "project-one-key",
        "project-two-key",
        "project-three-key",
        "project-three-key",
    ]


@pytest.mark.parametrize("failure", ["bad_request", "empty_response"])
def test_no_failover_for_bad_request_or_validation_failure(fake_sdk, failure):
    calls, _, configure = fake_sdk

    def respond(key, kwargs):
        if failure == "bad_request":
            raise ClientError(400, {"message": "bad input"})
        return SimpleNamespace(text="")

    configure(respond)
    with use_api_credentials(keys()):
        _, client = clients.create_llm_client(provider="gemini")
        with pytest.raises((ClientError, ValueError)):
            generate(client, max_attempts=1)
    assert [key for key, _ in calls] == ["project-one-key"]


def test_pool_exhaustion_is_shared_across_parallel_news_workers(fake_sdk):
    calls, _, configure = fake_sdk
    gate = Lock()
    counts = Counter()

    def respond(key, kwargs):
        with gate:
            counts[key] += 1
            exceeded = counts[key] > 5
        if exceeded:
            raise quota_error()
        return SimpleNamespace(text="done")

    def work():
        _, client = clients.create_llm_client(provider="gemini")
        return generate(client, max_attempts=1)

    configure(respond)
    with use_api_credentials(keys()), ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(copy_context().run, work) for _ in range(14)]
        assert [f.result(timeout=5) for f in futures] == ["done"] * 14
    assert counts["project-three-key"] == 4
    assert sum(min(value, 5) for value in counts.values()) == 14
    assert len(calls) >= 16


def test_temporary_quota_cycle_is_bounded_without_interactive_recovery(fake_sdk):
    calls, _, configure = fake_sdk
    configure(lambda key, kwargs: (_ for _ in ()).throw(quota_error(daily=False)))
    with use_api_credentials(keys()):
        _, client = clients.create_llm_client(provider="gemini")
        with pytest.raises(AttemptFailures):
            generate(client, max_attempts=1)
    assert [key for key, _ in calls] == [
        "project-one-key",
        "project-two-key",
        "project-three-key",
    ]


def test_quota_wait_can_be_cancelled(tmp_path):
    session = RecoverySession(
        lambda request: pytest.fail("Cancellation must not open recovery"),
        store=PermissionStore(tmp_path / "permissions"),
        cancel_check=lambda: True,
    )
    pool = GeminiKeyPool()
    available = {"primary": "one", "secondary": "two"}
    with use_recovery(session), pytest.raises(RecoveryCancelled):
        pool.select(pool.identity(available, "model"), available)


def test_temporary_quota_wait_and_pacific_daily_reset(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(gemini_keys.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        gemini_keys.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    monkeypatch.setattr(gemini_keys, "next_daily_reset", lambda: 1000.0)
    pool = GeminiKeyPool()
    available = {"primary": "one", "secondary": "two"}
    identity = pool.identity(available, "model")
    from misinformation_simulation.llm.recovery import RequestContext

    for slot in available:
        pool.failed(
            identity, slot, RequestContext("gemini", "model", slot), quota_error(daily=False)
        )
    assert pool.select(identity, available) in available
    assert 107 <= clock[0] < 108
    for slot in available:
        pool.failed(identity, slot, RequestContext("gemini", "model", slot), quota_error())
    with pytest.raises(AttemptFailures):
        pool.select(identity, available)
    clock[0] = 1000
    assert pool.select(identity, available) in available


def test_quota_classification_redaction_and_real_reset_timezone():
    assert daily_quota_exhausted(quota_error())
    assert not daily_quota_exhausted(quota_error(daily=False))
    assert quota_retry_delay(quota_error(daily=False)) == 7
    from datetime import datetime
    from zoneinfo import ZoneInfo

    reset = datetime.fromtimestamp(next_daily_reset(), ZoneInfo("America/Los_Angeles"))
    assert (reset.hour, reset.minute, reset.second) == (0, 0, 0)
    with use_api_credentials(keys()):
        for value in configured_gemini_keys().values():
            assert value not in repr(keys()) + safe_message(value)


def test_late_temporary_error_cannot_shorten_an_existing_daily_hold(monkeypatch):
    from misinformation_simulation.llm.recovery import RequestContext

    monkeypatch.setattr(gemini_keys, "next_daily_reset", lambda: gemini_keys.time.time() + 3600)
    pool = GeminiKeyPool()
    available = {"primary": "one", "secondary": "two"}
    identity = pool.identity(available, "model")
    context = RequestContext("gemini", "model", "primary")
    pool.failed(identity, "primary", context, quota_error())
    pool.failed(identity, "primary", context, quota_error(daily=False))
    assert pool.blocked[(identity, "primary")][1] is True
    assert pool.select(identity, available) == "secondary"
