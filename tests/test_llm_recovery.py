from __future__ import annotations

from collections import deque
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace

import httpx
import pytest
from google.genai.errors import ClientError
from openai import RateLimitError

from misinformation_simulation.apps.interaction_graph_recovery import RecoveryBroker
from misinformation_simulation.llm import clients
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.recovery import (
    ErrorPermitted,
    PermissionStore,
    RecoveryCancelled,
    RecoveryDecision,
    RecoveryRequest,
    RecoverySession,
    RequestContext,
    error_signature,
    openai_resource_exhausted,
    use_recovery,
)
from misinformation_simulation.llm.retry import (
    generate_and_parse_with_retry,
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)


def quota(code="insufficient_quota"):
    return RateLimitError(
        "Not enough resources",
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
        body={"code": code, "type": "insufficient_quota"},
    )


class FakeOpenAI:
    def __init__(self, outcomes, calls=None, api_key="first", base_url="https://api.openai.com/v1"):
        self.outcomes = outcomes
        self.calls = [] if calls is None else calls
        self.api_key = api_key
        self.base_url = base_url
        self._recovery_explicit_key = "first"
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, *, api_key, max_retries):
        assert max_retries == 0
        return FakeOpenAI(self.outcomes, self.calls, api_key, self.base_url)

    def create(self, **kwargs):
        self.calls.append((self.api_key, kwargs))
        result = self.outcomes[self.api_key].popleft()
        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict):
            return SimpleNamespace(**result)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])


def generate(client):
    return generate_openai_text_with_retry(
        client,
        model="test",
        prompt="unchanged input",
        system_instruction="system",
        temperature=0.1,
    )


@pytest.fixture
def recovery_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATGPT_API_KEY_FALLBACK", "second")
    monkeypatch.delenv("OPENAI_API_KEY_FALLBACK", raising=False)
    sessions = []

    def create(decisions=()):
        prompts = []
        choices = iter(decisions)

        def handler(request):
            prompts.append(request)
            return next(choices)

        session = RecoverySession(handler, store=PermissionStore(tmp_path / "permissions.json"))
        monkeypatch.setattr(session, "_read_file_keys", lambda: {})
        session._file_keys = {}
        sessions.append(session)
        return session, prompts

    return create


def test_failover_works_both_directions_and_remembers_active_key(recovery_factory):
    session, prompts = recovery_factory()
    client = FakeOpenAI(
        {"first": deque([quota(), "primary restored"]), "second": deque(["backup result", quota()])}
    )
    with use_recovery(session):
        assert generate(client) == "backup result"
        assert generate(client) == "primary restored"
    assert [key for key, _ in client.calls] == ["first", "second", "second", "first"]
    assert all(kwargs["messages"][1]["content"] == "unchanged input" for _, kwargs in client.calls)
    assert prompts == []


def test_manual_retry_tries_selected_key_then_other_even_for_different_errors(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("retry", "secondary")])
    client = FakeOpenAI(
        {
            "first": deque([quota(), "restored"]),
            "second": deque([quota(), RuntimeError("different failure")]),
        }
    )
    with use_recovery(session):
        assert generate(client) == "restored"
    assert [key for key, _ in client.calls] == ["first", "second", "second", "first"]
    assert len(prompts) == 1
    assert prompts[0].context.choices == ("primary", "secondary")


def test_manual_retry_failure_always_prompts_again(recovery_factory):
    session, prompts = recovery_factory(
        [
            RecoveryDecision("retry", "primary"),
            RecoveryDecision("allow"),
        ]
    )
    client = FakeOpenAI({"first": deque([quota(), quota()]), "second": deque([quota(), quota()])})
    with use_recovery(session), pytest.raises(ErrorPermitted):
        generate(client)
    assert len(prompts) == 2
    assert len(client.calls) == 4


def test_allowed_both_key_errors_persist_but_do_not_hide_different_error(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("allow")])
    client = FakeOpenAI({"first": deque([quota(), quota()]), "second": deque([quota(), quota()])})
    with use_recovery(session):
        for _ in range(2):
            with pytest.raises(ErrorPermitted):
                generate(client)
    assert len(prompts) == 1
    new_session, new_prompts = recovery_factory([RecoveryDecision("allow")])
    client.outcomes["first"].append(RuntimeError("a different error"))
    with use_recovery(new_session), pytest.raises(ErrorPermitted):
        generate(client)
    assert len(new_prompts) == 1
    new_session.store.clear()
    assert new_session.store.signatures() == set()


@pytest.mark.parametrize(
    "code, expected",
    [
        ("credit_balance_exhausted", True),
        ("insufficient_quota", True),
        ("rate_limit_exceeded", False),
        ("slow_down", False),
        ("project_spend_limit_exceeded", False),
        ("organization_spend_limit_exceeded", False),
        ("organization_usage_limit_exceeded", False),
    ],
)
def test_resource_detection_does_not_confuse_explicit_limits(code, expected):
    assert openai_resource_exhausted(quota(code)) is expected


def test_non_resource_error_does_not_rotate_before_user_retry(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("allow")])
    client = FakeOpenAI({"first": deque([quota("rate_limit_exceeded")]), "second": deque()})
    with use_recovery(session), pytest.raises(ErrorPermitted):
        generate(client)
    assert len(prompts) == len(client.calls) == 1


def test_validation_errors_retry_generation_without_duplicate_dialogs(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("retry", "primary")])
    client = FakeOpenAI({"first": deque(["bad json", "valid"]), "second": deque()})

    def parse(text):
        if text != "valid":
            raise ValueError("Invalid JSON")
        return text

    with use_recovery(session):
        assert generate_and_parse_with_retry(lambda: generate(client), parse) == "valid"
    assert len(prompts) == 1
    assert len(client.calls) == 2


def test_permitted_transport_error_is_not_retried_by_validation_layer(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("allow")])
    client = FakeOpenAI({"first": deque([RuntimeError("broken")]), "second": deque()})
    with use_recovery(session), pytest.raises(ErrorPermitted):
        generate_and_parse_with_retry(lambda: generate(client), lambda text: text)
    assert len(prompts) == len(client.calls) == 1


def test_manual_retry_tries_other_key_after_invalid_response(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("retry", "primary")])
    client = FakeOpenAI({"first": deque(["invalid", "invalid"]), "second": deque(["valid"])})

    def parse(value):
        if value != "valid":
            raise ValueError("Invalid response")
        return value

    with use_recovery(session):
        assert generate_and_parse_with_retry(lambda: generate(client), parse) == "valid"
    assert [key for key, _ in client.calls] == ["first", "first", "second"]
    assert len(prompts) == 1


def test_failover_remembers_slot_across_new_clients(recovery_factory):
    session, prompts = recovery_factory()
    outcomes = {"first": deque([quota()]), "second": deque(["one", "two"])}
    first = FakeOpenAI(outcomes)
    second = FakeOpenAI(outcomes)
    with use_recovery(session):
        assert generate(first) == "one"
        assert generate(second) == "two"
    assert [key for key, _ in second.calls] == ["second"]
    assert not prompts


@pytest.mark.parametrize(
    "provider, endpoint",
    [
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("deepseek", "https://api.deepseek.com"),
        ("grok", "https://api.x.ai/v1"),
    ],
)
def test_other_openai_compatible_providers_defer_any_error_to_user(
    recovery_factory,
    provider,
    endpoint,
):
    session, prompts = recovery_factory([RecoveryDecision("retry"), RecoveryDecision("allow")])
    client = FakeOpenAI(
        {"first": deque([RuntimeError("broken"), RuntimeError("broken")])}, base_url=endpoint
    )
    with use_recovery(session), pytest.raises(ErrorPermitted):
        generate(client)
    assert len(prompts) == len(client.calls) == 2
    assert all(prompt.context.provider == provider for prompt in prompts)


def test_gemini_uses_native_errors_and_remembers_permission(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("allow")])

    def broken(**kwargs):
        raise ClientError(402, {"error": {"status": "PAYMENT_REQUIRED", "message": "No credit"}})

    client = SimpleNamespace(models=SimpleNamespace(generate_content=broken))
    with use_recovery(session):
        for _ in range(2):
            with pytest.raises(ErrorPermitted):
                generate_gemini_text_with_retry(
                    client,
                    model="gemini",
                    prompt="input",
                    system_instruction="system",
                    temperature=0,
                )
    assert len(prompts) == 1
    assert prompts[0].context.provider == "gemini"


def test_http_200_error_envelope_is_shown_instead_of_choices_error(recovery_factory):
    session, prompts = recovery_factory([RecoveryDecision("allow")])
    client = FakeOpenAI(
        {"first": deque([{"error": {"code": 402, "message": "No credit"}}])},
        base_url="https://openrouter.ai/api/v1",
    )
    with use_recovery(session), pytest.raises(ErrorPermitted, match="No credit"):
        generate(client)
    assert "choices" not in prompts[0].message


def test_signatures_ignore_request_ids_but_keep_distinct_causes_and_slots():
    first = RequestContext("chatgpt", "model", "primary")
    second = RequestContext("chatgpt", "model", "secondary")
    assert error_signature(RuntimeError("bad req_123"), first) == error_signature(
        RuntimeError("bad req_456"),
        first,
    )
    assert error_signature(quota(), first) != error_signature(quota(), second)
    assert error_signature(RuntimeError("bad parameter"), first) != error_signature(
        RuntimeError("bad model"),
        first,
    )


def test_secrets_are_not_in_requests_permissions_or_permitted_errors(recovery_factory, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-123")
    session, prompts = recovery_factory([RecoveryDecision("allow")])

    def broken():
        raise RuntimeError("Failed with secret-value-123 and Bearer private-token")

    with pytest.raises(ErrorPermitted) as captured:
        session.run(broken)
    contents = str(captured.value) + prompts[0].message + session.store.path.read_text()
    assert "secret-value-123" not in contents
    assert "private-token" not in contents


def test_client_creation_error_can_be_fixed_without_restarting(recovery_factory, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    session, prompts = recovery_factory()

    def fix(request):
        prompts.append(request)
        monkeypatch.setenv("OPENROUTER_API_KEY", "new-key")
        return RecoveryDecision("retry")

    session.handler = fix
    with use_recovery(session):
        _, client = create_llm_client(provider="openrouter")
    try:
        assert client.api_key == "new-key"
        assert client.max_retries == 0
        assert len(prompts) == 1
    finally:
        client.close()


def test_existing_client_uses_corrected_environment_key_on_retry(recovery_factory, monkeypatch):
    calls = []

    class SDK:
        def __init__(self, *, api_key, **kwargs):
            self.api_key = api_key
            self.base_url = "https://api.deepseek.com"
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls.append(self.api_key)
            if self.api_key == "broken-key":
                raise RuntimeError("Invalid key")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="fixed"),
                    )
                ]
            )

    monkeypatch.setattr(clients, "OpenAI", SDK)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "broken-key")
    session, prompts = recovery_factory()

    def fix(request):
        prompts.append(request)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "corrected-key")
        return RecoveryDecision("retry")

    session.handler = fix
    with use_recovery(session):
        _, client = create_llm_client(provider="deepseek")
        assert generate(client) == "fixed"
    assert calls == ["broken-key", "corrected-key"]
    assert len(prompts) == 1


def test_changed_dotenv_values_reload_but_explicit_environment_wins(recovery_factory, monkeypatch):
    session, _ = recovery_factory()
    session._file_keys = {"CHATGPT_API_KEY": "old-file", "GEMINI_API_KEY": "old-google"}
    monkeypatch.setenv("CHATGPT_API_KEY", "old-file")
    monkeypatch.setenv("GEMINI_API_KEY", "explicit-environment")
    monkeypatch.setattr(
        session,
        "_read_file_keys",
        lambda: {
            "CHATGPT_API_KEY": "new-file",
            "GEMINI_API_KEY": "new-google",
        },
    )
    session.reload_keys()
    assert clients.os.getenv("CHATGPT_API_KEY") == "new-file"
    assert clients.os.getenv("GEMINI_API_KEY") == "explicit-environment"


def test_corrupt_permission_file_does_not_authorize_errors(tmp_path):
    store = PermissionStore(tmp_path / "permissions.json")
    for invalid in ("not json", '{"' + "a" * 64 + '": true}', '[null, 2, "short"]'):
        store.path.write_text(invalid)
        assert store.signatures() == set()


def test_unattended_openai_calls_still_try_other_credential(monkeypatch):
    from misinformation_simulation.llm import recovery

    monkeypatch.setenv("LLM_ERROR_RECOVERY", "auto")
    monkeypatch.setenv("CHATGPT_API_KEY_FALLBACK", "second")
    monkeypatch.setattr(recovery, "_automatic_session", None)
    monkeypatch.setattr(recovery.sys.stdin, "isatty", lambda: False)
    client = FakeOpenAI({"first": deque([quota()]), "second": deque(["success"])})
    assert generate(client) == "success"
    assert [key for key, _ in client.calls] == ["first", "second"]


def wait_for_request(broker):
    deadline = monotonic() + 2
    while broker.pending is None and monotonic() < deadline:
        sleep(0.01)
    assert broker.pending is not None


def test_broker_rejects_stale_duplicate_decisions_and_resumes_worker():
    broker = RecoveryBroker(Event())
    request = RecoveryRequest("one", RequestContext(), "error", ("signature",))
    decisions = []
    thread = Thread(target=lambda: decisions.append(broker.ask(request)))
    thread.start()
    try:
        wait_for_request(broker)
        assert not broker.submit("old", RecoveryDecision("allow"))
        assert broker.submit("one", RecoveryDecision("retry"))
        assert not broker.submit("one", RecoveryDecision("allow"))
    finally:
        broker.cancel_event.set()
        thread.join(2)
    assert not thread.is_alive()
    assert decisions == [RecoveryDecision("retry")]


def test_cancel_wakes_worker_waiting_for_decision():
    broker = RecoveryBroker(Event())
    request = RecoveryRequest("one", RequestContext(), "error", ())
    errors = []

    def work():
        try:
            broker.ask(request)
        except RecoveryCancelled as error:
            errors.append(error)

    thread = Thread(target=work)
    thread.start()
    wait_for_request(broker)
    broker.cancel_event.set()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert broker.pending is None


def test_graph_boundary_does_not_prompt_twice_for_allowed_llm_error(tmp_path):
    from misinformation_simulation.apps.interaction_graph_run_job import (
        GraphRunJob,
        _execute_graph,
    )

    job = GraphRunJob()
    job.recovery_session.store = PermissionStore(tmp_path / "permissions.json")
    errors = []

    def broken():
        raise RuntimeError("Provider failure")

    def work():
        with use_recovery(job.recovery_session):
            try:
                _execute_graph(
                    job,
                    lambda: job.recovery_session.run(
                        broken,
                        context=RequestContext("deepseek", "test"),
                    ),
                )
            except ErrorPermitted as error:
                errors.append(error)

    thread = Thread(target=work)
    thread.start()
    try:
        wait_for_request(job.recovery)
        request = job.recovery.pending
        assert request.context.provider == "deepseek"
        assert job.recovery.submit(request.id, RecoveryDecision("allow"))
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert job.recovery.pending is None
    finally:
        job.cancel_event.set()
        thread.join(2)
