from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from hashlib import sha256
from threading import Barrier
from types import SimpleNamespace

import pytest

from misinformation_simulation.llm.rate_limit import SharedRequestLimiter, use_shared_limiter
from misinformation_simulation.llm.recovery import (
    ErrorPermitted,
    PermissionStore,
    RecoveryDecision,
    RecoverySession,
    RequestContext,
    error_signature,
    use_recovery,
)
from misinformation_simulation.text_metrics import llm_vad


def response(quote):
    return json.dumps(
        dict(
            valence=3,
            arousal=2,
            dominance=3,
            rationales=dict(valence="A calm tone.", arousal="Low intensity.", dominance="Neutral."),
            evidence=dict(valence=[quote], arousal=[], dominance=[]),
            ambiguities=[],
        ),
        ensure_ascii=False,
    )


class Client:
    def __init__(self, outcomes, *, api_key="first", calls=None):
        self.outcomes = outcomes
        self.api_key = api_key
        self._recovery_explicit_key = "first"
        self.base_url = "https://api.openai.com/v1"
        self.calls = [] if calls is None else calls
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, *, api_key, max_retries):
        assert max_retries == 0
        return Client(self.outcomes, api_key=api_key, calls=self.calls)

    def create(self, **kwargs):
        self.calls.append((self.api_key, kwargs["messages"][1]["content"]))
        value = self.outcomes[self.api_key].popleft()
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value))])


def install(monkeypatch, outcomes):
    monkeypatch.setenv("CHATGPT_API_KEY_FALLBACK", "second")
    monkeypatch.delenv("OPENAI_API_KEY_FALLBACK", raising=False)
    client = Client({key: deque(values) for key, values in outcomes.items()})
    monkeypatch.setattr(llm_vad, "create_llm_client", lambda **_: ("chatgpt", client))
    return client


def recovery(tmp_path, handler):
    return RecoverySession(handler, store=PermissionStore(tmp_path / "permissions.json"))


@pytest.mark.parametrize("quote", ["calm ... report", "calmer", '"calm"', "", None, 42])
def test_invalid_evidence_remains_strict_and_identifies_dimension(quote):
    with pytest.raises(ValueError, match=r"evidence.valence\[0\]"):
        llm_vad.parse_llm_vad_assessment(response(quote), "A calm report.")


def test_different_invalid_quotes_keep_equivalent_recovery_signature():
    signatures = []
    for quote in ("wrong quote", "a different wrong quote"):
        with pytest.raises(llm_vad.VADEvidenceValidationError) as error:
            llm_vad.parse_llm_vad_assessment(response(quote), "A calm report.")
        signatures.append(error_signature(error.value, RequestContext("chatgpt", "test")))
    assert signatures[0] == signatures[1]


def test_interactive_vad_corrects_quotes_on_same_key_before_prompt(tmp_path, monkeypatch):
    client = install(monkeypatch, {"first": [response("calmer"), response("calm")]})
    session = recovery(tmp_path, lambda _: pytest.fail("Unexpected recovery popup"))
    limiter = SharedRequestLimiter(None)
    admissions = []
    original_acquire = limiter.acquire

    def acquire(provider):
        admissions.append(provider)
        original_acquire(provider)

    monkeypatch.setattr(limiter, "acquire", acquire)
    with use_recovery(session), use_shared_limiter(limiter):
        scorer = llm_vad.LLMVADScorer(model="test", provider="chatgpt")
        result = scorer.assess("A calm report.")
    assert result.evidence["valence"] == ["calm"]
    assert [key for key, _ in client.calls] == ["first", "first"]
    assert admissions == ["chatgpt", "chatgpt"]
    assert "Validation feedback" not in client.calls[0][1]
    assert "evidence.valence[0]" in client.calls[1][1]
    assert "calmer" in client.calls[1][1]
    assert scorer.last_attempts[0]["raw_response"] == response("calmer")
    assert scorer.last_attempts[1]["status"] == "valid"
    assert len({item["input_sha256"] for item in scorer.last_attempts}) == 1
    assert len({item["prompt_sha256"] for item in scorer.last_attempts}) == 2


def test_alignment_avoids_regeneration_and_keeps_raw_quote_and_audit(tmp_path, monkeypatch):
    raw = response("CALM")
    client = install(monkeypatch, {"first": [raw]})
    session = recovery(tmp_path, lambda _: pytest.fail("Unexpected recovery popup"))
    with use_recovery(session):
        scorer = llm_vad.LLMVADScorer(model="test", provider="chatgpt")
        assessment = scorer.assess("A calm report.")
    assert len(client.calls) == 1
    assert assessment.evidence["valence"] == ["calm"]
    assert scorer.last_attempts[0]["raw_response"] == raw
    assert scorer.last_attempts[0]["evidence_alignments"] == assessment.evidence_alignments
    assert "validation_error" not in scorer.last_attempts[0]


def test_exhausted_validation_prompts_after_four_calls_and_preserves_failure(tmp_path, monkeypatch):
    client = install(monkeypatch, {"first": [response("wrong")] * 4})
    prompts = []

    def decide(request):
        assert len(client.calls) == 4
        prompts.append(request)
        return RecoveryDecision("allow")

    with use_recovery(recovery(tmp_path, decide)), pytest.raises(ErrorPermitted):
        scorer = llm_vad.LLMVADScorer(model="test", provider="chatgpt", retry_attempts=10)
        scorer.assess("A calm report.")
    assert len(prompts) == 1
    assert [key for key, _ in client.calls] == ["first"] * 4
    assert len(scorer.last_attempts) == 4
    assert all("validation_error" in item for item in scorer.last_attempts)
    assert not any(item.get("status") == "valid" for item in scorer.last_attempts)


def test_manual_retry_keeps_credential_selection_after_validation_budget(tmp_path, monkeypatch):
    client = install(
        monkeypatch,
        {
            "first": [response("wrong")] * 4,
            "second": [response("calm")],
        },
    )
    prompts = []

    def decide(request):
        prompts.append(request)
        return RecoveryDecision("retry", "secondary")

    with use_recovery(recovery(tmp_path, decide)):
        result = llm_vad.LLMVADScorer(model="test", provider="chatgpt").assess("A calm report.")
    assert result.evidence["valence"] == ["calm"]
    assert [key for key, _ in client.calls] == ["first"] * 4 + ["second"]
    assert len(prompts) == 1


def test_transport_failure_is_not_automatically_retried_as_bad_evidence(tmp_path, monkeypatch):
    client = install(monkeypatch, {"first": [RuntimeError("Unavailable")]})
    with use_recovery(recovery(tmp_path, lambda _: RecoveryDecision("allow"))):
        with pytest.raises(ErrorPermitted):
            llm_vad.LLMVADScorer(model="test", provider="chatgpt").assess("A calm report.")
    assert len(client.calls) == 1


def test_noninteractive_retries_also_receive_validation_feedback(monkeypatch):
    monkeypatch.setenv("LLM_ERROR_RECOVERY", "off")
    monkeypatch.setattr("misinformation_simulation.llm.retry.time.sleep", lambda _: None)
    client = install(monkeypatch, {"first": [response("wrong"), response("calm")]})
    result = llm_vad.LLMVADScorer(model="test", provider="chatgpt").assess("A calm report.")
    assert result.evidence["valence"] == ["calm"]
    assert "Validation feedback" in client.calls[1][1]


def test_parallel_vad_feedback_and_input_hashes_stay_with_their_news(tmp_path, monkeypatch):
    barrier = Barrier(3)
    monkeypatch.setattr(llm_vad, "create_llm_client", lambda **_: ("gemini", object()))
    calls = {}

    def generate(client, *, prompt, **kwargs):
        text, _ = json.JSONDecoder().raw_decode(prompt.split("Text (JSON string):\n", 1)[1])
        attempts = calls.setdefault(text, [])
        attempts.append(prompt)
        if len(attempts) == 1:
            barrier.wait(timeout=3)
            return response("a quote from a different news item")
        assert "Validation feedback" in prompt
        return response(text)

    monkeypatch.setattr(llm_vad, "generate_gemini_text_with_retry", generate)

    def assess(index):
        text = f"Unique news text {index}."
        session = RecoverySession(
            lambda _: pytest.fail("Unexpected recovery popup"),
            store=PermissionStore(tmp_path / f"permissions_{index}.json"),
        )
        with use_recovery(session):
            scorer = llm_vad.LLMVADScorer(model="test", provider="gemini")
            result = scorer.assess(text)
        assert result.evidence["valence"] == [text]
        assert all(
            item["input_sha256"] == sha256(text.encode()).hexdigest()
            for item in scorer.last_attempts
        )
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(copy_context().run, assess, index) for index in range(3)]
        for future in futures:
            future.result(timeout=5)
    assert len(calls) == 3
    assert all(len(attempts) == 2 for attempts in calls.values())
