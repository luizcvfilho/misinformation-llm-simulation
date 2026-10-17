from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.config.prompts import (
    STRUCTURED_TOPIC_PROMPT_TEMPLATE,
    TOPIC_DRIFT_PROMPT_TEMPLATE,
)
from misinformation_simulation.llm import retry
from misinformation_simulation.llm.progress import observe_events, operation, use_run_identity
from misinformation_simulation.llm.recovery import (
    ErrorPermitted,
    PermissionStore,
    RecoveryDecision,
    RecoverySession,
    RequestContext,
    error_signature,
    use_recovery,
)
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift import extraction, semantic_comparison
from misinformation_simulation.topic_drift.comparison_workflow import run_comparison_workflow
from misinformation_simulation.topic_drift.models import empty_topic_structure


class FakeResponses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.requests = []

    def next(self):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def generate_content(self, **_kwargs):
        self.requests.append(_kwargs)
        return SimpleNamespace(text=self.next())

    def create(self, **_kwargs):
        self.requests.append(_kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.next()))]
        )


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _seconds: None)


def install_client(monkeypatch, operation, provider, responses):
    fake = FakeResponses(responses)
    client = SimpleNamespace(models=fake, chat=SimpleNamespace(completions=fake))
    module = extraction if operation == "extraction" else semantic_comparison

    def create_client(**kwargs):
        assert kwargs["max_retries"] == 0
        return provider, client

    monkeypatch.setattr(module, "create_llm_client", create_client)
    return fake


def valid_payload(operation):
    if operation == "extraction":
        return {
            "schema_version": 2,
            "main_topic": "Agency helps residents",
            "subtopics": [],
            "central_entities": ["Agency"],
            "central_relations": [{"subject": "Agency", "action": "helps", "object": "residents"}],
            "has_internal_contradiction": False,
            "internal_contradiction_score": 0,
            "opinions": [],
        }
    return dict.fromkeys(
        ("theme_drift", "subtopic_drift", "entity_drift", "relation_drift", "contradiction_drift"),
        0.5,
    )


def evaluate(operation, *, structured=True, retry_attempts=4, hook=None):
    kwargs = {
        "model": "test",
        "provider": "chatgpt",
        "structured": structured,
        "retry_attempts": retry_attempts,
        "before_request_hook": hook,
    }
    if operation == "extraction":
        return extraction.extract_topic_structure(text="Agency helps residents.", **kwargs)
    return semantic_comparison.compare_stdi_components_semantically(
        original_text="Agency helps residents.",
        modified_text="Agency refuses to help residents.",
        title=None,
        original_structure=empty_topic_structure(),
        modified_structure=empty_topic_structure(),
        **kwargs,
    )


@pytest.mark.parametrize("operation", ["extraction", "judge"])
@pytest.mark.parametrize("provider", ["chatgpt", "gemini"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError("provider error"), " ", "invalid JSON"])
def test_api_empty_and_parse_failures_recover(
    monkeypatch, operation, provider, structured, failure
):
    raw = json.dumps(valid_payload(operation))
    fake = install_client(monkeypatch, operation, provider, [failure, raw])
    hooks = []

    result = evaluate(operation, structured=structured, hook=lambda: hooks.append("request"))

    assert fake.calls == 2
    assert len(hooks) == 2
    if structured:
        assert result.provenance["raw_response"] == raw


@pytest.mark.parametrize("operation", ["extraction", "judge"])
@pytest.mark.parametrize("provider", ["chatgpt", "gemini"])
def test_api_and_validation_errors_share_capped_budget(monkeypatch, operation, provider):
    raw = json.dumps(valid_payload(operation))
    fake = install_client(
        monkeypatch,
        operation,
        provider,
        [TimeoutError("timeout"), " ", "invalid JSON", raw],
    )

    result = evaluate(operation, retry_attempts=10)

    assert fake.calls == 4
    assert result.provenance["raw_response"] == raw


@pytest.mark.parametrize("operation", ["extraction", "judge"])
@pytest.mark.parametrize("provider", ["chatgpt", "gemini"])
def test_exhausted_validation_preserves_last_response_and_error(monkeypatch, operation, provider):
    last_response = "last invalid response"
    fake = install_client(monkeypatch, operation, provider, ["invalid"] * 3 + [last_response])
    error_type = (
        extraction.ExtractionValidationError
        if operation == "extraction"
        else semantic_comparison.JudgeValidationError
    )

    with pytest.raises(error_type) as error:
        evaluate(operation, retry_attempts=10)

    assert fake.calls == 4
    assert error.value.provenance["raw_response"] == last_response
    assert str(error.value)


@pytest.mark.parametrize("operation", ["extraction", "judge"])
@pytest.mark.parametrize("structured", [False, True])
def test_exhausted_api_failure_preserves_exception(monkeypatch, operation, structured):
    last_error = RuntimeError("last provider failure")
    fake = install_client(monkeypatch, operation, "chatgpt", [last_error] * 4)

    with pytest.raises(RuntimeError) as error:
        evaluate(operation, structured=structured)

    assert fake.calls == 4
    assert error.value is last_error


@pytest.mark.parametrize("operation", ["extraction", "judge"])
def test_smaller_budget_and_invalid_budget(monkeypatch, operation):
    fake = install_client(monkeypatch, operation, "chatgpt", [RuntimeError("failure")] * 2)
    with pytest.raises(RuntimeError, match="failure"):
        evaluate(operation, retry_attempts=2)
    assert fake.calls == 2
    with pytest.raises(ValueError, match="greater than zero"):
        evaluate(operation, retry_attempts=0)
    assert fake.calls == 2


def test_partial_extraction_retries_then_preserves_partial_result(monkeypatch):
    payload = valid_payload("extraction")
    payload["central_relations"] = "invalid list"
    raw = json.dumps(payload)
    fake = install_client(monkeypatch, "extraction", "chatgpt", [raw] * 4)

    result = evaluate("extraction")

    assert fake.calls == 4
    assert result.extraction_status == "partial"
    assert "Invalid list: central_relations" in result.extraction_issues
    assert result.provenance["raw_response"] == raw


def test_partial_extraction_can_recover(monkeypatch):
    payload = valid_payload("extraction")
    incomplete = {**payload, "main_topic": None}
    raw = json.dumps(payload)
    fake = install_client(monkeypatch, "extraction", "gemini", [json.dumps(incomplete), raw])

    result = evaluate("extraction")

    assert fake.calls == 2
    assert result.main_topic == payload["main_topic"]


def test_optional_extraction_warnings_do_not_trigger_retry(monkeypatch):
    raw = json.dumps(valid_payload("extraction"))
    fake = install_client(monkeypatch, "extraction", "chatgpt", [raw])

    result = evaluate("extraction")

    assert fake.calls == 1
    assert result.extraction_issues
    assert result.extraction_status == "valid"


def test_invalid_judge_score_triggers_retry(monkeypatch):
    payload = valid_payload("judge")
    invalid = {**payload, "relation_drift": 2.0}
    fake = install_client(
        monkeypatch, "judge", "chatgpt", [json.dumps(invalid), json.dumps(payload)]
    )

    result = evaluate("judge")

    assert fake.calls == 2
    assert result.component_drifts["relation_drift"] == 0.5


def test_optional_judge_rationales_do_not_trigger_retry(monkeypatch):
    raw = json.dumps(valid_payload("judge"))
    fake = install_client(monkeypatch, "judge", "chatgpt", [raw])

    result = evaluate("judge")

    assert fake.calls == 1
    assert result.warnings


@pytest.mark.parametrize(
    "changes, issue",
    [
        ({"main_topic": None}, "Main topic is unavailable"),
        ({"central_relations": "invalid list"}, "Invalid list: central_relations"),
        ({"internal_contradiction_score": 2}, "Invalid contradiction score"),
        ({"schema_version": 1}, "A version-2 extraction is required"),
    ],
)
def test_interactive_extraction_corrects_required_fields_before_prompt(
    monkeypatch, tmp_path, changes, issue
):
    payload = valid_payload("extraction")
    invalid = json.dumps({**payload, **changes})
    fake = install_client(monkeypatch, "extraction", "chatgpt", [invalid, json.dumps(payload)])
    session = RecoverySession(
        lambda _: pytest.fail("Unexpected recovery dialog"),
        store=PermissionStore(tmp_path / "permissions.json"),
    )
    hooks = []
    with use_recovery(session):
        result = evaluate("extraction", hook=lambda: hooks.append("request"))
    assert fake.calls == len(hooks) == 2
    first, second = [request["messages"][1]["content"] for request in fake.requests]
    assert "Validation feedback" not in first
    assert issue in second
    assert "Do not invent facts" in second
    attempts = result.provenance["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["invalid", "valid"]
    assert attempts[0]["raw_response"] == invalid
    assert attempts[0]["prompt_sha256"] != attempts[1]["prompt_sha256"]
    assert attempts[0]["input_sha256"] == attempts[1]["input_sha256"]


def test_extraction_failure_preserves_full_diagnostics_and_skip_is_not_remembered(
    monkeypatch, tmp_path
):
    payload = {**valid_payload("extraction"), "main_topic": None, "opinions": ["x" * 7000]}
    raw = json.dumps(payload)
    fake = install_client(monkeypatch, "extraction", "chatgpt", [raw] * 4)
    requests = []
    events = []

    def decide(request):
        requests.append(request)
        return RecoveryDecision("skip")

    store = PermissionStore(tmp_path / "permissions.json")
    with (
        use_recovery(RecoverySession(decide, store=store)),
        use_run_identity(run_id="affected_run"),
        observe_events(events.append),
        operation("extraction"),
        pytest.raises(ErrorPermitted) as error,
    ):
        evaluate("extraction", retry_attempts=10)
    assert fake.calls == 4
    assert len(requests) == 1
    assert "Main topic is unavailable" in requests[0].message
    assert requests[0].location["run_id"] == "affected_run"
    assert requests[0].diagnostics["raw_response"] == raw
    assert len(error.value.provenance["attempts"]) == 4
    assert store.signatures() == set()
    failures = [event for event in events if event["kind"] == "validation_failed"]
    assert len(failures) == 4
    assert failures[0]["diagnostics"]["raw_response"] == raw
    assert failures[0]["diagnostics"]["validation_issues"] == ["Main topic is unavailable"]
    assert next(event for event in events if event["kind"] == "recovery_wait")["error"]


def test_extraction_diagnostics_redact_secrets(monkeypatch, tmp_path):
    secret = "test-secret-must-not-be-persisted"
    monkeypatch.setenv("TEST_API_KEY", secret)
    payload = {**valid_payload("extraction"), "opinions": [secret]}
    fake = install_client(monkeypatch, "extraction", "chatgpt", [json.dumps(payload)])
    result = evaluate("extraction")
    assert fake.calls == 1
    assert secret not in json.dumps(result.provenance)
    assert "[redacted]" in result.provenance["raw_response"]


def test_opinion_only_extraction_can_have_no_event_relations(monkeypatch):
    payload = {
        **valid_payload("extraction"),
        "central_relations": [],
        "opinions": ["The policy should change."],
    }
    fake = install_client(monkeypatch, "extraction", "chatgpt", [json.dumps(payload)])
    result = evaluate("extraction")
    assert fake.calls == 1
    assert result.central_relations == []
    assert result.extraction_status == "valid"


def test_structured_prompt_matches_topic_validation_without_changing_legacy_prompt():
    assert "main_topic: string or null" not in STRUCTURED_TOPIC_PROMPT_TEMPLATE
    assert "main_topic: non-empty string" in STRUCTURED_TOPIC_PROMPT_TEMPLATE
    assert "main_topic: string or null" in TOPIC_DRIFT_PROMPT_TEMPLATE


def test_different_extraction_issues_have_different_recovery_permissions():
    context = RequestContext("chatgpt", "test")
    first = extraction.ExtractionValidationError("Main topic is unavailable", {})
    second = extraction.ExtractionValidationError("Invalid list: central_relations", {})
    assert error_signature(first, context) != error_signature(second, context)


@pytest.mark.parametrize("method", ["cluster", "dual"])
def test_invalid_extraction_keeps_later_rewrites_and_available_comparisons(
    monkeypatch, tmp_path, method
):
    from misinformation_simulation.simulation import graph

    payload = valid_payload("extraction")
    valid = json.dumps(payload)
    invalid = json.dumps({**payload, "main_topic": None})
    responses = [valid] + [invalid] * (4 if method == "dual" else 1) + [valid]
    fake = install_client(monkeypatch, "extraction", "chatgpt", responses)
    monkeypatch.setattr(graph, "create_llm_client", lambda **_: ("chatgpt", object()))
    outputs = iter(["First rewritten text", "Second rewritten text"])
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_: next(outputs))
    session = RecoverySession(
        lambda _: RecoveryDecision("skip"), store=PermissionStore(tmp_path / "permissions.json")
    )

    class ConstantEmbedder:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    with use_recovery(session):
        result = graph.run_news_interaction_graph(
            pd.DataFrame([{"title": "Title", "description": "Original text"}]),
            nodes=[
                graph.SimulationNode("first", "test", "chatgpt", "persona"),
                graph.SimulationNode("second", "test", "chatgpt", "persona"),
            ],
            stdi_comparison_method=method,
            stdi_judge_fn=lambda **_: semantic_comparison.SemanticSTDIComparison(
                valid_payload("judge"), {}
            ),
            stdi_embedder=ConstantEmbedder(),
            vad_scorer=lambda _: VADScore(3, 3, 3),
            output_dir=tmp_path / "run",
        )
    assert fake.calls == len(responses)
    assert not result.summary["cancelled"]
    first, second = result.step_results
    assert first.rewritten_text == "First rewritten text"
    assert first.stdi_vs_original is None
    if method == "dual":
        assert first.rewritten_topic_structure_status == "error"
        failed_structure = json.loads(first.metadata["rewritten_json"])
        assert failed_structure["extraction_status"] == "failed"
        assert failed_structure["provenance"]["raw_response"] == invalid
        assert len(failed_structure["provenance"]["attempts"]) == 4
    assert second.rewritten_text == "Second rewritten text"
    assert second.rewritten_topic_structure_status == "success"
    assert second.stdi_vs_original is not None
    if method == "cluster":
        assert first.stdi_status_vs_original == "failed"
        assert second.stdi_status_incremental == "failed"
        assert second.stdi_incremental is None
    assert not session.store.signatures()


@pytest.mark.parametrize("failed_stage", ["extraction", "judge"])
@pytest.mark.parametrize("judge_repeats", [1, 3])
def test_workflow_continues_after_exhausted_retries(
    monkeypatch, tmp_path, failed_stage, judge_repeats
):
    extraction_raw = json.dumps(valid_payload("extraction"))
    judge_raw = json.dumps(valid_payload("judge"))
    extraction_responses = (
        ["invalid extraction"] * 4 + [extraction_raw] * 3
        if failed_stage == "extraction"
        else [extraction_raw] * 4
    )
    judge_responses = (
        ["invalid judgment"] * 4 + [judge_raw] * (2 * judge_repeats - 1)
        if failed_stage == "judge"
        else [judge_raw] * (2 * judge_repeats)
    )
    extractor = install_client(monkeypatch, "extraction", "chatgpt", extraction_responses)
    judge = install_client(monkeypatch, "judge", "chatgpt", judge_responses)

    class ConstantEmbedder:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    result = run_comparison_workflow(
        pd.DataFrame(
            [
                {"original_text": "Original A", "modified_text": "Modified A"},
                {"original_text": "Original B", "modified_text": "Modified B"},
            ]
        ),
        method="dual",
        embedder=ConstantEmbedder(),
        n_clusters=1,
        reuse_structures=False,
        vad_scorer=lambda _text: VADScore(3, 3, 3),
        cache_dir=tmp_path,
        judge_repeats=judge_repeats,
    ).results

    assert result.iloc[0]["comparison_status"] == "partial"
    assert result.iloc[1]["comparison_status"] == "valid"
    if failed_stage == "extraction":
        assert extractor.calls == 7
        details = json.loads(result.iloc[0]["original_json"])
        assert details["extraction_status"] == "failed"
        assert "JSON object" in details["extraction_issues"][0]
        assert details["provenance"]["raw_response"] == "invalid extraction"
    else:
        assert judge.calls == 4 + 2 * judge_repeats - 1
        details = json.loads(result.iloc[0]["dual_evaluation_json"])
        branch = details["llm_judge"]
        assert branch["status"] == ("failed" if judge_repeats == 1 else "partial")
        assert branch["valid_repeats"] == judge_repeats - 1
        assert branch["samples"][0]["provenance"]["raw_response"] == "invalid judgment"
        if judge_repeats == 1:
            assert branch["provenance"]["raw_response"] == "invalid judgment"
