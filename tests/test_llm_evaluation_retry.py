from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.llm import retry
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift import extraction, semantic_comparison
from misinformation_simulation.topic_drift.comparison_workflow import run_comparison_workflow
from misinformation_simulation.topic_drift.models import empty_topic_structure


class FakeResponses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def next(self):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def generate_content(self, **_kwargs):
        return SimpleNamespace(text=self.next())

    def create(self, **_kwargs):
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
