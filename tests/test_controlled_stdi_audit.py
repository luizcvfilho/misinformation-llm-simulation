from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from misinformation_simulation.topic_drift.semantic_comparison import JudgeValidationError

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "controlled_stdi_audit", ROOT / "scripts/audit_stdi_information_changes.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_audit_extraction_retries_invalid_response_before_checkpoint(tmp_path, monkeypatch):
    from misinformation_simulation.llm import retry
    from misinformation_simulation.topic_drift import extraction

    response = json.dumps(
        {
            "schema_version": 2,
            "main_topic": "Negotiations",
            "subtopics": [],
            "central_entities": ["union"],
            "central_relations": [],
            "opinions": [],
            "has_internal_contradiction": False,
            "internal_contradiction_score": 0.0,
        }
    )
    responses = iter(["invalid JSON", response])
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(extraction, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(extraction, "generate_openai_text_with_retry", generate)
    monkeypatch.setattr(retry.time, "sleep", lambda delay: None)
    job = {"job_id": "test-job", "text": "The union requested talks.", "replicate": 1}
    args = Namespace(output_dir=tmp_path, model="test", provider="chatgpt", structured=True)
    audit.extract_jobs([job], args, "test-configuration")
    checkpoint = json.loads((tmp_path / "extractions/test-job.json").read_text())
    assert len(calls) == 2
    assert checkpoint["structure"]["extraction_status"] == "valid"
    assert checkpoint["structure"]["provenance"]["raw_response"] == response


@pytest.mark.parametrize("judge_fails", [True, False])
def test_judge_checkpoints_preserve_failure_and_zero_scores(tmp_path, monkeypatch, judge_fails):
    structure = {"main_topic": "Negotiations", "central_relations": []}
    rows = [
        {
            "pair_id": "government_identity",
            "scenario_id": "government",
            "change_type": "identity",
            "comparison_kind": "original_vs_rewrite",
            "original_text": "The union requested talks.",
            "modified_text": "The union requested talks.",
            "original_structure_json": json.dumps(structure),
            "modified_structure_json": json.dumps(structure),
            "stdi": "0.0",
        }
    ]
    audit.write_csv(tmp_path / "scored_pairs.csv", rows)
    audit.write_json(
        tmp_path / "vad_scores.json",
        {rows[0]["original_text"]: {"valence": 3.0, "arousal": 3.0, "dominance": 3.0}},
    )
    calls = []

    def request(**kwargs):
        calls.append(kwargs)
        if judge_fails:
            raise JudgeValidationError(
                "Invalid component score", {"raw_response": "invalid score response"}
            )
        return SimpleNamespace(
            component_drifts=dict.fromkeys(
                (*audit.DEFAULT_STDI_WEIGHTS, "contradiction_drift"), 0.0
            ),
            rationales={},
            provenance={},
        )

    monkeypatch.setattr(audit, "compare_stdi_components_semantically", request)
    args = Namespace(output_dir=tmp_path, model="test", provider="chatgpt", structured=True)
    audit.score_semantic_comparisons(args)
    audit.score_semantic_comparisons(args)
    assert len(calls) == 1
    checkpoint = json.loads(
        (tmp_path / "semantic_comparisons/government_identity.json").read_text()
    )
    with (tmp_path / "semantic_method_comparison.csv").open(newline="") as handle:
        result = next(csv.DictReader(handle))
    if judge_fails:
        assert result["stdi"] == result["dual_stdi"] == result["method_gap"] == ""
        assert result["comparison_status"] == "failed"
        assert checkpoint["provenance"]["raw_response"] == "invalid score response"
    else:
        assert float(result["stdi"]) == float(result["dual_stdi"]) == 0.0
        assert result["comparison_status"] == "valid"


def test_new_prompt_configuration_cannot_overwrite_an_existing_audit(tmp_path, monkeypatch):
    target = tmp_path / "extraction_configuration.json"
    target.write_text('{"model": "previous-model"}', encoding="utf-8")
    original = target.read_bytes()
    monkeypatch.setattr(audit, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--structured",
            "--stage",
            "prepare",
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(ValueError, match="Choose a new output directory"):
        audit.main()
    assert target.read_bytes() == original
    assert not (tmp_path / "input_pairs.csv").exists()
