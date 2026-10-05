from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.types import SimulationNode
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift import extraction, semantic_comparison
from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
from misinformation_simulation.topic_drift.comparison_workflow import (
    run_comparison_workflow,
    topic_structure_from_row,
    write_comparison_output,
)
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    flatten_topic_structure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache
from misinformation_simulation.topic_drift.qualifiers import (
    adjust_relation_distance,
    compare_duration,
)
from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison
from misinformation_simulation.topic_drift.structured_comparison import (
    CONTENT_COMPONENTS,
    StructuredEmbeddingComparator,
    compare_dual_stdi,
    complete_stdi,
    shared_vad_drift,
)


class ConstantEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 2))


def relation(**changes):
    if changes.get("polarity") == "negated" and "negation_scope" not in changes:
        changes["negation_scope"] = "helps"
    return replace(
        TopicRelation(
            "Agency",
            "helps",
            "residents",
            polarity="affirmed",
            predicate="helps",
            signed_action="helps",
            base_action="helps",
            assertion_type="asserted",
            duration_status="absent",
        ),
        **changes,
    )


def structure(*relations):
    return TopicStructure(
        "public services",
        ["assistance"],
        ["Agency", "residents"],
        list(relations),
        schema_version=2,
        extraction_status="valid",
    )


def duration(value, unit="days"):
    return relation(
        duration_status="exact",
        duration_value=value,
        duration_unit=unit,
        duration_expression=f"{value} {unit}",
    )


def judge(**kwargs):
    return SemanticSTDIComparison(
        dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 0.5),
        dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), "Changed claim"),
    )


def compare(original=None, modified=None, **changes):
    original = original or structure(relation())
    modified = modified or structure(relation(polarity="negated"))
    comparator = StructuredEmbeddingComparator(embedder=ConstantEmbedder()).fit(
        [
            TopicStructurePair("pair", original, modified),
        ]
    )
    options = dict(
        original_text="Agency helps residents.",
        modified_text="Agency does not help residents.",
        title=None,
        original_structure=original,
        modified_structure=modified,
        original_vad=VADScore(3, 3, 3),
        modified_vad=VADScore(3, 3, 3),
        comparator=comparator,
        model="test",
        provider="chatgpt",
        judge_fn=judge,
    )
    return compare_dual_stdi(**(options | changes))


@pytest.mark.parametrize("baseline", [0.0, 0.2, 0.6, 1.0])
@pytest.mark.parametrize("different", [False, True])
def test_binary_polarity_direction_and_bounds(baseline, different):
    result = adjust_relation_distance(
        baseline,
        relation(),
        relation(polarity="negated" if different else "affirmed"),
    )
    assert result["distance"] == pytest.approx(0.8 * baseline + 0.2 * different)
    assert 0 <= result["distance"] <= 1
    assert result["distance"] >= baseline if different else result["distance"] <= baseline


def test_no_predicate_gate_and_zero_polarity_weight():
    result = adjust_relation_distance(
        0.6,
        relation(action="ignores", base_action="ignores"),
        relation(action="does not help", polarity="negated"),
    )
    assert result["distance"] == pytest.approx(0.68)
    assert (
        adjust_relation_distance(0.6, relation(), relation(), polarity_weight=0)["distance"] == 0.6
    )


@pytest.mark.parametrize(
    "a,b,expected,percentage",
    [
        (10, 11, 0.1, 10),
        (10, 20, 1, 100),
        (20, 10, 0.5, 50),
        (10, 30, 1, 200),
        (0, 0, 0, None),
        (0, 5, 1, None),
    ],
)
def test_directional_duration_and_zero_policy(a, b, expected, percentage):
    result = compare_duration(duration(a), duration(b))
    assert result["distance"] == pytest.approx(expected)
    assert result["percentage"] == percentage


@pytest.mark.parametrize("unit", ["hours", "horas"])
def test_equivalent_duration_units(unit):
    assert compare_duration(duration(1), duration(24, unit))["distance"] == 0


def test_duration_strength_and_remaining_distance():
    assert adjust_relation_distance(0, duration(10), duration(11))["distance"] == pytest.approx(
        0.02
    )
    result = adjust_relation_distance(0.6, duration(10), duration(20))
    assert result["polarity_adjusted_distance"] == pytest.approx(0.48)
    assert result["distance"] == pytest.approx(0.584)
    assert adjust_relation_distance(0, duration(10), duration(20))["distance"] == 0.2
    both = adjust_relation_distance(0, duration(10), replace(duration(20), polarity="negated"))
    assert both["distance"] == pytest.approx(0.36)


def test_absent_added_and_unknown_duration_are_distinct():
    assert compare_duration(relation(), relation())["distance"] == 0
    assert compare_duration(relation(), duration(10))["change"] == "addition"
    assert compare_duration(duration(10), relation())["change"] == "omission"
    for bad in (
        relation(duration_status="unknown"),
        duration(1, "months"),
        duration(float("nan")),
        duration(-1),
    ):
        assert compare_duration(duration(10), bad)["status"] == "partial"
    normalized = adjust_relation_distance(0.2, relation(polarity=None), relation())
    assert normalized["distance"] == pytest.approx(0.16)
    assert normalized["resolved_polarities"] == ["affirmed", "affirmed"]
    assert normalized["warnings"]


def test_greedy_matching_counts_unmatched_and_never_discounts_omissions():
    original, modified = structure(relation()), structure(relation(), relation(subject="City"))
    comparator = StructuredEmbeddingComparator(embedder=ConstantEmbedder()).fit(
        [
            TopicStructurePair("pair", original, modified),
        ]
    )
    result = comparator.compare_structured(original, modified)
    assert result["components"]["relation_drift"] == 0.5
    assert result["unmatched_relations"] == 1
    assert result["relation_denominator"] == 2
    assert len(result["relations"]) == 1
    assert comparator.compare_structured(original, structure())["components"]["relation_drift"] == 1
    assert (
        comparator.compare_structured(structure(), structure())["components"]["relation_drift"] == 0
    )


def test_base_action_scored_separately_without_mutating_signed_evidence():
    left = relation(action="helps for ten days", base_action="helps", signed_action="helps")
    right = relation(action="does not help", base_action="helps", polarity="negated")
    result = compare(structure(left), structure(right))
    assert result["embedding"]["relations"][0]["semantic_distance"] == pytest.approx(0)
    assert left.action == "helps for ten days"
    assert result["embedding"]["metrics"]["relation_drift"] == pytest.approx(0.2)


def test_dual_mean_uses_complete_branches_and_shared_vad():
    result = compare(modified_vad=VADScore(5, 5, 5))
    a, b = result["embedding"]["metrics"], result["llm_judge"]["metrics"]
    assert a["stdi"] == pytest.approx(0.145)
    assert b["stdi"] == pytest.approx(0.595)
    assert result["stdi"] == pytest.approx((a["stdi"] + b["stdi"]) / 2)
    assert a["vad_drift"] == b["vad_drift"] == 0.5
    assert min(a["stdi"], b["stdi"]) <= result["stdi"] <= max(a["stdi"], b["stdi"])


def test_failure_and_missing_qualifiers_preserve_available_branch():
    def fail(**kwargs):
        raise ValueError("Judge unavailable")

    result = compare(judge_fn=fail)
    assert result["embedding"]["metrics"] is not None
    assert result["llm_judge"]["status"] == "failed"
    assert result["stdi"] is None and result["status"] == "partial"
    result = compare(modified=structure(relation(polarity=None)))
    assert result["embedding"]["metrics"] is not None
    assert result["llm_judge"]["metrics"] is not None
    assert result["stdi"] is not None
    assert result["embedding"]["warnings"]
    result = compare(original_vad=VADScore(None, 3, 3))
    assert result["stdi"] is None
    assert result["llm_judge"]["metrics"] is None


def test_historical_structures_never_assume_qualifier_defaults():
    old = TopicStructure("topic", [], [], [TopicRelation("Agency", "helps", "residents")])
    restored = extraction._build_topic_structure(topic_structure_to_dict(old))
    assert restored.central_relations[0].polarity is None
    assert restored.central_relations[0].duration_status == "unknown"
    assert compare(original=old)["stdi"] is None


def test_identity_reuses_reference_and_skips_judge():
    def fail(**kwargs):
        pytest.fail("An identity pair must not call the judge")

    result = compare(
        original_text="same",
        modified_text="same",
        judge_fn=fail,
        modified=structure(relation(polarity="negated")),
    )
    assert result["identity_shortcut"]
    assert result["stdi"] == result["method_gap"] == 0


def test_judge_cache_is_durable_and_uncached_samples_do_not_replace_canonical(tmp_path):
    calls = []

    def counting_judge(**kwargs):
        calls.append(kwargs)
        return judge(**kwargs)

    a = compare(cache=EvaluationCache(tmp_path), judge_fn=counting_judge)
    b = compare(cache=EvaluationCache(tmp_path), judge_fn=counting_judge)
    assert len(calls) == 1
    assert not a["llm_judge"]["cache_hit"] and b["llm_judge"]["cache_hit"]
    assert a["stdi"] == b["stdi"]
    compare(cache=EvaluationCache(tmp_path), judge_fn=counting_judge, uncached_judge=True)
    assert len(calls) == 2
    compare(cache=EvaluationCache(tmp_path), judge_fn=counting_judge, title="New context")
    assert len(calls) == 3


def test_parser_preserves_qualifiers_and_incomplete_relations():
    payload = topic_structure_to_dict(structure(relation(), relation(polarity="negated")))
    payload["central_relations"].append({"subject": "Agency"})
    parsed = extraction._build_topic_structure(payload)
    assert len(parsed.central_relations) == 3
    assert parsed.central_relations[1].polarity == "negated"
    assert compare(original=parsed)["embedding"]["status"] == "partial"
    row = pd.Series(flatten_topic_structure(parsed, prefix="original"))
    assert topic_structure_from_row(row, prefix="original").schema_version == 2


def test_structured_extraction_prompt_and_raw_response_are_saved(monkeypatch):
    payload = json.dumps(topic_structure_to_dict(structure(relation())))
    monkeypatch.setattr(extraction, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        extraction, "generate_openai_text_with_retry", lambda *args, **kwargs: payload
    )
    result = extraction.extract_topic_structure(
        text="Agency helps residents.", structured=True, model="gpt-5-test"
    )
    assert result.schema_version == 2
    assert result.provenance["raw_response"] == payload
    assert result.provenance["temperature_sent"] is None


def test_structured_judge_validates_scores_and_preserves_raw_response(monkeypatch):
    payload = {
        **dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 0.5),
        "rationales": dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), "Change"),
    }
    monkeypatch.setattr(
        semantic_comparison, "create_llm_client", lambda **kwargs: ("chatgpt", object())
    )
    monkeypatch.setattr(
        semantic_comparison,
        "generate_openai_text_with_retry",
        lambda *args, **kwargs: json.dumps(payload),
    )
    kwargs = dict(
        original_text="Agency helps.",
        modified_text="Agency refuses.",
        title=None,
        original_structure=structure(),
        modified_structure=structure(),
        model="test",
        provider="chatgpt",
        structured=True,
    )
    result = semantic_comparison.compare_stdi_components_semantically(**kwargs)
    assert not hasattr(result, "evidence")
    assert result.component_drifts["theme_drift"] == 0.5
    assert result.provenance["raw_response"] == json.dumps(payload)
    payload["theme_drift"] = None
    with pytest.raises(semantic_comparison.JudgeValidationError) as error:
        semantic_comparison.compare_stdi_components_semantically(**kwargs)
    assert error.value.provenance["raw_response"] == json.dumps(payload)


def test_workflow_deduplicates_extraction_and_exports_both_methods(tmp_path):
    calls = []

    def extract(**kwargs):
        calls.append(kwargs["text"])
        return structure(relation())

    df = pd.DataFrame(
        [
            {"original_text": "Original", "modified_text": "Rewrite"},
            {"original_text": "Original", "modified_text": "Other rewrite"},
        ]
    )
    result = run_comparison_workflow(
        df,
        method="dual",
        extraction_fn=extract,
        llm_comparison_fn=judge,
        embedder=ConstantEmbedder(),
        vad_scorer=lambda text: VADScore(3, 3, 3),
        cache_dir=tmp_path / "cache",
    )
    assert calls == ["Original", "Rewrite", "Other rewrite"]
    assert result.manifest["valid_pairs"] == 2
    assert result.results["embedding_stdi"].notna().all()
    assert result.results["llm_judge_stdi"].notna().all()
    write_comparison_output(tmp_path / "output", result)
    assert (tmp_path / "output" / "comparison_results.csv").exists()


def test_graph_defaults_to_dual_and_retains_valid_branch_on_judge_failure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: ("chatgpt", object()))

    def extract(**kwargs):
        assert kwargs["structured"]
        calls.append(kwargs["text"])
        return structure(relation())

    monkeypatch.setattr(graph, "extract_topic_structure", extract)
    rewrites = iter(["First", "Second"])
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **kwargs: next(rewrites))
    judge_calls = []

    def sometimes_failed(**kwargs):
        judge_calls.append((kwargs["original_text"], kwargs["modified_text"]))
        if kwargs["modified_text"] == "Second":
            raise ValueError("Judge failed")
        return judge(**kwargs)

    result = graph.run_news_interaction_graph(
        pd.DataFrame([{"description": "Original"}]),
        nodes=[
            SimulationNode("a", "test", "chatgpt", "persona"),
            SimulationNode("b", "test", "chatgpt", "persona"),
        ],
        stdi_embedder=ConstantEmbedder(),
        stdi_judge_fn=sometimes_failed,
        vad_scorer=lambda text: VADScore(3, 3, 3),
        output_dir=tmp_path,
    )
    first, second = result.step_results
    assert result.summary["stdi_comparison_method"] == "dual"
    assert calls == ["Original", "First", "Second"]
    assert judge_calls.count(("Original", "First")) == 1
    assert first.stdi_vs_original is not None
    assert first.stdi_chain_complete
    assert second.rewrite_status == "success"
    assert second.stdi_vs_original is None
    assert second.stdi_embedding_vs_original is not None
    assert second.stdi_cumulative == first.stdi_incremental
    assert not second.stdi_chain_complete
    saved = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    assert saved[0]["stdi_vs_original"] == first.stdi_vs_original
    assert saved[1]["metadata_dual_stdi_vs_original"]["llm_judge"]["status"] == "failed"


def test_complete_formula_keeps_full_precision():
    components = dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 1 / 3)
    metrics = complete_stdi(components, shared_vad_drift(VADScore(3, 3, 3), VADScore(3, 3, 3)))
    assert metrics["content_drift"] == 1 / 3
    assert metrics["stdi"] != round(metrics["stdi"], 6)


@pytest.mark.parametrize("failed_stage", ["extraction", "vad"])
def test_graph_evaluation_failures_do_not_block_rewriting(failed_stage, monkeypatch, tmp_path):
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **kwargs: "Rewrite")

    def extract(**kwargs):
        if failed_stage == "extraction" and kwargs["text"] == "Original":
            raise ValueError("Extraction failed")
        return structure(relation())

    def vad(text):
        if failed_stage == "vad":
            raise ValueError("VAD failed")
        return VADScore(3, 3, 3)

    monkeypatch.setattr(graph, "extract_topic_structure", extract)
    result = graph.run_news_interaction_graph(
        pd.DataFrame([{"description": "Original"}]),
        nodes=[SimulationNode("a", "test", "chatgpt", "persona")],
        stdi_embedder=ConstantEmbedder(),
        stdi_judge_fn=judge,
        vad_scorer=vad,
        output_dir=tmp_path,
    )
    step = result.step_results[0]
    assert step.rewrite_status == "success"
    assert step.stdi_vs_original is None
    assert step.stdi_status_vs_original == "partial"
    if failed_stage == "extraction":
        assert step.original_topic_structure_status == "error"
        assert step.stdi_llm_judge_vs_original is not None
    else:
        assert step.original_vad_status == "error"
        assert step.stdi_llm_judge_vs_original is None


def test_workflow_preserves_judge_on_extraction_failure_and_handles_empty_input():
    def fail(**kwargs):
        raise ValueError("Extractor unavailable")

    options = dict(
        method="dual",
        extraction_fn=fail,
        llm_comparison_fn=judge,
        embedder=ConstantEmbedder(),
        vad_scorer=lambda text: VADScore(3, 3, 3),
    )
    df = pd.DataFrame([{"original_text": "Original", "modified_text": "Rewrite"}])
    result = run_comparison_workflow(df, **options)
    assert result.results.iloc[0]["comparison_status"] == "partial"
    assert result.results.iloc[0]["llm_judge_stdi"] is not None
    assert pd.isna(result.results.iloc[0]["stdi"])
    empty = run_comparison_workflow(df.head(0), **options)
    assert empty.manifest["valid_pairs"] == 0


def test_empty_relations_with_failed_extraction_are_not_zero_drift():
    failed = replace(
        structure(), extraction_status="failed", extraction_issues=["Missing response"]
    )
    result = compare(original=failed, modified=structure())
    assert result["embedding"]["metrics"] is None
    unknown = structure(relation(polarity=None))
    result = compare(original=unknown, modified=structure())
    assert result["embedding"]["metrics"]["relation_drift"] == 1


def test_nonfinite_scores_and_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        adjust_relation_distance(float("nan"), relation(), relation())
    with pytest.raises(ValueError):
        adjust_relation_distance(0.2, relation(), relation(), polarity_weight=1.1)
    components = dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), float("nan"))
    with pytest.raises(ValueError):
        complete_stdi(components, shared_vad_drift(VADScore(3, 3, 3), VADScore(3, 3, 3)))


def test_validation_report_separates_pending_annotations_and_uncached_variability():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/evaluate_dual_stdi.py")
    )
    canonical = pd.DataFrame(
        [
            {
                "pair_id": "a",
                "comparison_status": "valid",
                "method_gap": 0.2,
                "manual_review_status": "pending",
                "manual_expected_stdi": 0.9,
                "historical_embedding_stdi": 0.1,
                "embedding_stdi": 0.2,
                "llm_judge_stdi": 0.4,
                "stdi": 0.3,
            },
            {
                "pair_id": "b",
                "comparison_status": "partial",
                "method_gap": None,
                "manual_review_status": "reviewed",
                "manual_expected_stdi": 0.5,
                "historical_embedding_stdi": 0.4,
                "embedding_stdi": 0.2,
                "llm_judge_stdi": None,
                "stdi": None,
            },
        ]
    )
    repeats = pd.DataFrame(
        [{"pair_id": "a", "llm_judge_stdi": 0.2}, {"pair_id": "a", "llm_judge_stdi": 0.6}]
    )
    report = namespace["build_validation_report"](canonical, repeats)
    assert report["complete_pairs"] == 1
    assert report["human_reviewed_pairs"] == 1
    assert report["reviewed_mean_absolute_errors"]["stdi"]["pairs"] == 0
    assert report["reviewed_mean_absolute_errors"]["embedding_stdi"]["mae"] == pytest.approx(0.3)
    assert report["uncached_judge_variability"][0]["std"] > 0
    assert report["uncached_judge_variability"][0]["count"] == 2


def test_validation_fixtures_reserve_new_stories_and_do_not_claim_human_review():
    path = Path(__file__).resolve().parents[1] / "data/synthetic/dual_stdi_validation_pairs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    development = {case["story_family"] for case in cases if case["split"] == "development"}
    held_out = {case["story_family"] for case in cases if case["split"] == "held_out"}
    assert development.isdisjoint(held_out)
    assert all(case["manual_review_status"] == "pending" for case in cases)
    assert len(cases) == 18


def test_failed_extraction_keeps_raw_response_and_nonfinite_fields_remain_serializable(monkeypatch):
    monkeypatch.setattr(extraction, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        extraction, "generate_openai_text_with_retry", lambda *args, **kwargs: "invalid response"
    )
    with pytest.raises(extraction.ExtractionValidationError) as error:
        extraction.extract_topic_structure(text="Agency helps residents.", structured=True)
    assert error.value.provenance["raw_response"] == "invalid response"
    payload = topic_structure_to_dict(structure(relation()))
    payload["internal_contradiction_score"] = float("nan")
    parsed = extraction._build_topic_structure(payload)
    assert parsed.extraction_status == "partial"
    json.dumps(topic_structure_to_dict(parsed), allow_nan=False)


def test_workflow_reports_vad_failure_without_stale_scores():
    def vad(text):
        if text == "Original":
            raise ValueError("Model unavailable")
        return VADScore(3, 3, 3)

    source = pd.DataFrame(
        [{"original_text": "Original", "modified_text": "Rewrite", "stdi": 0.9, "theme_drift": 0.8}]
    )
    result = run_comparison_workflow(
        source,
        method="dual",
        extraction_fn=lambda **kwargs: structure(relation()),
        llm_comparison_fn=judge,
        embedder=ConstantEmbedder(),
        vad_scorer=vad,
    )
    row = result.results.iloc[0]
    assert row["comparison_status"] == "partial"
    assert pd.isna(row["stdi"])
    assert pd.isna(row["theme_drift"])
    assert row["original_vad_error"] == "Model unavailable"
