import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.analysis.stdi_evaluation import (
    available_evaluations,
    select_evaluation_steps,
)
from misinformation_simulation.apps.interaction_graph_results import load_saved_result
from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.types import SimulationNode
from misinformation_simulation.text_metrics import vad_evaluation as vad
from misinformation_simulation.text_metrics.llm_vad import LLMVADAssessment
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.comparison_workflow import run_comparison_workflow
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure
from misinformation_simulation.topic_drift.provenance import EvaluationCache
from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison
from misinformation_simulation.topic_drift.structured_comparison import (
    CONTENT_COMPONENTS,
)


def topic(text):
    negative = text != "Original"
    relation = TopicRelation(
        "Agency",
        "does not help" if negative else "helps",
        "residents",
        polarity="negated" if negative else "affirmed",
        predicate="helps",
        signed_action="does not help" if negative else "helps",
        base_action="helps",
        negation_scope="helps" if negative else None,
        duration_status="absent",
        assertion_type="asserted",
    )
    return TopicStructure(
        "services",
        ["assistance"],
        ["Agency", "residents"],
        [relation],
        schema_version=2,
        extraction_status="valid",
    )


class Embedder:
    def encode(self, texts):
        return np.ones((len(texts), 2))


def judge(**kwargs):
    return SemanticSTDIComparison(
        dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 0.5), {}
    )


MODEL_SCORES = {"Original": VADScore(2, 3, 3), "Rewrite": VADScore(4, 3, 3)}
LLM_SCORES = {"Original": VADScore(4, 1, 3), "Rewrite": VADScore(3, 3, 2)}


def evaluator(method):
    return vad.VADTextEvaluator(
        method=method, model_scorer=MODEL_SCORES.__getitem__, llm_scorer=LLM_SCORES.__getitem__
    )


def paired(method):
    scorer = evaluator(method)
    return vad.compare_vad_evaluations(
        scorer.evaluate("Original"), scorer.evaluate("Rewrite"), method=method
    )


def test_dual_averages_distances_without_cancelling_opposite_directions():
    scorer = vad.VADTextEvaluator(
        method="dual",
        model_scorer=lambda text: VADScore(2 if text == "a" else 4, 3, 3),
        llm_scorer=lambda text: VADScore(4 if text == "a" else 2, 3, 3),
    )
    result = vad.compare_vad_evaluations(scorer.evaluate("a"), scorer.evaluate("b"), method="dual")
    assert result["dual"]["valence_drift"] == 0.5
    assert result["dual"]["vad_drift"] == pytest.approx(1 / 6)
    assert result["model"]["valence_delta"] == 2
    assert result["llm"]["valence_delta"] == -2


def test_llm_failure_keeps_model_available_without_dual_fallback():
    def fail(text):
        raise ValueError("LLM unavailable")

    scorer = vad.VADTextEvaluator(
        method="dual", model_scorer=MODEL_SCORES.__getitem__, llm_scorer=fail
    )
    result = vad.compare_vad_evaluations(
        scorer.evaluate("Original"), scorer.evaluate("Rewrite"), method="dual"
    )
    assert result["model"]["status"] == "valid"
    assert result["llm"]["vad_drift"] is None
    assert result["dual"]["status"] == "partial"
    assert result["dual"]["vad_drift"] is None


def test_durable_llm_cache_uses_exact_text_model_provider_and_rubric(tmp_path, monkeypatch):
    calls = []

    class Scorer:
        last_attempts = [{"raw_response": "saved raw assessment"}]

        def __init__(self, **kwargs):
            pass

        def assess(self, text):
            calls.append(text)
            return LLMVADAssessment(VADScore(3, 2, 3), {}, {}, [])

    monkeypatch.setattr(vad, "LLMVADScorer", Scorer)
    first = vad.VADTextEvaluator(method="llm", cache=EvaluationCache(tmp_path), llm_model="first")
    first.evaluate("exact text")
    first.evaluate("exact text")
    second = vad.VADTextEvaluator(method="llm", cache=EvaluationCache(tmp_path), llm_model="first")
    assert second.evaluate("exact text")["llm"]["cache_hit"]
    assert calls == ["exact text"]
    changed = vad.VADTextEvaluator(
        method="llm", cache=EvaluationCache(tmp_path), llm_model="second"
    )
    changed.evaluate("exact text")
    changed.evaluate("changed text")
    assert len(calls) == 3
    assert (
        first.records["exact text"]["llm"]["attempts"][0]["raw_response"] == "saved raw assessment"
    )


@pytest.mark.parametrize("method", ["model", "llm", "dual"])
@pytest.mark.parametrize("stdi_method", ["cluster", "llm", "dual"])
def test_graph_pairs_selected_vad_with_each_stdi_and_persists_it(
    method, stdi_method, monkeypatch, tmp_path
):
    calls = {"model": [], "llm": []}

    def model(text):
        calls["model"].append(text)
        return MODEL_SCORES[text]

    def llm(text):
        calls["llm"].append(text)
        return LLM_SCORES[text]

    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **kwargs: topic(kwargs["text"]))
    monkeypatch.setattr(graph, "_extract_compared_structure", lambda **kwargs: topic("Rewrite"))
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **kwargs: "Rewrite")
    result = graph.run_news_interaction_graph(
        pd.DataFrame([{"description": "Original", "language": "en"}]),
        nodes=[SimulationNode("a", "test", "chatgpt", "persona")],
        stdi_comparison_method=stdi_method,
        stdi_judge_fn=judge,
        stdi_judge_repeats=1,
        stdi_embedder=Embedder(),
        vad_method=method,
        vad_scorer=model,
        vad_llm_scorer=llm,
        output_dir=tmp_path,
    )
    assert calls["model"] == (["Original", "Rewrite"] if method != "llm" else [])
    assert calls["llm"] == (["Original", "Rewrite"] if method != "model" else [])
    record = result.step_results[0].to_record()
    assert record["rewrite_status"] == "success"
    assert record["metadata_vad_method"] == method
    expected = paired(method)
    if stdi_method == "dual":
        details = record["metadata_dual_stdi_vs_original"]
        for branch in ("embedding", "llm_judge"):
            source = (
                {"embedding": "model", "llm_judge": "llm"}[branch] if method == "dual" else method
            )
            assert details[branch]["vad_source"] == source
            assert details[branch]["metrics"]["vad_drift"] == expected[source]["vad_drift"]
        assert details["metrics"]["vad_drift"] == expected[method]["vad_drift"]
        target = (
            details["embedding"]["dual_metrics"]["stdi"]
            + details["llm_judge"]["dual_metrics"]["stdi"]
        ) / 2
        assert details["stdi"] == target
        if method == "dual":
            assert details["stdi"] != pytest.approx(
                (details["embedding"]["metrics"]["stdi"] + details["llm_judge"]["metrics"]["stdi"])
                / 2
            )
    else:
        assert record["vad_drift_vs_original"] == pytest.approx(
            expected[method]["vad_drift"], abs=1e-6
        )
    before = result.steps_path.read_bytes()
    restored = load_saved_result(result.summary_path)
    assert result.steps_path.read_bytes() == before
    selected = "llm_judge" if stdi_method == "llm" else stdi_method
    assert selected in available_evaluations(restored["steps_df"], stdi_method)
    view = select_evaluation_steps(restored["steps_df"], selected, stdi_method)
    assert view.iloc[0].vad_drift_vs_original == pytest.approx(
        expected[method]["vad_drift"], abs=1e-6
    )
    assert record[f"vad_drift_{method}_vs_original"] == expected[method]["vad_drift"]


def test_invalid_vad_method_fails_before_creating_clients(monkeypatch):
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: pytest.fail("Unexpected API"))
    with pytest.raises(ValueError, match="vad_method"):
        graph.run_news_interaction_graph(
            pd.DataFrame([{"description": "text"}]), nodes=[], vad_method="unsupported"
        )


@pytest.mark.parametrize("method", ["model", "llm", "dual"])
@pytest.mark.parametrize("stdi_method", ["cluster", "llm_semantic", "dual"])
def test_pair_workflow_uses_selected_vad(method, stdi_method):
    result = run_comparison_workflow(
        pd.DataFrame([{"original_text": "Original", "modified_text": "Rewrite"}]),
        method=stdi_method,
        reuse_structures=False,
        extraction_fn=lambda **kwargs: topic(kwargs["text"]),
        embedder=Embedder(),
        llm_comparison_fn=judge,
        judge_repeats=1,
        vad_method=method,
        vad_scorer=MODEL_SCORES.__getitem__,
        vad_llm_scorer=LLM_SCORES.__getitem__,
    )
    row = result.results.iloc[0]
    expected = paired(method)
    assert row.comparison_status in {"success", "valid"}
    assert result.manifest["vad_method"] == method
    assert row[f"vad_drift_{method}"] == expected[method]["vad_drift"]
    if stdi_method == "dual":
        assert row.dual_vad_drift == expected[method]["vad_drift"]
        assert (
            row.cluster_vad_drift == expected["model" if method == "dual" else method]["vad_drift"]
        )
        assert (
            row.llm_judge_vad_drift == expected["llm" if method == "dual" else method]["vad_drift"]
        )
        assert row.stdi == row.dual_stdi
    else:
        assert row.vad_drift == expected[method]["vad_drift"]


@pytest.mark.parametrize("missing_source", ["model", "llm"])
def test_incomplete_vad_preserves_other_complete_stdi_branch(missing_source):
    from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
    from misinformation_simulation.topic_drift.structured_comparison import (
        StructuredEmbeddingComparator,
        compare_dual_stdi,
    )

    scores = {"model": MODEL_SCORES, "llm": LLM_SCORES}
    scores[missing_source] = dict.fromkeys(("Original", "Rewrite"), VADScore(3, None, 3))
    scorer = vad.VADTextEvaluator(
        method="dual",
        model_scorer=scores["model"].__getitem__,
        llm_scorer=scores["llm"].__getitem__,
    )
    evaluation = vad.compare_vad_evaluations(
        scorer.evaluate("Original"), scorer.evaluate("Rewrite"), method="dual"
    )
    original, rewrite = topic("Original"), topic("Rewrite")
    comparator = StructuredEmbeddingComparator(embedder=Embedder()).fit(
        [TopicStructurePair("pair", original, rewrite)]
    )
    result = compare_dual_stdi(
        original_text="Original",
        modified_text="Rewrite",
        title=None,
        original_structure=original,
        modified_structure=rewrite,
        original_vad=None,
        modified_vad=None,
        comparator=comparator,
        model="test",
        provider="chatgpt",
        judge_fn=judge,
        judge_repeats=1,
        vad_evaluation=evaluation,
    )
    unavailable = "embedding" if missing_source == "model" else "llm_judge"
    available = "llm_judge" if missing_source == "model" else "embedding"
    assert result["status"] == "partial"
    assert result["stdi"] is None
    assert result[unavailable]["metrics"] is None
    assert result[available]["status"] == "valid"
    assert result[available]["metrics"]["stdi"] is not None


@pytest.mark.parametrize("stdi_method", ["cluster", "llm_semantic", "dual"])
def test_failed_rerating_does_not_retain_previous_selected_metrics(stdi_method):
    source = pd.DataFrame(
        [
            {
                "original_text": "Original",
                "modified_text": "Rewrite",
                "vad_drift": 0.9,
                "valence_drift": 0.9,
                "dual_vad_drift": 0.9,
                "dual_stdi": 0.9,
            }
        ]
    )
    workflow = run_comparison_workflow(
        source,
        method=stdi_method,
        reuse_structures=False,
        extraction_fn=lambda **kwargs: topic(kwargs["text"]),
        embedder=Embedder(),
        llm_comparison_fn=judge,
        judge_repeats=1,
        vad_method="dual",
        vad_scorer=MODEL_SCORES.__getitem__,
        vad_llm_scorer=lambda _text: VADScore(3, None, 3),
    )
    row = workflow.results.iloc[0]
    assert pd.isna(row.stdi)
    if stdi_method == "dual":
        assert pd.isna(row.dual_stdi)
        assert pd.isna(row.dual_vad_drift)
    else:
        assert pd.isna(row.vad_drift)
        assert pd.isna(row.valence_drift)


@pytest.mark.parametrize("stdi_method", ["cluster", "dual"])
@pytest.mark.parametrize("same_provider", [True, False])
def test_vad_endpoint_inheritance_respects_provider(monkeypatch, stdi_method, same_provider):
    settings = []

    class Scorer:
        last_attempts = []

        def __init__(self, **kwargs):
            settings.append(kwargs)

        def assess(self, text):
            return LLMVADAssessment(LLM_SCORES[text], {}, {}, [])

    monkeypatch.setattr(vad, "LLMVADScorer", Scorer)
    run_comparison_workflow(
        pd.DataFrame([{"original_text": "Original", "modified_text": "Rewrite"}]),
        method=stdi_method,
        reuse_structures=False,
        extraction_fn=lambda **kwargs: topic(kwargs["text"]),
        embedder=Embedder(),
        llm_comparison_fn=judge,
        judge_repeats=1,
        extraction_provider="chatgpt",
        extraction_api_key="test-key",
        extraction_base_url="https://example.test/v1",
        vad_method="llm",
        vad_llm_provider="chatgpt" if same_provider else "gemini",
    )
    assert len(settings) == 1
    assert settings[0]["api_key"] == ("test-key" if same_provider else None)
    assert settings[0]["base_url"] == ("https://example.test/v1" if same_provider else None)
