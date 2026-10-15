import json

import numpy as np
import pandas as pd

from misinformation_simulation.config import prompts
from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.types import SimulationNode
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift import structured_comparison
from misinformation_simulation.topic_drift.cache_versions import runtime_cache_contracts
from misinformation_simulation.topic_drift.models import TopicStructure
from misinformation_simulation.topic_drift.provenance import EvaluationCache
from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison


def test_shared_cache_reuses_only_exact_stage_contracts(tmp_path):
    shared = tmp_path / "shared"
    inputs = {"text": "text", "model": "model"}
    contracts = {"extraction": {"version": "v1"}}
    output = {"extraction_status": "valid", "main_topic": "topic"}
    first = EvaluationCache(tmp_path / "run1", shared_directory=shared, compatibility=contracts)
    first.put("extraction", inputs, output)
    second = EvaluationCache(tmp_path / "run2", shared_directory=shared, compatibility=contracts)
    assert second.get("extraction", inputs) == output
    assert list((tmp_path / "run2").rglob("extraction_*.json"))
    changed = EvaluationCache(
        tmp_path / "run3",
        shared_directory=shared,
        compatibility={"extraction": {"version": "v2"}},
    )
    assert changed.get("extraction", inputs) is None
    assert second.get("extraction", {**inputs, "model": "new-model"}) is None


def test_legacy_partial_and_corrupt_entries_are_not_shared(tmp_path):
    inputs = {"text": "text"}
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    EvaluationCache(local).put("extraction", inputs, {"extraction_status": "valid"})
    cache = EvaluationCache(local, shared_directory=shared, compatibility={"extraction": {"v": 1}})
    assert cache.get("extraction", inputs) is None
    cache.put("extraction", inputs, {"extraction_status": "partial"})
    assert not list(shared.rglob("*.json"))
    cache.put("extraction", inputs, {"extraction_status": "valid"})
    path = next(shared.rglob("*.json"))
    payload = json.loads(path.read_text())
    payload["output"]["extra"] = "tampered"
    path.write_text(json.dumps(payload))
    fresh = EvaluationCache(shared_directory=shared, compatibility={"extraction": {"v": 1}})
    assert fresh.get("extraction", inputs) is None


def test_prompt_change_invalidates_only_affected_stage(monkeypatch):
    old = runtime_cache_contracts()
    monkeypatch.setattr(prompts, "STRUCTURED_JUDGE_PROMPT_TEMPLATE", "Changed judge rubric")
    new = runtime_cache_contracts()
    assert old["judge"] != new["judge"]
    for namespace in ("rewrite", "extraction", "vad_llm", "vad_model"):
        assert old[namespace] == new[namespace]


def test_graph_reuses_rewrites_and_recalculates_only_new_judge_version(monkeypatch, tmp_path):
    calls = {"rewrite": 0, "extraction": 0, "judge": 0}
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: ("chatgpt", object()))

    def rewrite(**kwargs):
        calls["rewrite"] += 1
        return f"Rewritten text {calls['rewrite']}"

    def extract(**kwargs):
        calls["extraction"] += 1
        return TopicStructure(
            "topic",
            [],
            ["Agency"],
            [],
            schema_version=2,
            extraction_status="valid",
            provenance={"model": "test", "raw_response": "audited extraction"},
        )

    def judge(**kwargs):
        calls["judge"] += 1
        return SemanticSTDIComparison(
            dict.fromkeys((*structured_comparison.CONTENT_COMPONENTS, "contradiction_drift"), 0.25),
            {},
        )

    class Embedder:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    monkeypatch.setattr(graph, "extract_topic_structure", extract)
    compare = graph.compare_dual_stdi
    monkeypatch.setattr(
        graph, "compare_dual_stdi", lambda **kwargs: compare(**kwargs, judge_fn=judge)
    )

    def run(name, *, reuse_rewrites=True, **changes):
        return graph.run_news_interaction_graph(
            pd.DataFrame([{"title": "Title", "description": "Original text", "language": "en"}]),
            nodes=[
                SimulationNode("first", "test", "chatgpt", "persona"),
                SimulationNode("second", "test", "chatgpt", "persona"),
            ],
            stdi_embedder=Embedder(),
            vad_scorer=lambda text: VADScore(3, 3, 3),
            shared_cache_dir=tmp_path / "shared",
            reuse_rewrites=reuse_rewrites,
            output_dir=tmp_path / name,
            **changes,
        )

    first = run("run1")
    initial_calls = calls.copy()
    assert initial_calls["rewrite"] == 2
    second = run("run2")
    assert calls == initial_calls
    assert second.summary["rewrite_operations_started"] == 0
    assert second.summary["api_usage"]["cache_hits"] > 0
    assert all(step.metadata["rewrite_cache_hit"] for step in second.step_results)
    assert [s.stdi_vs_original for s in second.step_results] == [
        s.stdi_vs_original for s in first.step_results
    ]
    assert json.loads(second.summary_path.read_text())["api_usage"] == second.summary["api_usage"]

    monkeypatch.setattr(prompts, "STRUCTURED_JUDGE_VERSION", "next-judge-version")
    monkeypatch.setattr(structured_comparison, "STRUCTURED_JUDGE_VERSION", "next-judge-version")
    third = run("run3")
    assert calls["rewrite"] == initial_calls["rewrite"]
    assert calls["extraction"] == initial_calls["extraction"]
    assert calls["judge"] == 2 * initial_calls["judge"]
    assert third.summary["cache_contracts"]["judge"]["prompt_version"] == "next-judge-version"
    assert (tmp_path / "run3" / "simulation_api_costs.json").is_file()
    fresh = run("fresh", reuse_rewrites=False)
    assert calls["rewrite"] == 2 * initial_calls["rewrite"]
    assert all(not step.metadata["rewrite_cache_hit"] for step in fresh.step_results)


def test_effective_local_endpoint_changes_invalidate_cache(monkeypatch, tmp_path):
    inputs = {"text": "text", "provider": "local", "model": "model", "base_url": None}
    contracts = {"extraction": {"v": 1}}
    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
    cache = EvaluationCache(shared_directory=tmp_path, compatibility=contracts)
    cache.put("extraction", inputs, {"extraction_status": "valid"})
    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11435/v1")
    fresh = EvaluationCache(shared_directory=tmp_path, compatibility=contracts)
    assert fresh.get("extraction", inputs) is None
