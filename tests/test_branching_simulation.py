from __future__ import annotations

from shutil import copytree
from threading import Event

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.analysis.interaction_graph_visualization import (
    discover_step_paths,
    load_interaction_graph_runs,
)
from misinformation_simulation.apps.interaction_graph_results import load_saved_result
from misinformation_simulation.enums import DefaultPersonality as Persona
from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.io import (
    resolve_result_reference,
)
from misinformation_simulation.simulation.topology import (
    _normalize_edges,
    _normalize_nodes,
    _root_to_leaf_paths,
    _topological_path,
)
from misinformation_simulation.simulation.types import SimulationEdge, SimulationNode
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.models import TopicStructure


def tree():
    ids = ["c1", "c2", "c3", "c4", "p3", "p4"]
    nodes = [
        SimulationNode(node, "model", "chatgpt", persona, label=node)
        for node, persona in zip(
            ids, [Persona.ConservativeRight] * 4 + [Persona.ProgressiveLeft] * 2, strict=True
        )
    ]
    edges = [
        SimulationEdge(source, target)
        for source, target in [("c1", "c2"), ("c2", "c3"), ("c3", "c4"), ("c2", "p3"), ("p3", "p4")]
    ]
    return nodes, edges


class ConstantEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 3)) / np.sqrt(3)


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        graph,
        "extract_topic_structure",
        lambda **kwargs: TopicStructure(
            main_topic="topic",
            subtopics=["subtopic"],
            central_entities=["entity"],
            central_relations=[],
            narrative_frame="frame",
        ),
    )
    monkeypatch.setattr(graph, "_extract_compared_structure", graph.extract_topic_structure)
    calls = []

    def rewrite(**kwargs):
        calls.append(kwargs["prompt"])
        return f"output_{len(calls)}"

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)

    def compare(**kwargs):
        value = int(kwargs["modified_text"].split("_")[-1]) / 10
        metrics = dict.fromkeys(graph.STDI_COMPONENTS, 0.0)
        metrics["stdi"] = value
        return {
            "stdi": value,
            "status": "valid",
            "method_gap": 0.0,
            "metrics": metrics,
            "embedding": {"metrics": metrics, "status": "valid"},
            "llm_judge": {"metrics": metrics, "status": "valid"},
        }

    monkeypatch.setattr(graph, "compare_dual_stdi", compare)
    return calls


def run_tree(tmp_path, method="lexical", **kwargs):
    nodes, edges = tree()
    return graph.run_news_interaction_graph(
        pd.DataFrame([{"description": "original", "title": "title", "language": "en"}]),
        nodes=nodes,
        edges=edges,
        stdi_comparison_method=method,
        vad_scorer=lambda text: VADScore(2.0 if text == "original" else 3.0, 3.0, 3.0),
        stdi_embedder=ConstantEmbedder(),
        output_dir=tmp_path,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["lexical", "cluster", "llm", "dual"])
def test_shared_prefix_executes_once_and_paths_keep_parent_text_and_metrics(
    tmp_path,
    stub_models,
    method,
):
    progress = []
    result = run_tree(tmp_path, method, work_progress_callback=lambda *args: progress.append(args))
    assert len(stub_models) == result.summary["rewrite_operations_started"] == 6
    assert result.summary["planned_linear_rewrites"] == 8
    assert result.summary["planned_rewrites_saved"] == 2
    steps = {step.node_id: step for step in result.step_results}
    assert steps["p3"].source_node_id == "c2"
    assert steps["p3"].source_text == steps["c3"].source_text == "output_2"
    assert "output_2" in stub_models[4]
    assert "output_4" not in stub_models[4]
    assert [step.step_index for step in result.step_results] == [1, 2, 3, 4, 3, 4]
    assert progress[-1] == (1, 1, 6, 6)
    assert [path.summary["chain_code"] for path in result.path_results] == ["CCCC", "CCPP"]
    left, right = result.path_results
    assert left.step_results[1].rewritten_text == right.step_results[1].rewritten_text
    assert left.step_results[1] is not right.step_results[1]
    assert right.step_results[1].metadata["graph_step_reused"] is True
    assert sum(path.summary["rewrite_operations_started"] for path in result.path_results) == 6
    if method in {"llm", "dual"}:
        assert steps["c4"].stdi_cumulative == pytest.approx(1.0)
        assert steps["p4"].stdi_cumulative == pytest.approx(1.4)
        assert steps["p4"].stdi_cumulative_valid_steps == 4
        assert steps["p4"].stdi_cluster_cumulative == pytest.approx(1.4)
        assert steps["p4"].stdi_llm_judge_cumulative == pytest.approx(1.4)
    else:
        assert steps["p4"].stdi_cumulative == pytest.approx(
            sum(steps[node].stdi_incremental for node in ("c1", "c2", "p3", "p4")), abs=1e-6
        )
    right.step_results[0].metadata["changed"] = True
    assert "changed" not in left.step_results[0].metadata
    assert "changed" not in result.step_results[0].metadata


def test_saved_graph_and_linear_paths_load_without_counting_overview_as_another_chain(
    tmp_path,
    stub_models,
):
    result = run_tree(tmp_path)
    assert load_saved_result(result.summary_path)["summary"]["steps_total"] == 6
    paths = discover_step_paths(tmp_path)
    assert len(paths) == 2
    assert result.steps_path not in paths
    runs = load_interaction_graph_runs(tmp_path)
    assert len(runs.steps) == 8
    assert set(runs.steps.chain_code) == {"CCCC", "CCPP"}
    assert runs.steps.execution_id.nunique() == 1
    assert runs.steps.graph_id.nunique() == 2
    for path in result.path_results:
        loaded = load_saved_result(path.summary_path)
        assert len(loaded["steps_df"]) == 4
        assert loaded["name"] == path.summary["chain_code"]


def test_branch_failure_blocks_only_its_descendants(tmp_path, stub_models, monkeypatch):
    def rewrite(**kwargs):
        stub_models.append(kwargs["prompt"])
        if len(stub_models) == 3:
            raise RuntimeError("branch unavailable")
        return f"output_{len(stub_models)}"

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    result = run_tree(tmp_path)
    steps = {step.node_id: step for step in result.step_results}
    assert steps["c3"].rewrite_status == "error"
    assert steps["c4"].rewrite_status == "blocked"
    assert steps["c4"].source_node_id == "c3"
    assert steps["p3"].rewrite_status == steps["p4"].rewrite_status == "success"
    assert steps["p3"].source_text == "output_2"
    assert len(stub_models) == 5


def test_cancelled_graph_persists_partial_paths(tmp_path, stub_models):
    cancelled = Event()

    def progress(row, total_rows, done, total_steps):
        if done == 3:
            cancelled.set()

    result = run_tree(tmp_path, work_progress_callback=progress, cancel_check=cancelled.is_set)
    assert result.summary["cancelled"]
    assert len(stub_models) == 3
    assert [len(path.step_results) for path in result.path_results] == [3, 2]
    assert all(path.summary["cancelled"] for path in result.path_results)
    assert all(path.steps_path.is_file() for path in result.path_results)


def test_new_graph_run_generates_its_own_prefix(tmp_path, stub_models):
    run_tree(tmp_path / "first")
    run_tree(tmp_path / "second")
    assert len(stub_models) == 12


def test_news_rows_have_separate_prefixes_and_cumulative_state(tmp_path, stub_models):
    nodes, edges = tree()
    result = graph.run_news_interaction_graph(
        pd.DataFrame([{"description": "original"}, {"description": "original"}]),
        nodes=nodes,
        edges=edges,
        stdi_comparison_method="dual",
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        stdi_embedder=ConstantEmbedder(),
        output_dir=tmp_path,
    )
    assert len(stub_models) == 12
    leaves = [step for step in result.step_results if step.node_id == "p4"]
    assert [step.stdi_cumulative for step in leaves] == pytest.approx([1.4, 3.8])
    assert len(result.path_results[1].step_results) == 8
    assert result.summary["planned_rewrites_saved"] == 4


def test_copied_result_folder_preserves_path_references_and_execution_grouping(
    tmp_path, stub_models
):
    result = run_tree(tmp_path / "original")
    copied = tmp_path / "copied"
    copytree(result.summary_path.parent, copied)
    bundle = load_saved_result(copied / result.summary_path.name)
    descriptor = bundle["summary"]["path_results"][1]
    path = resolve_result_reference(descriptor["summary_path"], bundle["summary_path"])
    assert path.is_relative_to(copied)
    assert load_saved_result(path)["name"] == "CCPP"
    runs = load_interaction_graph_runs(copied)
    assert runs.steps.execution_id.unique().tolist() == [str(copied.resolve())]


def test_arbitrary_fanout_nested_branches_and_different_depths():
    nodes, edges = tree()
    for node, parent in [("extra", "c2"), ("nested-a", "p3"), ("nested-b", "nested-a")]:
        nodes.append(SimulationNode(node, "model", "chatgpt", "custom"))
        edges.append(SimulationEdge(parent, node))
    by_id = _normalize_nodes(nodes)
    order = _topological_path(by_id, _normalize_edges(by_id, edges), "c1")
    paths = _root_to_leaf_paths(order, edges)
    assert len(order) == 9
    assert len(paths) == 4
    assert ["c1", "c2", "p3", "nested-a", "nested-b"] in paths
    assert ["c1", "c2", "extra"] in paths


@pytest.mark.parametrize("kind", ["merge", "cycle", "disconnected"])
def test_invalid_topology_fails_before_creating_model_clients(monkeypatch, kind):
    nodes, edges = tree()
    if kind == "merge":
        edges.append(SimulationEdge("p4", "c4"))
    elif kind == "cycle":
        edges.append(SimulationEdge("p4", "c1"))
    else:
        edges = [edge for edge in edges if edge.target != "p3"]
    monkeypatch.setattr(graph, "create_llm_client", lambda **kwargs: pytest.fail("Unexpected API"))
    with pytest.raises(ValueError):
        graph.run_news_interaction_graph(
            pd.DataFrame([{"description": "original"}]),
            nodes=nodes,
            edges=edges,
            start_node_id="c1",
        )
