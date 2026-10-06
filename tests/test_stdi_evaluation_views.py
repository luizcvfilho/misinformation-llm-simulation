from __future__ import annotations

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.analysis.interaction_graph_visualization import (
    load_interaction_graph_runs,
)
from misinformation_simulation.analysis.stdi_evaluation import (
    available_evaluations,
    select_evaluation_steps,
)
from misinformation_simulation.apps.interaction_graph_results import evaluation_result_bundle
from misinformation_simulation.simulation.types import EVALUATION_METRICS


def _record(step=1, *, failed_llm=False):
    record = {
        "news_id": "news-1",
        "step_index": step,
        "node_id": "node-1",
        "node_label": "Conservative",
        "source_node_label": "original",
        "provider": "chatgpt",
        "model": "model",
        "personality": "persona",
        "source_text": "Original",
        "rewritten_text": "Rewrite",
        "rewrite_status": "success",
        "rewrite_error": None,
        "metadata_title": "Example",
        "metadata_category": "politics",
        "metadata_stdi_comparison_version": "dual_stdi_v4",
        "metadata_rewrite_mode": "faithful",
        "stdi_cumulative": 0.5 * step,
        "stdi_chain_complete": True,
    }
    for suffix in ("vs_original", "incremental"):
        cluster = dict.fromkeys(EVALUATION_METRICS, 0.2)
        llm = dict.fromkeys(EVALUATION_METRICS, 0.8)
        record[f"metadata_dual_stdi_{suffix}"] = {
            "status": "partial" if failed_llm else "valid",
            "stdi": None if failed_llm else 0.5,
            "method_gap": None if failed_llm else 0.6,
            "embedding": {"status": "valid", "metrics": cluster},
            "llm_judge": {
                "status": "failed" if failed_llm else "valid",
                "metrics": None if failed_llm else llm,
                "error": "Judge unavailable" if failed_llm else None,
            },
        }
        for metric in EVALUATION_METRICS:
            record[f"{metric}_{suffix}"] = cluster[metric]
        record[f"stdi_{suffix}"] = None if failed_llm else 0.5
    return record


@pytest.mark.parametrize(
    "evaluation,expected", [("cluster", 0.2), ("llm_judge", 0.8), ("dual", 0.5)]
)
def test_selection_switches_all_metrics_without_mutating_saved_scores(evaluation, expected):
    steps = pd.DataFrame([_record(2), _record(1)])
    before = steps.copy(deep=True)
    selected = select_evaluation_steps(steps, evaluation)
    assert available_evaluations(steps) == ["dual", "cluster", "llm_judge"]
    for suffix in ("vs_original", "incremental"):
        for metric in EVALUATION_METRICS:
            assert selected[f"{metric}_{suffix}"].tolist() == [expected, expected]
    assert selected["stdi_cumulative"].tolist() == pytest.approx([2 * expected, expected])
    pd.testing.assert_frame_equal(steps, before)


def test_unavailable_branch_remains_missing_instead_of_using_other_scores():
    steps = pd.DataFrame([_record(), _record(2, failed_llm=True)])
    llm = select_evaluation_steps(steps, "llm_judge")
    dual = select_evaluation_steps(steps, "dual")
    cluster = select_evaluation_steps(steps, "cluster")
    assert llm.loc[1, "stdi_status_vs_original"] == "failed"
    assert llm.loc[1, "stdi_error_vs_original"] == "Judge unavailable"
    for metric in EVALUATION_METRICS:
        assert pd.isna(llm.loc[1, f"{metric}_vs_original"])
        assert pd.isna(dual.loc[1, f"{metric}_vs_original"])
        assert cluster.loc[1, f"{metric}_vs_original"] == 0.2
    assert llm["stdi_cumulative"].tolist() == [0.8, 0.8]
    assert llm["stdi_cumulative_valid_steps"].tolist() == [1, 1]
    assert not llm.loc[1, "stdi_chain_complete"]
    assert cluster.loc[1, "stdi_cumulative"] == 0.4
    assert available_evaluations(pd.DataFrame([_record(failed_llm=True)])) == ["cluster"]


def test_mixed_runs_are_not_pooled_as_different_methods():
    dual = _record()
    cluster = {
        key: value for key, value in _record().items() if not key.startswith("metadata_dual_")
    }
    cluster["metadata_stdi_comparison_version"] = "cluster_v4"
    cluster["stdi_vs_original"] = cluster["stdi_incremental"] = 0.3
    cluster["news_id"] = "cluster-news"
    steps = pd.DataFrame([dual, cluster])
    selected = select_evaluation_steps(steps, "dual")
    assert selected.loc[0, "stdi_vs_original"] == 0.5
    assert pd.isna(selected.loc[1, "stdi_vs_original"])
    assert select_evaluation_steps(steps, "cluster").loc[1, "stdi_vs_original"] == 0.3
    assert pd.isna(select_evaluation_steps(steps, "llm_judge").loc[1, "stdi_cumulative"])


def test_unknown_legacy_method_retains_saved_view_and_zero_scores_are_available():
    legacy = pd.DataFrame([{"news_id": "n", "step_index": 1, "stdi_vs_original": 0.0}])
    assert available_evaluations(legacy) == ["saved"]
    assert available_evaluations(legacy, "cluster") == ["cluster"]
    assert select_evaluation_steps(legacy, "saved").iloc[0]["stdi_vs_original"] == 0
    assert select_evaluation_steps(legacy, "cluster", "cluster").iloc[0]["stdi_vs_original"] == 0
    assert pd.isna(select_evaluation_steps(legacy, "dual").iloc[0]["stdi_vs_original"])


def test_loader_recovers_historical_branches_and_keeps_news_and_chains_separate(tmp_path):
    for graph in ("01_first", "02_second"):
        path = tmp_path / f"{graph}_steps.jsonl"
        path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
        path.with_name(f"{graph}_summary.json").write_text(
            json.dumps({"stdi_comparison_method": "dual"}), encoding="utf-8"
        )
    runs = load_interaction_graph_runs(tmp_path)
    assert "relation_drift_llm_judge_vs_original" in runs.steps
    assert "metadata_dual_stdi_vs_original" not in runs.steps
    llm = select_evaluation_steps(runs.steps, "llm_judge")
    assert llm["stdi_cumulative"].tolist() == [0.8, 0.8]
    assert llm["relation_drift_vs_original"].tolist() == [0.8, 0.8]


def _bundle():
    return {
        "name": "Graph",
        "status": "completed",
        "steps_df": pd.DataFrame([_record()]),
        "summary": {
            "stdi_comparison_method": "dual",
            "rows_processed": 1,
            "steps_total": 1,
            "steps_success": 1,
            "steps_error": 0,
        },
        "summary_path": None,
        "steps_path": None,
        "output_prefix": "test",
        "graph_payload": None,
    }


def test_result_bundle_switches_existing_summaries_and_category_charts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = _bundle()
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_sections import render_results_tab\n"
        "render_results_tab()"
    )
    app.session_state["run_bundles"] = [bundle]
    app.run(timeout=30)
    assert not app.exception
    selector = app.selectbox(key="results_stdi_evaluation")
    assert selector.options == ["Dual", "Cluster", "LLM"]
    for evaluation, expected in (("cluster", 0.2), ("llm_judge", 0.8), ("dual", 0.5)):
        selector.set_value(evaluation).run(timeout=30)
        assert not app.exception
        node_table = next(
            table.value
            for table in app.dataframe
            if "mean_stdi_incremental" in table.value and "provider" in table.value
        )
        assert node_table.iloc[0]["mean_stdi_incremental"] == expected
        categories = next(table.value for table in app.dataframe if "category" in table.value)
        assert categories.iloc[0]["mean_contradiction_drift_vs_original"] == expected
        assert next(
            metric for metric in app.metric if metric.label == "STDI vs original"
        ).value == (f"{expected:.3f}")
    assert bundle["steps_df"].iloc[0]["stdi_vs_original"] == 0.5


def test_bundle_export_identifies_the_selected_method():
    selected = evaluation_result_bundle(_bundle(), "llm_judge")
    exported = selected["steps_df"]
    assert exported.iloc[0]["stdi_evaluation"] == "llm_judge"
    assert exported.iloc[0]["relation_drift_vs_original"] == 0.8
    assert exported.iloc[0]["relation_drift_cluster_vs_original"] == 0.2
