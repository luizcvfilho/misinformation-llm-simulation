from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.apps.interaction_graph_categories import (
    build_category_comparison_dataframe,
    build_category_summary_dataframe,
    split_news_categories,
)
from misinformation_simulation.apps.interaction_graph_ui import build_news_summary_dataframe


def test_category_summary_counts_each_news_category_without_duplicate_labels() -> None:
    steps = pd.DataFrame(
        [
            {
                "news_id": "first",
                "metadata_title": "First",
                "metadata_category": "Politics; top; POLITICS",
                "step_index": 1,
                "node_label": "Node A",
                "rewrite_status": "success",
                "stdi_vs_original": 0.2,
                "vad_drift_vs_original": 0.1,
            },
            {
                "news_id": "second",
                "metadata_title": "Second",
                "metadata_category": "politics",
                "step_index": 1,
                "node_label": "Node A",
                "rewrite_status": "error",
                "stdi_vs_original": None,
            },
            {
                "news_id": "first",
                "metadata_title": "First",
                "metadata_category": "Politics; top",
                "step_index": 2,
                "node_label": "Node B",
                "rewrite_status": "success",
                "stdi_vs_original": 0.4,
            },
        ]
    )
    steps["stdi_incremental"] = steps["stdi_vs_original"]
    steps["stdi_cumulative"] = steps["stdi_vs_original"]

    summary = build_category_summary_dataframe(steps)
    politics_first = summary[
        (summary["category"] == "politics") & (summary["step_index"] == 1)
    ].iloc[0]
    top_first = summary[(summary["category"] == "top") & (summary["step_index"] == 1)].iloc[0]
    politics_second = summary[
        (summary["category"] == "politics") & (summary["step_index"] == 2)
    ].iloc[0]

    assert split_news_categories("Politics; top; POLITICS") == ["politics", "top"]
    assert politics_first["news_count"] == 2
    assert politics_first["runs"] == 2
    assert politics_first["success_rate"] == pytest.approx(0.5)
    assert politics_first["mean_stdi_vs_original"] == pytest.approx(0.2)
    assert top_first["news_count"] == 1
    assert politics_second["mean_stdi_vs_original"] == pytest.approx(0.4)
    assert build_news_summary_dataframe(steps).set_index("news_id").loc["first", "category"] == (
        "Politics; top; POLITICS"
    )


def test_category_comparison_keeps_graph_runs_separate_and_skips_legacy_results() -> None:
    steps = pd.DataFrame(
        [
            {
                "news_id": "first",
                "metadata_category": "health; top",
                "step_index": 1,
                "node_label": "Node A",
                "rewrite_status": "success",
                "stdi_vs_original": 0.3,
            }
        ]
    )
    comparison = build_category_comparison_dataframe(
        [
            {"name": "Graph", "steps_df": steps},
            {"name": "Old result", "steps_df": steps.drop(columns="metadata_category")},
            {"name": "Graph", "steps_df": steps},
        ]
    )

    assert set(comparison["run"]) == {"1. Graph", "3. Graph"}
    assert set(comparison["category"]) == {"health", "top"}
    assert len(comparison) == 4
    assert build_category_summary_dataframe(steps.drop(columns="metadata_category")).empty
