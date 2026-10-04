from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.analysis.interaction_graph_groups import (
    ORIGINAL_CATEGORY_GROUPING,
    available_news_groupings,
    news_chain_proximity,
    summarize_chain_pairs_by_group,
    summarize_group_proximity,
)
from misinformation_simulation.analysis.interaction_graph_plotly import (
    build_chain_pair_heatmap,
    build_group_composition_figure,
    build_group_proximity_boxplot,
    build_group_proximity_interval_figure,
)


def _steps() -> pd.DataFrame:
    final_values = {
        "01 · SSSS": {"news-1": 0.2, "news-2": 0.3, "news-3": 0.5},
        "02 · CCCC": {"news-1": 0.4, "news-2": 0.35, "news-3": 0.4},
        "03 · PPPP": {"news-1": 0.3, "news-2": 0.5, "news-3": 0.45},
    }
    categories = {
        "news-1": "business; top",
        "news-2": "business",
        "news-3": "top",
    }
    domains = {
        "news-1": {
            "01 · SSSS": "politics",
            "02 · CCCC": "politics",
            "03 · PPPP": "economy_business_and_finance",
        },
        "news-2": {chain: "economy_business_and_finance" for chain in final_values},
        "news-3": {chain: "politics" for chain in final_values},
    }
    records = []
    for chain_index, (chain_label, news_values) in enumerate(final_values.items(), start=1):
        run_id = f"run-{chain_index}"
        for news_id, final_value in news_values.items():
            for step_index, value in ((1, final_value / 2), (2, final_value)):
                records.append(
                    {
                        "run_id": run_id,
                        "chain_label": chain_label,
                        "news_id": news_id,
                        "metadata_title": f"Title {news_id}",
                        "metadata_category": categories[news_id],
                        "metadata_original_topic_domain": domains[news_id][chain_label],
                        "step_index": step_index,
                        "rewrite_status": "success",
                        "stdi_vs_original": value,
                    }
                )
    return pd.DataFrame(records)


def test_available_groupings_prefer_persisted_original_categories() -> None:
    groupings = available_news_groupings(_steps())

    assert list(groupings) == [ORIGINAL_CATEGORY_GROUPING]


def test_news_proximity_splits_original_categories() -> None:
    steps = _steps()

    categories = news_chain_proximity(steps, "stdi_vs_original", ORIGINAL_CATEGORY_GROUPING)

    business = categories.loc[categories["group_value"].eq("business")]
    top = categories.loc[categories["group_value"].eq("top")]
    assert set(business["news_id"]) == {"news-1", "news-2"}
    assert set(top["news_id"]) == {"news-1", "news-3"}
    assert business.loc[business["news_id"].eq("news-1"), "range_between_chains"].item() == (
        pytest.approx(0.2)
    )


def test_group_and_chain_pair_summaries_keep_news_as_the_unit() -> None:
    steps = _steps()
    proximity = news_chain_proximity(steps, "stdi_vs_original", ORIGINAL_CATEGORY_GROUPING)

    summary = summarize_group_proximity(proximity, bootstrap_iterations=100)
    pairs = summarize_chain_pairs_by_group(
        steps,
        "stdi_vs_original",
        ORIGINAL_CATEGORY_GROUPING,
    )

    business = summary.loc[summary["group_value"].eq("business")].iloc[0]
    assert business["news_items"] == 2
    assert business["mean_range"] == pytest.approx(0.2)
    assert pairs.loc[pairs["group_value"].eq("business"), "paired_news"].eq(2).all()


def test_group_plotly_figures_render_expected_views() -> None:
    steps = _steps()
    proximity = news_chain_proximity(steps, "stdi_vs_original", ORIGINAL_CATEGORY_GROUPING)
    summary = summarize_group_proximity(proximity, bootstrap_iterations=20)
    pairs = summarize_chain_pairs_by_group(
        steps,
        "stdi_vs_original",
        ORIGINAL_CATEGORY_GROUPING,
    )

    composition = build_group_composition_figure(summary)
    intervals = build_group_proximity_interval_figure(summary)
    boxes = build_group_proximity_boxplot(proximity, "range_between_chains")
    heatmap = build_chain_pair_heatmap(pairs, "business")

    assert len(composition.data) == 1
    assert len(intervals.data) == 1
    assert len(boxes.data) == 2
    assert len(heatmap.data) == 1
    assert heatmap.data[0].type == "heatmap"
