from __future__ import annotations

import pandas as pd

from misinformation_simulation.analysis.interaction_graph_plotly import (
    build_component_figure,
    build_distribution_figure,
    build_evolution_figure,
    build_iteration_distribution_figure,
)


def _steps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "news_id": news_id,
                "step_index": iteration,
                "rewrite_status": "success",
                "chain_label": chain_label,
                "stdi_vs_original": stdi,
                "stdi_incremental": stdi,
                "stdi_cumulative": stdi * iteration,
                "theme_drift_vs_original": 0.1,
                "subtopic_drift_vs_original": 0.2,
                "entity_drift_vs_original": 0.3,
                "relation_drift_vs_original": 0.4,
                "contradiction_drift_vs_original": 0.0,
                "vad_drift_vs_original": 0.05,
                "valence_drift_vs_original": 0.04,
                "arousal_drift_vs_original": 0.06,
                "dominance_drift_vs_original": 0.05,
            }
            for news_id, chain_label, iteration, stdi in (
                ("news-1", "01 · SSSS", 1, 0.2),
                ("news-1", "01 · SSSS", 2, 0.3),
                ("news-2", "02 · CCCC", 1, 0.25),
                ("news-2", "02 · CCCC", 2, 0.35),
            )
        ]
    )


def test_plotly_figures_include_the_requested_views() -> None:
    steps = _steps()

    evolution = build_evolution_figure(steps, "stdi_vs_original")
    components = build_component_figure(
        steps,
        ["theme_drift_vs_original", "arousal_drift_vs_original"],
    )
    distribution = build_distribution_figure(steps, "stdi_incremental", 2)
    iteration_distribution = build_iteration_distribution_figure(steps, "stdi_incremental")

    assert evolution.layout.hovermode == "x unified"
    assert {trace.name for trace in evolution.data} >= {
        "01 · SSSS",
        "02 · CCCC",
        "Média global",
    }
    assert len(components.data) == 6
    assert components.layout.height == 440
    assert {trace.name for trace in distribution.data} == {"01 · SSSS", "02 · CCCC"}
    assert {trace.name for trace in iteration_distribution.data} == {"1", "2"}
    assert "incremental" in iteration_distribution.layout.title.text
