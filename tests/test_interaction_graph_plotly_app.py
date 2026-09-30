from __future__ import annotations

import pandas as pd

from misinformation_simulation.analysis.interaction_graph_personas import (
    summarize_scenario_contrasts,
)
from misinformation_simulation.analysis.interaction_graph_plotly import (
    build_case_component_figure,
    build_component_figure,
    build_contrast_interval_figure,
    build_distribution_figure,
    build_evolution_figure,
    build_iteration_distribution_figure,
    build_persona_boxplot,
    build_persona_position_boxplot,
    build_scenario_difference_boxplot,
    build_transition_pair_figure,
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


def _paired_scenario_steps() -> pd.DataFrame:
    steps = _steps().copy()
    steps["news_id"] = "news-1"
    steps["metadata_title"] = "Example news"
    steps["source_text"] = "source"
    steps["rewritten_text"] = "rewrite"
    steps["run_id"] = steps["chain_label"]
    first_chain = steps["chain_label"].eq("01 · SSSS")
    steps.loc[first_chain, "chain_code"] = "CCPP"
    steps.loc[~first_chain, "chain_code"] = "PPCC"
    steps.loc[first_chain & steps["step_index"].eq(1), "node_label"] = "1. Conservative"
    steps.loc[first_chain & steps["step_index"].eq(2), "node_label"] = "2. Progressive"
    steps.loc[~first_chain & steps["step_index"].eq(1), "node_label"] = "1. Progressive"
    steps.loc[~first_chain & steps["step_index"].eq(2), "node_label"] = "2. Conservative"
    steps.loc[steps["step_index"].eq(1), "source_node_label"] = "description"
    steps.loc[first_chain & steps["step_index"].eq(2), "source_node_label"] = "1. Conservative"
    steps.loc[~first_chain & steps["step_index"].eq(2), "source_node_label"] = "1. Progressive"
    return steps


def test_persona_transition_and_contrast_figures() -> None:
    steps = _paired_scenario_steps()
    summary = summarize_scenario_contrasts(steps, bootstrap_iterations=20)

    personas = build_persona_boxplot(steps, "stdi_incremental")
    positions = build_persona_position_boxplot(steps, "stdi_incremental")
    transitions = build_transition_pair_figure(
        steps,
        "stdi_incremental",
        "Conservative -> Progressive",
        "Progressive -> Conservative",
    )
    contrast_boxes = build_scenario_difference_boxplot(steps, "stdi_vs_original")
    intervals = build_contrast_interval_figure(summary)
    case_components = build_case_component_figure({"Theme": 0.1, "Entities": 0.2})

    assert len(personas.data) == 2
    assert len(positions.data) == 4
    assert len(transitions.data) == 3
    assert len(contrast_boxes.data) == 1
    assert len(intervals.data) == 1
    assert list(case_components.data[0].y) == [0.1, 0.2]
