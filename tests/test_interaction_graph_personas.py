from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.analysis.interaction_graph_personas import (
    build_transition_matrix,
    normalize_persona_label,
    persona_news_values,
    prepare_persona_steps,
    scenario_contrast_cases,
    scenario_contrast_values,
    summarize_persona_components,
    summarize_personas,
    summarize_scenario_contrasts,
    summarize_transition_asymmetry,
    summarize_transitions,
    transition_pair_values,
)


def _record(
    *,
    news_id: str,
    run_id: str,
    chain_code: str,
    step_index: int,
    node_label: str,
    source_node_label: str,
    incremental: float,
    versus_original: float,
) -> dict[str, object]:
    return {
        "news_id": news_id,
        "run_id": run_id,
        "chain_code": chain_code,
        "chain_label": chain_code,
        "step_index": step_index,
        "node_label": node_label,
        "source_node_label": source_node_label,
        "rewrite_status": "success",
        "stdi_incremental": incremental,
        "stdi_vs_original": versus_original,
        "theme_drift_incremental": incremental / 2,
        "subtopic_drift_incremental": incremental / 4,
        "entity_drift_incremental": incremental,
        "relation_drift_incremental": incremental / 3,
        "contradiction_drift_incremental": 0.0,
        "vad_drift_incremental": incremental / 5,
        "source_text": f"source-{run_id}-{news_id}-{step_index}",
        "rewritten_text": f"rewrite-{run_id}-{news_id}-{step_index}",
        "metadata_title": f"Title {news_id}",
    }


def _steps() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for news_id, cp_incremental, pc_incremental in (
        ("news-1", 0.3, 0.1),
        ("news-2", 0.5, 0.2),
    ):
        rows.extend(
            [
                _record(
                    news_id=news_id,
                    run_id="run-cp",
                    chain_code="CP",
                    step_index=1,
                    node_label="1. Conservative",
                    source_node_label="description",
                    incremental=0.1,
                    versus_original=0.1,
                ),
                _record(
                    news_id=news_id,
                    run_id="run-cp",
                    chain_code="CP",
                    step_index=2,
                    node_label="2. Progressive",
                    source_node_label="1. Conservative",
                    incremental=cp_incremental,
                    versus_original=cp_incremental + 0.1,
                ),
                _record(
                    news_id=news_id,
                    run_id="run-pc",
                    chain_code="PC",
                    step_index=1,
                    node_label="1. Progressive",
                    source_node_label="description",
                    incremental=0.2,
                    versus_original=0.2,
                ),
                _record(
                    news_id=news_id,
                    run_id="run-pc",
                    chain_code="PC",
                    step_index=2,
                    node_label="2. Conservative",
                    source_node_label="1. Progressive",
                    incremental=pc_incremental,
                    versus_original=pc_incremental + 0.2,
                ),
            ]
        )
    return pd.DataFrame(rows)


def test_prepares_personas_transitions_and_original_text() -> None:
    prepared = prepare_persona_steps(_steps())

    cp_second = prepared.loc[prepared["chain_code"].eq("CP") & prepared["step_index"].eq(2)].iloc[0]
    assert cp_second["persona_code"] == "P"
    assert cp_second["previous_persona_code"] == "C"
    assert cp_second["transition_code"] == "C -> P"
    assert cp_second["original_text"] == "source-run-cp-news-1-1"


def test_summarizes_personas_components_and_transitions() -> None:
    steps = _steps()
    personas = summarize_personas(steps, bootstrap_iterations=200)
    components = summarize_persona_components(steps)
    transitions = summarize_transitions(steps, bootstrap_iterations=200)
    matrix = build_transition_matrix(transitions)

    progressive = personas.loc[personas["persona_code"].eq("P")].iloc[0]
    assert progressive["news_items"] == 2
    assert progressive["positions"] == "1, 2"
    assert components.loc[components["persona_code"].eq("P"), "dominant_component"].item() == (
        "Entities"
    )
    assert matrix.loc["C", "P"] == pytest.approx(0.4)
    assert matrix.loc["P", "C"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Right", "Conservative"),
        ("2. LEFT", "Progressive"),
        ("Conspiracy", "Conspiratorial"),
        ("Skeptic", "Investigative skeptic"),
        ("ConservativeRight", "Conservative"),
        ("1. Custom commentator", "Custom commentator"),
    ],
)
def test_normalizes_known_aliases_and_preserves_custom_labels(label: str, expected: str) -> None:
    assert normalize_persona_label(label) == expected


def test_alias_runs_share_canonical_transitions_without_reweighting_news() -> None:
    canonical = _steps()
    aliases = canonical.copy()
    aliases["run_id"] = "alias-" + aliases["run_id"]
    for column in ("node_label", "source_node_label"):
        aliases[column] = (
            aliases[column].str.replace("Conservative", "Right").str.replace("Progressive", "Left")
        )
    aliases.loc[aliases["node_label"].str.contains("Left"), "stdi_incremental"] = 0.8
    summary = summarize_transitions(
        pd.concat([canonical, aliases], ignore_index=True), bootstrap_iterations=20
    )
    matrix = build_transition_matrix(summary)

    cp = summary.loc[summary["transition_code"].eq("C -> P")].iloc[0]
    assert len(summary) == 2
    assert cp["observations"] == 4
    assert cp["news_items"] == 2
    assert matrix.loc["C", "P"] == pytest.approx(0.6)
    assert matrix.loc["P", "C"] == pytest.approx(0.15)


def test_transition_matrix_retains_distinct_custom_personas() -> None:
    steps = _steps().copy()
    for column in ("node_label", "source_node_label"):
        steps[column] = (
            steps[column]
            .str.replace("Conservative", "Custom A")
            .str.replace("Progressive", "Custom B")
        )
    summary = summarize_transitions(steps, bootstrap_iterations=20)
    before = summary.copy(deep=True)
    matrix = build_transition_matrix(summary)

    assert matrix.loc["? · Custom A", "? · Custom B"] == pytest.approx(0.4)
    assert matrix.loc["? · Custom B", "? · Custom A"] == pytest.approx(0.15)
    assert pd.isna(matrix.loc["C", "P"])
    pd.testing.assert_frame_equal(summary, before)


def test_summarizes_directional_asymmetry_with_paired_news() -> None:
    asymmetry = summarize_transition_asymmetry(_steps(), bootstrap_iterations=200)

    assert len(asymmetry) == 1
    assert asymmetry.loc[0, "paired_news"] == 2
    assert asymmetry.loc[0, "difference"] == pytest.approx(0.25)
    assert asymmetry.loc[0, "win_rate_a"] == 1.0


def test_summarizes_and_expands_scenario_contrast_cases() -> None:
    steps = _steps()
    summary = summarize_scenario_contrasts(
        steps,
        contrasts=(("CP", "PC", "Reverse order"),),
        bootstrap_iterations=200,
    )
    cases = scenario_contrast_cases(steps, "stdi_vs_original", "CP", "PC")

    assert summary.loc[0, "paired_news"] == 2
    assert summary.loc[0, "difference"] == pytest.approx(0.15)
    assert summary.loc[0, "description"] == "Reverse order"
    assert cases["difference"].tolist() == pytest.approx([0.2, 0.1])
    assert set(cases["original_text_a"]) == {
        "source-run-cp-news-1-1",
        "source-run-cp-news-2-1",
    }


def test_incremental_scenario_contrast_uses_the_final_step_only() -> None:
    summary = summarize_scenario_contrasts(
        _steps(),
        metric="stdi_incremental",
        contrasts=(("CP", "PC", "Reverse order"),),
        bootstrap_iterations=200,
    )

    assert summary.loc[0, "mean_a"] == pytest.approx(0.4)
    assert summary.loc[0, "mean_b"] == pytest.approx(0.15)
    assert summary.loc[0, "difference"] == pytest.approx(0.25)


def test_returns_news_level_values_for_boxplots() -> None:
    steps = _steps()
    personas = persona_news_values(steps, by_position=True)
    transitions = transition_pair_values(
        steps,
        "stdi_incremental",
        "Conservative -> Progressive",
        "Progressive -> Conservative",
    )
    contrasts = scenario_contrast_values(
        steps,
        contrasts=(("CP", "PC", "Reverse order"),),
    )

    assert len(personas) == 8
    assert transitions["difference"].tolist() == pytest.approx([0.3, 0.2])
    assert contrasts["difference"].tolist() == pytest.approx([0.1, 0.2])
