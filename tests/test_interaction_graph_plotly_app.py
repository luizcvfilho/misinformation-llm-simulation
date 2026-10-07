from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

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
from misinformation_simulation.apps import interaction_graph_analysis_plotly_app as analysis_app
from misinformation_simulation.simulation.types import EVALUATION_METRICS


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
    steps["execution_id"] = "example-execution"
    steps["execution_label"] = "Example execution"
    steps["source_path"] = steps["run_id"]
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


def test_analysis_filters_transmission_modes_before_rendering(monkeypatch) -> None:
    legacy = _paired_scenario_steps().assign(metadata_rewrite_mode="legacy")
    interpretive = _paired_scenario_steps().assign(metadata_rewrite_mode="interpretive")
    interpretive["stdi_vs_original"] = 0.8
    steps = pd.concat([legacy, interpretive], ignore_index=True)
    monkeypatch.setattr(analysis_app, "load_steps", lambda *_args: (steps, 4))
    rendered = []

    def capture(frame, *_args):
        rendered.append(frame.copy())

    for render_name in (
        "_render_overview",
        "_render_news_group_analysis",
        "_render_persona_analysis",
        "_render_transition_analysis",
        "_render_scenario_contrasts",
        "_render_case_explorer",
    ):
        monkeypatch.setattr(analysis_app, render_name, capture)

    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app import main\n"
        "main()"
    ).run(timeout=30)
    assert not app.exception
    assert len(rendered) == 6
    assert all(set(frame["metadata_rewrite_mode"]) == {"interpretive"} for frame in rendered)

    rendered.clear()
    control = next(control for control in app.selectbox if control.label == "Transmission mode")
    control.set_value("legacy").run(timeout=30)
    assert not app.exception
    assert len(rendered) == 6
    assert all(set(frame["metadata_rewrite_mode"]) == {"legacy"} for frame in rendered)


def test_analysis_evaluation_selector_updates_every_tab_and_component(monkeypatch):
    steps = _paired_scenario_steps().assign(
        metadata_rewrite_mode="faithful", metadata_stdi_comparison_version="dual_stdi_v4"
    )
    for suffix in ("vs_original", "incremental"):
        for metric in EVALUATION_METRICS:
            steps[f"{metric}_cluster_{suffix}"] = 0.2
            steps[f"{metric}_llm_judge_{suffix}"] = 0.8
        steps[f"stdi_{suffix}"] = 0.5
    steps["stdi_cumulative"] = steps["step_index"] * 0.5
    monkeypatch.setattr(analysis_app, "load_steps", lambda *_args: (steps, 2))
    rendered = []
    for name in (
        "_render_overview",
        "_render_news_group_analysis",
        "_render_persona_analysis",
        "_render_transition_analysis",
        "_render_scenario_contrasts",
        "_render_case_explorer",
    ):
        monkeypatch.setattr(analysis_app, name, lambda frame, *_args: rendered.append(frame.copy()))
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app import main\n"
        "main()"
    ).run(timeout=30)
    assert not app.exception
    selector = app.selectbox(key="analysis_stdi_evaluation")
    assert selector.options == ["Dual", "Cluster", "LLM"]
    for evaluation, expected in (("cluster", 0.2), ("llm_judge", 0.8), ("dual", 0.5)):
        rendered.clear()
        selector.set_value(evaluation).run(timeout=30)
        assert not app.exception
        assert len(rendered) == 6
        for frame in rendered:
            for suffix in ("vs_original", "incremental"):
                for metric in EVALUATION_METRICS:
                    assert frame[f"{metric}_{suffix}"].eq(expected).all()
            assert (frame["stdi_cumulative"] == frame["step_index"] * expected).all()
    assert steps["stdi_vs_original"].eq(0.5).all()
    missing_chain = steps["chain_label"].iloc[0]
    steps.loc[steps["chain_label"].eq(missing_chain), "stdi_llm_judge_vs_original"] = float("nan")
    steps.loc[steps["chain_label"].eq(missing_chain), "stdi_llm_judge_incremental"] = float("nan")
    app.run(timeout=30)
    app.selectbox(key="analysis_stdi_evaluation").set_value("llm_judge").run(timeout=30)
    rendered.clear()
    chains = next(control for control in app.multiselect if control.label == "Cadeias")
    chains.set_value([missing_chain]).run(timeout=30)
    assert not app.exception
    assert app.warning
    assert not rendered
    chains.set_value(steps["chain_label"].unique().tolist()).run(timeout=30)
    cluster_only = _paired_scenario_steps().assign(
        metadata_rewrite_mode="faithful",
        metadata_stdi_comparison_version="cluster_v4",
        execution_id="cluster-execution",
        execution_label="Cluster execution",
        stdi_vs_original=0.3,
        stdi_incremental=0.3,
    )
    monkeypatch.setattr(
        analysis_app, "load_steps", lambda *_args: (pd.concat([steps, cluster_only]), 4)
    )
    app.run(timeout=30)
    app.selectbox(key="analysis_stdi_evaluation").set_value("llm_judge").run(timeout=30)
    rendered.clear()
    active = next(control for control in app.selectbox if control.label == "Active execution")
    active.set_value("cluster-execution").run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="analysis_stdi_evaluation").options == ["Cluster"]
    assert len(rendered) == 6
    assert all(frame["stdi_vs_original"].eq(0.3).all() for frame in rendered)


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


def test_analysis_switches_executions_without_pooling_shared_news_and_chain_ids(
    tmp_path,
    monkeypatch,
) -> None:
    newest = str(tmp_path / "simulation_ui_20261004_000000")
    older = str(tmp_path / "simulation_ui_20261003_223837")
    first = _paired_scenario_steps().assign(
        metadata_rewrite_mode="legacy", execution_id=newest, execution_label="Latest execution"
    )
    second = first.assign(execution_id=older, execution_label="Older execution")
    second["stdi_vs_original"] = 0.9
    steps = pd.concat([second, first], ignore_index=True)
    monkeypatch.setattr(analysis_app, "load_steps", lambda *_args: (steps, 4))
    rendered = []

    def capture(frame, *_args):
        rendered.append(frame.copy())

    for render_name in (
        "_render_overview",
        "_render_news_group_analysis",
        "_render_persona_analysis",
        "_render_transition_analysis",
        "_render_scenario_contrasts",
        "_render_case_explorer",
    ):
        monkeypatch.setattr(analysis_app, render_name, capture)

    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app import main\n"
        "main()"
    ).run(timeout=30)
    assert not app.exception
    assert len(rendered) == 6
    assert all(set(frame["execution_id"]) == {newest} for frame in rendered)

    rendered.clear()
    control = next(control for control in app.selectbox if control.label == "Active execution")
    control.set_value(older).run(timeout=30)
    assert not app.exception
    assert len(rendered) == 6
    assert all(set(frame["stdi_vs_original"]) == {0.9} for frame in rendered)

    selection = next(control for control in app.multiselect if control.label == "Executions")
    rendered.clear()
    selection.set_value([]).run(timeout=30)
    assert not app.exception
    assert app.warning
    assert not rendered


def test_custom_chain_pairs_render_summary_boxplot_and_cases() -> None:
    steps = _paired_scenario_steps()
    steps["chain_code"] = steps["chain_code"].replace({"CCPP": "custom_a", "PPCC": "custom_b"})
    contrasts = (("custom_a", "custom_b", "Custom chain comparison"),)
    summary = summarize_scenario_contrasts(steps, contrasts=contrasts, bootstrap_iterations=20)
    figure = build_scenario_difference_boxplot(steps, "stdi_vs_original", contrasts=contrasts)

    assert summary["paired_news"].tolist() == [1]
    assert [trace.name for trace in figure.data] == ["custom_a - custom_b"]
    assert list(figure.data[0].y) == list(summary["difference"])

    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app "
        "import _render_scenario_contrasts, _render_contrast_cases\n"
        "from misinformation_simulation.analysis.interaction_graph_visualization "
        "import available_metrics\n"
        "import streamlit as st\n"
        "steps = st.session_state['steps']\n"
        "_render_scenario_contrasts(steps, available_metrics(steps))\n"
        "_render_contrast_cases(steps, available_metrics(steps))\n"
    )
    app.session_state["steps"] = steps
    app.run(timeout=30)
    assert not app.exception
    assert any(control.label == "Chain A" for control in app.selectbox)
    assert any(control.label == "Caso para leitura" for control in app.selectbox)


def test_neutral_relay_renders_personas_transitions_and_preset_contrasts() -> None:
    steps = _paired_scenario_steps()
    steps["chain_code"] = steps["chain_code"].replace({"CCPP": "DDNN", "PPCC": "NNDD"})
    for column in ("node_label", "source_node_label"):
        steps[column] = (
            steps[column]
            .str.replace("Conservative", "Conspiratorial")
            .str.replace("Progressive", "Neutral relay")
        )
    summary = summarize_scenario_contrasts(steps, bootstrap_iterations=20)
    assert summary["label_a"].tolist() == ["DDNN"]
    assert summary["label_b"].tolist() == ["NNDD"]
    figure = build_persona_boxplot(steps, "stdi_incremental")
    assert {trace.name for trace in figure.data} == {"D · Conspiratorial", "N · Neutral relay"}

    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app import "
        "_render_persona_analysis, _render_transition_analysis, "
        "_render_scenario_contrasts, _render_contrast_cases\n"
        "from misinformation_simulation.analysis.interaction_graph_visualization "
        "import available_metrics\n"
        "import streamlit as st\n"
        "steps = st.session_state['steps']\n"
        "_render_persona_analysis(steps)\n"
        "_render_transition_analysis(steps)\n"
        "_render_scenario_contrasts(steps, available_metrics(steps))\n"
        "_render_contrast_cases(steps, available_metrics(steps))\n"
    )
    app.session_state["steps"] = steps
    app.run(timeout=30)
    assert not app.exception
    comparisons = [control for control in app.radio if control.label == "Chain comparisons"]
    assert comparisons
    assert all(control.value == "Preset contrasts" for control in comparisons)
    assert any(control.label == "Caso para leitura" for control in app.selectbox)


def test_overview_handles_a_single_iteration() -> None:
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app "
        "import _render_overview\n"
        "from misinformation_simulation.analysis.interaction_graph_visualization "
        "import available_metrics\n"
        "import streamlit as st\n"
        "steps = st.session_state['steps']\n"
        "_render_overview(steps, 'stdi_vs_original', available_metrics(steps))\n"
    )
    app.session_state["steps"] = _steps().loc[lambda frame: frame["step_index"].eq(1)]
    app.run(timeout=30)
    assert not app.exception


def test_analysis_loads_multiple_folder_inputs(tmp_path, monkeypatch) -> None:
    steps = _paired_scenario_steps()
    steps["chain_code"] = steps["chain_code"].replace({"CCPP": "custom_a", "PPCC": "custom_b"})
    for execution in ("first", "second"):
        folder = tmp_path / execution
        folder.mkdir()
        for index, (chain, rows) in enumerate(steps.groupby("chain_code"), start=1):
            rows.to_json(
                folder / f"batch_{index:02d}_{chain}_steps.jsonl",
                orient="records",
                lines=True,
            )
    monkeypatch.setattr(analysis_app, "DEFAULT_RUNS_DIR", tmp_path)
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis_plotly_app import main\n"
        "main()"
    ).run(timeout=30)
    assert not app.exception

    folder_input = next(
        control for control in app.text_area if control.label.startswith("Run folders")
    )
    folder_input.set_value(f"{tmp_path / 'first'}\n{tmp_path / 'second'}\n{tmp_path}").run(
        timeout=30
    )
    assert not app.exception
    selection = next(control for control in app.multiselect if control.label == "Executions")
    assert len(selection.value) == 2
    assert next(metric for metric in app.metric if metric.label == "Chain step files").value == "2"

    active = next(control for control in app.selectbox if control.label == "Active execution")
    active.set_value(str(tmp_path / "second")).run(timeout=30)
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label == "Chain step files").value == "2"
