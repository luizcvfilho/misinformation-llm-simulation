from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps import interaction_graph_analysis as analysis
from misinformation_simulation.apps import interaction_graph_app as studio
from misinformation_simulation.apps.interaction_graph_results import load_saved_result
from misinformation_simulation.apps.interaction_graph_run_job import GraphRunJob
from misinformation_simulation.simulation.types import SimulationStepResult


def _saved_run(root: Path, *, score: float = 0.3) -> Path:
    folder = root / "simulation_ui_20261007_100000" / "01_SSSS"
    folder.mkdir(parents=True, exist_ok=True)
    summary = folder / "01_SSSS_summary.json"
    summary.write_text(
        json.dumps(
            {
                "graph_name": "Saved graph",
                "rows_processed": 2,
                "steps_total": 2,
                "steps_success": 1,
                "steps_error": 1,
                "cancelled": True,
                "stdi_comparison_method": "cluster",
            }
        ),
        encoding="utf-8",
    )
    records = []
    for index, status in enumerate(("success", "error"), start=1):
        records.append(
            SimulationStepResult(
                news_id=f"article-{index}",
                step_index=1,
                node_id="node-1",
                node_label="Investigative skeptic",
                source_node_id="original",
                source_node_label="description",
                provider="test",
                model="test",
                personality="persona",
                source_text="Original text",
                rewritten_text="Rewritten text" if status == "success" else None,
                target_language="en",
                target_language_source="row.language",
                rewrite_status=status,
                rewrite_error="Example failure" if status == "error" else None,
                stdi_vs_original=score if status == "success" else None,
                stdi_incremental=score if status == "success" else None,
                stdi_cumulative=score if status == "success" else None,
                metadata={
                    "title": "Article",
                    "rewrite_mode": "faithful",
                    "stdi_comparison_version": "cluster_v4",
                },
            ).to_record()
        )
    summary.with_name("01_SSSS_steps.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return summary


def _capture_charts(monkeypatch):
    rendered = []
    for name in (
        "_render_overview",
        "_render_news_group_analysis",
        "_render_persona_analysis",
        "_render_transition_analysis",
        "_render_scenario_contrasts",
        "_render_case_explorer",
    ):
        monkeypatch.setattr(analysis, name, lambda frame, *_args: rendered.append(frame.copy()))
    return rendered


def test_analysis_cache_detects_saved_changes_and_retains_failures(tmp_path):
    _saved_run(tmp_path)
    directories = (str(tmp_path),)
    fingerprint = analysis.source_fingerprint(directories)
    before, _ = analysis.load_steps(
        directories, analysis.ANALYSIS_CACHE_SCHEMA_VERSION, fingerprint
    )
    assert before["rewrite_status"].tolist() == ["success", "error"]
    _saved_run(tmp_path, score=0.75)
    updated_fingerprint = analysis.source_fingerprint(directories)
    assert updated_fingerprint != fingerprint
    after, _ = analysis.load_steps(
        directories,
        analysis.ANALYSIS_CACHE_SCHEMA_VERSION,
        updated_fingerprint,
    )
    assert after.loc[after["rewrite_status"].eq("success"), "stdi_vs_original"].tolist() == [0.75]


def test_analysis_details_keep_errors_outside_analytical_charts(tmp_path, monkeypatch):
    _saved_run(tmp_path)
    monkeypatch.setattr(analysis, "DEFAULT_RUNS_DIR", tmp_path)
    rendered = _capture_charts(monkeypatch)
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis import render_analysis\n"
        "render_analysis()"
    ).run(timeout=45)
    assert not app.exception
    assert [tab.label for tab in app.tabs][-1] == "Run details"
    assert len(rendered) == 6
    assert all(frame["rewrite_status"].tolist() == ["success"] for frame in rendered)
    assert next(metric for metric in app.metric if metric.label == "Errors").value == "1"
    assert any("cancelled" in warning.value for warning in app.warning)
    news = app.selectbox(key="selected_news_id")
    assert set(news.options) == {"article-1", "article-2"}
    news.set_value("article-2").run(timeout=45)
    assert not app.exception
    assert any(error.value == "Example failure" for error in app.error)


def test_analyze_run_switches_page_and_preserves_simulation_settings(tmp_path, monkeypatch):
    summary = _saved_run(tmp_path)
    monkeypatch.setattr(analysis, "DEFAULT_RUNS_DIR", tmp_path)
    monkeypatch.setattr(studio, "_refresh_graph_backend_if_stale", lambda: None)
    monkeypatch.setattr(
        studio.interaction_graph_sections,
        "render_sidebar",
        lambda: (pd.DataFrame({"description": ["News"], "title": ["Title"]}), "test.csv"),
    )
    _capture_charts(monkeypatch)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps import interaction_graph_app as studio\n"
        "studio.main()\n"
        "if st.button('Return to Simulation'):\n"
        "    st.switch_page(st.Page(studio.render_simulation_page, title='Simulation', "
        "url_path='simulation'))\n"
    )
    app.session_state["run_bundles"] = [load_saved_result(summary)]
    active_job = GraphRunJob()
    app.session_state["run_job"] = active_job
    counts_key = "_dataset_builder_requests_description_;_1_counts"
    app.session_state[counts_key] = {"News": 1}
    app.run(timeout=45)
    assert not app.exception
    app.text_input(key="current_graph_name").set_value("Edited graph").run(timeout=45)
    app.number_input(key="simulation_execution_settings_max_rows").set_value(7).run(timeout=45)
    next(button for button in app.button if button.label == "Analyze this run").click().run(
        timeout=45
    )
    assert not app.exception
    assert any(header.value == "STDI exploratory analysis" for header in app.header)
    assert app.session_state["run_job"] is active_job
    assert any(expander.label == "Simulation progress" for expander in app.expander)
    assert app.selectbox(key="analysis_main_active_execution").value == str(summary.parent.parent)
    assert app.selectbox(key="analysis_detail_source").value == str(
        summary.with_name("01_SSSS_steps.jsonl")
    )
    next(button for button in app.button if button.label == "Return to Simulation").click().run(
        timeout=45,
    )
    assert not app.exception
    assert app.text_input(key="current_graph_name").value == "Edited graph"
    assert app.number_input(key="simulation_execution_settings_max_rows").value == 7
    assert app.session_state[counts_key] == {"News": 1}


def test_analysis_keeps_details_when_every_step_failed(tmp_path, monkeypatch):
    summary = _saved_run(tmp_path)
    summary.write_text(
        json.dumps(
            {
                "rows_processed": 2,
                "steps_total": 2,
                "steps_success": 0,
                "steps_error": 2,
            }
        ),
        encoding="utf-8",
    )
    path = summary.with_name("01_SSSS_steps.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record.update(rewrite_status="error", rewrite_error="Example failure")
        for metric in ("stdi_vs_original", "stdi_incremental", "stdi_cumulative"):
            record[metric] = None
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    monkeypatch.setattr(analysis, "DEFAULT_RUNS_DIR", tmp_path)
    rendered = _capture_charts(monkeypatch)
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_analysis import render_analysis\n"
        "render_analysis()"
    ).run(timeout=45)
    assert not app.exception
    assert not rendered
    assert next(metric for metric in app.metric if metric.label == "Errors").value == "2"
    assert any("Run details" in warning.value for warning in app.warning)
