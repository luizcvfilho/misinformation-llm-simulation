import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps import interaction_graph_sections
from misinformation_simulation.apps.interaction_graph_categories import (
    build_category_comparison_dataframe,
)
from misinformation_simulation.apps.interaction_graph_results import (
    clear_imported_results,
    find_saved_results,
    load_saved_result,
    remove_imported_result,
)
from misinformation_simulation.simulation.types import SimulationStepResult


def test_import_saved_result_from_named_folder(tmp_path):
    folder = tmp_path / "batch" / "simulation_01_graph"
    folder.mkdir(parents=True)
    summary_path = folder / "simulation_01_graph_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "rows_processed": 1,
                "steps_total": 0,
                "steps_success": 0,
                "steps_error": 0,
                "cancelled": True,
            }
        )
    )
    (folder / "simulation_01_graph_steps.jsonl").write_text("")

    assert find_saved_results(tmp_path) == [summary_path]
    bundle = load_saved_result(summary_path)
    assert bundle["status"] == "cancelled"
    assert bundle["source"] == "imported"
    assert bundle["steps_df"].empty
    assert bundle["output_prefix"] == "simulation_01_graph"
    assert bundle["graph_payload"] is None


def test_imported_result_retains_news_categories_for_comparison(tmp_path) -> None:
    summary_path = tmp_path / "run_summary.json"
    steps_path = tmp_path / "run_steps.jsonl"
    summary_path.write_text(json.dumps({"rows_processed": 1, "steps_total": 1}))
    step = SimulationStepResult(
        news_id="article-1",
        step_index=1,
        node_id="node-1",
        node_label="Node 1",
        source_node_id="original",
        source_node_label="description",
        provider="gemini",
        model="model",
        personality="persona",
        source_text="original",
        rewritten_text="rewritten",
        target_language="en",
        target_language_source="row.language",
        rewrite_status="success",
        rewrite_error=None,
        stdi_vs_original=0.25,
        metadata={"title": "Article", "category": "politics; top"},
    )
    steps_path.write_text(json.dumps(step.to_record()) + "\n")

    bundle = load_saved_result(summary_path)
    comparison = build_category_comparison_dataframe([bundle])

    assert bundle["news_summary_df"].iloc[0]["category"] == "politics; top"
    assert set(comparison["category"]) == {"politics", "top"}
    assert set(comparison["mean_stdi_vs_original"]) == {0.25}

    stale_bundle = {
        **bundle,
        "steps_df": bundle["steps_df"].drop(columns="metadata_category"),
        "news_summary_df": bundle["news_summary_df"].drop(columns="category"),
    }
    interaction_graph_sections._refresh_category_data_from_saved_results([stale_bundle])
    assert stale_bundle["steps_df"].iloc[0]["metadata_category"] == "politics; top"
    assert stale_bundle["news_summary_df"].iloc[0]["category"] == "politics; top"


def test_import_requires_matching_steps_file(tmp_path):
    summary_path = tmp_path / "old_summary.json"
    summary_path.write_text(json.dumps({"rows_processed": 1, "steps_total": 1}))
    with pytest.raises(ValueError, match="Matching steps file"):
        load_saved_result(summary_path)


def test_remove_and_clear_only_imported_results() -> None:
    current = {"name": "current"}
    first = {"name": "first", "source": "imported"}
    second = {"name": "second", "source": "imported"}
    bundles = [current, first, second]

    assert not remove_imported_result(bundles, 0)
    assert bundles == [current, first, second]
    assert remove_imported_result(bundles, 1)
    assert bundles == [current, second]
    assert clear_imported_results(bundles) == 1
    assert bundles == [current]


def test_results_ui_imports_compact_runs_and_distinguishes_executions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph_name = "A descriptive graph name that exceeds the filename limit"
    prefix = "01_a_descriptive_graph_name_that_exc"
    executions = ["simulation_ui_20261003_223837", "simulation_ui_20261004_000000"]
    paths = []
    for execution in executions:
        folder = Path("output/interaction_graph/app_runs") / execution / prefix
        folder.mkdir(parents=True)
        path = folder / f"{prefix}_summary.json"
        path.write_text(
            json.dumps(
                {
                    "rows_processed": 0,
                    "steps_total": 0,
                    "steps_success": 0,
                    "steps_error": 0,
                    "graph_name": graph_name,
                }
            ),
            encoding="utf-8",
        )
        (folder / f"{prefix}_steps.jsonl").write_text("", encoding="utf-8")
        paths.append(path)

    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_sections "
        "import render_results_tab\n"
        "if 'run_bundles' not in st.session_state:\n"
        "    st.session_state.run_bundles = []\n"
        "render_results_tab()\n"
    ).run(timeout=30)
    assert not app.exception
    for path in paths:
        app.text_input(key="saved_result_path").set_value(str(path)).run(timeout=30)
        next(button for button in app.button if button.label == "Import result").click().run(
            timeout=30
        )
        assert not app.exception

    control = next(control for control in app.selectbox if control.label == "Inspect graph result")
    assert control.options == [
        f"{index}. {graph_name} — {execution}"
        for index, execution in enumerate(executions, start=1)
    ]
    assert app.dataframe[0].value["execution"].tolist() == executions
    control.set_value(1).run(timeout=30)
    assert not app.exception
    assert any(executions[1] in caption.value for caption in app.caption)
