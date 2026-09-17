import json

import pytest

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
    folder = tmp_path / "simulation_01_graph"
    folder.mkdir()
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
