import json

import pytest

from misinformation_simulation.apps.interaction_graph_results import (
    clear_imported_results,
    find_saved_results,
    load_saved_result,
    remove_imported_result,
)


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
