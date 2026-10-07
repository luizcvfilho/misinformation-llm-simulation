from __future__ import annotations

import json
from pathlib import Path

import pytest

from misinformation_simulation.analysis.interaction_graph_visualization import (
    create_static_figures,
    discover_step_paths,
    export_analysis_tables,
    load_interaction_graph_runs,
    summarize_components,
    summarize_metric,
)


def _write_steps(path, *, stdi: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "news_id": "news-1",
                "step_index": 1,
                "rewrite_status": "success",
                "metadata_category": "business; top",
                "metadata_original_topic_domain": "economy_business_and_finance",
                "stdi_vs_original": stdi,
                "stdi_incremental": stdi,
                "stdi_cumulative": stdi,
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
        )
        + "\n",
        encoding="utf-8",
    )


def test_loads_current_runs_and_excludes_old_runs_case_insensitively(tmp_path) -> None:
    current = (
        tmp_path / "simulation_ui_20260918_000717" / "simulation_ui_20260918_000717_01_01_ssss"
    )
    archived = tmp_path / "OLD_RUNS"
    _write_steps(current / "simulation_ui_20260918_000717_01_01_ssss_steps.jsonl", stdi=0.2)
    _write_steps(archived / "archived_steps.jsonl", stdi=0.9)

    paths = discover_step_paths(tmp_path)
    runs = load_interaction_graph_runs(tmp_path)

    assert paths == [current / "simulation_ui_20260918_000717_01_01_ssss_steps.jsonl"]
    assert runs.steps["chain_label"].tolist() == ["01 · SSSS"]
    assert runs.steps["graph_id"].tolist() == ["01"]
    assert runs.steps["metadata_category"].tolist() == ["business; top"]
    assert runs.steps["metadata_rewrite_mode"].tolist() == ["legacy"]
    assert "metadata_original_topic_domain" not in runs.steps
    assert runs.steps["metadata_stdi_comparison_version"].tolist() == ["legacy"]


def test_loads_single_and_multi_graph_run_layouts_together(tmp_path) -> None:
    single = tmp_path / "single_01_01_ssss" / "single_01_01_ssss_steps.jsonl"
    queued = tmp_path / "batch" / "batch_01_02_cccc" / "batch_01_02_cccc_steps.jsonl"
    _write_steps(single, stdi=0.2)
    _write_steps(queued, stdi=0.3)

    runs = load_interaction_graph_runs(tmp_path)

    assert runs.source_paths == (queued, single)
    assert set(runs.steps["chain_code"]) == {"SSSS", "CCCC"}


def test_loader_retains_transmission_mode_alongside_legacy_runs(tmp_path) -> None:
    legacy_path = tmp_path / "legacy_steps.jsonl"
    interpretive_path = tmp_path / "interpretive_steps.jsonl"
    _write_steps(legacy_path, stdi=0.1)
    _write_steps(interpretive_path, stdi=0.4)
    record = json.loads(interpretive_path.read_text(encoding="utf-8"))
    record["metadata_rewrite_mode"] = "interpretive"
    interpretive_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    runs = load_interaction_graph_runs(tmp_path)

    assert set(runs.steps["metadata_rewrite_mode"]) == {"legacy", "interpretive"}
    interpretive = runs.steps.loc[runs.steps["metadata_rewrite_mode"].eq("interpretive")]
    assert interpretive["stdi_vs_original"].tolist() == [0.4]


def test_exports_summaries_and_static_figures(tmp_path, monkeypatch) -> None:
    current = tmp_path / "simulation_ui_20260918_000717_01_01_ssss"
    steps_path = current / "simulation_ui_20260918_000717_01_01_ssss_steps.jsonl"
    _write_steps(steps_path, stdi=0.2)
    runs = load_interaction_graph_runs(tmp_path)

    def write_pngs(_figures, paths, **_kwargs) -> None:
        for path in paths:
            Path(path).write_bytes(b"png")

    monkeypatch.setattr(
        "misinformation_simulation.analysis.interaction_graph_visualization"
        "._is_kaleido_browser_available",
        lambda: True,
    )
    monkeypatch.setattr("plotly.io.write_images", write_pngs)

    metric_summary = summarize_metric(runs.steps, "stdi_vs_original")
    component_summary = summarize_components(runs.steps)
    output_dir = tmp_path / "analysis"
    output_paths = {
        **create_static_figures(runs, output_dir),
        **export_analysis_tables(runs, output_dir),
    }

    assert metric_summary.loc[0, "mean"] == 0.2
    assert len(component_summary) == 9
    assert all(path.is_file() for path in output_paths.values())
    assert output_paths["stdi_evolution_html"].suffix == ".html"
    assert "plotly" in output_paths["stdi_evolution_html"].read_text(encoding="utf-8").lower()
    assert output_paths["figure_export_status"].is_file()
    assert output_paths["stdi_evolution_png"].suffix == ".png"
    assert (output_dir / "final_stdi_by_chain.csv").is_file()
    exported = (output_dir / "successful_steps.csv").read_text(encoding="utf-8")
    assert "metadata_category" in exported
    assert "metadata_original_topic_domain" not in exported


@pytest.mark.parametrize(
    ("run_id", "graph_id", "chain_code"),
    [
        (
            "simulation_ui_20261003_223837_01_progressive_conservative",
            "01",
            "progressive_conservative",
        ),
        ("simulation_ui_20261003_223837_02_02_custom_chain", "02", "custom_chain"),
        ("batch_01_03_cadeia_ação", "03", "cadeia_ação"),
        ("batch_04_My chain.name", "04", "My chain.name"),
        ("simulation_ui_20261003_223837_04_meme", "04", "meme"),
        ("simulation_ui_20261003_223837_01_01_ssss", "01", "SSSS"),
        ("01_progressive_conservative", "01", "progressive_conservative"),
        ("02_02_custom_chain", "02", "custom_chain"),
        ("01_01_ssss", "01", "SSSS"),
        ("01_01_nnnn", "01", "NNNN"),
        ("07_07_ddnn", "07", "DDNN"),
        ("batch_08_08_nndd", "08", "NNDD"),
        ("simulation_ui_20261006_223837_07_07_ddnn", "07", "DDNN"),
        ("free chain name", "free chain name", "free chain name"),
    ],
)
def test_loads_arbitrary_chain_names(tmp_path, run_id, graph_id, chain_code) -> None:
    _write_steps(tmp_path / f"{run_id}_steps.jsonl", stdi=0.2)

    runs = load_interaction_graph_runs(tmp_path)

    assert runs.steps["chain_code"].tolist() == [chain_code]
    assert runs.steps["graph_id"].tolist() == [graph_id]


def test_multiple_folders_keep_executions_distinct_and_deduplicate_overlaps(tmp_path) -> None:
    run_id = "batch_01_01_custom_chain"
    paths = [tmp_path / execution / run_id / f"{run_id}_steps.jsonl" for execution in ("a", "b")]
    _write_steps(paths[0], stdi=0.2)
    _write_steps(paths[1], stdi=0.8)

    runs = load_interaction_graph_runs([tmp_path, tmp_path / "a", tmp_path / "b" / run_id])

    assert runs.source_paths == tuple(paths)
    assert len(runs.steps) == 2
    assert runs.steps["execution_id"].nunique() == 2
    assert runs.steps.groupby("execution_label")["stdi_vs_original"].first().to_dict() == {
        "a": 0.2,
        "b": 0.8,
    }


def test_compact_layouts_keep_executions_distinct(tmp_path) -> None:
    run_id = "01_progressive_conservative"
    batch = tmp_path / "simulation_ui_20261003_223837"
    single = tmp_path / f"simulation_ui_20261004_000000_{run_id}"
    _write_steps(batch / run_id / f"{run_id}_steps.jsonl", stdi=0.2)
    _write_steps(single / f"{run_id}_steps.jsonl", stdi=0.8)

    runs = load_interaction_graph_runs(tmp_path)

    assert set(runs.steps["execution_id"]) == {str(batch), str(single)}
    assert runs.steps["graph_id"].tolist() == ["01", "01"]
    assert set(runs.steps["chain_code"]) == {"progressive_conservative"}


def test_analysis_retains_full_graph_names_from_compact_summaries(tmp_path) -> None:
    graph_name = "A descriptive graph name that exceeds the filename limit"
    run_id = "01_a_descriptive_graph_name_that_exc"
    graph_dir = tmp_path / "simulation_ui_20261004_000000" / run_id
    _write_steps(graph_dir / f"{run_id}_steps.jsonl", stdi=0.2)
    (graph_dir / f"{run_id}_summary.json").write_text(
        json.dumps({"graph_name": graph_name}), encoding="utf-8"
    )

    runs = load_interaction_graph_runs(tmp_path)

    assert runs.steps["chain_label"].tolist() == [f"01 · {graph_name}"]
    assert runs.steps["execution_label"].tolist() == ["simulation_ui_20261004_000000"]
