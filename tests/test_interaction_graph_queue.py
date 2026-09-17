from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from misinformation_simulation.apps import interaction_graph_queue as queue
from misinformation_simulation.apps import interaction_graph_run_job as run_job
from misinformation_simulation.apps import interaction_graph_sections as sections
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form
from misinformation_simulation.simulation.persistence import _persist_results
from misinformation_simulation.simulation.types import SimulationResult


def test_queue_keeps_independent_snapshots_and_order() -> None:
    nodes = [create_default_node_form(1)]
    graphs = []
    queue.add_graph(graphs, "First", nodes)
    nodes[0]["label"] = "Changed"
    queue.add_graph(graphs, "Second", nodes)
    queue.move_graph(graphs, 1, -1)

    assert [graph["name"] for graph in graphs] == ["Second", "First"]
    assert graphs[1]["nodes"][0]["label"] == "Node 1"
    assert queue.output_prefix_for_graph("batch", 1, "A/B") == "batch_01_a_b"
    assert queue.output_prefix_for_graph("batch", 2, "A/B") == "batch_02_a_b"


def test_queue_rejects_invalid_graph() -> None:
    with pytest.raises(ValueError, match="model"):
        queue.add_graph([], "Bad", [{**create_default_node_form(1), "model": ""}])


def test_run_queue_continues_after_one_graph_fails(tmp_path) -> None:
    nodes = [create_default_node_form(1)]
    graphs = []
    queue.add_graph(graphs, "First", nodes)
    queue.add_graph(graphs, "Second", deepcopy(nodes))
    calls = []
    comparison_methods = []
    output_dirs = []

    def fake_run(**kwargs):
        calls.append(kwargs["output_prefix"])
        comparison_methods.append(kwargs["stdi_comparison_method"])
        output_dirs.append(kwargs["output_dir"])
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        return SimpleNamespace(
            step_results=[],
            summary={"rows_processed": 1, "steps_total": 0, "cancelled": False},
            summary_path=None,
            steps_path=None,
        )

    settings = {
        "text_column": "description",
        "title_column": "title",
        "news_id_column": "",
        "max_rows": 1,
        "sleep_seconds": 0,
        "max_requests_per_minute": 0,
        "retry_attempts": 1,
        "allow_title_fallback": True,
        "topic_drift_model": "model",
        "topic_drift_provider": "gemini",
        "output_dir": str(tmp_path),
        "output_prefix": "batch",
    }

    job = run_job.GraphRunJob()
    run_job._run_graph_queue(
        job=job,
        df=pd.DataFrame([{"title": "t", "description": "d"}]),
        graphs=graphs,
        settings=settings,
        runner=fake_run,
        queue_mode=True,
    )
    events = list(job.events.queue)
    bundles = [payload for kind, payload in events if kind == "bundle"]
    messages = [payload for kind, payload in events if kind == "progress"]

    assert calls == ["batch_01_first", "batch_02_second"]
    assert comparison_methods == ["cluster", "cluster"]
    assert output_dirs == [tmp_path / name for name in calls]
    assert all(directory.is_dir() for directory in output_dirs)
    assert [bundle["status"] for bundle in bundles] == ["failed", "completed"]
    assert bundles[1]["name"] == "Second"
    assert any("failed: provider unavailable" in message for message in messages)
    assert events[-1] == ("done", {"completed": 1, "failed": 1, "cancelled": False})


def test_cancelled_run_skips_remaining_graphs(tmp_path) -> None:
    nodes = [create_default_node_form(1)]
    graphs = []
    queue.add_graph(graphs, "First", nodes)
    queue.add_graph(graphs, "Second", nodes)
    calls = []
    job = run_job.GraphRunJob()

    def fake_run(**kwargs):
        calls.append(kwargs["output_prefix"])
        job.cancel_event.set()
        return SimpleNamespace(
            step_results=[],
            summary={"rows_processed": 1, "steps_total": 0, "cancelled": True},
            summary_path=None,
            steps_path=None,
        )

    settings = {
        "text_column": "description",
        "title_column": "title",
        "news_id_column": "",
        "max_rows": 1,
        "sleep_seconds": 0,
        "max_requests_per_minute": 0,
        "retry_attempts": 1,
        "allow_title_fallback": True,
        "topic_drift_model": "model",
        "topic_drift_provider": "gemini",
        "output_dir": str(tmp_path),
        "output_prefix": "batch",
    }
    run_job._run_graph_queue(
        job=job,
        df=pd.DataFrame([{"title": "t", "description": "d"}]),
        graphs=graphs,
        settings=settings,
        runner=fake_run,
        queue_mode=True,
    )
    events = list(job.events.queue)

    assert calls == ["batch_01_first"]
    assert [payload["status"] for kind, payload in events if kind == "bundle"] == ["cancelled"]
    assert events[-1] == ("done", {"completed": 0, "failed": 0, "cancelled": True})


def test_single_graph_run_creates_named_folder_and_avoids_overwrite(tmp_path) -> None:
    nodes = [create_default_node_form(1)]
    graphs = [{"name": "Investigative skeptic", "nodes": nodes}]
    settings = {
        "text_column": "description",
        "title_column": "title",
        "news_id_column": "",
        "max_rows": 1,
        "sleep_seconds": 0,
        "max_requests_per_minute": 0,
        "retry_attempts": 1,
        "allow_title_fallback": True,
        "topic_drift_model": "model",
        "topic_drift_provider": "gemini",
        "output_dir": str(tmp_path),
        "output_prefix": "simulation_ui_20260917_123456",
    }

    def fake_run(**kwargs):
        result = SimulationResult(
            summary={"rows_processed": 0, "steps_total": 0, "cancelled": False},
            step_results=[],
        )
        result.summary_path, result.steps_path = _persist_results(
            result=result,
            output_dir=kwargs["output_dir"],
            output_prefix=kwargs["output_prefix"],
        )
        return result

    folders = []
    for _ in range(2):
        job = run_job.GraphRunJob()
        run_job._run_graph_queue(
            job=job,
            df=pd.DataFrame([{"title": "t", "description": "d"}]),
            graphs=graphs,
            settings=settings,
            runner=fake_run,
            queue_mode=False,
        )
        bundle = next(payload for kind, payload in job.events.queue if kind == "bundle")
        summary_path = Path(bundle["summary_path"])
        steps_path = Path(bundle["steps_path"])
        folders.append(summary_path.parent)
        assert summary_path.is_file()
        assert steps_path.is_file()
        assert summary_path.name == f"{summary_path.parent.name}_summary.json"
        assert steps_path.name == f"{steps_path.parent.name}_steps.jsonl"

    assert folders == [
        tmp_path / "simulation_ui_20260917_123456_01_investigative_skeptic",
        tmp_path / "simulation_ui_20260917_123456_01_investigative_skeptic_02",
    ]


def test_cancel_button_signals_active_job(monkeypatch) -> None:
    job = run_job.GraphRunJob()
    messages = []
    state = SimpleNamespace(
        run_job=job,
        run_messages=[],
        run_bundles=[],
        run_bundle=None,
    )
    fake_st = SimpleNamespace(
        session_state=state,
        button=lambda label, **_kwargs: label == "Cancel simulation",
        warning=messages.append,
        info=messages.append,
        code=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(sections, "st", fake_st)

    sections._render_run_monitor.__wrapped__()

    assert job.cancel_event.is_set()
    assert any("Cancellation requested" in message for message in messages)


def test_add_graphs_from_folder_in_filename_order_and_reports_bad_files(tmp_path) -> None:
    import json

    payload = {
        "nodes": [
            {
                "node_id": "node-a",
                "model": "model",
                "provider": "gemini",
                "personality": "persona",
            }
        ]
    }
    (tmp_path / "b.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("ignored", encoding="utf-8")
    graphs = []

    added, errors = queue.add_graphs_from_directory(graphs, tmp_path)

    assert added == ["a.json", "b.json"]
    assert [graph["name"] for graph in graphs] == ["a", "b"]
    assert len(errors) == 1 and errors[0].startswith("broken.json:")
    assert graphs[0]["nodes"][0]["node_id"] == "node-a"


def test_add_graphs_from_folder_requires_json_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="No JSON graph files"):
        queue.add_graphs_from_directory([], tmp_path)
    with pytest.raises(ValueError, match="Select a graph folder"):
        queue.add_graphs_from_directory([], "")
