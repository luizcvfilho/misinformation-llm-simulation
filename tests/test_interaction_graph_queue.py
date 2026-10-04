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
    rewrite_modes = []
    output_dirs = []

    def fake_run(**kwargs):
        calls.append(kwargs["output_prefix"])
        comparison_methods.append(kwargs["stdi_comparison_method"])
        rewrite_modes.append(kwargs["rewrite_mode"])
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
        "rewrite_mode": "interpretive",
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
    assert rewrite_modes == ["interpretive", "interpretive"]
    assert output_dirs == [tmp_path / "batch" / name for name in calls]
    assert all(directory.is_dir() for directory in output_dirs)
    assert sorted(path.name for path in (tmp_path / "batch").iterdir()) == sorted(calls)
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


def test_graph_queue_reserves_a_batch_folder_and_avoids_overwrite(tmp_path) -> None:
    nodes = [create_default_node_form(1)]
    graphs = [{"name": "First", "nodes": nodes}]
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
    output_dirs = []

    def fake_run(**kwargs):
        output_dirs.append(kwargs["output_dir"])
        return SimpleNamespace(
            step_results=[],
            summary={"rows_processed": 1, "steps_total": 0, "cancelled": False},
            summary_path=None,
            steps_path=None,
        )

    for _ in range(2):
        run_job._run_graph_queue(
            job=run_job.GraphRunJob(),
            df=pd.DataFrame([{"title": "t", "description": "d"}]),
            graphs=graphs,
            settings=settings,
            runner=fake_run,
            queue_mode=True,
        )

    assert output_dirs == [
        tmp_path / "batch" / "batch_01_first",
        tmp_path / "batch_02" / "batch_01_first",
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


def test_work_progress_covers_news_nodes_and_failed_graph(tmp_path) -> None:
    graphs = [
        {"name": "First", "nodes": [create_default_node_form(1), create_default_node_form(2)]},
        {"name": "Second", "nodes": [create_default_node_form(1)]},
    ]
    settings = {
        "text_column": "description",
        "title_column": "title",
        "news_id_column": "",
        "max_rows": 2,
        "sleep_seconds": 0,
        "max_requests_per_minute": 0,
        "retry_attempts": 1,
        "allow_title_fallback": True,
        "topic_drift_model": "model",
        "topic_drift_provider": "gemini",
        "output_dir": str(tmp_path),
        "output_prefix": "batch",
    }
    calls = 0
    observed = []
    starts = []
    job = run_job.GraphRunJob()

    def fake_run(**kwargs):
        nonlocal calls
        calls += 1
        starts.append(job.latest_progress["completed"])
        callback = kwargs["work_progress_callback"]
        for row, step in [(1, 0), (1, 1)]:
            callback(row, 2, step, len(kwargs["nodes"]))
            observed.append(job.latest_progress["completed"])
        if calls == 2:
            raise RuntimeError("provider unavailable")
        for row, step in [(1, 2), (2, 0), (2, 1), (2, 2)]:
            callback(row, 2, step, 2)
            observed.append(job.latest_progress["completed"])
        return SimpleNamespace(
            step_results=[],
            summary={"rows_processed": 2, "steps_total": 0, "cancelled": False},
            summary_path=None,
            steps_path=None,
        )

    run_job._run_graph_queue(
        job=job,
        df=pd.DataFrame([{"title": "a", "description": "a"}, {"title": "b", "description": "b"}]),
        graphs=graphs,
        settings=settings,
        runner=fake_run,
        queue_mode=True,
    )
    assert starts == [0, 7]
    assert observed == [1, 2, 3, 4, 5, 6, 8, 9]
    assert job.latest_progress["graph_index"] == 2
    assert job.latest_progress["completed"] == job.latest_progress["total"] == 12
    assert not any(kind == "work" for kind, _ in job.events.queue)


def test_progress_display_estimates_remaining_time(monkeypatch) -> None:
    bars = []
    captions = []
    monkeypatch.setattr(
        sections,
        "st",
        SimpleNamespace(
            progress=lambda value, **kwargs: bars.append((value, kwargs["text"])),
            caption=captions.append,
        ),
    )
    sections._render_run_progress(
        {
            "completed": 3,
            "total": 12,
            "graph_index": 1,
            "graph_total": 2,
            "row": 1,
            "row_total": 2,
            "step": 2,
            "step_total": 2,
        },
        elapsed=30,
    )

    assert bars == [(0.25, "Graph 1/2 · news 1/2 · node 2/2 · 25% complete")]
    assert captions == ["Estimated time remaining: about 1m 30s"]


def test_monitor_bounds_message_history(monkeypatch) -> None:
    job = run_job.GraphRunJob()
    for index in range(30):
        job.events.put(("progress", f"Step {index}"))
    state = SimpleNamespace(
        run_job=job,
        run_messages=[],
        run_bundles=[],
        run_bundle=None,
    )
    fake_st = SimpleNamespace(
        session_state=state,
        button=lambda *_args, **_kwargs: False,
        warning=lambda *_args: None,
        info=lambda *_args: None,
        code=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(sections, "st", fake_st)

    sections._render_run_monitor.__wrapped__()

    assert state.run_messages == [f"Step {index}" for index in range(10, 30)]
