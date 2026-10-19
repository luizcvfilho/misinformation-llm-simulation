from __future__ import annotations

import json
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps import interaction_graph_queue as queue
from misinformation_simulation.apps import interaction_graph_run_job as run_job
from misinformation_simulation.apps import interaction_graph_sections as sections
from misinformation_simulation.apps.interaction_graph_results import load_saved_result
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    build_linear_graph_payload,
    create_default_node_form,
    resolve_personality_text,
    set_tree_mode,
)
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


def test_compact_prefix_bounds_long_names_and_keeps_queue_positions_distinct() -> None:
    name = "A descriptive graph name " * 20
    first = queue.output_prefix_for_graph(
        "simulation_ui_20261003_223837", 1, name, include_base=False
    )
    second = queue.output_prefix_for_graph(
        "simulation_ui_20261003_223837", 2, name, include_base=False
    )

    assert len(first) <= 35
    assert first.startswith("01_a_descriptive_graph_name")
    assert second.startswith("02_")
    assert first != second


def test_queue_rejects_invalid_graph() -> None:
    with pytest.raises(ValueError, match="model"):
        queue.add_graph([], "Bad", [{**create_default_node_form(1), "model": ""}])


@pytest.mark.parametrize("shape", ["single", "linear", "branching"])
def test_queue_details_show_snapshots_and_full_personality_prompts(shape) -> None:
    nodes = [create_default_node_form(index) for index in range(1, 4)]
    if shape == "single":
        nodes = nodes[:1]
    elif shape == "branching":
        set_tree_mode(nodes, True)
        nodes[2]["parent_uid"] = nodes[0]["uid"]
    nodes[-1].update(
        model="custom-rewrite-model",
        personality_mode="custom",
        personality_custom="Use careful explanations. " * 25,
    )
    graphs = []
    queue.add_graph(graphs, "Same name", nodes)
    queue.add_graph(graphs, "Same name", nodes)
    snapshot = deepcopy(graphs)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps import interaction_graph_sections as sections\n"
        "sections._render_graph_queue()"
    )
    app.session_state["graph_nodes"] = [create_default_node_form(1)]
    app.session_state["graph_queue"] = graphs
    app.run(timeout=30)

    assert not app.exception
    assert len([item for item in app.expander if item.label.startswith("Graph details")]) == 2
    detail = app.dataframe[0].value
    assert detail["Model"].tolist() == [node["model"] for node in nodes]
    assert detail["Provider"].tolist() == [node["provider"] for node in nodes]
    assert detail["Node ID"].tolist() == [node["node_id"] for node in nodes]
    assert detail["Receive text from"].tolist() == (
        ["Source news (root)"]
        if shape == "single"
        else ["Source news (root)", "node_1", "node_1" if shape == "branching" else "node_2"]
    )
    assert app.code[0].value == resolve_personality_text(nodes[0])
    app.selectbox(key=f"queue_detail_node_{graphs[0]['id']}").set_value(nodes[-1]["node_id"]).run(
        timeout=30
    )
    assert not app.exception
    assert app.code[0].value == nodes[-1]["personality_custom"].strip()
    assert app.session_state["graph_queue"] == snapshot


def test_load_queue_copy_preserves_tree_and_isolates_model_variants() -> None:
    nodes = [create_default_node_form(index) for index in range(1, 4)]
    set_tree_mode(nodes, True)
    nodes[2]["parent_uid"] = nodes[0]["uid"]
    nodes[0].update(flow_x="25.5", flow_y="40.0")
    nodes[1].update(personality_mode="custom", personality_custom="Read carefully.")
    graphs = []
    queue.add_graph(graphs, "Original", nodes)
    snapshot = deepcopy(graphs)
    graph_id = graphs[0]["id"]
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps import interaction_graph_sections as sections\n"
        "from misinformation_simulation.apps import interaction_graph_components as components\n"
        "st.text_input('Current graph name', key='current_graph_name')\n"
        "for index, node in enumerate(st.session_state.graph_nodes):\n"
        "    components.render_node_editor(index, node)\n"
        "sections._render_graph_queue()"
    )
    app.session_state["graph_nodes"] = deepcopy(nodes)
    app.session_state["graph_queue"] = graphs
    app.session_state["_graph_flow_state"] = "stale canvas"
    app.run(timeout=30)
    assert not app.exception
    app.button(key=f"queue_load_copy_{graph_id}").click().run(timeout=30)

    assert not app.exception
    loaded = app.session_state["graph_nodes"]
    assert build_editor_graph_payload(loaded) == build_editor_graph_payload(nodes)
    assert {node["uid"] for node in loaded}.isdisjoint(node["uid"] for node in nodes)
    assert "_graph_flow_state" not in app.session_state
    assert app.text_input(key="current_graph_name").value == "Original copy"
    assert app.text_input(key="queued_graph_name").value == "Original copy"
    assert app.session_state["graph_queue"] == snapshot

    uid = loaded[0]["uid"]
    app.selectbox(key=f"model_option_{uid}").set_value("gpt-4.1-mini").run(timeout=30)
    app.selectbox(key=f"provider_{uid}").set_value("chatgpt").run(timeout=30)
    next(button for button in app.button if button.label == "Add current graph").click().run(
        timeout=30
    )
    assert not app.exception
    updated_queue = app.session_state["graph_queue"]
    assert updated_queue[0] == snapshot[0]
    assert updated_queue[1]["id"] != graph_id
    assert updated_queue[1]["name"] == "Original copy"
    assert updated_queue[1]["nodes"][0]["model"] == "gpt-4.1-mini"
    assert updated_queue[1]["nodes"][0]["provider"] == "chatgpt"
    assert (
        build_editor_graph_payload(updated_queue[1]["nodes"])["edges"]
        == (build_editor_graph_payload(nodes)["edges"])
    )


def _queue_editor_app() -> AppTest:
    nodes = [create_default_node_form(index) for index in range(1, 4)]
    set_tree_mode(nodes, True)
    nodes[2]["parent_uid"] = nodes[0]["uid"]
    nodes[0].update(flow_x="12.0", flow_y="24.0")
    graphs = []
    queue.add_graph(graphs, "Same name", nodes)
    queue.add_graph(graphs, "Same name", nodes)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps import interaction_graph_sections as sections\n"
        "from misinformation_simulation.apps import interaction_graph_components as components\n"
        "st.text_input('Current graph name', key='current_graph_name')\n"
        "for index, node in enumerate(st.session_state.graph_nodes):\n"
        "    components.render_node_editor(index, node)\n"
        "sections._render_graph_queue()"
    )
    app.session_state["graph_nodes"] = deepcopy(nodes)
    app.session_state["graph_queue"] = graphs
    return app.run(timeout=30)


def test_edit_queued_graph_saves_in_place_after_reordering() -> None:
    app = _queue_editor_app()
    snapshot = deepcopy(app.session_state["graph_queue"])
    target_id = snapshot[0]["id"]
    app.button(key=f"queue_edit_{target_id}").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["editing_queue_graph_id"] == target_id
    assert app.text_input(key="current_graph_name").value == "Same name"
    uid = app.session_state["graph_nodes"][0]["uid"]
    app.selectbox(key=f"model_option_{uid}").set_value("gpt-4.1-mini").run(timeout=30)
    app.selectbox(key=f"provider_{uid}").set_value("chatgpt").run(timeout=30)
    app.text_input(key="current_graph_name").set_value("Renamed graph").run(timeout=30)
    assert app.session_state["graph_queue"] == snapshot
    app.button(key=f"queue_down_{target_id}").click().run(timeout=30)
    app.button(key="queue_save_changes").click().run(timeout=30)

    assert not app.exception
    saved = app.session_state["graph_queue"]
    assert len(saved) == 2
    assert saved[0] == snapshot[1]
    assert saved[1]["id"] == target_id
    assert saved[1]["name"] == "Renamed graph"
    assert saved[1]["nodes"][0]["model"] == "gpt-4.1-mini"
    assert saved[1]["nodes"][0]["provider"] == "chatgpt"
    payload = build_editor_graph_payload(saved[1]["nodes"])
    original = build_editor_graph_payload(snapshot[0]["nodes"])
    assert payload["edges"] == original["edges"]
    assert payload["layout"] == original["layout"]
    assert app.success[0].value == "Graph changes saved to the queue."
    app.text_input(key=f"label_{uid}").set_value("Unsaved label").run(timeout=30)
    assert app.session_state["graph_queue"] == saved


@pytest.mark.parametrize("invalid", ["name", "model", "connection"])
def test_edit_queued_graph_rejects_invalid_changes_atomically(invalid) -> None:
    app = _queue_editor_app()
    snapshot = deepcopy(app.session_state["graph_queue"])
    app.button(key=f"queue_edit_{snapshot[0]['id']}").click().run(timeout=30)
    form = app.session_state["graph_nodes"][0]
    if invalid == "name":
        app.text_input(key="current_graph_name").set_value(" ").run(timeout=30)
    elif invalid == "model":
        app.selectbox(key=f"model_option_{form['uid']}").set_value("__custom__").run(timeout=30)
        app.text_input(key=f"model_custom_{form['uid']}").set_value("").run(timeout=30)
    else:
        app.selectbox(key=f"parent_uid_{form['uid']}").set_value(
            app.session_state["graph_nodes"][1]["uid"]
        ).run(timeout=30)
    app.button(key="queue_save_changes").click().run(timeout=30)
    assert not app.exception
    assert app.error
    assert app.session_state["graph_queue"] == snapshot
    assert app.session_state["editing_queue_graph_id"] == snapshot[0]["id"]


def test_cancel_editing_restores_saved_graph_and_detaches_copy() -> None:
    app = _queue_editor_app()
    snapshot = deepcopy(app.session_state["graph_queue"])
    graph_id = snapshot[0]["id"]
    app.button(key=f"queue_edit_{graph_id}").click().run(timeout=30)
    uid = app.session_state["graph_nodes"][0]["uid"]
    app.text_input(key=f"label_{uid}").set_value("Draft").run(timeout=30)
    app.text_input(key="current_graph_name").set_value("Draft name").run(timeout=30)
    app.button(key="queue_cancel_edit").click().run(timeout=30)
    assert not app.exception
    assert "editing_queue_graph_id" not in app.session_state
    assert app.text_input(key="current_graph_name").value == "Same name"
    assert build_editor_graph_payload(app.session_state["graph_nodes"]) == (
        build_editor_graph_payload(snapshot[0]["nodes"])
    )
    assert app.session_state["graph_queue"] == snapshot
    assert not any(button.key == "queue_save_changes" for button in app.button)

    app.button(key=f"queue_edit_{graph_id}").click().run(timeout=30)
    app.button(key=f"queue_load_copy_{graph_id}").click().run(timeout=30)
    assert not app.exception
    assert "editing_queue_graph_id" not in app.session_state
    assert app.text_input(key="current_graph_name").value == "Same name copy"
    assert app.session_state["graph_queue"] == snapshot


def test_removing_edited_graph_ends_editing_without_changing_other_entries() -> None:
    app = _queue_editor_app()
    snapshot = deepcopy(app.session_state["graph_queue"])
    graph_id = snapshot[0]["id"]
    app.button(key=f"queue_edit_{graph_id}").click().run(timeout=30)
    app.button(key=f"queue_remove_{graph_id}").click().run(timeout=30)
    assert not app.exception
    assert "editing_queue_graph_id" not in app.session_state
    assert app.session_state["graph_queue"] == snapshot[1:]
    assert not any(button.key == "queue_save_changes" for button in app.button)


def test_update_graph_rejects_missing_target() -> None:
    with pytest.raises(ValueError, match="no longer in the queue"):
        queue.update_graph([], "missing", "Name", [create_default_node_form(1)])


def test_queue_archive_preserves_snapshots_and_can_be_imported_from_folder(tmp_path) -> None:
    nodes = [create_default_node_form(1), create_default_node_form(2)]
    nodes[0].update(
        label="Nó de análise",
        model="custom-model",
        personality_mode="custom",
        personality_custom="Use precise explanations in Portuguese.",
    )
    graphs = []
    queue.add_graph(graphs, "Análise / graph", nodes)
    nodes.reverse()
    queue.add_graph(graphs, "Análise / graph", nodes)
    queue.move_graph(graphs, 1, -1)
    original = deepcopy(graphs)

    with ZipFile(BytesIO(queue.build_graph_queue_archive(graphs))) as archive:
        assert archive.namelist() == ["01_análise_graph.json", "02_análise_graph.json"]
        for graph, file_name in zip(graphs, archive.namelist(), strict=True):
            payload = json.loads(archive.read(file_name).decode("utf-8"))
            assert payload == build_linear_graph_payload(graph["nodes"])
            (tmp_path / file_name).write_bytes(archive.read(file_name))

    restored = []
    added, errors = queue.add_graphs_from_directory(restored, tmp_path)

    assert len(added) == 2
    assert errors == []
    assert [build_linear_graph_payload(graph["nodes"]) for graph in restored] == [
        build_linear_graph_payload(graph["nodes"]) for graph in graphs
    ]
    assert graphs == original


def test_queue_archive_preserves_folder_import_order_with_over_99_graphs() -> None:
    nodes = [create_default_node_form(1)]
    graphs = [{"name": "Same name", "nodes": nodes} for _ in range(101)]

    with ZipFile(BytesIO(queue.build_graph_queue_archive(graphs))) as archive:
        names = archive.namelist()

    assert names == sorted(names)
    assert names[0] == "001_same_name.json"
    assert names[-1] == "101_same_name.json"
    assert len(set(names)) == 101


def test_queue_archive_is_stable_over_time_and_changes_with_exported_content(monkeypatch):
    graphs = []
    nodes = [create_default_node_form(1)]
    queue.add_graph(graphs, "First", nodes)
    queue.add_graph(graphs, "Second", nodes)
    monkeypatch.setattr("zipfile.time.localtime", lambda *_: (2026, 10, 19, 12, 0, 0, 0, 0, 0))
    original = queue.build_graph_queue_archive(graphs)
    monkeypatch.setattr("zipfile.time.localtime", lambda *_: (2026, 10, 19, 16, 0, 0, 0, 0, 0))
    assert queue.build_graph_queue_archive(graphs) == original

    graphs[0]["nodes"][0]["model"] = "updated-model"
    edited = queue.build_graph_queue_archive(graphs)
    assert edited != original
    graphs[0]["name"] = "Renamed"
    renamed = queue.build_graph_queue_archive(graphs)
    assert renamed != edited
    queue.move_graph(graphs, 1, -1)
    assert queue.build_graph_queue_archive(graphs) != renamed


def test_queue_zip_is_deferred_and_exports_the_rendered_snapshot(monkeypatch):
    downloads = []
    builds = []
    download_button = sections.st.download_button

    def capture_download(label, **kwargs):
        if kwargs.get("key") == "download_graph_queue_zip":
            downloads.append(kwargs)
        return download_button(label, **kwargs)

    def build_archive(graphs):
        builds.append(deepcopy(graphs))
        return queue.build_graph_queue_archive(graphs)

    monkeypatch.setattr(sections.st, "download_button", capture_download)
    monkeypatch.setattr(sections, "build_graph_queue_archive", build_archive)
    graphs = []
    nodes = [create_default_node_form(1)]
    queue.add_graph(graphs, "First", nodes)
    snapshot = deepcopy(graphs)
    app = AppTest.from_string(
        "from misinformation_simulation.apps import interaction_graph_sections as sections\n"
        "sections._render_graph_queue()"
    )
    app.session_state["graph_nodes"] = nodes
    app.session_state["graph_queue"] = graphs
    app.run(timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert builds == []
    button = next(item for item in app.get("download_button") if item.proto.label.endswith("ZIP"))
    assert button.proto.url == ""
    assert button.proto.deferred_file_id
    assert button.proto.ignore_rerun

    graphs[0]["nodes"][0]["model"] = "updated-model"
    with ZipFile(BytesIO(downloads[-1]["data"]())) as archive:
        assert json.loads(archive.read(archive.namelist()[0])) == build_editor_graph_payload(
            snapshot[0]["nodes"]
        )
    assert builds == [snapshot]

    app.session_state["graph_queue"] = graphs
    app.run(timeout=30)
    assert not app.exception
    assert len(builds) == 1
    with ZipFile(BytesIO(downloads[-1]["data"]())) as archive:
        assert json.loads(archive.read(archive.namelist()[0])) == build_editor_graph_payload(
            graphs[0]["nodes"]
        )


@pytest.mark.parametrize("judge_repeats", [None, 1, 5])
def test_run_queue_continues_after_one_graph_fails(tmp_path, judge_repeats) -> None:
    nodes = [create_default_node_form(1)]
    graphs = []
    queue.add_graph(graphs, "First", nodes)
    queue.add_graph(graphs, "Second", deepcopy(nodes))
    calls = []
    comparison_methods = []
    rewrite_modes = []
    judge_counts = []
    processing_modes = []
    output_dirs = []

    def fake_run(**kwargs):
        calls.append(kwargs["output_prefix"])
        comparison_methods.append(kwargs["stdi_comparison_method"])
        rewrite_modes.append(kwargs["rewrite_mode"])
        judge_counts.append(kwargs["stdi_judge_repeats"])
        processing_modes.append(kwargs["openai_service_tier"])
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
        "openai_service_tier": "flex",
        "topic_drift_model": "model",
        "topic_drift_provider": "gemini",
        "output_dir": str(tmp_path),
        "output_prefix": "batch",
    }

    job = run_job.GraphRunJob()
    if judge_repeats is not None:
        settings["stdi_judge_repeats"] = judge_repeats
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

    assert calls == ["01_first", "02_second"]
    assert comparison_methods == ["dual", "dual"]
    assert rewrite_modes == ["interpretive", "interpretive"]
    assert judge_counts == [judge_repeats or 3, judge_repeats or 3]
    assert processing_modes == ["flex", "flex"]
    assert output_dirs == [tmp_path / "batch" / name for name in calls]
    assert all(directory.is_dir() for directory in output_dirs)
    assert sorted(path.name for path in (tmp_path / "batch").iterdir() if path.is_dir()) == sorted(
        calls
    )
    assert (tmp_path / "batch" / "execution.log").is_file()
    assert (tmp_path / "batch" / "timings.json").is_file()
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

    assert calls == ["01_first"]
    assert [payload["status"] for kind, payload in events if kind == "bundle"] == ["cancelled"]
    assert events[-1] == ("done", {"completed": 0, "failed": 0, "cancelled": True})


@pytest.mark.parametrize("queue_mode", [False, True])
def test_one_graph_uses_execution_folder_and_avoids_overwrite(tmp_path, queue_mode) -> None:
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
            queue_mode=queue_mode,
        )
        bundle = next(payload for kind, payload in job.events.queue if kind == "bundle")
        summary_path = Path(bundle["summary_path"])
        steps_path = Path(bundle["steps_path"])
        folders.append(summary_path.parent)
        assert summary_path.is_file()
        assert steps_path.is_file()
        assert summary_path.name == "01_investigative_skeptic_summary.json"
        assert steps_path.name == "01_investigative_skeptic_steps.jsonl"
        assert bundle["summary"]["graph_name"] == "Investigative skeptic"
        assert load_saved_result(summary_path)["name"] == "Investigative skeptic"
        assert bundle["execution_name"] == summary_path.parent.parent.name
        assert load_saved_result(summary_path)["execution_name"] == bundle["execution_name"]

    assert folders == [
        tmp_path / "simulation_ui_20260917_123456" / "01_investigative_skeptic",
        tmp_path / "simulation_ui_20260917_123456_02" / "01_investigative_skeptic",
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
        tmp_path / "batch" / "01_first",
        tmp_path / "batch_02" / "01_first",
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
        expander=lambda *_args: nullcontext(),
        text_input=lambda *_args, **_kwargs: "",
        download_button=lambda *_args, **_kwargs: None,
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


def test_progress_display_waits_for_operation_timings(monkeypatch) -> None:
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
    assert captions == ["Calibrating time estimate: waiting for operation timings."]


def test_progress_display_shows_judge_wait_repetition_and_time_range(monkeypatch) -> None:
    captions = []
    bars = []
    monkeypatch.setattr(
        sections,
        "st",
        SimpleNamespace(
            progress=lambda value, **kwargs: bars.append(kwargs["text"]),
            caption=captions.append,
        ),
    )
    sections._render_run_progress(
        dict(
            completed=3,
            total=12,
            graph_index=1,
            graph_total=2,
            row=1,
            row_total=2,
            step=2,
            step_total=2,
            phase="judge",
            node_id="C2",
            estimate_low=50,
            estimate_high=90,
            current=dict(
                kind="rate_limit_wait",
                stage="judge",
                news_id="n1",
                node_id="C2",
                comparison="vs original",
                model="test-model",
                repeat=2,
                repeats=3,
                delay=55,
                age=5,
            ),
            stages={"judge": dict(completed=1, total=6, cache_hits=0, skipped=0, failed=0)},
        )
    )
    assert "repeat 2/3" in captions[1]
    assert "vs original" in captions[1]
    assert "waiting for request limit · 50s" in captions[1]
    assert "50s–1m 30s" in captions[2]
    assert "Graph 1/2 · judge · news 1/2 · node C2" in bars[0]


def test_monitor_preserves_complete_message_history(monkeypatch) -> None:
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
        expander=lambda *_args: nullcontext(),
        text_input=lambda *_args, **_kwargs: "",
        download_button=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(sections, "st", fake_st)

    sections._render_run_monitor.__wrapped__()

    assert state.run_messages == [f"Step {index}" for index in range(30)]
