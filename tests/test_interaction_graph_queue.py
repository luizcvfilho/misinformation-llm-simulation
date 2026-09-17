from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from misinformation_simulation.apps import interaction_graph_queue as queue
from misinformation_simulation.apps import interaction_graph_sections as sections
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form


class SessionState(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


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


def test_run_queue_continues_after_one_graph_fails(monkeypatch) -> None:
    nodes = [create_default_node_form(1)]
    graphs = []
    queue.add_graph(graphs, "First", nodes)
    queue.add_graph(graphs, "Second", deepcopy(nodes))
    state = SessionState(graph_queue=graphs, graph_nodes=nodes, run_bundle=None, run_bundles=[])
    messages = []
    placeholder = SimpleNamespace(
        info=lambda message: messages.append(message),
        warning=lambda message: messages.append(message),
        success=lambda message: messages.append(message),
        code=lambda *args, **kwargs: None,
    )
    fake_st = SimpleNamespace(
        session_state=state,
        empty=lambda: placeholder,
        button=lambda *args, **kwargs: True,
        error=lambda message: messages.append(message),
    )
    monkeypatch.setattr(sections, "st", fake_st)
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs["output_prefix"])
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        return SimpleNamespace(
            step_results=[],
            summary={"rows_processed": 1, "steps_total": 0},
            summary_path=None,
            steps_path=None,
        )

    monkeypatch.setattr(sections, "run_news_interaction_graph", fake_run)
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
        "output_dir": "output/interaction_graph/app_runs",
        "output_prefix": "batch",
    }

    sections._render_run_controls(pd.DataFrame([{"title": "t", "description": "d"}]), settings)

    assert calls == ["batch_01_first", "batch_02_second"]
    assert [bundle["status"] for bundle in state.run_bundles] == ["failed", "completed"]
    assert state.run_bundle["name"] == "Second"
    assert any("1 completed, 1 failed" in message for message in messages)
