from __future__ import annotations

from types import SimpleNamespace

from misinformation_simulation.apps import interaction_graph_state as state


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        return self[name]

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def test_initialize_state_sets_defaults(monkeypatch) -> None:
    fake_st = SimpleNamespace(session_state=FakeSessionState())
    monkeypatch.setattr(state, "st", fake_st)
    monkeypatch.setattr(state, "load_initial_graph_nodes", lambda: [{"node_id": "node_1"}])

    state.initialize_state()

    assert fake_st.session_state.dataset_path == state.DEFAULT_DATASET_PATH
    assert fake_st.session_state.graph_config_path == state.DEFAULT_GRAPH_CONFIG_PATH
    assert fake_st.session_state.graph_nodes == [{"node_id": "node_1"}]
    assert fake_st.session_state.run_bundle is None


def test_move_and_remove_node_update_session_state(monkeypatch) -> None:
    fake_st = SimpleNamespace(
        session_state=FakeSessionState(
            graph_nodes=[{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}]
        )
    )
    monkeypatch.setattr(state, "st", fake_st)

    state.move_node(0, 1)
    state.move_node(2, 1)
    state.remove_node(1)

    assert fake_st.session_state.graph_nodes == [{"node_id": "b"}, {"node_id": "c"}]


def test_remove_node_keeps_single_node(monkeypatch) -> None:
    fake_st = SimpleNamespace(session_state=FakeSessionState(graph_nodes=[{"node_id": "only"}]))
    monkeypatch.setattr(state, "st", fake_st)

    state.remove_node(0)

    assert fake_st.session_state.graph_nodes == [{"node_id": "only"}]


def test_reset_graph_and_import_payload_replace_nodes(monkeypatch) -> None:
    fake_st = SimpleNamespace(session_state=FakeSessionState(graph_nodes=[{"node_id": "old"}]))
    monkeypatch.setattr(state, "st", fake_st)
    monkeypatch.setattr(state, "load_initial_graph_nodes", lambda: [{"node_id": "reset"}])

    state.reset_graph()

    assert fake_st.session_state.graph_nodes == [{"node_id": "reset"}]

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

    state.import_graph_payload(payload)

    assert fake_st.session_state.graph_nodes[0]["node_id"] == "node-a"
