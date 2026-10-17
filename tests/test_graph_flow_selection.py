from __future__ import annotations

from pathlib import Path

import pytest
import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from misinformation_simulation.apps import interaction_graph_flow as flow


def test_selection_frontend_adapts_installed_bundle_without_modifying_dependency(tmp_path):
    source = Path(streamlit_flow.__file__).parent / "frontend" / "build"
    original_html = (source / "index.html").read_text(encoding="utf-8")
    build = flow.prepare_selection_frontend(source, tmp_path)
    html = (build / "index.html").read_text(encoding="utf-8")
    script = build / f"static/js/main.selection-{build.name}.js"
    content = script.read_text(encoding="utf-8")
    assert script.name in html
    assert flow.SELECTION_CSS in html
    for _, replacement in flow.SELECTION_CALLBACK_PATCHES:
        assert replacement in content
    assert flow.prepare_selection_frontend(source, tmp_path) == build
    assert (source / "index.html").read_text(encoding="utf-8") == original_html


def test_selection_frontend_fails_explicitly_on_incompatible_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "asset-manifest.json").write_text('{"files": {"main.js": "main.js"}}')
    (source / "main.js").write_text("incompatible frontend")
    with pytest.raises(RuntimeError, match="1.6.1"):
        flow.prepare_selection_frontend(source, tmp_path / "cache")


def test_component_serializes_and_restores_multiple_selected_nodes(monkeypatch):
    nodes = [StreamlitFlowNode(str(index), (index * 300, 0), {}) for index in range(3)]
    state = StreamlitFlowState(nodes, [])
    captured = {}

    def component(**arguments):
        captured.update(arguments)
        returned = [node | {"selected": node["id"] in {"0", "2"}} for node in arguments["nodes"]]
        return {
            "nodes": returned,
            "edges": [],
            "selectedId": None,
            "timestamp": state.timestamp + 1,
        }

    monkeypatch.setattr(flow, "_selection_component", lambda: component)
    returned = flow.selectable_streamlit_flow(
        "test", state, layout=ManualLayout(), show_minimap=True, get_node_on_click=True
    )
    assert [node.id for node in returned.nodes if node.selected] == ["0", "2"]
    assert captured["showMiniMap"] is True
    assert captured["getNodeOnClick"] is True
    assert captured["layoutOptions"] == ManualLayout().__to_dict__()
