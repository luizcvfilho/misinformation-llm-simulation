from __future__ import annotations

from copy import deepcopy

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps import interaction_graph_canvas as canvas
from misinformation_simulation.apps.interaction_graph_ui import (
    CUSTOM_OPTION,
    PREDEFINED_PERSONALITIES,
    apply_bulk_node_updates,
    build_editor_graph_payload,
    create_default_node_form,
    resolve_personality_text,
    set_tree_mode,
)


def node_forms():
    forms = [create_default_node_form(index) for index in range(1, 4)]
    set_tree_mode(forms, True)
    for index, form in enumerate(forms):
        form.update(flow_x=str(index * 300.0), flow_y="80.0")
    return forms


def editor_app(mode="Node forms"):
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_bulk_editor "
        "import render_bulk_node_editor\n"
        "from misinformation_simulation.apps.interaction_graph_components "
        "import render_node_editor\n"
        "from misinformation_simulation.apps.interaction_graph_canvas import render_graph_canvas\n"
        "from misinformation_simulation.apps.interaction_graph_ui "
        "import build_editor_graph_payload\n"
        "from misinformation_simulation.apps.interaction_graph_workspace "
        "import retain_workspace_widgets\n"
        "retain_workspace_widgets()\n"
        "if st.session_state.test_mode == 'Visual canvas':\n"
        "    render_graph_canvas()\n"
        "else:\n"
        "    render_bulk_node_editor()\n"
        "    for index, form in enumerate(st.session_state.graph_nodes):\n"
        "        render_node_editor(index, form)\n"
        "st.session_state.test_payload = build_editor_graph_payload(st.session_state.graph_nodes)"
    )
    app.session_state["graph_nodes"] = node_forms()
    app.session_state["test_mode"] = mode
    return app.run(timeout=45)


def test_bulk_updates_preserve_unselected_nodes_topology_layout_and_input():
    forms = node_forms()
    before = deepcopy(forms)
    selected = [forms[0]["uid"], forms[2]["uid"]]
    updated = apply_bulk_node_updates(forms, selected, provider="gemini", model="custom-model")

    assert forms == before
    assert updated[1] == before[1]
    for index in (0, 2):
        assert updated[index] == before[index] | {"provider": "gemini", "model": "custom-model"}
    payload = build_editor_graph_payload(updated)
    assert payload["edges"] == build_editor_graph_payload(before)["edges"]
    assert payload["layout"] == build_editor_graph_payload(before)["layout"]
    assert [node["model"] for node in payload["nodes"]] == [
        "custom-model",
        before[1]["model"],
        "custom-model",
    ]


def test_bulk_personality_can_switch_between_custom_and_preset():
    forms = node_forms()
    uids = [form["uid"] for form in forms]
    custom = apply_bulk_node_updates(forms, uids, personality_custom="  New personality  ")
    assert all(form["personality_mode"] == "custom" for form in custom)
    assert all(resolve_personality_text(form) == "New personality" for form in custom)
    preset = next(iter(PREDEFINED_PERSONALITIES))
    restored = apply_bulk_node_updates(custom, uids, personality_preset=preset)
    assert all(form["personality_custom"] == "" for form in restored)
    assert all(
        resolve_personality_text(form) == PREDEFINED_PERSONALITIES[preset] for form in restored
    )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"provider": "unsupported"},
        {"provider": " "},
        {"model": " "},
        {"personality_custom": "\n "},
        {"personality_preset": "missing"},
        {"personality_preset": "NeutralRelay", "personality_custom": "Custom"},
        {"provider": "gemini", "model": " "},
    ],
)
def test_invalid_bulk_updates_are_atomic(changes):
    forms = node_forms()
    before = deepcopy(forms)
    with pytest.raises(ValueError):
        apply_bulk_node_updates(forms, [form["uid"] for form in forms], **changes)
    assert forms == before


@pytest.mark.parametrize("selected", [[], ["node-from-another-graph"]])
def test_bulk_updates_require_current_nodes(selected):
    with pytest.raises(ValueError):
        apply_bulk_node_updates(node_forms(), selected, model="custom-model")


@pytest.mark.parametrize("mode", ["Node forms", "Visual canvas"])
def test_bulk_provider_and_model_survive_individual_widgets_and_canvas_reruns(mode):
    app = editor_app(mode)
    assert not app.exception
    before = deepcopy(app.session_state["graph_nodes"])
    uids = [form["uid"] for form in before[:2]]
    app.multiselect(key="graph_bulk_selected_uids").set_value(uids).run(timeout=45)
    app.checkbox(key="graph_bulk_change_provider").check().run(timeout=45)
    app.selectbox(key="graph_bulk_provider").set_value("gemini").run(timeout=45)
    app.checkbox(key="graph_bulk_change_model").check().run(timeout=45)
    app.selectbox(key="graph_bulk_model_option").set_value("gemini-3.5-flash-lite").run(timeout=45)
    assert app.session_state["graph_nodes"] == before
    app.button(key="graph_bulk_apply").click().run(timeout=45)
    app.run(timeout=45)

    assert not app.exception
    after = app.session_state["graph_nodes"]
    for old, new in zip(before[:2], after[:2], strict=True):
        assert new["provider"] == "gemini"
        assert new["model"] == "gemini-3.5-flash-lite"
        for key in ("uid", "node_id", "label", "parent_uid", "flow_x", "flow_y"):
            assert new[key] == old[key]
        assert resolve_personality_text(new) == resolve_personality_text(old)
    assert after[2] == before[2]
    assert app.selectbox(key=f"provider_{uids[0]}").value == "gemini"
    assert app.selectbox(key=f"model_option_{uids[0]}").value == "gemini-3.5-flash-lite"
    assert app.session_state["test_payload"] == build_editor_graph_payload(after)
    if mode == "Visual canvas":
        state = app.session_state["_graph_flow_state"]
        assert "gemini-3.5-flash-lite" in state.nodes[0].data["content"]
        assert state.nodes[0].position == {"x": 0.0, "y": 80.0}


def test_bulk_custom_values_validate_then_update_all_and_switch_back_to_preset():
    app = editor_app()
    before = deepcopy(app.session_state["graph_nodes"])
    app.button(key="graph_bulk_select_all").click().run(timeout=45)
    assert len(app.multiselect(key="graph_bulk_selected_uids").value) == 3
    app.checkbox(key="graph_bulk_change_model").check().run(timeout=45)
    app.selectbox(key="graph_bulk_model_option").set_value(CUSTOM_OPTION).run(timeout=45)
    app.text_input(key="graph_bulk_model_custom").set_value("local/custom-model").run(timeout=45)
    app.checkbox(key="graph_bulk_change_personality").check().run(timeout=45)
    app.radio(key="graph_bulk_personality_mode").set_value("Custom").run(timeout=45)
    app.button(key="graph_bulk_apply").click().run(timeout=45)
    assert not app.exception
    assert any("cannot be empty" in error.value for error in app.error)
    assert app.session_state["graph_nodes"] == before

    app.text_area(key="graph_bulk_personality_custom").set_value("Custom persona").run(timeout=45)
    app.button(key="graph_bulk_apply").click().run(timeout=45)
    assert not app.exception
    for form in app.session_state["graph_nodes"]:
        uid = form["uid"]
        assert form["model"] == "local/custom-model"
        assert resolve_personality_text(form) == "Custom persona"
        assert app.text_input(key=f"model_custom_{uid}").value == "local/custom-model"
        assert app.radio(key=f"personality_mode_{uid}").value == "Custom"
        assert app.text_area(key=f"personality_custom_{uid}").value == "Custom persona"

    app.checkbox(key="graph_bulk_change_model").uncheck().run(timeout=45)
    app.radio(key="graph_bulk_personality_mode").set_value("Predefined").run(timeout=45)
    app.selectbox(key="graph_bulk_personality_preset").set_value("NeutralRelay").run(timeout=45)
    app.button(key="graph_bulk_apply").click().run(timeout=45)
    assert not app.exception
    for form in app.session_state["graph_nodes"]:
        assert form["model"] == "local/custom-model"
        assert resolve_personality_text(form) == PREDEFINED_PERSONALITIES["NeutralRelay"]
        assert app.radio(key=f"personality_mode_{form['uid']}").value == "Predefined"
    assert app.success[0].value == "Updated 3 nodes."


def test_selection_controls_disable_apply_and_drop_nodes_from_previous_graph():
    app = editor_app()
    assert app.button(key="graph_bulk_apply").disabled
    app.button(key="graph_bulk_select_all").click().run(timeout=45)
    assert app.button(key="graph_bulk_apply").disabled
    app.checkbox(key="graph_bulk_change_provider").check().run(timeout=45)
    assert not app.button(key="graph_bulk_apply").disabled
    app.button(key="graph_bulk_clear").click().run(timeout=45)
    assert app.multiselect(key="graph_bulk_selected_uids").value == []
    assert app.button(key="graph_bulk_apply").disabled

    app.button(key="graph_bulk_select_all").click().run(timeout=45)
    app.session_state["graph_nodes"] = node_forms()
    app.run(timeout=45)
    assert not app.exception
    assert app.multiselect(key="graph_bulk_selected_uids").value == []
    assert app.button(key="graph_bulk_apply").disabled


def test_canvas_box_selection_populates_bulk_editor_and_survives_updates(monkeypatch):
    def canvas_event(_key, state, **_options):
        event = st.session_state.pop("test_canvas_selection", None)
        if event is None:
            return state
        returned = deepcopy(state)
        returned.timestamp += 1
        for node in returned.nodes:
            node.selected = node.id in event
            if node.selected:
                node.position["y"] = 240.0
        return returned

    monkeypatch.setattr(canvas, "streamlit_flow", canvas_event)
    app = editor_app("Visual canvas")
    before = deepcopy(app.session_state["graph_nodes"])
    uids = [before[index]["uid"] for index in (0, 2)]
    app.session_state["test_canvas_selection"] = uids
    app.run(timeout=45)
    assert not app.exception
    assert app.multiselect(key="graph_bulk_selected_uids").value == uids
    bulk_panel = next(panel for panel in app.expander if panel.label == "Bulk edit nodes")
    assert bulk_panel.proto.expanded
    app.checkbox(key="graph_bulk_change_model").check().run(timeout=45)
    app.selectbox(key="graph_bulk_model_option").set_value("gpt-4.1-mini").run(timeout=45)
    app.button(key="graph_bulk_apply").click().run(timeout=45)
    assert not app.exception
    for old, new in zip(before, app.session_state["graph_nodes"], strict=True):
        assert new["model"] == ("gpt-4.1-mini" if old["uid"] in uids else old["model"])
        assert new["parent_uid"] == old["parent_uid"]
    selected_nodes = [
        node for node in app.session_state["_graph_flow_state"].nodes if node.selected
    ]
    assert [node.id for node in selected_nodes] == uids
    assert all(node.position["y"] == 240.0 for node in selected_nodes)

    app.multiselect(key="graph_bulk_selected_uids").set_value(uids[:1]).run(timeout=45)
    assert [
        node.id for node in app.session_state["_graph_flow_state"].nodes if node.selected
    ] == uids[:1]
    app.session_state["test_canvas_selection"] = []
    app.run(timeout=45)
    assert not app.exception
    assert app.multiselect(key="graph_bulk_selected_uids").value == []
    assert app.button(key="graph_bulk_apply").disabled

    app.button(key="graph_bulk_select_all").click().run(timeout=45)
    assert all(node.selected for node in app.session_state["_graph_flow_state"].nodes)
    app.button(key="graph_bulk_clear").click().run(timeout=45)
    assert not any(node.selected for node in app.session_state["_graph_flow_state"].nodes)
