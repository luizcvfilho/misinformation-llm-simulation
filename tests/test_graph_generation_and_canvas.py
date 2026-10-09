from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from streamlit.elements.lib import policies
from streamlit.testing.v1 import AppTest
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode

from misinformation_simulation.apps import interaction_graph_app as studio
from misinformation_simulation.apps import interaction_graph_sections as sections
from misinformation_simulation.apps.interaction_graph_canvas import (
    automatic_positions,
    flow_state_to_forms,
    forms_fingerprint,
    synchronize_flow_state,
)
from misinformation_simulation.apps.interaction_graph_state import graph_nodes_to_forms
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    restore_editor_layout,
    validate_node_forms,
)
from misinformation_simulation.simulation.generation import generate_branching_graphs
from misinformation_simulation.simulation.io import graph_config_from_payload
from misinformation_simulation.simulation.topology import (
    _normalize_nodes,
    _root_to_leaf_paths,
    _topological_path,
)


def generated_forms(sequences="CCCC CCPP"):
    payload = generate_branching_graphs(sequences)[0]
    nodes, edges, start = graph_config_from_payload(payload)
    return graph_nodes_to_forms(nodes, edges, start)


def test_generator_shares_exact_prefixes_and_keeps_two_linear_paths():
    (payload,) = generate_branching_graphs("cccc, ccpp; CCCC")
    nodes, edges, root = graph_config_from_payload(payload)
    order = _topological_path(_normalize_nodes(nodes), edges, root)
    paths = _root_to_leaf_paths(order, edges)
    assert len(nodes) == 6
    assert [[node.removeprefix("prefix_") for node in path] for path in paths] == [
        ["C", "CC", "CCC", "CCCC"],
        ["C", "CC", "CCP", "CCPP"],
    ]
    assert len(payload["edges"]) == 5


def test_generator_groups_different_initial_personas_into_separate_trees():
    payloads = generate_branching_graphs("NNNN CCCC PPPP DDDD CCPP PPCC DDNN NNDD")
    assert [payload["start_node_id"] for payload in payloads] == [
        "prefix_N",
        "prefix_C",
        "prefix_P",
        "prefix_D",
    ]
    assert sum(len(payload["nodes"]) for payload in payloads) == 24
    assert all(
        validate_node_forms(graph_nodes_to_forms(*graph_config_from_payload(payload))) == []
        for payload in payloads
    )


@pytest.mark.parametrize("sequences", ["", "CCXP", "CC CCPP"])
def test_generator_rejects_invalid_codes_and_ambiguous_endpoints(sequences):
    with pytest.raises(ValueError):
        generate_branching_graphs(sequences)


def test_dragging_nodes_preserves_configuration_and_does_not_reset_canvas_state():
    forms = generated_forms()
    state = synchronize_flow_state(forms)
    before = forms_fingerprint(forms)
    state.nodes[2].position = {"x": 912.5, "y": 401.0}
    moved = flow_state_to_forms(forms, state)
    assert moved[2]["flow_x"] == "912.5"
    assert forms_fingerprint(moved) == before
    timestamp = state.timestamp
    same_state = synchronize_flow_state(moved, state)
    assert same_state is state
    assert same_state.timestamp > timestamp
    assert same_state.nodes[2].position == {"x": 912.5, "y": 401.0}


def test_drawing_new_parent_connection_rewires_only_that_branch():
    forms = generated_forms()
    state = synchronize_flow_state(forms)
    state.edges = [edge for edge in state.edges if edge.target != forms[4]["uid"]]
    state.edges.append(StreamlitFlowEdge("new", forms[2]["uid"], forms[4]["uid"]))
    updated = flow_state_to_forms(forms, state)
    assert updated[4]["parent_uid"] == forms[2]["uid"]
    assert updated[5]["parent_uid"] == forms[4]["uid"]
    assert forms[4]["parent_uid"] == forms[1]["uid"]
    assert validate_node_forms(updated) == []


@pytest.mark.parametrize("invalid", ["merge", "cycle", "delete_root"])
def test_invalid_canvas_changes_leave_simulation_configuration_unchanged(invalid):
    forms = generated_forms()
    snapshot = deepcopy(forms)
    state = synchronize_flow_state(forms)
    if invalid == "merge":
        state.edges.append(StreamlitFlowEdge("merge", forms[3]["uid"], forms[4]["uid"]))
    elif invalid == "cycle":
        state.edges.append(StreamlitFlowEdge("cycle", forms[5]["uid"], forms[0]["uid"]))
    else:
        state.nodes.pop(0)
    with pytest.raises(ValueError):
        flow_state_to_forms(forms, state)
    assert forms == snapshot


def test_new_canvas_node_gets_model_and_persona_fields_and_requires_a_connection():
    forms = generated_forms()
    state = synchronize_flow_state(forms)
    state.nodes.append(StreamlitFlowNode("from-canvas", (900, 450), {"content": "New persona"}))
    updated = flow_state_to_forms(forms, state)
    assert len(updated) == 7
    assert updated[-1]["uid"] == "from-canvas"
    assert updated[-1]["model"]
    assert updated[-1]["personality_preset"]
    assert validate_node_forms(updated)
    state.edges.append(StreamlitFlowEdge("connect", forms[1]["uid"], "from-canvas"))
    connected = flow_state_to_forms(forms, state)
    assert validate_node_forms(connected) == []


def test_automatic_positions_and_json_layout_round_trip():
    forms = generated_forms()
    positions = automatic_positions(forms)
    assert positions[forms[4]["uid"]][0] == positions[forms[2]["uid"]][0]
    assert positions[forms[4]["uid"]][1] != positions[forms[2]["uid"]][1]
    state = synchronize_flow_state(forms, arrange=True)
    arranged = flow_state_to_forms(forms, state)
    payload = build_editor_graph_payload(arranged)
    restored = graph_nodes_to_forms(*graph_config_from_payload(payload))
    restore_editor_layout(restored, payload)
    assert build_editor_graph_payload(restored) == payload


def test_visual_canvas_node_label_edits_update_configuration():
    forms = generated_forms()
    state = synchronize_flow_state(forms)
    state.nodes[1].data["content"] = "Shared conservative prefix"
    updated = flow_state_to_forms(forms, state)
    assert updated[1]["label"] == "Shared conservative prefix"
    assert updated[1]["personality_preset"] == forms[1]["personality_preset"]


def test_idle_canvas_reruns_do_not_resend_graph_or_reset_positions():
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_state import initialize_state\n"
        "from misinformation_simulation.apps.interaction_graph_canvas import render_graph_canvas\n"
        "initialize_state()\nrender_graph_canvas()"
    ).run(timeout=45)
    assert not app.exception
    state = app.session_state["_graph_flow_state"]
    state.nodes[1].position = {"x": 912.5, "y": 401.0}
    state.selected_id = state.nodes[1].id
    timestamp = state.timestamp
    for _ in range(3):
        app.run(timeout=45)
        assert not app.exception
        returned = app.session_state["_graph_flow_state"]
        assert returned.timestamp == timestamp
        assert returned.nodes[1].position == {"x": 912.5, "y": 401.0}
        assert returned.selected_id == state.nodes[1].id


def test_generator_and_canvas_controls_work_in_configuration_page(monkeypatch):
    monkeypatch.setattr(sections, "_render_graph_queue", lambda: None)
    monkeypatch.setattr(sections, "_render_run_controls", lambda *_args: None)
    app = AppTest.from_string(
        "import pandas as pd\n"
        "from misinformation_simulation.apps.interaction_graph_state import initialize_state\n"
        "from misinformation_simulation.apps.interaction_graph_sections "
        "import render_configuration_tab\n"
        "initialize_state()\n"
        "df = pd.DataFrame({'description': ['News'], 'title': ['Title']})\n"
        "render_configuration_tab(df, 'test')"
    ).run(timeout=45)
    assert not app.exception
    assert not any(radio.label == "Graph structure" for radio in app.radio)
    assert not any(heading.value == "Graph preview" for heading in app.subheader)
    assert app.radio(key="graph_editing_mode").value == "Visual canvas"
    initial_forms = app.session_state["graph_nodes"]
    assert initial_forms[0]["parent_uid"] == ""
    assert all(
        form["parent_uid"] == initial_forms[index - 1]["uid"]
        for index, form in enumerate(initial_forms[1:], start=1)
    )
    app.button(key="generate_editor_tree").click().run(timeout=45)
    assert not app.exception
    assert len(app.session_state["graph_nodes"]) == 6
    assert app.radio(key="graph_editing_mode").value == "Visual canvas"
    app.button(key="canvas_add_node").click().run(timeout=45)
    assert not app.exception
    assert len(app.session_state["graph_nodes"]) == 7
    app.button(key="canvas_arrange").click().run(timeout=45)
    assert not app.exception
    assert all("flow_x" in form for form in app.session_state["graph_nodes"])


def test_generator_can_queue_all_eight_scenarios():
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_state import initialize_state\n"
        "from misinformation_simulation.apps.interaction_graph_generator "
        "import render_graph_generator\n"
        "initialize_state()\nrender_graph_generator()"
    ).run(timeout=45)
    app.text_area(key="graph_generator_sequences").set_value(
        "NNNN CCCC PPPP DDDD CCPP PPCC DDNN NNDD"
    ).run(timeout=45)
    app.button(key="generate_queue_trees").click().run(timeout=45)
    assert not app.exception
    assert len(app.session_state["graph_queue"]) == 4
    assert sum(len(item["nodes"]) for item in app.session_state["graph_queue"]) == 24


def test_generator_preserves_restored_values_without_duplicate_default_warning(monkeypatch, caplog):
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_generator "
        "import render_graph_generator\nrender_graph_generator()"
    )
    app.session_state["graph_generator_sequences"] = "PPPP\nPPCC"
    app.session_state["graph_generator_model"] = "custom-model"
    app.session_state["graph_generator_provider"] = "gemini"
    app.run(timeout=45)
    assert not app.exception
    assert app.text_area(key="graph_generator_sequences").value == "PPPP\nPPCC"
    assert app.text_input(key="graph_generator_model").value == "custom-model"
    assert app.selectbox(key="graph_generator_provider").value == "gemini"
    assert not any("default value but also" in record.getMessage() for record in caplog.records)


def test_workspace_reruns_preserve_widget_values_without_duplicate_default_warning(
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(studio, "_refresh_graph_backend_if_stale", lambda: None)
    monkeypatch.setattr(
        studio.interaction_graph_sections,
        "render_sidebar",
        lambda: (pd.DataFrame({"description": ["News"], "title": ["Title"]}), "test.csv"),
    )
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_app import main\nmain()"
    ).run(timeout=45)
    assert not app.exception
    app.text_area(key="graph_generator_sequences").set_value("NNNN\nNNDD").run(timeout=45)
    app.number_input(key="simulation_execution_settings_max_rows").set_value(7).run(timeout=45)
    app.run(timeout=45)
    assert not app.exception
    assert app.text_area(key="graph_generator_sequences").value == "NNNN\nNNDD"
    assert app.number_input(key="simulation_execution_settings_max_rows").value == 7
    assert not any("default value but also" in record.getMessage() for record in caplog.records)
