from __future__ import annotations

from copy import deepcopy
from math import isfinite
from time import time_ns

import networkx as nx
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from misinformation_simulation.apps.interaction_graph_components import render_node_editor
from misinformation_simulation.apps.interaction_graph_ui import (
    PREDEFINED_PERSONALITIES,
    append_editor_node,
    resolve_personality_text,
)
from misinformation_simulation.simulation.paths import PERSONALITY_CODES
from misinformation_simulation.topic_drift.provenance import input_hash

NODE_STYLE = {
    "width": 230,
    "borderRadius": "8px",
    "background": "var(--secondary-background-color, #262730)",
    "color": "var(--text-color, #fafafa)",
    "borderColor": "color-mix(in srgb, var(--text-color, #fafafa) 25%, transparent)",
}


def forms_fingerprint(forms: list[dict[str, str]]) -> str:
    return input_hash(
        [
            {key: value for key, value in form.items() if key not in {"flow_x", "flow_y"}}
            for form in forms
        ]
    )


def node_content(form: dict[str, str]) -> str:
    code = PERSONALITY_CODES.get(resolve_personality_text(form), "X")
    persona = (
        form.get("personality_preset", "Custom")
        if form.get("personality_mode") == "preset"
        else "Custom"
    )
    return f"**{form['label']}**\n\n{code} · {persona}\n\n{form['model']}"


def automatic_positions(forms: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(form["uid"] for form in forms)
    graph.add_edges_from(
        (form["parent_uid"], form["uid"]) for form in forms if form.get("parent_uid") in graph
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Remove graph cycles before arranging nodes.")
    order = list(nx.topological_sort(graph))
    depth = {}
    for node in order:
        depth[node] = max((depth[parent] + 1 for parent in graph.predecessors(node)), default=0)
    roots = [node for node in order if graph.in_degree(node) == 0]
    leaves = [
        node
        for root in roots
        for node in nx.dfs_preorder_nodes(graph, root)
        if graph.out_degree(node) == 0
    ]
    vertical = {node: float(index * 170) for index, node in enumerate(leaves)}
    for node in reversed(order):
        children = list(graph.successors(node))
        if children:
            vertical[node] = (vertical[children[0]] + vertical[children[-1]]) / 2
    return {node: (float(depth[node] * 300), vertical[node]) for node in order}


def synchronize_flow_state(
    forms: list[dict[str, str]],
    state: StreamlitFlowState | None = None,
    *,
    arrange: bool = False,
) -> StreamlitFlowState:
    """Update the persistent component state after Python-side edits."""
    if state is None:
        state = StreamlitFlowState([], [])
    previous = {node.id: node for node in state.nodes}
    positions = automatic_positions(forms)
    root = next((form["uid"] for form in forms if not form.get("parent_uid")), None)
    nodes = []
    for form in forms:
        uid = form["uid"]
        position = positions[uid]
        if not arrange:
            if "flow_x" in form and "flow_y" in form:
                position = float(form["flow_x"]), float(form["flow_y"])
            elif uid in previous:
                position = previous[uid].position["x"], previous[uid].position["y"]
        content = node_content(form)
        nodes.append(
            StreamlitFlowNode(
                uid,
                position,
                {"content": content},
                node_type="input" if uid == root else "default",
                source_position="right",
                target_position="left",
                connectable=True,
                selectable=True,
                deletable=uid != root,
                style=NODE_STYLE.copy(),
            )
        )
    state.nodes = nodes
    state.edges = [
        StreamlitFlowEdge(
            f"{form['parent_uid']}:{form['uid']}",
            form["parent_uid"],
            form["uid"],
            edge_type="smoothstep",
            marker_end={"type": "arrowclosed"},
            deletable=True,
        )
        for form in forms
        if form.get("parent_uid")
    ]
    state.timestamp = max(time_ns() // 1_000_000, state.timestamp + 1)
    return state


def flow_state_to_forms(
    forms: list[dict[str, str]],
    state: StreamlitFlowState,
) -> list[dict[str, str]]:
    """Accept an editable forest, but reject cycles, merges and removal of the source root."""
    node_ids = [node.id for node in state.nodes]
    if not node_ids or len(node_ids) != len(set(node_ids)) or not all(node_ids):
        raise ValueError("The canvas must contain nodes with unique IDs.")
    root = next((form["uid"] for form in forms if not form.get("parent_uid")), None)
    if root is not None and root not in node_ids:
        raise ValueError("Keep the source root node in the graph.")
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    for edge in state.edges:
        if edge.source not in graph or edge.target not in graph:
            raise ValueError("Connect only nodes that exist on the canvas.")
        graph.add_edge(edge.source, edge.target)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("That connection creates a cycle. Each branch must continue separately.")
    if any(graph.in_degree(node) > 1 for node in graph):
        raise ValueError("A node can have only one parent. Remove its previous connection first.")
    if root is not None and graph.in_degree(root):
        raise ValueError("The source root cannot receive text from another node.")
    existing = {form["uid"]: deepcopy(form) for form in forms}
    result = [existing[uid] for uid in node_ids if uid in existing]
    for node in state.nodes:
        if node.id not in existing:
            form = append_editor_node(result)
            form["uid"] = node.id
            form["label"] = str(node.data.get("content", form["label"]))[:120]
            existing[node.id] = form
    result = [existing[node.id] for node in state.nodes]
    for node, form in zip(state.nodes, result, strict=True):
        parent = next(graph.predecessors(node.id), "")
        x, y = float(node.position["x"]), float(node.position["y"])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Node positions must be finite numbers.")
        form.update(parent_uid=parent, flow_x=str(x), flow_y=str(y))
        content = str(node.data.get("content", ""))
        if node.id in {item["uid"] for item in forms} and content != node_content(form):
            form["label"] = content.strip() or form["label"]
    return result


def render_graph_canvas() -> None:
    forms = st.session_state.graph_nodes
    fingerprint = forms_fingerprint(forms)
    state = st.session_state.get("_graph_flow_state")
    if (
        state is None
        or st.session_state.get("_graph_flow_source") != fingerprint
        or any(
            node.style.get(key) != value
            for node in state.nodes
            for key, value in NODE_STYLE.items()
        )
    ):
        try:
            state = synchronize_flow_state(forms, state)
        except ValueError as exc:
            st.error(str(exc))
            st.info("Use Node forms to correct the connections.")
            return
        st.session_state._graph_flow_state = state
        st.session_state._graph_flow_source = fingerprint

    controls = st.columns([2, 1, 1])
    new_persona = controls[0].selectbox(
        "New node persona",
        list(PREDEFINED_PERSONALITIES),
        key="graph_canvas_new_persona",
    )
    if controls[1].button("Add child node", key="canvas_add_node", width="stretch"):
        selected = st.session_state.get("graph_canvas_selected") or forms[0]["uid"]
        parent = next((form for form in forms if form["uid"] == selected), forms[0])
        new_form = append_editor_node(forms, parent_uid=parent["uid"])
        new_form.update(
            personality_preset=new_persona,
            label=f"{new_persona} {len(forms)}",
            model=parent["model"],
            provider=parent["provider"],
        )
        st.rerun()
    if controls[2].button("Arrange automatically", key="canvas_arrange", width="stretch"):
        state = synchronize_flow_state(forms, state, arrange=True)
        st.session_state.graph_nodes = flow_state_to_forms(forms, state)
        st.session_state._graph_flow_source = forms_fingerprint(st.session_state.graph_nodes)
        st.rerun()
    st.caption(
        "Drag nodes to move them. Drag from the right handle to another node's left handle "
        "to connect them. Right-click a connection to remove it. Click a node to configure it."
    )
    returned = streamlit_flow(
        "simulation_graph_canvas",
        state,
        height=500,
        fit_view=True,
        allow_new_edges=True,
        get_node_on_click=True,
        enable_pane_menu=True,
        enable_node_menu=True,
        enable_edge_menu=True,
        show_minimap=True,
        layout=ManualLayout(),
        min_zoom=0.1,
    )
    if returned.timestamp >= state.timestamp:
        try:
            updated = flow_state_to_forms(forms, returned)
        except ValueError as exc:
            st.session_state._graph_flow_error = str(exc)
            state.timestamp = max(state.timestamp, returned.timestamp) + 1
        else:
            changed = forms_fingerprint(forms) != forms_fingerprint(updated)
            for old_form in forms:
                new_form = next((item for item in updated if item["uid"] == old_form["uid"]), None)
                if new_form is not None and new_form.get("parent_uid") != old_form.get(
                    "parent_uid"
                ):
                    st.session_state.pop(f"parent_uid_{old_form['uid']}", None)
                if new_form is not None and new_form.get("label") != old_form.get("label"):
                    st.session_state.pop(f"label_{old_form['uid']}", None)
            forms = updated
            st.session_state.graph_nodes = forms
            st.session_state._graph_flow_state = returned
            st.session_state._graph_flow_source = None if changed else forms_fingerprint(forms)
            st.session_state.pop("_graph_flow_error", None)
            if returned.selected_id in {form["uid"] for form in forms}:
                if returned.selected_id != st.session_state.get("_graph_flow_clicked"):
                    st.session_state.graph_canvas_selected = returned.selected_id
                    st.session_state._graph_flow_clicked = returned.selected_id
            if changed:
                st.rerun()
    if st.session_state.get("_graph_flow_error"):
        st.warning(st.session_state._graph_flow_error)
    options = {form["uid"]: form for form in forms}
    if st.session_state.get("graph_canvas_selected") not in options:
        st.session_state.graph_canvas_selected = next(iter(options))
    selected = st.selectbox(
        "Node to configure",
        list(options),
        key="graph_canvas_selected",
        format_func=lambda uid: f"{options[uid]['label']} ({options[uid]['node_id']})",
    )
    with st.expander("Selected node settings", expanded=True):
        before = forms_fingerprint(forms)
        render_node_editor(forms.index(options[selected]), options[selected])
        if forms_fingerprint(forms) != before:
            st.rerun()
