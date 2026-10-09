from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from misinformation_simulation.apps.interaction_graph_io import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GRAPH_CONFIG_PATH,
    PROJECT_ROOT,
    load_local_graph_payload_cached,
)
from misinformation_simulation.apps.interaction_graph_ui import (
    create_default_node_form,
    normalize_node_form,
)
from misinformation_simulation.config.interaction_chains import (
    DEFAULT_INTERACTION_CHAINS_PATH,
    INTERACTION_CHAINS_ROOT,
)
from misinformation_simulation.simulation.graph import (
    SimulationEdge,
    SimulationNode,
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
    _topological_path,
)
from misinformation_simulation.simulation.io import graph_config_from_payload
from misinformation_simulation.simulation.topology import _root_to_leaf_paths


def initialize_state() -> None:
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = DEFAULT_DATASET_PATH
    if "graph_config_path" not in st.session_state:
        st.session_state.graph_config_path = DEFAULT_GRAPH_CONFIG_PATH
    if st.session_state.get("graph_queue_folder_path", "") in {"", INTERACTION_CHAINS_ROOT}:
        st.session_state.graph_queue_folder_path = DEFAULT_INTERACTION_CHAINS_PATH
    if "graph_nodes" not in st.session_state:
        st.session_state.graph_nodes = load_initial_graph_nodes()
    if "current_graph_name" not in st.session_state:
        st.session_state.current_graph_name = Path(DEFAULT_GRAPH_CONFIG_PATH).stem
    if "run_bundle" not in st.session_state:
        st.session_state.run_bundle = None
    if "graph_queue" not in st.session_state:
        st.session_state.graph_queue = []
    if "run_bundles" not in st.session_state:
        st.session_state.run_bundles = []
    if "run_job" not in st.session_state:
        st.session_state.run_job = None
    if "run_messages" not in st.session_state:
        st.session_state.run_messages = []
    if "run_progress" not in st.session_state:
        st.session_state.run_progress = None


def load_initial_graph_nodes() -> list[dict[str, str]]:
    default_config = PROJECT_ROOT / DEFAULT_GRAPH_CONFIG_PATH
    if default_config.exists():
        payload = load_local_graph_payload_cached(DEFAULT_GRAPH_CONFIG_PATH)
        nodes, edges, start_node_id = graph_config_from_payload(payload)
        if nodes:
            return graph_nodes_to_forms(nodes, edges, start_node_id)
    return [create_default_node_form(1), create_default_node_form(2)]


def graph_nodes_to_forms(
    nodes: list[SimulationNode],
    edges: list[SimulationEdge] | None,
    start_node_id: str | None,
) -> list[dict[str, str]]:
    nodes_by_id = _normalize_nodes(nodes)
    normalized_edges = _normalize_edges(nodes_by_id, edges)
    resolved_start_node = _resolve_start_node(nodes_by_id, normalized_edges, start_node_id)
    ordered_node_ids = _topological_path(nodes_by_id, normalized_edges, resolved_start_node)
    forms = [
        normalize_node_form(nodes_by_id[node_id], position)
        for position, node_id in enumerate(ordered_node_ids, start=1)
    ]
    if len(_root_to_leaf_paths(ordered_node_ids, normalized_edges)) > 1:
        forms_by_id = {form["node_id"]: form for form in forms}
        parents = {edge.target: edge.source for edge in normalized_edges}
        for form in forms:
            parent = parents.get(form["node_id"])
            form["parent_uid"] = forms_by_id[parent]["uid"] if parent else ""
    return forms


def move_node(index: int, direction: int) -> None:
    target_index = index + direction
    if target_index < 0 or target_index >= len(st.session_state.graph_nodes):
        return
    nodes = st.session_state.graph_nodes
    nodes[index], nodes[target_index] = nodes[target_index], nodes[index]


def remove_node(index: int) -> None:
    if len(st.session_state.graph_nodes) == 1:
        return
    st.session_state.graph_nodes.pop(index)


def reset_graph() -> None:
    st.session_state.graph_nodes = load_initial_graph_nodes()


def import_graph_payload(payload: dict[str, Any]) -> None:
    nodes, edges, start_node_id = graph_config_from_payload(payload)
    if not nodes:
        raise ValueError("The selected graph config does not contain nodes.")
    st.session_state.graph_nodes = graph_nodes_to_forms(nodes, edges, start_node_id)
