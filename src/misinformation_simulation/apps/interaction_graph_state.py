from __future__ import annotations

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
from misinformation_simulation.simulation.io import graph_config_from_payload


def initialize_state() -> None:
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = DEFAULT_DATASET_PATH
    if "graph_config_path" not in st.session_state:
        st.session_state.graph_config_path = DEFAULT_GRAPH_CONFIG_PATH
    if "graph_nodes" not in st.session_state:
        st.session_state.graph_nodes = load_initial_graph_nodes()
    if "run_bundle" not in st.session_state:
        st.session_state.run_bundle = None


def load_initial_graph_nodes() -> list[dict[str, str]]:
    default_config = PROJECT_ROOT / DEFAULT_GRAPH_CONFIG_PATH
    if default_config.exists():
        payload = load_local_graph_payload_cached(DEFAULT_GRAPH_CONFIG_PATH)
        nodes, _, _ = graph_config_from_payload(payload)
        if nodes:
            return [
                normalize_node_form(node, position) for position, node in enumerate(nodes, start=1)
            ]
    return [create_default_node_form(1), create_default_node_form(2)]


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
    nodes, _, _ = graph_config_from_payload(payload)
    if not nodes:
        raise ValueError("The selected graph config does not contain nodes.")
    st.session_state.graph_nodes = [
        normalize_node_form(node, position) for position, node in enumerate(nodes, start=1)
    ]
