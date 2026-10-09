from __future__ import annotations

import streamlit as st

from misinformation_simulation.apps.interaction_graph_queue import add_graph
from misinformation_simulation.apps.interaction_graph_state import (
    graph_nodes_to_forms,
    import_graph_payload,
)
from misinformation_simulation.apps.interaction_graph_ui import AVAILABLE_PROVIDERS, set_tree_mode
from misinformation_simulation.apps.interaction_graph_workspace import widget_default
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.simulation.generation import generate_branching_graphs
from misinformation_simulation.simulation.io import graph_config_from_payload


def render_graph_generator() -> None:
    with st.expander("Generate graphs from sequences"):
        sequences = st.text_area(
            "Persona sequences",
            value=widget_default("graph_generator_sequences", "CCCC\nCCPP", ""),
            key="graph_generator_sequences",
            help="One sequence per line. Codes: C, P, D, S, E, M, N. Shared prefixes run once.",
        )
        columns = st.columns(2)
        model = columns[0].text_input(
            "Generated nodes model",
            value=widget_default("graph_generator_model", DEFAULT_LLM_MODEL.value, ""),
            key="graph_generator_model",
        )
        provider = columns[1].selectbox(
            "Generated nodes provider",
            AVAILABLE_PROVIDERS,
            index=widget_default(
                "graph_generator_provider", AVAILABLE_PROVIDERS.index(DEFAULT_LLM_PROVIDER.value), 0
            ),
            key="graph_generator_provider",
        )
        st.caption(
            "Sequences with different first personas produce separate trees in the queue. "
            "Generation does not call an LLM. Every node remains editable."
        )
        actions = st.columns(2)
        edit = actions[0].button(
            "Generate tree in editor", key="generate_editor_tree", width="stretch"
        )
        queue = actions[1].button(
            "Generate trees in queue", key="generate_queue_trees", width="stretch"
        )
        if edit or queue:
            try:
                payloads = generate_branching_graphs(sequences, model=model, provider=provider)
                if edit and len(payloads) != 1:
                    raise ValueError(
                        "Use sequences with the same first persona to edit one tree, "
                        "or choose 'Generate trees in queue' to create all trees."
                    )
                if edit:
                    import_graph_payload(payloads[0])
                    set_tree_mode(st.session_state.graph_nodes, True)
                    st.session_state.current_graph_name = payloads[0]["start_node_id"] + "_tree"
                else:
                    for payload in payloads:
                        nodes, edges, start = graph_config_from_payload(payload)
                        forms = graph_nodes_to_forms(nodes, edges, start)
                        set_tree_mode(forms, True)
                        add_graph(st.session_state.graph_queue, f"{start}_tree", forms)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
