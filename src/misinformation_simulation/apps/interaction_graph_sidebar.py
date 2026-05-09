from __future__ import annotations

import pandas as pd
import streamlit as st

from misinformation_simulation.apps.interaction_graph_io import (
    load_local_dataframe_cached,
    load_local_graph_payload_cached,
    load_uploaded_dataframe,
    load_uploaded_graph_payload,
    select_local_file,
)
from misinformation_simulation.apps.interaction_graph_state import (
    import_graph_payload,
    reset_graph,
)
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form


def render_sidebar() -> tuple[pd.DataFrame | None, str]:
    with st.sidebar:
        st.header("Data")
        df, dataset_label = _render_dataset_loader()

        st.divider()
        st.header("Import graph")
        _render_graph_importer()
        _render_graph_actions()

    return df, dataset_label


def _render_dataset_loader() -> tuple[pd.DataFrame | None, str]:
    dataset_mode = st.radio("Dataset source", ["Project file", "Upload"], horizontal=True)
    dataset_error = None
    df: pd.DataFrame | None = None
    dataset_label = ""
    if dataset_mode == "Project file":
        dataset_path_cols = st.columns([3, 1], vertical_alignment="bottom")
        dataset_path_cols[0].text_input(
            "Dataset path",
            key="dataset_path",
            help="CSV, JSON, or JSONL file with the news rows used as simulation input.",
        )
        if dataset_path_cols[1].button(
            "Browse...",
            key="browse_dataset_path",
            use_container_width=True,
        ):
            try:
                selected_path = select_local_file(
                    title="Select dataset file",
                    filetypes=[
                        ("Data files", "*.csv *.json *.jsonl"),
                        ("CSV files", "*.csv"),
                        ("JSON files", "*.json *.jsonl"),
                        ("All files", "*.*"),
                    ],
                )
                if selected_path is not None:
                    st.session_state.dataset_path = selected_path
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))
        dataset_label = st.session_state.dataset_path
        try:
            df = load_local_dataframe_cached(dataset_label)
        except Exception as exc:
            dataset_error = str(exc)
    else:
        uploaded_dataset = st.file_uploader(
            "Upload CSV, JSON, or JSONL", type=["csv", "json", "jsonl"]
        )
        dataset_label = uploaded_dataset.name if uploaded_dataset is not None else "upload"
        if uploaded_dataset is not None:
            try:
                df = load_uploaded_dataframe(uploaded_dataset)
            except Exception as exc:
                dataset_error = str(exc)

    if dataset_error:
        st.error(dataset_error)
    elif df is not None:
        st.success(f"{len(df)} row(s) loaded")
        st.caption(f"Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}")

    return df, dataset_label


def _render_graph_importer() -> None:
    graph_mode = st.radio("Graph source", ["Project file", "Upload"], horizontal=True)
    if graph_mode == "Project file":
        graph_path_cols = st.columns([3, 1], vertical_alignment="bottom")
        graph_path_cols[0].text_input(
            "Graph config path",
            key="graph_config_path",
            help="JSON file with nodes, edges, and optional start_node_id for the graph.",
        )
        if graph_path_cols[1].button(
            "Browse...",
            key="browse_graph_config_path",
            use_container_width=True,
        ):
            try:
                selected_path = select_local_file(
                    title="Select graph config file",
                    filetypes=[
                        ("JSON files", "*.json"),
                        ("All files", "*.*"),
                    ],
                )
                if selected_path is not None:
                    st.session_state.graph_config_path = selected_path
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if st.button("Load graph config", use_container_width=True):
            try:
                import_graph_payload(
                    load_local_graph_payload_cached(st.session_state.graph_config_path)
                )
                st.success("Graph config loaded into the editor.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        uploaded_graph = st.file_uploader("Upload graph JSON", type=["json"], key="graph_upload")
        if st.button("Import uploaded graph", use_container_width=True):
            try:
                import_graph_payload(load_uploaded_graph_payload(uploaded_graph))
                st.success("Uploaded graph config loaded into the editor.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _render_graph_actions() -> None:
    sidebar_action_cols = st.columns(2)
    if sidebar_action_cols[0].button("Add node", use_container_width=True):
        st.session_state.graph_nodes.append(
            create_default_node_form(len(st.session_state.graph_nodes) + 1)
        )
        st.rerun()
    if sidebar_action_cols[1].button("Reset graph", use_container_width=True):
        reset_graph()
        st.rerun()
