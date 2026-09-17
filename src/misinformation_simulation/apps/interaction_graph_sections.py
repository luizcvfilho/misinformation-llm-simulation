from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd
import streamlit as st

from misinformation_simulation.apps.interaction_graph_components import (
    render_node_editor,
    render_result_bundle,
)
from misinformation_simulation.apps.interaction_graph_io import select_local_directory
from misinformation_simulation.apps.interaction_graph_preview import render_graph_preview
from misinformation_simulation.apps.interaction_graph_queue import (
    add_graph,
    add_graphs_from_directory,
    move_graph,
)
from misinformation_simulation.apps.interaction_graph_results import (
    clear_imported_results,
    find_saved_results,
    load_saved_result,
    remove_imported_result,
)
from misinformation_simulation.apps.interaction_graph_run_job import start_graph_run_job
from misinformation_simulation.apps.interaction_graph_sidebar import render_sidebar
from misinformation_simulation.apps.interaction_graph_ui import (
    AVAILABLE_MODELS,
    AVAILABLE_PROVIDERS,
    CUSTOM_OPTION,
    build_linear_graph_payload,
    validate_node_forms,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.simulation import run_news_interaction_graph

GRAPH_OUTPUT_LAYOUT_VERSION = 4

__all__ = ["render_sidebar", "render_configuration_tab", "render_results_tab"]


def render_configuration_tab(df: pd.DataFrame | None, dataset_label: str) -> None:
    dataset_info_col, execution_col = st.columns([1.2, 1])

    with dataset_info_col:
        available_columns = _render_dataset_preview(df, dataset_label)

    with execution_col:
        settings = _render_execution_settings(df, available_columns)

    st.subheader("Graph editor")
    for index, node_form in enumerate(st.session_state.graph_nodes):
        render_node_editor(index, node_form)

    st.subheader("Graph preview")
    render_graph_preview(st.session_state.graph_nodes)

    graph_payload = build_linear_graph_payload(st.session_state.graph_nodes)
    _render_graph_export(graph_payload)
    _render_graph_queue()
    _render_run_controls(df, settings)


def _render_dataset_preview(df: pd.DataFrame | None, dataset_label: str) -> list[str]:
    st.subheader("Dataset preview")
    if df is None:
        st.info("Load a dataset from the sidebar to configure execution options.")
        return []

    st.caption(f"Source: `{dataset_label}`")
    preview_cols = [column for column in df.columns[:8]]
    st.dataframe(df[preview_cols].head(8), use_container_width=True)
    return df.columns.tolist()


def _render_execution_settings(
    df: pd.DataFrame | None,
    available_columns: list[str],
) -> dict[str, Any]:
    st.subheader("Execution settings")
    graph_name = st.text_input(
        "Current graph name",
        key="current_graph_name",
        help="Name used for this graph in the results folder and filenames.",
    )
    default_text_column = (
        "description" if "description" in available_columns else _first_column(available_columns)
    )
    default_title_column = (
        "title" if "title" in available_columns else _first_column(available_columns)
    )
    default_news_id_column = "article_id" if "article_id" in available_columns else ""

    text_column = st.selectbox(
        "Text column",
        available_columns or [""],
        index=_option_index(available_columns, default_text_column),
        help="Column containing the source text that will be rewritten by the graph nodes.",
    )
    title_column = st.selectbox(
        "Title column",
        available_columns or [""],
        index=_option_index(available_columns, default_title_column),
        help="Column used as the article title in prompts and topic drift extraction.",
    )
    news_id_options = [""] + available_columns if available_columns else [""]
    news_id_column = st.selectbox(
        "News ID column",
        news_id_options,
        index=_option_index(news_id_options, default_news_id_column),
        format_func=lambda value: "Use row index" if value == "" else value,
        help="Optional stable identifier for grouping results by news item.",
    )
    max_rows = st.number_input(
        "Max rows",
        min_value=1,
        value=min(len(df), 5) if df is not None and not df.empty else 1,
        step=1,
        help="Maximum number of dataset rows to process in this run.",
    )
    allow_title_fallback = st.checkbox(
        "Allow title fallback when the text column is empty",
        value=True,
        help=(
            "When enabled, the simulation can use the title if the selected text column is empty."
        ),
    )
    st.caption(
        "STDI compares topic structures with local MiniLM embeddings after all graph steps "
        "are generated. The first run may download the model."
    )

    advanced_settings = _render_advanced_settings()
    return {
        "graph_name": graph_name,
        "text_column": text_column,
        "title_column": title_column,
        "news_id_column": news_id_column,
        "max_rows": max_rows,
        "allow_title_fallback": allow_title_fallback,
        **advanced_settings,
    }


def _render_advanced_settings() -> dict[str, Any]:
    advanced_label = datetime.now().strftime("simulation_ui_%Y%m%d_%H%M%S")
    with st.expander("Advanced options"):
        sleep_seconds = st.number_input(
            "Sleep between requests (seconds)",
            min_value=0.0,
            value=0.0,
            step=0.25,
            help="Delay inserted between simulation steps to reduce API pressure.",
        )
        max_requests_per_minute = st.number_input(
            "Max requests per minute",
            min_value=0,
            value=0,
            step=1,
            help="Use 0 to disable rate limiting.",
        )
        retry_attempts = st.number_input(
            "Retry attempts",
            min_value=1,
            value=5,
            step=1,
            help="Number of retry attempts when a provider request fails temporarily.",
        )
        topic_drift_provider = st.selectbox(
            "Topic drift provider",
            AVAILABLE_PROVIDERS,
            index=(
                AVAILABLE_PROVIDERS.index(DEFAULT_LLM_PROVIDER.value)
                if DEFAULT_LLM_PROVIDER.value in AVAILABLE_PROVIDERS
                else 0
            ),
            help="Provider used to extract topic structures for STDI/topic drift metrics.",
        )
        topic_drift_model = _render_topic_drift_model_selector()
        output_dir = st.text_input(
            "Output directory",
            value="output/interaction_graph/app_runs",
            help="Parent directory for a named subfolder per graph run.",
        )
        output_prefix = st.text_input(
            "Output prefix",
            value=advanced_label,
            help=(
                "Date and time prefix for the run folder and files; "
                "the graph name is added automatically."
            ),
        )

    return {
        "sleep_seconds": sleep_seconds,
        "max_requests_per_minute": max_requests_per_minute,
        "retry_attempts": retry_attempts,
        "topic_drift_provider": topic_drift_provider,
        "topic_drift_model": topic_drift_model,
        "output_dir": output_dir,
        "output_prefix": output_prefix,
    }


def _render_topic_drift_model_selector() -> str:
    topic_drift_model_default = DEFAULT_LLM_MODEL.value
    topic_drift_model_options = AVAILABLE_MODELS + [CUSTOM_OPTION]
    selected_topic_drift_model_option = (
        topic_drift_model_default
        if topic_drift_model_default in AVAILABLE_MODELS
        else CUSTOM_OPTION
    )
    topic_drift_model_cols = st.columns([1, 1])
    selected_topic_drift_model_option = topic_drift_model_cols[0].selectbox(
        "Topic drift model preset",
        topic_drift_model_options,
        index=topic_drift_model_options.index(selected_topic_drift_model_option),
        key="topic_drift_model_option",
        format_func=lambda value: "Custom model" if value == CUSTOM_OPTION else value,
        help="Model used for topic structure extraction before calculating topic drift.",
    )
    if selected_topic_drift_model_option == CUSTOM_OPTION:
        return topic_drift_model_cols[1].text_input(
            "Custom topic drift model",
            value=topic_drift_model_default,
            key="topic_drift_model_custom",
            help="Provider-specific model id for topic drift extraction.",
        )

    topic_drift_model_cols[1].text_input(
        "Resolved topic drift model",
        value=selected_topic_drift_model_option,
        key="topic_drift_model_resolved",
        disabled=True,
        help="Model id that will be passed to the topic drift extraction step.",
    )
    return selected_topic_drift_model_option


def _render_graph_export(graph_payload: dict[str, Any]) -> None:
    export_cols = st.columns([1, 1, 2])
    export_cols[0].download_button(
        "Download graph JSON",
        data=json.dumps(graph_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="graph_config_ui.json",
        mime="application/json",
        key="download_editor_graph_json",
        use_container_width=True,
    )
    export_cols[1].caption(f"Start node: `{graph_payload.get('start_node_id', '-')}`")


def _render_graph_queue() -> None:
    st.subheader("Graph queue")
    st.caption(
        "Add a copy of the current graph, then edit or import another graph and add it. "
        "The queued graphs run in the order shown below."
    )
    add_cols = st.columns([3, 1])
    name = add_cols[0].text_input("Graph name", key="queued_graph_name")
    if add_cols[1].button("Add current graph", use_container_width=True):
        try:
            add_graph(st.session_state.graph_queue, name, st.session_state.graph_nodes)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    with st.expander("Add graphs from a folder"):
        folder_cols = st.columns([3, 1], vertical_alignment="bottom")
        folder_cols[0].text_input(
            "Graph folder path",
            key="graph_queue_folder_path",
            help="Adds every JSON graph config directly inside this folder, in filename order.",
        )
        if folder_cols[1].button("Browse...", key="browse_graph_queue_folder"):
            try:
                selected_folder = select_local_directory(title="Select folder with graph configs")
                if selected_folder:
                    st.session_state.graph_queue_folder_path = selected_folder
                    st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if st.button("Add folder to queue", disabled=st.session_state.get("run_job") is not None):
            try:
                added, errors = add_graphs_from_directory(
                    st.session_state.graph_queue, st.session_state.graph_queue_folder_path
                )
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                if added:
                    st.success(f"Added {len(added)} graph(s): {', '.join(added)}")
                for error in errors:
                    st.error(error)

    for index, graph in enumerate(st.session_state.graph_queue):
        cols = st.columns([5, 1, 1, 1])
        cols[0].write(f"{index + 1}. **{graph['name']}** ({len(graph['nodes'])} nodes)")
        if cols[1].button("↑", key=f"queue_up_{graph['id']}", disabled=index == 0):
            move_graph(st.session_state.graph_queue, index, -1)
            st.rerun()
        if cols[2].button(
            "↓",
            key=f"queue_down_{graph['id']}",
            disabled=index == len(st.session_state.graph_queue) - 1,
        ):
            move_graph(st.session_state.graph_queue, index, 1)
            st.rerun()
        if cols[3].button("Remove", key=f"queue_remove_{graph['id']}"):
            st.session_state.graph_queue.pop(index)
            st.rerun()


def _render_run_controls(
    df: pd.DataFrame | None,
    settings: dict[str, Any],
) -> None:
    queue = st.session_state.graph_queue
    active_job = st.session_state.get("run_job")
    run_button = st.button(
        "Run graph queue" if queue else "Run simulation",
        type="primary",
        use_container_width=True,
        disabled=active_job is not None,
    )
    if run_button:
        graphs = (
            deepcopy(queue)
            if queue
            else [
                {
                    "name": settings["graph_name"].strip(),
                    "nodes": deepcopy(st.session_state.graph_nodes),
                }
            ]
        )
        validation_errors = _validate_run_inputs(df, settings, graphs)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return
        st.session_state.run_bundles = []
        st.session_state.run_bundle = None
        st.session_state.run_messages = []
        st.session_state.run_job = start_graph_run_job(
            df=df,
            graphs=graphs,
            settings=settings,
            runner=run_news_interaction_graph,
            queue_mode=bool(queue),
        )

    if st.session_state.get("run_job") is not None:
        _render_run_monitor()
    elif st.session_state.get("run_messages"):
        st.info(st.session_state.run_messages[-1])
        st.code("\n".join(st.session_state.run_messages[-20:]), language="text")


@st.fragment(run_every="1s")
def _render_run_monitor() -> None:
    job = st.session_state.run_job
    finished = None
    while True:
        try:
            kind, payload = job.events.get_nowait()
        except Empty:
            break
        if kind == "progress":
            st.session_state.run_messages.append(payload)
        elif kind == "bundle":
            st.session_state.run_bundles.append(payload)
            if payload["status"] in {"completed", "cancelled"}:
                st.session_state.run_bundle = payload
        elif kind == "done":
            finished = payload

    if finished is not None:
        if finished["cancelled"]:
            message = "Simulation cancelled. Open Results for any completed steps."
        else:
            message = (
                f"Queue finished: {finished['completed']} completed, "
                f"{finished['failed']} failed. Open the Results tab."
            )
        st.session_state.run_messages.append(message)
        st.session_state.run_job = None
        st.rerun()

    if job.cancel_event.is_set():
        st.warning("Cancellation requested. Waiting for the current operation to finish.")
    elif st.button("Cancel simulation", type="secondary", use_container_width=True):
        job.cancel_event.set()
        st.warning("Cancellation requested. Waiting for the current operation to finish.")
    if st.session_state.run_messages:
        st.info(st.session_state.run_messages[-1])
        st.code("\n".join(st.session_state.run_messages[-20:]), language="text")


def _validate_run_inputs(
    df: pd.DataFrame | None,
    settings: dict[str, Any],
    graphs: list[dict[str, Any]],
) -> list[str]:
    validation_errors = [
        f"Graph {index} ({graph['name']}): {error}"
        for index, graph in enumerate(graphs, start=1)
        for error in validate_node_forms(graph["nodes"])
    ]
    if df is None:
        validation_errors.append("Load a valid dataset before running the simulation.")
    if not st.session_state.graph_queue and not settings["graph_name"].strip():
        validation_errors.append("Enter a name for the current graph.")
    if not settings["text_column"]:
        validation_errors.append("Select a text column before running the simulation.")
    if not settings["title_column"]:
        validation_errors.append("Select a title column before running the simulation.")
    if not settings["topic_drift_model"].strip():
        validation_errors.append("Select or enter a topic drift model before running.")
    if not settings["output_prefix"].strip():
        validation_errors.append("Enter an output prefix before running.")
    return validation_errors


def render_results_tab() -> None:
    st.subheader("Simulation outputs")
    with st.expander("Import a saved simulation"):
        output_dir = Path("output/interaction_graph/app_runs")
        saved_results = find_saved_results(output_dir)
        if saved_results:
            selected_path = st.selectbox(
                "Saved results",
                saved_results,
                format_func=lambda path: str(path.relative_to(output_dir)),
                key="saved_result_selection",
            )
        else:
            selected_path = None
            st.caption("No saved simulations found in the default output directory.")
        custom_path = st.text_input(
            "Or enter a summary JSON path",
            key="saved_result_path",
            help="The matching _steps.jsonl file must be in the same folder.",
        )
        if st.button("Import result", disabled=not (custom_path.strip() or selected_path)):
            try:
                path = Path(custom_path.strip()) if custom_path.strip() else selected_path
                if path is None:
                    raise ValueError("Select a saved result.")
                bundle = load_saved_result(path)
            except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
                st.error(f"Could not import result: {exc}")
            else:
                existing = st.session_state.run_bundles
                existing[:] = [
                    item
                    for item in existing
                    if item.get("source") != "imported"
                    or item.get("summary_path") != bundle["summary_path"]
                ]
                existing.append(bundle)
                st.success(f"Imported {bundle['output_prefix']}.")
    bundles = st.session_state.run_bundles
    if not bundles:
        st.info("Run a simulation or import a saved result to populate this dashboard.")
    else:
        rows = []
        for index, bundle in enumerate(bundles, start=1):
            summary = bundle.get("summary", {})
            rows.append(
                {
                    "order": index,
                    "graph": bundle["name"],
                    "status": bundle["status"],
                    "source": bundle.get("source", "current run"),
                    "rows": summary.get("rows_processed"),
                    "steps": summary.get("steps_total"),
                    "errors": summary.get("steps_error"),
                    "output_prefix": bundle["output_prefix"],
                    "failure": bundle.get("error", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        selected_index = st.selectbox(
            "Inspect graph result",
            range(len(bundles)),
            format_func=lambda index: f"{index + 1}. {bundles[index]['name']}",
        )
        selected = bundles[selected_index]
        action_cols = st.columns(2)
        if action_cols[0].button(
            "Remove selected imported result",
            disabled=selected.get("source") != "imported",
            use_container_width=True,
        ):
            remove_imported_result(bundles, selected_index)
            st.rerun()
        if action_cols[1].button(
            "Clear imported results",
            disabled=not any(bundle.get("source") == "imported" for bundle in bundles),
            use_container_width=True,
        ):
            clear_imported_results(bundles)
            st.rerun()
        st.caption("Removing imported results from this view does not delete saved files.")
        if selected["status"] == "failed":
            st.error(selected["error"])
        else:
            if selected["status"] == "cancelled":
                st.warning("Simulation cancelled; showing completed steps.")
            render_result_bundle(selected)


def _first_column(available_columns: list[str]) -> str:
    return available_columns[0] if available_columns else ""


def _option_index(options: list[str], selected: str) -> int:
    return options.index(selected) if selected in options else 0
