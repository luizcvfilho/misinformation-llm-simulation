from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import Empty
from time import monotonic
from typing import Any

import pandas as pd
import streamlit as st

from misinformation_simulation.analysis.stdi_evaluation import (
    EVALUATION_LABELS,
    available_evaluations,
)
from misinformation_simulation.apps.evaluation_selector import render_evaluation_selector
from misinformation_simulation.apps.interaction_graph_attempts import find_attempts
from misinformation_simulation.apps.interaction_graph_bulk_editor import render_bulk_node_editor
from misinformation_simulation.apps.interaction_graph_canvas import render_graph_canvas
from misinformation_simulation.apps.interaction_graph_categories import (
    build_category_comparison_dataframe,
)
from misinformation_simulation.apps.interaction_graph_components import (
    render_node_editor,
    render_result_bundle,
)
from misinformation_simulation.apps.interaction_graph_credentials import render_api_keys
from misinformation_simulation.apps.interaction_graph_generator import render_graph_generator
from misinformation_simulation.apps.interaction_graph_io import render_path_input
from misinformation_simulation.apps.interaction_graph_preview import render_graph_preview
from misinformation_simulation.apps.interaction_graph_queue import (
    add_graph,
    add_graphs_from_directory,
    build_graph_queue_archive,
    move_graph,
    update_graph,
)
from misinformation_simulation.apps.interaction_graph_recovery import render_recovery_dialog
from misinformation_simulation.apps.interaction_graph_results import (
    clear_imported_results,
    evaluation_result_bundle,
    find_saved_results,
    load_saved_result,
    remove_imported_result,
)
from misinformation_simulation.apps.interaction_graph_run_job import start_graph_run_job
from misinformation_simulation.apps.interaction_graph_sidebar import render_sidebar
from misinformation_simulation.apps.interaction_graph_state import import_graph_payload
from misinformation_simulation.apps.interaction_graph_ui import (
    AVAILABLE_PROVIDERS,
    build_editor_graph_payload,
    resolve_personality_text,
    set_tree_mode,
    validate_node_forms,
)
from misinformation_simulation.apps.interaction_graph_workspace import (
    open_run_analysis,
    widget_default,
)
from misinformation_simulation.apps.model_selector import render_model_selector
from misinformation_simulation.config.prompts import (
    GRAPH_REWRITE_MODE_LABELS,
    GRAPH_REWRITE_MODES,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.enums.models import validate_provider_model
from misinformation_simulation.llm.recovery import PermissionStore
from misinformation_simulation.llm.service_tier import (
    OPENAI_SERVICE_TIERS,
    configured_openai_service_tier,
)
from misinformation_simulation.simulation import run_news_interaction_graph
from misinformation_simulation.simulation.io import resolve_project_path
from misinformation_simulation.text_metrics.vad import DEFAULT_VAD_MODEL_NAME

GRAPH_OUTPUT_LAYOUT_VERSION = 41

__all__ = [
    "render_sidebar",
    "render_configuration_tab",
    "render_results_tab",
    "render_run_recovery_tab",
]


def render_configuration_tab(df: pd.DataFrame | None, dataset_label: str) -> None:
    render_graph_generator()
    dataset_info_col, execution_col = st.columns([1.2, 1])

    with dataset_info_col:
        available_columns = _render_dataset_preview(df, dataset_label)

    with execution_col:
        settings = _render_execution_settings(df, available_columns)

    editor_header_cols = st.columns([3, 1], vertical_alignment="center")
    editor_header_cols[0].subheader("Graph editor")
    export_placeholder = editor_header_cols[1].empty()
    set_tree_mode(st.session_state.graph_nodes, True)
    editor = st.radio(
        "Graph editing",
        ["Visual canvas", "Node forms"],
        horizontal=True,
        key="graph_editing_mode",
    )
    if editor == "Visual canvas":
        render_graph_canvas()
    else:
        render_bulk_node_editor()
        for index, node_form in enumerate(st.session_state.graph_nodes):
            render_node_editor(index, node_form)

    graph_payload = build_editor_graph_payload(st.session_state.graph_nodes)
    with export_placeholder.container():
        _render_graph_export(graph_payload)

    for error in validate_node_forms(st.session_state.graph_nodes):
        st.error(error)
    st.caption(f"Start node: `{graph_payload.get('start_node_id', '-')}`")

    _render_graph_queue()
    _render_run_controls(df, settings)


def _render_dataset_preview(df: pd.DataFrame | None, dataset_label: str) -> list[str]:
    st.subheader("Dataset preview")
    if df is None:
        st.info("Load a dataset from the sidebar to configure execution options.")
        return []

    st.caption(f"Source: `{dataset_label}`")
    preview_cols = [column for column in df.columns[:8]]
    st.dataframe(df[preview_cols].head(10), width="stretch", height="content")
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
    max_rows = st.number_input(
        "Max rows",
        min_value=1,
        value=widget_default(
            "simulation_execution_settings_max_rows",
            min(len(df), 5) if df is not None and not df.empty else 1,
            "min",
        ),
        step=1,
        help="Maximum number of dataset rows to process in this run.",
        key="simulation_execution_settings_max_rows",
    )
    rewrite_mode = st.selectbox(
        "Transmission mode",
        GRAPH_REWRITE_MODES,
        index=widget_default("graph_rewrite_mode", GRAPH_REWRITE_MODES.index("interpretive"), 0),
        format_func=GRAPH_REWRITE_MODE_LABELS.__getitem__,
        key="graph_rewrite_mode",
        help="Applies to every node and every queued graph in this execution.",
    )
    if rewrite_mode == "interpretive":
        st.caption(
            "Each person interprets and passes along only the previous message according to "
            "their personality. They may omit context and add unsupported interpretations "
            "of causes or intentions. Retained names, numbers, dates, and quotes must remain "
            "accurate, without inventing concrete details."
        )
    else:
        st.caption(
            "Each person rewrites the received text according to their personality, changing "
            "framing, emphasis, tone, and vocabulary while preserving factual content and "
            "adding no new information."
        )
    with st.expander("Dataset columns"):
        default_text_column = (
            "description"
            if "description" in available_columns
            else _first_column(available_columns)
        )
        default_title_column = (
            "title" if "title" in available_columns else _first_column(available_columns)
        )
        default_news_id_column = "article_id" if "article_id" in available_columns else ""

        text_column = st.selectbox(
            "Text column",
            available_columns or [""],
            index=widget_default(
                "simulation_execution_settings_text_column",
                _option_index(available_columns, default_text_column),
                0,
            ),
            help="Column containing the source text that will be rewritten by the graph nodes.",
            key="simulation_execution_settings_text_column",
        )
        title_column = st.selectbox(
            "Title column",
            available_columns or [""],
            index=widget_default(
                "simulation_execution_settings_title_column",
                _option_index(available_columns, default_title_column),
                0,
            ),
            help=(
                "Article title used in topic drift extraction and faithful rewrite prompts. "
                "Interpretive relay does not supply the original title to the rewriting nodes."
            ),
            key="simulation_execution_settings_title_column",
        )
        news_id_options = [""] + available_columns if available_columns else [""]
        news_id_column = st.selectbox(
            "News ID column",
            news_id_options,
            index=widget_default(
                "simulation_execution_settings_news_id_column",
                _option_index(news_id_options, default_news_id_column),
                0,
            ),
            format_func=lambda value: "Use row index" if value == "" else value,
            help="Optional stable identifier for grouping results by news item.",
            key="simulation_execution_settings_news_id_column",
        )
        allow_title_fallback = st.checkbox(
            "Allow title fallback when the text column is empty",
            value=widget_default(
                "simulation_execution_settings_allow_title_fallback_when_the_text_column_is_empty",
                True,
                False,
            ),
            help=(
                "When enabled, the simulation can use the title "
                "if the selected text column is empty."
            ),
            key="simulation_execution_settings_allow_title_fallback_when_the_text_column_is_empty",
        )

    with st.expander("STDI evaluation"):
        st.caption(
            "STDI compares topic structures with local MiniLM embeddings after all graph steps "
            "are generated. The first run may download the model."
        )

        stdi_method = render_evaluation_selector(
            "STDI method",
            key="stdi_comparison_method",
            methods={"cluster": "Cluster", "llm": "LLM", "dual": "Dual"},
            help=(
                "Cluster uses embeddings; LLM uses a judge. Dual calculates both structural "
                "evaluations and averages their complete scores using the selected VAD."
            ),
        )
        judge_requested = stdi_method in {"dual", "llm"}
        repeat_judge = st.checkbox(
            "Repeat LLM judge evaluation",
            value=widget_default("repeat_stdi_judge", True, False),
            key="repeat_stdi_judge",
            disabled=not judge_requested,
            help="Evaluate each text pair multiple times with the same model and rubric.",
        )
        judge_repeats = st.number_input(
            "LLM judge evaluations per pair",
            min_value=2,
            value=widget_default("stdi_judge_repeats", 3, "min"),
            step=1,
            key="stdi_judge_repeats",
            disabled=not judge_requested or not repeat_judge,
            help=(
                "Total evaluations per pair, including the first. "
                "Disable repetition for one evaluation."
            ),
        )
        if judge_requested:
            effective_repeats = int(judge_repeats) if repeat_judge else 1
            st.caption(
                f"Each pair uses {effective_repeats} judge evaluation(s). Scores are averaged; "
                "individual judgments and variability are saved. Additional evaluations increase "
                "judge API cost approximately in proportion to their count."
            )

    vad_expander = st.expander("VAD evaluation")
    advanced_settings = _render_advanced_settings()
    with vad_expander:
        vad_method = render_evaluation_selector(
            "VAD method",
            key="vad_method",
            methods={
                "model": DEFAULT_VAD_MODEL_NAME,
                "llm": "LLM",
                "dual": "Dual",
            },
            help=(
                "Dual averages VAD drifts. With Dual STDI, each branch uses its corresponding VAD."
            ),
        )
        vad_llm_provider = st.selectbox(
            "VAD evaluator provider",
            options=AVAILABLE_PROVIDERS,
            key="vad_llm_provider",
            disabled=vad_method == "model",
            index=widget_default(
                "vad_llm_provider",
                AVAILABLE_PROVIDERS.index(str(advanced_settings["topic_drift_provider"])),
                0,
            ),
        )
        vad_llm_model = render_model_selector(
            vad_llm_provider,
            label="VAD evaluator model",
            option_key="vad_llm_model_option",
            custom_key="vad_llm_model",
            default=str(advanced_settings["topic_drift_model"]),
            custom_label="Custom VAD evaluator model",
            disabled=vad_method == "model",
        )

    return {
        "graph_name": graph_name,
        "text_column": text_column,
        "title_column": title_column,
        "news_id_column": news_id_column,
        "max_rows": max_rows,
        "allow_title_fallback": allow_title_fallback,
        "rewrite_mode": rewrite_mode,
        "stdi_comparison_method": stdi_method,
        "vad_method": vad_method,
        "vad_llm_model": vad_llm_model,
        "vad_llm_provider": vad_llm_provider,
        "stdi_judge_repeats": int(judge_repeats) if judge_requested and repeat_judge else 1,
        **advanced_settings,
    }


def _render_advanced_settings() -> dict[str, Any]:
    advanced_label = datetime.now().strftime("simulation_ui_%Y%m%d_%H%M%S")
    with st.expander("Advanced options"):
        openai_service_tier = st.selectbox(
            "OpenAI processing mode",
            options=OPENAI_SERVICE_TIERS,
            index=widget_default(
                "openai_service_tier",
                OPENAI_SERVICE_TIERS.index(configured_openai_service_tier()),
                0,
            ),
            format_func=lambda tier: {
                "auto": "Project default (Auto)",
                "default": "Standard",
                "flex": "Flex (lower cost, slower)",
            }[tier],
            key="openai_service_tier",
            disabled=st.session_state.get("run_job") is not None,
            help=(
                "Applies to all official OpenAI calls in this execution: rewrites, topic "
                "extraction, STDI judge and LLM VAD. Other providers are unaffected. "
                "GPT-6 Luna supports Flex; check compatibility when mixing other models."
            ),
        )
        openai_flex_fallback = False
        openai_flex_retry_minutes = 5.0
        if openai_service_tier == "flex":
            openai_flex_fallback = st.checkbox(
                "Allow temporary fallback to Standard",
                value=False,
                key="openai_flex_fallback",
                disabled=st.session_state.get("run_job") is not None,
                help=(
                    "On Flex capacity errors, all runs in this execution temporarily use "
                    "Standard at its higher price. Other API errors keep normal recovery."
                ),
            )
            openai_flex_retry_minutes = st.number_input(
                "Try Flex again after (minutes)",
                min_value=0.1,
                value=widget_default("openai_flex_retry_minutes", 5.0, "min"),
                step=0.5,
                key="openai_flex_retry_minutes",
                disabled=not openai_flex_fallback or st.session_state.get("run_job") is not None,
                help=(
                    "One request tests Flex after this interval. Other requests continue in "
                    "Standard until the test succeeds. Another capacity error restarts "
                    "the interval."
                ),
            )
            st.caption(
                "Flex uses a 15-minute timeout per request and may return resource unavailable "
                "errors. With fallback disabled, retries keep Flex. "
                "Reused results do not make new API calls."
            )
        reuse_previous_results = st.checkbox(
            "Reuse compatible results from previous executions",
            value=True,
            key="reuse_previous_results",
            help=(
                "Reuse saved rewrites and evaluations when inputs, model, prompts and stage "
                "implementation match. Incompatible stages are recalculated automatically. "
                "Disable for a new independent simulation replicate."
            ),
        )
        api_pricing_path = render_path_input(
            "API pricing JSON (optional)",
            key="api_pricing_path",
            title="Select API pricing JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            help="Override dated USD-per-million-token rates; unpriced requests remain unknown.",
        )
        sleep_seconds = st.number_input(
            "Sleep between requests (seconds)",
            min_value=0.0,
            value=widget_default(
                "simulation_advanced_settings_sleep_between_requests_seconds", 0.0, "min"
            ),
            step=0.25,
            help="Delay inserted between simulation steps to reduce API pressure.",
            key="simulation_advanced_settings_sleep_between_requests_seconds",
        )
        max_simultaneous_runs = st.number_input(
            "Simultaneous news runs",
            min_value=1,
            value=widget_default("simulation_advanced_settings_max_simultaneous_runs", 1, "min"),
            step=1,
            help=(
                "Global limit across all graphs. News from the first graph is scheduled first; "
                "spare slots start news from the next graph. Use 1 for sequential execution."
            ),
            key="simulation_advanced_settings_max_simultaneous_runs",
        )
        provider_rpm_limits = _render_provider_rpm_limits()
        retry_attempts = st.number_input(
            "Retry attempts",
            min_value=1,
            value=widget_default("simulation_advanced_settings_retry_attempts", 5, "min"),
            step=1,
            help="Number of retry attempts when a provider request fails temporarily.",
            key="simulation_advanced_settings_retry_attempts",
        )
        topic_drift_provider = st.selectbox(
            "Topic drift provider",
            AVAILABLE_PROVIDERS,
            index=(
                widget_default(
                    "simulation_advanced_settings_topic_drift_provider",
                    AVAILABLE_PROVIDERS.index(DEFAULT_LLM_PROVIDER.value)
                    if DEFAULT_LLM_PROVIDER.value in AVAILABLE_PROVIDERS
                    else 0,
                    0,
                )
            ),
            help="Provider used for topic extraction and the dual STDI judge.",
            key="simulation_advanced_settings_topic_drift_provider",
        )
        topic_drift_model = _render_topic_drift_model_selector(topic_drift_provider)
        output_dir = render_path_input(
            "Output directory",
            title="Select output directory",
            directory=True,
            default="output/interaction_graph/app_runs",
            help=(
                "Each execution gets a folder containing one subfolder per graph, "
                "including executions with a single graph."
            ),
            key="simulation_advanced_settings_output_directory",
        )
        output_prefix = st.text_input(
            "Output prefix",
            value=widget_default("simulation_advanced_settings_output_prefix", advanced_label, ""),
            help=(
                "Date and time prefix for the execution folder. Graph folders and files "
                "use the queue position and up to 32 characters of the graph name."
            ),
            key="simulation_advanced_settings_output_prefix",
        )

    return {
        "sleep_seconds": sleep_seconds,
        "openai_service_tier": openai_service_tier,
        "openai_flex_fallback": openai_flex_fallback,
        "openai_flex_retry_seconds": float(openai_flex_retry_minutes) * 60,
        "reuse_previous_results": reuse_previous_results,
        "api_pricing_path": api_pricing_path,
        "provider_rpm_limits": provider_rpm_limits,
        "max_simultaneous_runs": max_simultaneous_runs,
        "retry_attempts": retry_attempts,
        "topic_drift_provider": topic_drift_provider,
        "topic_drift_model": topic_drift_model,
        "output_dir": output_dir,
        "output_prefix": output_prefix,
    }


def _render_provider_rpm_limits() -> dict[str, int]:
    legacy_rpm = st.session_state.get("simulation_advanced_settings_max_requests_per_minute", 0)
    legacy_provider = st.session_state.get("simulation_advanced_settings_rpm_provider", "gemini")
    providers = st.multiselect(
        "Providers with RPM limits",
        options=[provider for provider in AVAILABLE_PROVIDERS if provider != "local"],
        default=widget_default(
            "simulation_advanced_settings_rpm_providers",
            [legacy_provider] if legacy_rpm else [],
            [],
        ),
        key="simulation_advanced_settings_rpm_providers",
        help="Select any number of providers and set a separate RPM limit for each.",
    )
    st.caption(
        "Each provider has an independent budget shared by its models, keys, news runs, "
        "evaluations and retries. Requests are spaced evenly. "
        "Providers left unselected or set to 0 have no RPM cap."
    )
    limits = {}
    for provider in providers:
        key = f"simulation_advanced_settings_provider_rpm_{provider}"
        limits[provider] = int(
            st.number_input(
                f"Max requests per minute ({provider})",
                min_value=0,
                value=widget_default(key, legacy_rpm if provider == legacy_provider else 0, "min"),
                step=1,
                key=key,
            )
        )
    return limits


def _render_topic_drift_model_selector(provider: str = DEFAULT_LLM_PROVIDER.value) -> str:
    return render_model_selector(
        provider,
        label="Topic drift model preset",
        option_key="topic_drift_model_option",
        custom_key="topic_drift_model_custom",
        default=DEFAULT_LLM_MODEL.value,
        custom_label="Custom topic drift model",
        help="Model used for topic structure extraction before calculating topic drift.",
    )


def _render_graph_export(graph_payload: dict[str, Any]) -> None:
    graph_name = st.session_state.get("current_graph_name", "")
    file_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", graph_name).strip(" .") or "graph_config"
    file_name = file_stem if file_stem.lower().endswith(".json") else f"{file_stem}.json"
    st.download_button(
        "Export graph JSON",
        data=json.dumps(graph_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=file_name,
        mime="application/json",
        key="download_editor_graph_json",
        type="primary",
        width="stretch",
        help=(
            "Download the current nodes, personalities, models, connections, and start node. "
            "Import this JSON from the sidebar to reuse the graph without running a simulation."
        ),
    )


def _render_graph_queue() -> None:
    queue_header_cols = st.columns([3, 1], vertical_alignment="center")
    queue_header_cols[0].subheader("Graph queue")
    export_placeholder = queue_header_cols[1].empty()
    _render_queue_edit_controls()
    st.caption(
        "Add a copy of the current graph, then edit or import another graph and add it. "
        "The queued graphs run in the order shown below."
    )
    add_cols = st.columns([3, 1])
    name = add_cols[0].text_input("Graph name", key="queued_graph_name")
    if add_cols[1].button("Add current graph", width="stretch"):
        try:
            add_graph(st.session_state.graph_queue, name, st.session_state.graph_nodes)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    _render_queue_folder_import()

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
            if st.session_state.get("editing_queue_graph_id") == graph["id"]:
                st.session_state.pop("editing_queue_graph_id", None)
            st.rerun()
        with st.expander(f"Graph details · {index + 1}. {graph['name']}"):
            _render_queued_graph_details(graph)

    with export_placeholder.container():
        st.download_button(
            "Export graph queue ZIP",
            data=build_graph_queue_archive(st.session_state.graph_queue),
            file_name="graph_queue.zip",
            mime="application/zip",
            key="download_graph_queue_zip",
            disabled=not st.session_state.graph_queue,
            width="stretch",
            help=(
                "Download every queued graph as a separate JSON file, numbered in queue order. "
                "Extract the ZIP and use Add graphs from a folder to reuse the queue."
            ),
        )


def _load_queued_graph_copy(graph_id: str) -> None:
    graph = next(graph for graph in st.session_state.graph_queue if graph["id"] == graph_id)
    import_graph_payload(build_editor_graph_payload(graph["nodes"]))
    st.session_state.current_graph_name = f"{graph['name']} copy"
    st.session_state.queued_graph_name = f"{graph['name']} copy"


def _edit_queued_graph(graph_id: str) -> None:
    graph = next(graph for graph in st.session_state.graph_queue if graph["id"] == graph_id)
    import_graph_payload(build_editor_graph_payload(graph["nodes"]))
    st.session_state.current_graph_name = graph["name"]
    st.session_state.queued_graph_name = graph["name"]
    st.session_state.editing_queue_graph_id = graph_id


def _cancel_queued_graph_edit() -> None:
    _edit_queued_graph(st.session_state.editing_queue_graph_id)
    st.session_state.pop("editing_queue_graph_id", None)


def _render_queue_edit_controls() -> None:
    graph_id = st.session_state.get("editing_queue_graph_id")
    if graph_id is None:
        return
    graph = next((graph for graph in st.session_state.graph_queue if graph["id"] == graph_id), None)
    if graph is None:
        st.session_state.pop("editing_queue_graph_id", None)
        return
    st.info(
        f"Editing queued graph: {graph['name']}. "
        "Save changes to update this queue entry; the queue runs its saved version. "
        "Use Current graph name in Execution settings to rename it."
    )
    columns = st.columns(2)
    if columns[0].button("Save changes to queue", key="queue_save_changes", width="stretch"):
        try:
            update_graph(
                st.session_state.graph_queue,
                graph_id,
                st.session_state.current_graph_name,
                st.session_state.graph_nodes,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.queued_graph_name = graph["name"]
            st.session_state.pop(f"queue_detail_node_{graph_id}", None)
            st.success("Graph changes saved to the queue.")
    columns[1].button(
        "Cancel editing",
        key="queue_cancel_edit",
        on_click=_cancel_queued_graph_edit,
        width="stretch",
        help="Discard unsaved changes and restore the saved graph in the editor.",
    )
    if st.session_state.get("run_job") is not None:
        st.caption("Saved changes apply to the next execution; the active run keeps its snapshot.")


def _render_queued_graph_details(graph: dict[str, Any]) -> None:
    forms = graph["nodes"]
    payload = build_editor_graph_payload(forms)
    st.caption(f"Start node: `{payload['start_node_id']}`")
    render_graph_preview(forms)
    parents = {edge["target"]: edge["source"] for edge in payload.get("edges", [])}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Node ID": node["node_id"],
                    "Label": node["label"],
                    "Receive text from": parents.get(node["node_id"], "Source news (root)"),
                    "Provider": node["provider"],
                    "Model": node["model"],
                    "Personality": (
                        form["personality_preset"]
                        if form.get("personality_mode") == "preset"
                        else "Custom"
                    ),
                }
                for form, node in zip(forms, payload["nodes"], strict=True)
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    forms_by_id = {form["node_id"].strip(): form for form in forms}
    node_id = st.selectbox(
        "Personality prompt for node",
        list(forms_by_id),
        key=f"queue_detail_node_{graph['id']}",
    )
    st.code(resolve_personality_text(forms_by_id[node_id]), language=None, wrap_lines=True)
    action_cols = st.columns(3)
    action_cols[0].button(
        "Edit queued graph",
        key=f"queue_edit_{graph['id']}",
        on_click=_edit_queued_graph,
        args=(graph["id"],),
        width="stretch",
        help="Open this graph in the editor, then use Save changes to queue to update it.",
    )
    action_cols[1].button(
        "Load copy into editor",
        key=f"queue_load_copy_{graph['id']}",
        on_click=_load_queued_graph_copy,
        args=(graph["id"],),
        width="stretch",
        help="Replace the current editor contents with an independent copy of this queued graph.",
    )
    action_cols[2].download_button(
        "Export queued graph JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{re.sub(r'[^\w-]+', '_', graph['name']).strip('_') or 'graph'}.json",
        mime="application/json",
        key=f"queue_download_{graph['id']}",
        width="stretch",
    )
    st.caption(
        "To test another rewrite model, load a copy into the editor, change each node's "
        "model and provider, then add it to the queue with a distinct name. "
        "To update this entry instead, use Edit queued graph, then Save changes to queue. "
        "Transmission mode and evaluation settings apply to the entire execution."
    )


def _render_run_controls(
    df: pd.DataFrame | None,
    settings: dict[str, Any],
) -> None:
    queue = st.session_state.graph_queue
    active_job = st.session_state.get("run_job")
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
    with st.expander("API keys", expanded=False):
        api_credentials, missing_keys = render_api_keys(
            graphs, settings, disabled=active_job is not None
        )
    if missing_keys:
        st.info(
            "Expand **API keys** and enter credentials for "
            + ", ".join(missing_keys)
            + " to run the selected graphs."
        )
    permissions = PermissionStore()
    permission_count = len(permissions.signatures())
    if permission_count:
        st.caption(f"Remembered error permissions: {permission_count}")
        if st.button("Clear remembered error permissions"):
            permissions.clear()
            st.rerun()
    run_button = st.button(
        "Run graph queue" if queue else "Run simulation",
        type="primary",
        width="stretch",
        disabled=active_job is not None or bool(missing_keys),
    )
    if run_button:
        validation_errors = _validate_run_inputs(df, settings, graphs)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return
        st.session_state.run_bundles = []
        st.session_state.run_bundle = None
        st.session_state.run_messages = []
        st.session_state.run_progress = None
        st.session_state.last_run_output_dir = settings.get(
            "output_dir", "output/interaction_graph/app_runs"
        )
        st.session_state.run_job = start_graph_run_job(
            df=df,
            graphs=graphs,
            settings=settings,
            runner=run_news_interaction_graph,
            queue_mode=bool(queue),
            api_credentials=api_credentials,
        )

    if st.session_state.get("run_job") is not None:
        _render_run_monitor()
    elif st.session_state.get("run_messages"):
        if st.session_state.get("run_progress") is not None:
            _render_run_progress(st.session_state.run_progress, active=False)
        _render_run_log()


@st.fragment(run_every="3s")
def _render_run_monitor() -> None:
    job = st.session_state.run_job
    finished = None
    while True:
        try:
            kind, payload = job.events.get_nowait()
        except Empty:
            break
        finished = _consume_job_event(kind, payload, finished)

    _update_job_progress(job)

    if finished is not None:
        if "completed_runs" in finished:
            message = (
                f"News runs finished: {finished['completed_runs']} completed, "
                f"{finished['failed_runs']} failed, {finished['cancelled_runs']} cancelled. "
                "Open Results for graph summaries and available steps."
            )
        elif finished["cancelled"]:
            message = "Simulation cancelled. Open Results for any completed steps."
        else:
            message = (
                f"Queue finished: {finished['completed']} completed, "
                f"{finished['failed']} failed. Open the Results tab."
            )
        st.session_state.run_messages.append(message)
        st.session_state.run_job = None
        st.rerun()

    progress = getattr(st.session_state, "run_progress", None)
    if progress is not None:
        _render_run_progress(
            progress, elapsed=monotonic() - job.started_at, active=not job.cancel_event.is_set()
        )

    _render_cancellation_controls(job, progress)

    if st.session_state.run_messages:
        _render_run_log()


def _render_run_log() -> None:
    messages = st.session_state.run_messages
    st.info(messages[-1])
    st.code("\n".join(messages[-20:]), language="text")
    with st.expander(f"Execution history ({len(messages)} messages)"):
        query = st.text_input("Filter history", key="run_log_filter").strip().casefold()
        visible = [message for message in messages if query in message.casefold()]
        st.code("\n".join(visible[-500:]), language="text")
        if len(visible) > 500:
            st.caption(
                "Showing the latest 500 matching messages. Download the complete history below."
            )
        st.download_button(
            "Download complete execution log",
            data="\n".join(messages),
            file_name="execution.log",
            mime="text/plain",
            key="download_execution_log",
        )


def _format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(max(0, round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _render_flex_fallback_status(progress, active):
    if active and (fallback := progress.get("flex_fallback")):
        if fallback["probing"]:
            st.caption("Testing Flex availability; other requests continue in temporary Standard.")
        elif fallback["temporary_standard"]:
            remaining = _format_duration(fallback["retry_in_seconds"])
            st.caption(f"Temporary Standard active · next Flex attempt in {remaining}.")


def _render_run_progress(
    progress: dict[str, Any], *, elapsed: float = 0, active: bool = True
) -> None:
    _render_flex_fallback_status(progress, active)
    if progress.get("parallel"):
        _render_parallel_progress(progress, active)
        return

    completed = progress["completed"]
    total = progress["total"]
    percent = completed / total if total else 1.0
    graph = f"Graph {progress['graph_index']}/{progress['graph_total']}"
    if progress.get("phase"):
        graph += f" · {progress['phase']}"
    if progress["row"]:
        detail = f" · news {progress['row']}/{progress['row_total']}"
        if progress.get("node_id"):
            detail += f" · node {progress['node_id']}"
        elif progress["step"] and not progress.get("phase"):
            detail += f" · node {progress['step']}/{progress['step_total']}"
    else:
        detail = ""
    st.progress(percent, text=f"{graph}{detail} · {percent:.0%} complete")
    if progress.get("stages"):
        st.caption(
            " · ".join(
                f"{stage}: {counts['completed']}/{counts['total']} resolved "
                f"({counts['cache_hits']} cached, {counts['skipped']} skipped, "
                f"{counts['failed']} failed)"
                for stage, counts in progress["stages"].items()
            )
        )
    _render_current_operation(progress, active)

    if active and completed < total:
        if progress.get("estimate_low") is not None:
            st.caption(
                f"Estimated time remaining: {_format_duration(progress['estimate_low'])}–"
                f"{_format_duration(progress['estimate_high'])} (observed operation durations)"
            )
        else:
            stages = ", ".join(progress.get("calibrating", []))
            st.caption(
                "Calibrating time estimate"
                + (
                    f": waiting for timings from {stages}."
                    if stages
                    else ": waiting for operation timings."
                )
            )


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
    evaluations = [("Topic drift", settings["topic_drift_provider"], settings["topic_drift_model"])]
    if settings.get("vad_method", "model") in {"llm", "dual"}:
        evaluations.append(
            (
                "VAD evaluator",
                settings.get("vad_llm_provider") or settings["topic_drift_provider"],
                settings.get("vad_llm_model", settings["topic_drift_model"]),
            )
        )
    for label, provider, model in evaluations:
        try:
            validate_provider_model(provider, model)
        except ValueError as exc:
            validation_errors.append(f"{label}: {exc}")
    if not settings["output_prefix"].strip():
        validation_errors.append("Enter an output prefix before running.")
    return validation_errors


def render_run_recovery_tab() -> None:
    from misinformation_simulation.apps.interaction_graph_attempt_ui import render_run_history

    render_run_history(run_news_interaction_graph)


def render_results_tab() -> None:
    process_pending_saved_result_import()
    st.subheader("Simulation outputs")
    _render_saved_result_import()

    bundles = st.session_state.run_bundles
    _render_imported_result_controls(bundles)
    if not bundles:
        st.info("Run a simulation or import a saved result to populate this dashboard.")
    else:
        _refresh_category_data_from_saved_results(bundles)
        rows = []
        for index, bundle in enumerate(bundles, start=1):
            summary = bundle.get("summary", {})
            rows.append(
                {
                    "order": index,
                    "graph": bundle["name"],
                    "execution": bundle.get("execution_name", ""),
                    "status": bundle["status"],
                    "source": bundle.get("source", "current run"),
                    "rows": summary.get("rows_processed"),
                    "steps": summary.get("steps_total"),
                    "errors": summary.get("steps_error"),
                    "api_requests": summary.get("api_usage", {}).get("requests"),
                    "cache_hits": summary.get("api_usage", {}).get("cache_hits"),
                    "estimated_cost_usd": summary.get("api_usage", {}).get("estimated_cost_usd"),
                    "output_prefix": bundle["output_prefix"],
                    "failure": bundle.get("error", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")
        available = {
            evaluation
            for bundle in bundles
            if isinstance(bundle.get("steps_df"), pd.DataFrame)
            for evaluation in available_evaluations(
                bundle["steps_df"], bundle.get("summary", {}).get("stdi_comparison_method")
            )
        }
        evaluations = [evaluation for evaluation in EVALUATION_LABELS if evaluation in available]
        selected_evaluation = None
        if evaluations:
            selected_evaluation = st.selectbox(
                "STDI evaluation",
                evaluations,
                format_func=EVALUATION_LABELS.__getitem__,
                key="results_stdi_evaluation",
            )
        displayed_bundles = [
            evaluation_result_bundle(bundle, selected_evaluation)
            if selected_evaluation and isinstance(bundle.get("steps_df"), pd.DataFrame)
            else bundle
            for bundle in bundles
        ]
        _render_category_comparison(displayed_bundles)
        selected_index = st.selectbox(
            "Inspect graph result",
            range(len(bundles)),
            format_func=lambda index: (
                f"{index + 1}. {bundles[index]['name']}"
                + (
                    f" — {bundles[index]['execution_name']}"
                    if bundles[index].get("execution_name")
                    else ""
                )
            ),
            key="simulation_results_tab_inspect_graph_result",
        )
        selected = bundles[selected_index]
        _render_result_usage(selected)

        if selected.get("summary_path") and "_studio_analysis_page" in st.session_state:
            if st.button(
                "Analyze this run",
                type="primary",
            ):
                open_run_analysis(selected["summary_path"], selected_evaluation)
        _render_selected_result(selected, displayed_bundles[selected_index])


def _render_selected_result(selected, displayed):
    if selected["status"] == "failed":
        st.error(selected.get("error") or "One or more news runs failed.")
    if selected["status"] == "cancelled":
        st.warning("Simulation cancelled; showing completed steps.")
    if failures := selected.get("summary", {}).get("run_failures"):
        st.warning(f"{len(failures)} news run(s) failed; showing available results.")
    if cancellations := selected.get("summary", {}).get("run_cancellations"):
        st.warning(
            f"{len(cancellations)} news run(s) were cancelled; see Run recovery for details."
        )
    if isinstance(selected.get("steps_df"), pd.DataFrame):
        render_result_bundle(displayed)
    else:
        st.info("This graph has no completed news results.")


def _refresh_category_data_from_saved_results(bundles: list[dict[str, Any]]) -> None:
    for bundle in bundles:
        steps_df = bundle.get("steps_df")
        if (
            not isinstance(steps_df, pd.DataFrame)
            or steps_df.empty
            or "metadata_category" in steps_df.columns
            or bundle.get("_category_refresh_attempted")
        ):
            continue
        bundle["_category_refresh_attempted"] = True
        summary_path = bundle.get("summary_path")
        if not summary_path:
            continue
        try:
            saved_bundle = load_saved_result(Path(summary_path))
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            continue
        if "metadata_category" in saved_bundle["steps_df"].columns:
            bundle["steps_df"] = saved_bundle["steps_df"]
            bundle["news_summary_df"] = saved_bundle["news_summary_df"]


def _render_category_comparison(bundles: list[dict[str, Any]]) -> None:
    st.subheader("News category comparison")
    comparison = build_category_comparison_dataframe(bundles)
    if comparison.empty:
        st.info(
            "Category analysis is available for runs made with a dataset containing 'category'."
        )
        return

    st.caption(
        "Categories come from the input dataset. NewsData may assign several categories to one "
        "article, so category counts overlap. Runs without saved category data are omitted. "
        "Metric means use available step scores."
    )
    st.download_button(
        "Download category comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="graph_category_comparison.csv",
        mime="text/csv",
    )
    categories = sorted(comparison["category"].unique())
    selected_categories = st.multiselect(
        "News categories",
        categories,
        default=widget_default("category_comparison_categories", categories, None),
        key="category_comparison_categories",
    )
    if not selected_categories:
        st.info("Select at least one category to compare.")
        return

    filtered = comparison[comparison["category"].isin(selected_categories)].copy()
    step_options = ["Final step of each graph", *sorted(filtered["step_index"].unique())]
    selected_step = st.selectbox("Graph step", step_options, key="category_comparison_step")
    if selected_step == "Final step of each graph":
        filtered = filtered[
            filtered["step_index"] == filtered.groupby("run")["step_index"].transform("max")
        ]
    else:
        filtered = filtered[filtered["step_index"] == selected_step]

    metric_labels = {
        "success_rate": "Success rate",
        "mean_stdi_vs_original": "Mean STDI vs original",
        "mean_stdi_incremental": "Mean incremental STDI",
        "mean_stdi_cumulative": "Mean cumulative STDI",
        "mean_vad_drift_vs_original": "Mean VAD drift vs original",
        "mean_contradiction_drift_vs_original": "Mean contradiction drift vs original",
    }
    selected_metric = st.selectbox(
        "Comparison metric",
        list(metric_labels),
        format_func=metric_labels.get,
        key="category_comparison_metric",
    )
    chart_data = (
        filtered.groupby(["category", "run"], as_index=False)[selected_metric]
        .mean()
        .dropna(subset=[selected_metric])
    )
    if chart_data.empty:
        st.info("No numeric values are available for this metric and step.")
    else:
        if (chart_data[selected_metric] == 0).all():
            st.caption("All compared values are zero for this metric.")
        scale = {"domain": [0, 1]} if selected_metric == "success_rate" else {"zero": True}
        st.vega_lite_chart(
            chart_data,
            spec={
                "mark": "bar",
                "encoding": {
                    "x": {"field": "category", "type": "nominal", "title": "News category"},
                    "xOffset": {"field": "run", "type": "nominal"},
                    "y": {
                        "field": selected_metric,
                        "type": "quantitative",
                        "title": metric_labels[selected_metric],
                        "scale": scale,
                    },
                    "color": {
                        "field": "run",
                        "type": "nominal",
                        "title": "Graph",
                        "legend": {"orient": "top"},
                    },
                    "tooltip": [
                        {"field": "category", "type": "nominal", "title": "Category"},
                        {"field": "run", "type": "nominal", "title": "Graph"},
                        {
                            "field": selected_metric,
                            "type": "quantitative",
                            "title": metric_labels[selected_metric],
                            "format": ".3f",
                        },
                    ],
                },
            },
            width="stretch",
            height=320,
        )
    st.dataframe(filtered, width="stretch")


def _first_column(available_columns: list[str]) -> str:
    return available_columns[0] if available_columns else ""


def _option_index(options: list[str], selected: str) -> int:
    return options.index(selected) if selected in options else 0


def _render_queue_folder_import():
    with st.expander("Add graphs from a folder"):
        render_path_input(
            "Graph folder path",
            key="graph_queue_folder_path",
            title="Select folder with graph configs",
            directory=True,
            browse_key="browse_graph_queue_folder",
            help="Adds every JSON graph config directly inside this folder, in filename order.",
        )
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


def _update_job_progress(job):
    progress_lock = getattr(job, "progress_lock", None)
    if progress_lock is not None:
        with progress_lock:
            if job.latest_progress is not None:
                if getattr(job, "observing", False):
                    job.latest_progress = {**job.latest_progress, **job.tracker.snapshot()}
                st.session_state.run_progress = {
                    **job.latest_progress,
                    "flex_fallback": job.flex_fallback.snapshot()
                    if getattr(job, "flex_fallback", None)
                    else None,
                }


def _render_cancellation_controls(job, progress):
    if job.cancel_event.is_set():
        st.warning("Cancellation requested. Waiting for the current operation to finish.")
    elif st.button("Cancel simulation", type="secondary", width="stretch"):
        job.cancel_event.set()
        st.warning("Cancellation requested. Waiting for the current operation to finish.")
    if request := job.recovery.pending:
        st.warning("Some runs are waiting for an error recovery decision; others may continue.")
        render_recovery_dialog(job, request)
    if progress and progress.get("parallel"):
        running = [
            item["run_id"]
            for item in progress.get("runs", [])
            if item["state"] not in {"completed", "failed", "cancelled"}
        ]
        if running:
            selected = st.selectbox("Run to cancel", running, key="cancel_selected_news_run")
            if st.button("Cancel selected news run", key="cancel_news_run"):
                job.run_cancellations[selected].set()
                st.info(f"Cancellation requested for {selected}.")


def _render_parallel_progress(progress, active):
    if progress.get("parallel"):
        total = progress["total"]
        resolved = progress["completed"]
        st.progress(
            resolved / total if total else 1.0, text=f"News runs resolved: {resolved}/{total}"
        )
        st.caption(
            f"Simultaneous run limit: {progress['max_simultaneous_runs']} · "
            f"Operations resolved: {progress['operation_completed']}/{progress['operation_total']}"
        )
        runs = []
        for item in progress.get("runs", []):
            current = item.get("current") or {}
            runs.append(
                {
                    **{key: value for key, value in item.items() if key != "current"},
                    "stage": current.get("stage", ""),
                    "node": current.get("node_id", ""),
                    "requested_service_tier": current.get("requested_service_tier", ""),
                }
            )
        if runs:
            st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
        if active and resolved < total:
            if progress.get("awaiting_decisions"):
                st.caption("Time estimate suspended while a recovery decision is pending.")
            elif progress.get("estimate_low") is not None:
                st.caption(
                    f"Estimated time remaining: {_format_duration(progress['estimate_low'])}–"
                    f"{_format_duration(progress['estimate_high'])} "
                    "(observed batch throughput, including quota waits)"
                )
            else:
                st.caption("Calibrating time estimate from observed batch throughput.")
        return


def _render_current_operation(progress, active):
    if active and (current := progress.get("current")):
        context = " · ".join(
            str(current[key])
            for key in ("stage", "news_id", "node_id", "comparison", "model")
            if current.get(key)
        )
        if current.get("repeat"):
            context += f" · repeat {current['repeat']}/{current['repeats']}"
        state = current["kind"]
        if state in {"rate_limit_wait", "retry_wait"}:
            remaining = max(0, current["delay"] - current["age"])
            detail = (
                f"waiting for request limit · {_format_duration(remaining)}"
                if state == "rate_limit_wait"
                else (
                    f"retry {current['attempt'] + 1}/{current['max_attempts']} "
                    f"in {_format_duration(remaining)} · {current['reason']}"
                )
            )
        elif state == "request_started":
            detail = f"awaiting API response for {_format_duration(current['age'])}"
        else:
            detail = f"processing for {_format_duration(current['age'])}"
        st.caption(f"Current operation: {context} · {detail}")


def _render_saved_result_import():
    with st.expander("Import a saved simulation"):
        output_dir = resolve_project_path("output/interaction_graph/app_runs")
        saved_results = find_saved_results(output_dir)
        if saved_results:
            selected_path = st.selectbox(
                "Saved results",
                saved_results,
                format_func=lambda path: str(path.relative_to(output_dir)),
                key="saved_result_selection",
                on_change=_select_saved_result,
            )
        else:
            selected_path = None
            st.caption("No saved simulations found in the default output directory.")
        custom_path = render_path_input(
            "Or enter a summary JSON path",
            key="saved_result_path",
            title="Select simulation summary JSON",
            filetypes=[("Summary files", "*_summary.json"), ("JSON files", "*.json")],
            help="The matching _steps.jsonl file must be in the same folder.",
        )
        chosen_path = resolve_project_path(custom_path) if custom_path.strip() else selected_path
        if chosen_path is not None:
            st.caption(f"Selected summary: {chosen_path}")
        st.button(
            "Import result",
            key="import_saved_result",
            disabled=chosen_path is None,
            on_click=_request_saved_result_import,
        )

    if error := st.session_state.get("saved_result_import_error"):
        st.error(error)
    if feedback := st.session_state.get("saved_result_import_feedback"):
        st.success(feedback["message"])
        if feedback["attempt_count"]:
            st.info(f"Loaded {feedback['attempt_count']} saved attempt(s) in the Run recovery tab.")
        else:
            st.info(
                "This summary loads saved results for inspection. This execution has no recovery "
                "snapshots, so importing it cannot enable Retry or Restart. Recovery snapshots "
                "are saved by new app executions."
            )
        for error in feedback["errors"]:
            st.warning(error)


def _select_saved_result():
    st.session_state.saved_result_path = ""


def _request_saved_result_import():
    st.session_state.pop("saved_result_import_feedback", None)
    st.session_state.pop("saved_result_import_error", None)
    custom_path = st.session_state.get("saved_result_path", "").strip()
    path = (
        resolve_project_path(custom_path)
        if custom_path
        else st.session_state.get("saved_result_selection")
    )
    st.session_state["_pending_saved_result_import"] = str(path) if path is not None else None


def process_pending_saved_result_import():
    if "_pending_saved_result_import" not in st.session_state:
        return
    path = st.session_state["_pending_saved_result_import"]
    try:
        with st.spinner("Loading saved simulation and recovery history..."):
            _import_saved_result(Path(path) if path else None)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.session_state.saved_result_import_error = f"Could not import result: {exc}"
    st.session_state.pop("_pending_saved_result_import", None)


def _import_saved_result(path):
    if path is None:
        raise ValueError("Select a saved result.")
    bundle = load_saved_result(path)
    records, errors = find_attempts(Path(bundle["summary_path"]).parent)
    existing = st.session_state.setdefault("run_bundles", [])
    existing[:] = [
        item
        for item in existing
        if item.get("source") != "imported" or item.get("summary_path") != bundle["summary_path"]
    ]
    existing.append(bundle)
    st.session_state.simulation_results_tab_inspect_graph_result = len(existing) - 1
    attempts = st.session_state.setdefault("run_attempts", {})
    attempts.update({record["manifest_path"]: record for record in records})
    recoverable = [record for record in records if record["state"] in {"failed", "cancelled"}]
    if recoverable:
        st.session_state.recovery_selected_attempt = recoverable[0]["manifest_path"]
    summary = bundle["summary"]
    st.session_state.saved_result_import_feedback = dict(
        message=(
            f"Imported {bundle['name']} from {bundle['execution_name']}: "
            f"{summary.get('rows_processed', 0)} news row(s), "
            f"{summary.get('steps_total', 0)} step(s). Selected below in Inspect graph result."
        ),
        attempt_count=len(records),
        errors=errors,
    )


def _render_result_usage(selected):
    usage = selected.get("summary", {}).get("api_usage")
    if usage:
        st.subheader("API usage and estimated cost")
        st.caption(usage["note"])
        st.dataframe(
            pd.DataFrame.from_dict(usage["stages"], orient="index").rename_axis("stage"),
            width="stretch",
        )
        if usage["estimated_cost_usd"] is None:
            st.info(
                f"Total cost is unavailable for {usage['unpriced_requests']} request(s). "
                f"Known estimated subtotal: USD {usage['known_estimated_cost_usd']:.6f}."
            )
        else:
            st.caption(f"Estimated total: USD {usage['estimated_cost_usd']:.6f}")


def _render_imported_result_controls(bundles):
    if message := st.session_state.pop("saved_result_removal_feedback", None):
        st.success(message)
    imported = {
        bundle["summary_path"]: bundle for bundle in bundles if bundle.get("source") == "imported"
    }
    if not imported:
        return
    with st.container(border=True):
        st.subheader("Manage imported results")
        st.selectbox(
            "Imported result to remove",
            list(imported),
            format_func=lambda path: (
                f"{imported[path]['name']} — {imported[path].get('execution_name') or path}"
            ),
            key="imported_result_to_remove",
        )
        action_cols = st.columns(2)
        action_cols[0].button(
            "Remove selected imported result",
            key="remove_imported_result",
            on_click=_remove_imported_results,
            width="stretch",
        )
        action_cols[1].button(
            "Clear imported results",
            key="clear_imported_results",
            on_click=_remove_imported_results,
            kwargs={"clear_all": True},
            width="stretch",
        )
        st.caption("Removing imported results from this view does not delete saved files.")


def _remove_imported_results(*, clear_all: bool = False) -> None:
    bundles = st.session_state.run_bundles
    inspect_key = "simulation_results_tab_inspect_graph_result"
    previous_index = st.session_state.get(inspect_key, 0)
    selected = bundles[previous_index] if 0 <= previous_index < len(bundles) else None
    if clear_all:
        removed = clear_imported_results(bundles)
    else:
        path = st.session_state.get("imported_result_to_remove")
        index = next(
            (
                i
                for i, bundle in enumerate(bundles)
                if bundle.get("source") == "imported" and bundle.get("summary_path") == path
            ),
            None,
        )
        removed = int(index is not None and remove_imported_result(bundles, index))
    if not removed:
        return
    st.session_state[inspect_key] = next(
        (i for i, bundle in enumerate(bundles) if bundle is selected),
        max(0, min(previous_index, len(bundles) - 1)),
    )
    st.session_state.pop("imported_result_to_remove", None)
    st.session_state.pop("saved_result_import_feedback", None)
    st.session_state.pop("saved_result_import_error", None)
    st.session_state.saved_result_removal_feedback = (
        f"Removed {removed} imported result(s) from Results. Saved files were preserved."
    )


def _consume_job_event(kind, payload, finished):
    if kind == "progress":
        st.session_state.run_messages.append(payload)
    elif kind == "work":
        st.session_state.run_progress = payload
    elif kind == "bundle":
        st.session_state.run_bundles.append(payload)
        if payload["status"] in {"completed", "cancelled"}:
            st.session_state.run_bundle = payload
    elif kind == "attempt":
        if not hasattr(st.session_state, "run_attempts"):
            st.session_state.run_attempts = {}
        st.session_state.run_attempts[payload["manifest_path"]] = payload
    elif kind == "done":
        finished = payload
    return finished
