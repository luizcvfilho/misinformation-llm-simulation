from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.apps.interaction_graph_ui import (  # noqa: E402
    AVAILABLE_MODELS,
    AVAILABLE_PROVIDERS,
    CUSTOM_OPTION,
    PREDEFINED_PERSONALITIES,
    build_linear_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    create_default_node_form,
    normalize_node_form,
    steps_to_dataframe,
    validate_node_forms,
)
from misinformation_simulation.simulation import run_news_interaction_graph  # noqa: E402
from misinformation_simulation.simulation.io import (  # noqa: E402
    DEFAULT_PROJECT_ROOT,
    graph_config_from_payload,
    load_graph_config_payload,
    load_news_dataframe,
)

DEFAULT_DATASET_PATH = "data/graph_news.csv"
DEFAULT_GRAPH_CONFIG_PATH = "data/graph_config.json"


@st.cache_data(show_spinner=False)
def load_local_dataframe_cached(path_str: str) -> pd.DataFrame:
    return load_news_dataframe(path_str, project_root=DEFAULT_PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_local_graph_payload_cached(path_str: str) -> dict[str, Any]:
    return load_graph_config_payload(path_str, project_root=DEFAULT_PROJECT_ROOT)


def load_uploaded_dataframe(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Upload a dataset file to continue.")

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(io.BytesIO(uploaded_file.getvalue()), lines=suffix == ".jsonl")
    else:
        raise ValueError("Unsupported file type. Use CSV, JSON, or JSONL.")

    if "article_id" in df.columns:
        article_ids = df["article_id"].fillna("").astype(str).str.strip().str.lower()
        df = df[article_ids != "__query_metadata__"].copy()
    return df


def load_uploaded_graph_payload(uploaded_file: Any) -> dict[str, Any]:
    if uploaded_file is None:
        raise ValueError("Upload a graph JSON file to continue.")
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def select_local_file(
    *,
    title: str,
    filetypes: list[tuple[str, str]],
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Could not open the local file explorer.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        selected_path = filedialog.askopenfilename(
            title=title,
            initialdir=str(PROJECT_ROOT),
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    return selected_path or None


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


def render_graph_preview(node_forms: list[dict[str, str]]) -> None:
    if not node_forms:
        st.info("Add nodes to preview the interaction chain.")
        return

    preview_parts = [
        """
        <style>
        body {
            margin: 0;
            background: transparent;
        }
        .graph-preview-shell {
            --preview-bg: var(--background-color, transparent);
            --preview-node-bg: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
            --preview-border: color-mix(in srgb, var(--text-color, currentColor) 22%, transparent);
            --preview-text: var(--text-color, currentColor);
            --preview-muted: color-mix(in srgb, var(--text-color, currentColor) 72%, transparent);
            --preview-accent: var(--primary-color, #ff4b4b);
            --preview-accent-soft: color-mix(
                in srgb,
                var(--primary-color, #ff4b4b) 16%,
                var(--background-color, transparent)
            );
            border: 1px solid var(--preview-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--preview-bg);
            color: var(--preview-text);
        }
        .graph-preview-chain {
            display: grid;
            gap: 1rem;
        }
        .graph-preview-row {
            display: flex;
            align-items: stretch;
            width: 100%;
        }
        .graph-preview-unit {
            display: flex;
            flex: 1 1 0;
            min-width: 0;
        }
        .graph-preview-node {
            width: 100%;
            border: 1px solid var(--preview-border);
            border-radius: 8px;
            background: var(--preview-node-bg);
            min-height: 100%;
            padding: 0.7rem 0.75rem;
            box-sizing: border-box;
            position: relative;
        }
        .graph-preview-node-number {
            display: inline-grid;
            place-items: center;
            width: 1.45rem;
            height: 1.45rem;
            margin-bottom: 0.45rem;
            border: 2px solid var(--preview-accent);
            border-radius: 50%;
            background: var(--preview-accent-soft);
            color: var(--preview-text);
            font-size: 0.72rem;
            font-weight: 800;
        }
        .graph-preview-connector {
            align-self: center;
            flex: 0 0 clamp(34px, 3.5vw, 58px);
            height: 2px;
            margin: 0 -1px;
            background:
                linear-gradient(
                    90deg,
                    color-mix(in srgb, var(--preview-accent) 32%, var(--preview-border)),
                    var(--preview-accent)
                );
            font-size: 0;
            position: relative;
            z-index: 2;
        }
        .graph-preview-connector::before {
            content: "";
            position: absolute;
            left: -3px;
            top: -3px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--preview-accent);
        }
        .graph-preview-connector::after {
            content: "";
            position: absolute;
            right: -5px;
            top: -5px;
            border-left: 11px solid var(--preview-accent);
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            filter: drop-shadow(0 0 1px var(--preview-bg));
        }
        .graph-preview-row-bridge {
            height: 58px;
            margin: -0.15rem 0;
            position: relative;
        }
        .graph-preview-bridge-exit,
        .graph-preview-bridge-run,
        .graph-preview-bridge-entry {
            position: absolute;
            background: var(--preview-accent);
        }
        .graph-preview-bridge-exit {
            height: 24px;
            right: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 0;
            width: 2px;
        }
        .graph-preview-bridge-exit::before {
            content: "";
            position: absolute;
            left: -3px;
            top: -3px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--preview-accent);
        }
        .graph-preview-bridge-run {
            height: 2px;
            left: clamp(0.4rem, 1.5vw, 1.1rem);
            right: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 24px;
        }
        .graph-preview-bridge-entry {
            height: 28px;
            left: clamp(0.4rem, 1.5vw, 1.1rem);
            top: 24px;
            width: 2px;
        }
.graph-preview-bridge-entry::after {
    content: "";
    position: absolute;
    bottom: -9px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 10px solid var(--preview-accent);
    filter: drop-shadow(0 0 1px var(--preview-bg));
}
        .graph-preview-kicker {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
            opacity: 0.72;
            text-transform: uppercase;
            color: var(--preview-muted);
        }
        .graph-preview-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.45rem;
            white-space: normal;
            color: var(--preview-text);
        }
        .graph-preview-meta {
            display: grid;
            gap: 0.28rem;
            font-size: 0.78rem;
            color: var(--preview-muted);
        }
        .graph-preview-meta span {
            display: block;
            overflow-wrap: anywhere;
        }
        .graph-preview-personality {
            margin-top: 0.45rem;
            padding-top: 0.45rem;
            border-top: 1px solid var(--preview-border);
            font-size: 0.78rem;
            overflow-wrap: anywhere;
            color: var(--preview-text);
        }
        @media (max-width: 760px) {
            .graph-preview-row {
                display: grid;
                gap: 0.65rem;
            }
            .graph-preview-connector {
                width: 2px;
                height: 34px;
                justify-self: center;
                background:
                    linear-gradient(
                        180deg,
                        var(--preview-border),
                        var(--preview-accent)
                    );
            }
            .graph-preview-connector::before {
                left: -3px;
                top: -3px;
            }
    .graph-preview-connector::after {
        right: auto;
        top: auto;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 9px solid var(--preview-accent);
        border-bottom: 0;
    }
            .graph-preview-row-bridge {
                height: 42px;
                margin: -0.1rem 0;
            }
            .graph-preview-bridge-exit,
            .graph-preview-bridge-run {
                display: none;
            }
            .graph-preview-bridge-entry {
                height: 34px;
                left: 50%;
                top: 0;
            }
        }
        </style>
        <div class="graph-preview-shell">
            <div class="graph-preview-chain">
        """,
    ]
    total_nodes = len(node_forms)
    nodes_per_row = 4
    for row_start in range(0, total_nodes, nodes_per_row):
        row_node_forms = node_forms[row_start : row_start + nodes_per_row]
        preview_parts.append('<div class="graph-preview-row">')
        for row_position, node_form in enumerate(row_node_forms):
            index = row_start + row_position + 1
            is_preset_personality = node_form.get("personality_mode") == "preset"
            personality_label = node_form.get("personality_preset", "Custom")
            if not is_preset_personality:
                personality_label = node_form.get("personality_custom", "").strip()
                personality_label = personality_label[:140] or "Custom personality"
            node_label = node_form.get("label", "").strip() or "Untitled node"
            node_id = node_form.get("node_id", "").strip() or "missing"
            provider = node_form.get("provider", "").strip() or "missing"
            model = node_form.get("model", "").strip() or "missing"

            if row_position > 0:
                preview_parts.append('<div class="graph-preview-connector"></div>')
            preview_parts.append(
                f"""
                    <div class="graph-preview-unit">
                        <div class="graph-preview-node">
                            <div class="graph-preview-node-number">{index}</div>
                            <div class="graph-preview-kicker">Node {index} of {total_nodes}</div>
                            <div class="graph-preview-title">{escape(node_label)}</div>
                            <div class="graph-preview-meta">
                                <span><strong>ID:</strong> {escape(node_id)}</span>
                                <span><strong>Provider:</strong> {escape(provider)}</span>
                                <span><strong>Model:</strong> {escape(model)}</span>
                            </div>
                            <div class="graph-preview-personality">
                                <strong>Characteristic:</strong> {escape(personality_label)}
                            </div>
                        </div>
                    </div>
                """
            )
        preview_parts.append("</div>")
        if row_start + nodes_per_row < total_nodes:
            preview_parts.append(
                """
                <div class="graph-preview-row-bridge" aria-hidden="true">
                    <div class="graph-preview-bridge-exit"></div>
                    <div class="graph-preview-bridge-run"></div>
                    <div class="graph-preview-bridge-entry"></div>
                </div>
                """
            )

    preview_parts.append(
        """
            </div>
        </div>
        """
    )
    st.html("".join(preview_parts))


def render_node_editor(index: int, node_form: dict[str, str]) -> None:
    uid = node_form["uid"]
    with st.container(border=True):
        action_cols = st.columns([5, 1, 1, 1])
        action_cols[0].markdown(f"#### Node {index + 1}")
        if action_cols[1].button(
            "↑", key=f"up_{uid}", disabled=index == 0, use_container_width=True
        ):
            move_node(index, -1)
            st.rerun()
        if action_cols[2].button(
            "↓",
            key=f"down_{uid}",
            disabled=index == len(st.session_state.graph_nodes) - 1,
            use_container_width=True,
        ):
            move_node(index, 1)
            st.rerun()
        if action_cols[3].button(
            "Remove",
            key=f"remove_{uid}",
            disabled=len(st.session_state.graph_nodes) == 1,
            use_container_width=True,
        ):
            remove_node(index)
            st.rerun()

        identity_cols = st.columns(3)
        node_form["label"] = identity_cols[0].text_input(
            "Label",
            value=node_form.get("label", ""),
            key=f"label_{uid}",
        )
        node_form["node_id"] = identity_cols[1].text_input(
            "Node ID",
            value=node_form.get("node_id", ""),
            key=f"node_id_{uid}",
            help="Unique identifier used internally by the simulation.",
        )
        provider_index = (
            AVAILABLE_PROVIDERS.index(node_form.get("provider", AVAILABLE_PROVIDERS[0]))
            if node_form.get("provider") in AVAILABLE_PROVIDERS
            else 0
        )
        node_form["provider"] = identity_cols[2].selectbox(
            "Provider",
            AVAILABLE_PROVIDERS,
            index=provider_index,
            key=f"provider_{uid}",
        )

        model_default = node_form.get("model", "")
        model_options = AVAILABLE_MODELS + [CUSTOM_OPTION]
        selected_model_option = (
            model_default if model_default in AVAILABLE_MODELS else CUSTOM_OPTION
        )
        model_cols = st.columns([1, 1])
        selected_model_option = model_cols[0].selectbox(
            "Model preset",
            model_options,
            index=model_options.index(selected_model_option),
            key=f"model_option_{uid}",
            format_func=lambda value: "Custom model" if value == CUSTOM_OPTION else value,
        )
        if selected_model_option == CUSTOM_OPTION:
            node_form["model"] = model_cols[1].text_input(
                "Custom model",
                value=model_default,
                key=f"model_custom_{uid}",
            )
        else:
            node_form["model"] = selected_model_option
            model_cols[1].text_input(
                "Resolved model",
                value=node_form["model"],
                key=f"model_resolved_{uid}",
                disabled=True,
            )

        personality_mode = st.radio(
            "Personality source",
            ["Predefined", "Custom"],
            index=0 if node_form.get("personality_mode") == "preset" else 1,
            horizontal=True,
            key=f"personality_mode_{uid}",
        )
        node_form["personality_mode"] = "preset" if personality_mode == "Predefined" else "custom"

        if node_form["personality_mode"] == "preset":
            preset_names = list(PREDEFINED_PERSONALITIES)
            current_preset = node_form.get("personality_preset", preset_names[0])
            if current_preset not in preset_names:
                current_preset = preset_names[0]
            node_form["personality_preset"] = st.selectbox(
                "Preset personality",
                preset_names,
                index=preset_names.index(current_preset),
                key=f"personality_preset_{uid}",
            )
            st.caption(PREDEFINED_PERSONALITIES[node_form["personality_preset"]])
        else:
            node_form["personality_custom"] = st.text_area(
                "Custom personality prompt",
                value=node_form.get("personality_custom", ""),
                key=f"personality_custom_{uid}",
                height=130,
            )


def render_result_bundle(run_bundle: dict[str, Any]) -> None:
    summary = run_bundle["summary"]
    steps_df = run_bundle["steps_df"]
    node_summary_df = run_bundle["node_summary_df"]
    news_summary_df = run_bundle["news_summary_df"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows processed", summary["rows_processed"])
    metric_cols[1].metric("Total steps", summary["steps_total"])
    metric_cols[2].metric("Successful steps", summary["steps_success"])
    metric_cols[3].metric("Errors", summary["steps_error"])

    path_cols = st.columns(2)
    path_cols[0].caption(f"Summary path: `{run_bundle['summary_path'] or 'not persisted'}`")
    path_cols[1].caption(f"Steps path: `{run_bundle['steps_path'] or 'not persisted'}`")

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download summary JSON",
        data=json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{run_bundle['output_prefix']}_summary.json",
        mime="application/json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download steps CSV",
        data=steps_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_bundle['output_prefix']}_steps.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "Download graph JSON",
        data=json.dumps(run_bundle["graph_payload"], ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="graph_config_ui.json",
        mime="application/json",
        use_container_width=True,
    )

    st.subheader("Node performance")
    if node_summary_df.empty:
        st.info("No node summary available yet.")
    else:
        st.dataframe(
            node_summary_df[
                [
                    "step_index",
                    "node_label",
                    "provider",
                    "model",
                    "runs",
                    "successes",
                    "errors",
                    "success_rate",
                    "mean_stdi_vs_original",
                    "mean_stdi_incremental",
                ]
            ],
            use_container_width=True,
        )
        chart_df = node_summary_df.set_index("node_label")[
            ["mean_stdi_vs_original", "mean_stdi_incremental"]
        ]
        st.bar_chart(chart_df)

    st.subheader("News overview")
    if news_summary_df.empty:
        st.info("No per-news summary available yet.")
    else:
        st.dataframe(news_summary_df, use_container_width=True)
        selected_news_id = st.selectbox(
            "Inspect one news item",
            news_summary_df["news_id"].tolist(),
            key="selected_news_id",
        )
        selected_steps = steps_df[steps_df["news_id"] == selected_news_id].copy()
        if not selected_steps.empty:
            st.line_chart(
                selected_steps.set_index("node_label")[["stdi_vs_original", "stdi_incremental"]]
            )
            st.dataframe(
                selected_steps[
                    [
                        "step_index",
                        "node_label",
                        "rewrite_status",
                        "stdi_vs_original",
                        "stdi_incremental",
                        "rewrite_error",
                    ]
                ],
                use_container_width=True,
            )
            for _, row in selected_steps.iterrows():
                with st.expander(
                    f"Step {int(row['step_index'])}: {row['node_label']} ({row['rewrite_status']})"
                ):
                    info_cols = st.columns(4)
                    info_cols[0].metric("STDI vs original", format_metric(row["stdi_vs_original"]))
                    info_cols[1].metric("Incremental STDI", format_metric(row["stdi_incremental"]))
                    info_cols[2].metric("Provider", str(row["provider"]))
                    info_cols[3].metric("Model", str(row["model"]))
                    if pd.notna(row.get("rewrite_error")) and str(row["rewrite_error"]).strip():
                        st.error(str(row["rewrite_error"]))
                    text_cols = st.columns(2)
                    text_cols[0].text_area(
                        "Input text for this step",
                        value=str(row.get("source_text") or ""),
                        height=220,
                        disabled=True,
                    )
                    text_cols[1].text_area(
                        "Rewritten text",
                        value=str(row.get("rewritten_text") or ""),
                        height=220,
                        disabled=True,
                    )

    st.subheader("All step records")
    st.dataframe(steps_df, use_container_width=True)


def format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    st.set_page_config(
        page_title="Interaction Graph Studio",
        page_icon="",
        layout="wide",
    )
    initialize_state()

    st.title("Interaction Graph Studio")
    st.caption(
        "Configure a chain of personas, run the graph simulation over a news dataset, "
        "and inspect topic drift and rewrite quality without leaving the browser."
    )
    st.info(
        "The current backend supports a single connected chain of nodes. The UI reflects that "
        "constraint while still letting you add, reorder, and compare as many nodes as you need."
    )

    with st.sidebar:
        st.header("Data")
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
            dataset_path = st.session_state.dataset_path
            dataset_label = dataset_path
            try:
                df = load_local_dataframe_cached(dataset_path)
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
            st.caption(
                f"Columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}"
            )

        st.divider()
        st.header("Import graph")
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
            graph_path = st.session_state.graph_config_path
            if st.button("Load graph config", use_container_width=True):
                try:
                    import_graph_payload(load_local_graph_payload_cached(graph_path))
                    st.success("Graph config loaded into the editor.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            uploaded_graph = st.file_uploader(
                "Upload graph JSON", type=["json"], key="graph_upload"
            )
            if st.button("Import uploaded graph", use_container_width=True):
                try:
                    import_graph_payload(load_uploaded_graph_payload(uploaded_graph))
                    st.success("Uploaded graph config loaded into the editor.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        sidebar_action_cols = st.columns(2)
        if sidebar_action_cols[0].button("Add node", use_container_width=True):
            st.session_state.graph_nodes.append(
                create_default_node_form(len(st.session_state.graph_nodes) + 1)
            )
            st.rerun()
        if sidebar_action_cols[1].button("Reset graph", use_container_width=True):
            reset_graph()
            st.rerun()

    config_tab, results_tab = st.tabs(["Configuration", "Results"])

    with config_tab:
        dataset_info_col, execution_col = st.columns([1.2, 1])

        with dataset_info_col:
            st.subheader("Dataset preview")
            if df is None:
                st.info("Load a dataset from the sidebar to configure execution options.")
                available_columns: list[str] = []
            else:
                st.caption(f"Source: `{dataset_label}`")
                preview_cols = [column for column in df.columns[:8]]
                st.dataframe(df[preview_cols].head(8), use_container_width=True)
                available_columns = df.columns.tolist()

        with execution_col:
            st.subheader("Execution settings")
            default_text_column = (
                "description"
                if "description" in available_columns
                else (available_columns[0] if available_columns else "")
            )
            default_title_column = (
                "title"
                if "title" in available_columns
                else (available_columns[0] if available_columns else "")
            )
            default_news_id_column = "article_id" if "article_id" in available_columns else ""

            text_column = st.selectbox(
                "Text column",
                available_columns or [""],
                index=(
                    available_columns.index(default_text_column)
                    if default_text_column in available_columns
                    else 0
                ),
                help="Column containing the source text that will be rewritten by the graph nodes.",
            )
            title_column = st.selectbox(
                "Title column",
                available_columns or [""],
                index=(
                    available_columns.index(default_title_column)
                    if default_title_column in available_columns
                    else 0
                ),
                help="Column used as the article title in prompts and topic drift extraction.",
            )
            news_id_options = [""] + available_columns if available_columns else [""]
            news_id_column = st.selectbox(
                "News ID column",
                news_id_options,
                index=(
                    news_id_options.index(default_news_id_column)
                    if default_news_id_column in news_id_options
                    else 0
                ),
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
                    "When enabled, the simulation can use the title if the selected text "
                    "column is empty."
                ),
            )

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
                    index=AVAILABLE_PROVIDERS.index("gemini")
                    if "gemini" in AVAILABLE_PROVIDERS
                    else 0,
                    help="Provider used to extract topic structures for STDI/topic drift metrics.",
                )
                topic_drift_model_default = "gemini-3.1-flash-lite-preview"
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
                    help=(
                        "Model used for topic structure extraction before calculating topic drift."
                    ),
                )
                if selected_topic_drift_model_option == CUSTOM_OPTION:
                    topic_drift_model = topic_drift_model_cols[1].text_input(
                        "Custom topic drift model",
                        value=topic_drift_model_default,
                        key="topic_drift_model_custom",
                        help="Provider-specific model id for topic drift extraction.",
                    )
                else:
                    topic_drift_model = selected_topic_drift_model_option
                    topic_drift_model_cols[1].text_input(
                        "Resolved topic drift model",
                        value=topic_drift_model,
                        key="topic_drift_model_resolved",
                        disabled=True,
                        help="Model id that will be passed to the topic drift extraction step.",
                    )
                output_dir = st.text_input(
                    "Output directory",
                    value="output/interaction_graph/app_runs",
                    help="Directory where the summary JSON and step records JSONL will be saved.",
                )
                output_prefix = st.text_input(
                    "Output prefix",
                    value=advanced_label,
                    help=(
                        "Prefix used in generated result filenames, such as "
                        "<prefix>_summary.json and <prefix>_steps.jsonl."
                    ),
                )

        st.subheader("Graph editor")
        for index, node_form in enumerate(st.session_state.graph_nodes):
            render_node_editor(index, node_form)

        st.subheader("Graph preview")
        render_graph_preview(st.session_state.graph_nodes)

        graph_payload = build_linear_graph_payload(st.session_state.graph_nodes)
        export_cols = st.columns([1, 1, 2])
        export_cols[0].download_button(
            "Download graph JSON",
            data=json.dumps(graph_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="graph_config_ui.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[1].caption(f"Start node: `{graph_payload.get('start_node_id', '-')}`")

        run_placeholder = st.empty()
        log_placeholder = st.empty()
        run_button = st.button("Run simulation", type="primary", use_container_width=True)

        if run_button:
            validation_errors = validate_node_forms(st.session_state.graph_nodes)
            if df is None:
                validation_errors.append("Load a valid dataset before running the simulation.")
            if not text_column:
                validation_errors.append("Select a text column before running the simulation.")
            if not title_column:
                validation_errors.append("Select a title column before running the simulation.")
            if not topic_drift_model.strip():
                validation_errors.append("Select or enter a topic drift model before running.")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                progress_messages: list[str] = []

                def progress_callback(message: str) -> None:
                    progress_messages.append(message)
                    run_placeholder.info(message)
                    log_placeholder.code("\n".join(progress_messages[-20:]), language="text")

                try:
                    nodes = build_simulation_nodes(st.session_state.graph_nodes)
                    result = run_news_interaction_graph(
                        df=df,
                        nodes=nodes,
                        start_node_id=nodes[0].node_id,
                        text_column=text_column,
                        title_column=title_column,
                        news_id_column=news_id_column or None,
                        max_rows=int(max_rows),
                        sleep_seconds=float(sleep_seconds),
                        max_requests_per_minute=(
                            None
                            if int(max_requests_per_minute) == 0
                            else int(max_requests_per_minute)
                        ),
                        retry_attempts=int(retry_attempts),
                        allow_title_fallback=allow_title_fallback,
                        topic_drift_model=topic_drift_model,
                        topic_drift_provider=topic_drift_provider,
                        output_dir=output_dir,
                        output_prefix=output_prefix,
                        progress_callback=progress_callback,
                    )
                    steps_df = steps_to_dataframe(result.step_results)
                    st.session_state.run_bundle = {
                        "summary": result.summary,
                        "summary_path": str(result.summary_path)
                        if result.summary_path is not None
                        else None,
                        "steps_path": str(result.steps_path)
                        if result.steps_path is not None
                        else None,
                        "steps_df": steps_df,
                        "node_summary_df": build_node_summary_dataframe(steps_df),
                        "news_summary_df": build_news_summary_dataframe(steps_df),
                        "output_prefix": output_prefix,
                        "graph_payload": graph_payload,
                    }
                    run_placeholder.success(
                        "Simulation finished. Open the Results tab to inspect the outputs."
                    )
                except Exception as exc:
                    run_placeholder.error(f"Simulation failed: {exc}")

    with results_tab:
        st.subheader("Simulation outputs")
        if st.session_state.run_bundle is None:
            st.info("Run the simulation from the Configuration tab to populate this dashboard.")
        else:
            render_result_bundle(st.session_state.run_bundle)


if __name__ == "__main__":
    main()
