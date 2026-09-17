from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from misinformation_simulation.apps.interaction_graph_state import move_node, remove_node
from misinformation_simulation.apps.interaction_graph_ui import (
    AVAILABLE_MODELS,
    AVAILABLE_PROVIDERS,
    CUSTOM_OPTION,
    PREDEFINED_PERSONALITIES,
)


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
        key="download_results_summary_json",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "Download steps CSV",
        data=steps_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_bundle['output_prefix']}_steps.csv",
        mime="text/csv",
        key="download_results_steps_csv",
        use_container_width=True,
    )
    if run_bundle.get("graph_payload") is not None:
        download_cols[2].download_button(
            "Download graph JSON",
            data=json.dumps(run_bundle["graph_payload"], ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            file_name="graph_config_ui.json",
            mime="application/json",
            key="download_results_graph_json",
            use_container_width=True,
        )

    if steps_df.empty:
        st.info("This simulation has no completed step records.")
        return

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
                    "mean_stdi_cumulative",
                    "mean_vad_drift_vs_original",
                    "mean_contradiction_drift_vs_original",
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
        selected_steps = (
            steps_df[steps_df["news_id"] == selected_news_id].copy().sort_values("step_index")
        )
        if not selected_steps.empty:
            st.line_chart(
                selected_steps.set_index("step_index")[["stdi_vs_original", "stdi_incremental"]]
            )
            st.dataframe(
                selected_steps[
                    [
                        "step_index",
                        "node_label",
                        "rewrite_status",
                        "stdi_vs_original",
                        "stdi_incremental",
                        "stdi_cumulative",
                        "vad_drift_vs_original",
                        "vad_drift_incremental",
                        "contradiction_drift_vs_original",
                        "contradiction_drift_incremental",
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
                    st.metric("Cumulative STDI", format_metric(row["stdi_cumulative"]))
                    component_cols = st.columns(4)
                    component_cols[0].metric(
                        "VAD vs original", format_metric(row["vad_drift_vs_original"])
                    )
                    component_cols[1].metric(
                        "Incremental VAD", format_metric(row["vad_drift_incremental"])
                    )
                    component_cols[2].metric(
                        "Contradiction vs original",
                        format_metric(row["contradiction_drift_vs_original"]),
                    )
                    component_cols[3].metric(
                        "Incremental contradiction",
                        format_metric(row["contradiction_drift_incremental"]),
                    )
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
                    render_topic_comparison(row)

    st.subheader("All step records")
    st.dataframe(steps_df, use_container_width=True)


def render_topic_comparison(row: pd.Series) -> None:
    st.markdown("#### Extracted topics and category scores")
    st.caption(
        "Scores measure category drift: 0 means unchanged and 1 means maximum change. "
        "The original score compares with the source article; the input score compares "
        "with the text entering this node. Individual items do not have separate scores."
    )
    original = topic_structure_from_step(row, "original")
    rewritten = topic_structure_from_step(row, "rewritten")
    if original is None and rewritten is None:
        st.info("Topic extraction is unavailable for this step.")
        return

    categories = (
        ("Main topic", "main_topic", "theme_drift"),
        ("Subtopics", "subtopics", "subtopic_drift"),
        ("Entities", "central_entities", "entity_drift"),
        ("Relations", "central_relations", "relation_drift"),
    )
    for label, field, metric in categories:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            columns = st.columns([3, 3, 1, 1])
            columns[0].caption("Original article")
            columns[0].text(format_topic_items(original, field))
            columns[1].caption("Rewritten text")
            columns[1].text(format_topic_items(rewritten, field))
            columns[2].metric("vs original", format_metric(row.get(f"{metric}_vs_original")))
            columns[3].metric("vs input", format_metric(row.get(f"{metric}_incremental")))

    if original is None:
        st.caption(f"Original topic extraction: {row.get('original_topic_structure_status', '-')}")
    if rewritten is None:
        st.caption(
            f"Rewritten topic extraction: {row.get('rewritten_topic_structure_status', '-')}"
        )


def topic_structure_from_step(row: pd.Series, prefix: str) -> dict[str, Any] | None:
    value = row.get(f"metadata_{prefix}_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass

    fields = ("main_topic", "subtopics", "central_entities", "central_relations")
    structure = {}
    for field in fields:
        value = row.get(f"metadata_{prefix}_{field}")
        if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, str) and field != "main_topic":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        structure[field] = value
    return structure or None


def format_topic_items(structure: dict[str, Any] | None, field: str) -> str:
    if structure is None:
        return "Unavailable"
    value = structure.get(field)
    if not value:
        return "—"
    if field == "main_topic":
        return str(value)
    if not isinstance(value, list):
        return str(value)
    if field == "central_relations":
        return "\n".join(
            f"• {item.get('subject', '')} — {item.get('action', '')} — {item.get('object', '')}"
            if isinstance(item, dict)
            else f"• {item}"
            for item in value
        )
    return "\n".join(f"• {item}" for item in value)


def format_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
