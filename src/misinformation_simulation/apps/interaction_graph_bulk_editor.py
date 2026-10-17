from __future__ import annotations

import streamlit as st

from misinformation_simulation.apps.interaction_graph_ui import (
    AVAILABLE_MODELS,
    AVAILABLE_PROVIDERS,
    CUSTOM_OPTION,
    PREDEFINED_PERSONALITIES,
    apply_bulk_node_updates,
)
from misinformation_simulation.apps.interaction_graph_workspace import widget_default


def render_bulk_node_editor() -> None:
    forms = st.session_state.graph_nodes
    options = {form["uid"]: form for form in forms}
    selection_key = "graph_bulk_selected_uids"
    selected = st.session_state.get(selection_key, [])
    current_selection = [uid for uid in selected if uid in options]
    if selected != current_selection:
        st.session_state[selection_key] = current_selection

    with st.expander("Bulk edit nodes", expanded=len(current_selection) > 1):
        message = st.session_state.pop("_graph_bulk_success", None)
        if message:
            st.success(message)
        actions = st.columns(2)
        actions[0].button(
            "Select all",
            key="graph_bulk_select_all",
            width="stretch",
            on_click=_set_bulk_selection,
            args=(list(options),),
        )
        actions[1].button(
            "Clear selection",
            key="graph_bulk_clear",
            width="stretch",
            on_click=_set_bulk_selection,
            args=([],),
        )
        selected = st.multiselect(
            "Nodes to edit",
            list(options),
            key=selection_key,
            format_func=lambda uid: f"{options[uid]['label']} ({options[uid]['node_id']})",
            help=(
                "Hold Shift and drag on empty canvas to select several nodes, choose them "
                "here, or use Select all to edit the entire open graph."
            ),
        )
        st.caption(
            f"{len(selected)} of {len(forms)} nodes selected. "
            "Only checked fields will change when you apply."
        )
        if not forms:
            return
        sample = options[selected[0]] if selected else forms[0]
        changes = _render_bulk_fields(sample)
        if st.button(
            f"Apply to {len(selected)} selected nodes",
            key="graph_bulk_apply",
            disabled=not selected or not changes,
            type="primary",
        ):
            try:
                updated = apply_bulk_node_updates(forms, selected, **changes)
            except ValueError as exc:
                st.error(str(exc))
            else:
                _clear_bulk_node_widgets(selected, changes)
                st.session_state.graph_nodes = updated
                st.session_state._graph_bulk_success = f"Updated {len(selected)} nodes."
                st.rerun()


def _set_bulk_selection(selected: list[str]) -> None:
    st.session_state.graph_bulk_selected_uids = selected


def _render_bulk_fields(sample: dict[str, str]) -> dict[str, str]:
    columns = st.columns(3)
    change_provider = columns[0].checkbox("Change provider", key="graph_bulk_change_provider")
    change_model = columns[1].checkbox("Change model", key="graph_bulk_change_model")
    change_personality = columns[2].checkbox(
        "Change personality", key="graph_bulk_change_personality"
    )
    changes = {}
    if change_provider:
        provider = sample.get("provider", AVAILABLE_PROVIDERS[0])
        changes["provider"] = st.selectbox(
            "Provider for selected nodes",
            AVAILABLE_PROVIDERS,
            index=widget_default(
                "graph_bulk_provider",
                AVAILABLE_PROVIDERS.index(provider) if provider in AVAILABLE_PROVIDERS else 0,
                0,
            ),
            key="graph_bulk_provider",
        )
    if change_model:
        model = sample.get("model", "")
        model_options = AVAILABLE_MODELS + [CUSTOM_OPTION]
        preset = model if model in AVAILABLE_MODELS else CUSTOM_OPTION
        chosen = st.selectbox(
            "Model for selected nodes",
            model_options,
            index=widget_default("graph_bulk_model_option", model_options.index(preset), 0),
            key="graph_bulk_model_option",
            format_func=lambda value: "Custom model" if value == CUSTOM_OPTION else value,
        )
        changes["model"] = (
            st.text_input(
                "Custom model for selected nodes",
                value=widget_default("graph_bulk_model_custom", model, ""),
                key="graph_bulk_model_custom",
            )
            if chosen == CUSTOM_OPTION
            else chosen
        )
    if change_personality:
        mode = st.radio(
            "Personality source for selected nodes",
            ["Predefined", "Custom"],
            horizontal=True,
            key="graph_bulk_personality_mode",
        )
        if mode == "Predefined":
            changes["personality_preset"] = st.selectbox(
                "Personality for selected nodes",
                list(PREDEFINED_PERSONALITIES),
                key="graph_bulk_personality_preset",
            )
        else:
            changes["personality_custom"] = st.text_area(
                "Custom personality prompt for selected nodes",
                key="graph_bulk_personality_custom",
                height=130,
            )
    return changes


def _clear_bulk_node_widgets(selected: list[str], changes: dict[str, str]) -> None:
    prefixes = []
    if "provider" in changes:
        prefixes.append("provider")
    if "model" in changes:
        prefixes.extend(["model_option", "model_custom", "model_resolved"])
    if "personality_preset" in changes or "personality_custom" in changes:
        prefixes.extend(["personality_mode", "personality_preset", "personality_custom"])
    for uid in selected:
        for prefix in prefixes:
            st.session_state.pop(f"{prefix}_{uid}", None)
