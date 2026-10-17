from __future__ import annotations

from typing import Any

import streamlit as st

from misinformation_simulation.apps.interaction_graph_ui import CUSTOM_OPTION
from misinformation_simulation.apps.interaction_graph_workspace import widget_default
from misinformation_simulation.enums.models import (
    default_model_for_provider,
    models_for_provider,
    validate_provider_model,
)


def render_model_selector(
    provider: str,
    *,
    label: str,
    option_key: str,
    custom_key: str,
    default: str,
    custom_label: str = "Custom model",
    disabled: bool = False,
    help: str | None = None,
    selector_container: Any = None,
    custom_container: Any = None,
) -> str:
    """Keep model choices and restored widget state consistent with the provider."""
    models = models_for_provider(provider)
    options = models + [CUSTOM_OPTION]
    provider_key = f"_model_provider_{option_key}"
    previous_provider = st.session_state.get(provider_key)
    provider_changed = previous_provider is not None and previous_provider != provider
    candidate = st.session_state.get(custom_key, default)
    if provider_changed:
        candidate = default_model_for_provider(provider)
    try:
        validate_provider_model(provider, candidate)
    except ValueError:
        candidate = default_model_for_provider(provider)
    initial = candidate if candidate in models else CUSTOM_OPTION
    current = st.session_state.get(option_key, initial)
    if provider_changed or current not in options:
        st.session_state[option_key] = initial
        st.session_state[custom_key] = candidate
    st.session_state[provider_key] = provider
    chosen = (selector_container or st).selectbox(
        label,
        options,
        index=widget_default(option_key, options.index(initial), 0),
        key=option_key,
        format_func=lambda value: "Custom model" if value == CUSTOM_OPTION else value,
        disabled=disabled,
        help=help,
    )
    if chosen != CUSTOM_OPTION:
        st.session_state[custom_key] = chosen
        return chosen
    model = (
        (custom_container or st)
        .text_input(
            custom_label,
            value=widget_default(custom_key, candidate, ""),
            key=custom_key,
            disabled=disabled,
            help="Provider-specific model ID outside the preset catalog.",
        )
        .strip()
    )
    if not disabled:
        try:
            validate_provider_model(provider, model)
        except ValueError as exc:
            st.error(str(exc))
    return model
