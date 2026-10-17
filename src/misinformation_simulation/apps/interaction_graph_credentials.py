"""Request only the credentials needed by the selected graph execution."""

from __future__ import annotations

from typing import Any

import streamlit as st

from misinformation_simulation.enums import Provider
from misinformation_simulation.llm.credentials import (
    API_KEY_NAMES,
    OPENAI_FALLBACK_NAMES,
    ProviderAPIKeys,
    environment_api_key,
    environment_openai_fallback,
)
from misinformation_simulation.llm.recovery import register_secret

PROVIDER_LABELS = {
    "chatgpt": "OpenAI / GPT",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "deepseek": "DeepSeek",
    "grok": "Grok / xAI",
}
_SESSION_KEYS = "_session_api_keys"
_SESSION_OVERRIDES = "_session_api_key_overrides"
_WIDGET_PREFIX = "session_api_key_"


def required_provider_uses(graphs: list[dict], settings: dict[str, Any]) -> dict[str, list[str]]:
    uses: dict[str, list[str]] = {}

    def add(provider, description):
        name = str(provider or "").strip().lower()
        if name in PROVIDER_LABELS and description not in uses.setdefault(name, []):
            uses[name].append(description)

    for graph in graphs:
        for index, node in enumerate(graph.get("nodes", []), 1):
            label = node.get("label") or node.get("node_id") or f"Node {index}"
            add(
                node.get("provider"),
                f"{graph.get('name', 'Graph')} / {label}: {node.get('model', '')}",
            )

    extraction_provider = settings.get("topic_drift_provider")
    extraction_model = settings.get("topic_drift_model", "")
    add(extraction_provider, f"Topic extraction: {extraction_model}")
    if settings.get("stdi_comparison_method", "dual") in {"dual", "llm"}:
        add(extraction_provider, f"STDI LLM judge: {extraction_model}")
    if settings.get("vad_method", "model") != "model":
        add(
            settings.get("vad_llm_provider") or extraction_provider,
            f"VAD evaluator: {settings.get('vad_llm_model') or extraction_model}",
        )
    return {provider.value: uses[provider.value] for provider in Provider if provider.value in uses}


def clear_session_api_keys() -> None:
    st.session_state.pop(_SESSION_KEYS, None)
    st.session_state.pop(_SESSION_OVERRIDES, None)
    for key in list(st.session_state):
        if key.startswith(_WIDGET_PREFIX):
            del st.session_state[key]


def session_api_credentials(graphs, settings):
    saved = st.session_state.get(_SESSION_KEYS, {})
    overrides = st.session_state.get(_SESSION_OVERRIDES, {})
    credentials, missing = {}, []
    for provider in required_provider_uses(graphs, settings):
        slots = saved.get(provider, {})
        primary = (
            slots.get("primary")
            if (overrides.get(f"{provider}_primary") or not environment_api_key(provider))
            else None
        )
        fallback = (
            slots.get("fallback")
            if (overrides.get(f"{provider}_fallback") or not environment_openai_fallback())
            else None
        )
        if primary or fallback:
            credentials[provider] = ProviderAPIKeys(primary or None, fallback or None)
        if not primary and (
            overrides.get(f"{provider}_primary") or not environment_api_key(provider)
        ):
            missing.append(PROVIDER_LABELS[provider])
        if provider == "chatgpt" and overrides.get("chatgpt_fallback") and not fallback:
            missing.append(f"{PROVIDER_LABELS[provider]} fallback")
    return credentials, missing


def _override_environment_key(provider: str, slot: str, *, disabled: bool) -> bool:
    saved = st.session_state.setdefault(_SESSION_OVERRIDES, {})
    choice_key = f"{provider}_{slot}"
    widget_key = f"{_WIDGET_PREFIX}override_{choice_key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved.get(choice_key, False)
    label = "fallback key" if slot == "fallback" else "primary key"
    selected = st.checkbox(
        f"Use a different {PROVIDER_LABELS[provider]} {label} for this session",
        key=widget_key,
        disabled=disabled,
        help="Uncheck to use the key configured in the environment again.",
    )
    saved[choice_key] = selected
    return selected


def _key_input(provider: str, slot: str, *, disabled: bool, replacement: bool = False) -> str:
    saved = st.session_state.setdefault(_SESSION_KEYS, {})
    widget_key = f"{_WIDGET_PREFIX}{provider}_{slot}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved.get(provider, {}).get(slot, "")
    names = OPENAI_FALLBACK_NAMES if slot == "fallback" else API_KEY_NAMES[provider]
    suffix = "fallback API key" if slot == "fallback" else "API key"
    if slot == "fallback" and not replacement:
        suffix += " (optional)"
    label = f"{PROVIDER_LABELS[provider]} {suffix}"
    value = st.text_input(
        label,
        type="password",
        key=widget_key,
        disabled=disabled,
        help=f"Session value for {' / '.join(names)}. The environment is not modified.",
    ).strip()
    saved.setdefault(provider, {})[slot] = value
    register_secret(value)
    return value


def render_api_keys(
    graphs: list[dict],
    settings: dict[str, Any],
    *,
    disabled: bool = False,
) -> tuple[dict[str, ProviderAPIKeys], list[str]]:
    required = required_provider_uses(graphs, settings)
    st.caption(
        "Keys entered here apply to this session. They are not saved to .env, graph "
        "exports, or execution results. One key is shared by models of the same provider."
    )
    if any(st.session_state.get(_SESSION_OVERRIDES, {}).values()) or any(
        value
        for slots in st.session_state.get(_SESSION_KEYS, {}).values()
        for value in slots.values()
    ):
        st.button("Clear session API keys", disabled=disabled, on_click=clear_session_api_keys)
    if not required:
        st.caption("The selected execution uses local providers; no API keys are required.")
    credentials = {}
    missing = []
    for provider, roles in required.items():
        st.markdown(f"**{PROVIDER_LABELS[provider]}**")
        st.caption("Used by: " + "; ".join(roles))
        primary = None
        fallback = None
        primary_configured = bool(environment_api_key(provider))
        override_primary = False
        if primary_configured:
            names = " / ".join(API_KEY_NAMES[provider])
            st.caption(f"Primary key configured in the environment ({names}).")
            override_primary = _override_environment_key(provider, "primary", disabled=disabled)
        if not primary_configured or override_primary:
            primary = _key_input(provider, "primary", disabled=disabled) or None
            if not primary:
                missing.append(PROVIDER_LABELS[provider])
        if provider == "chatgpt":
            fallback = _render_fallback_key(provider, disabled, missing)

        if primary or fallback:
            credentials[provider] = ProviderAPIKeys(primary, fallback)
    return credentials, missing


def _render_fallback_key(provider, disabled, missing):
    fallback = None
    fallback_configured = bool(environment_openai_fallback())
    override_fallback = False
    if fallback_configured:
        st.caption("OpenAI fallback key configured in the environment.")
        override_fallback = _override_environment_key(provider, "fallback", disabled=disabled)
    if not fallback_configured or override_fallback:
        fallback = (
            _key_input(provider, "fallback", disabled=disabled, replacement=override_fallback)
            or None
        )
        if override_fallback and not fallback:
            missing.append(f"{PROVIDER_LABELS[provider]} fallback")
    return fallback
