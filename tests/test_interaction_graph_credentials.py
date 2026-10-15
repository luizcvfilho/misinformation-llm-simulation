from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps.interaction_graph_credentials import required_provider_uses
from misinformation_simulation.llm.credentials import API_KEY_NAMES, OPENAI_FALLBACK_NAMES

SCRIPT = """
import streamlit as st
from misinformation_simulation.apps.interaction_graph_credentials import render_api_keys
from misinformation_simulation.apps.interaction_graph_ui import (
    create_default_node_form, build_editor_graph_payload,
)

providers = st.multiselect("Node providers",
    ["chatgpt", "gemini", "openrouter", "deepseek", "grok", "local"],
    default=["chatgpt", "openrouter", "deepseek"], key="providers")
extraction = st.selectbox("Extraction provider", ["gemini", "chatgpt", "local"], key="extraction")
vad = st.selectbox("VAD provider", ["grok", "chatgpt", "local"], key="vad")
vad_method = st.selectbox("VAD method", ["dual", "model"], key="method")
forms = []
for index, provider in enumerate(providers, 1):
    form = create_default_node_form(index)
    form.update(provider=provider, model=f"model-{index}")
    forms.append(form)
graphs = [{"name": "Selected graph", "nodes": forms}]
settings = {"topic_drift_provider": extraction, "topic_drift_model": "extract-model",
    "vad_llm_provider": vad, "vad_llm_model": "vad-model", "vad_method": vad_method,
    "stdi_comparison_method": "dual"}
credentials, missing = render_api_keys(graphs, settings)
st.session_state.test_credentials = credentials
st.session_state.test_missing = missing
st.session_state.graph_export = build_editor_graph_payload(forms) if forms else {}
st.button("Run", disabled=bool(missing))
"""


@pytest.fixture(autouse=True)
def empty_environment(monkeypatch):
    for names in [*API_KEY_NAMES.values(), OPENAI_FALLBACK_NAMES]:
        for name in names:
            monkeypatch.delenv(name, raising=False)


def app(tmp_path):
    path = tmp_path / "credentials_app.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(path)).run(timeout=25)


def test_all_selected_providers_have_password_fields_and_block_missing_keys(tmp_path):
    ui = app(tmp_path)
    assert not ui.exception
    assert len(ui.text_input) == 6
    for item in ui.text_input:
        assert item.proto.type == item.proto.PASSWORD
    assert ui.button[-1].disabled
    for provider in ("gemini", "chatgpt", "openrouter", "deepseek", "grok"):
        ui.text_input(key=f"session_api_key_{provider}_primary").input(f"ui-{provider}").run()
    assert not ui.exception
    assert not ui.button[-1].disabled
    assert ui.session_state.test_missing == []
    assert ui.session_state.test_credentials["chatgpt"].fallback is None


def test_environment_aliases_hide_inputs_without_showing_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-environment-primary")
    monkeypatch.setenv("OPENAI_API_KEY_FALLBACK", "secret-environment-backup")
    monkeypatch.setenv("XAI_API_KEY", "secret-environment-grok")
    ui = app(tmp_path)
    assert not ui.exception
    keys = [item.key for item in ui.text_input]
    assert "session_api_key_chatgpt_primary" not in keys
    assert "session_api_key_chatgpt_fallback" not in keys
    assert "session_api_key_grok_primary" not in keys
    assert len(ui.text_input) == 3
    rendered = " ".join(item.value for item in ui.caption)
    assert "secret-environment" not in rendered
    assert "chatgpt" not in ui.session_state.test_credentials


def test_openai_can_add_optional_fallback_with_environment_primary(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATGPT_API_KEY", "environment-primary")
    ui = app(tmp_path)
    ui.text_input(key="session_api_key_chatgpt_fallback").input("session-backup").run()
    assert not ui.exception
    assert ui.session_state.test_credentials["chatgpt"].primary is None
    assert ui.session_state.test_credentials["chatgpt"].fallback == "session-backup"


@pytest.mark.parametrize("provider", ["chatgpt", "gemini", "openrouter", "deepseek", "grok"])
def test_environment_primary_can_be_replaced_and_restored(tmp_path, monkeypatch, provider):
    monkeypatch.setenv(API_KEY_NAMES[provider][0], "environment-primary")
    ui = app(tmp_path)
    choice = f"session_api_key_override_{provider}_primary"
    field = f"session_api_key_{provider}_primary"
    assert not ui.checkbox(key=choice).value
    assert field not in [item.key for item in ui.text_input]
    ui.checkbox(key=choice).check().run()
    assert provider not in ui.session_state.test_credentials
    assert ui.button[-1].disabled
    ui.text_input(key=field).input("replacement-key").run()
    assert not ui.exception
    assert ui.session_state.test_credentials[provider].primary == "replacement-key"
    ui.checkbox(key=choice).uncheck().run()
    credentials = ui.session_state.test_credentials
    assert provider not in credentials or credentials[provider].primary is None
    assert field not in [item.key for item in ui.text_input]
    ui.checkbox(key=choice).check().run()
    assert ui.text_input(key=field).value == "replacement-key"


def test_openai_replacement_choices_are_independent_and_clear_restores_defaults(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-primary")
    monkeypatch.setenv("OPENAI_API_KEY_FALLBACK", "environment-fallback")
    ui = app(tmp_path)
    ui.multiselect(key="providers").set_value(["chatgpt"]).run()
    ui.selectbox(key="extraction").set_value("local").run()
    ui.selectbox(key="method").set_value("model").run()
    assert not ui.button[-1].disabled
    primary_choice = "session_api_key_override_chatgpt_primary"
    fallback_choice = "session_api_key_override_chatgpt_fallback"
    ui.checkbox(key=fallback_choice).check().run()
    assert ui.button[-1].disabled
    assert ui.session_state.test_missing == ["OpenAI / GPT fallback"]
    ui.text_input(key="session_api_key_chatgpt_fallback").input("replacement-fallback").run()
    assert not ui.button[-1].disabled
    assert ui.session_state.test_credentials["chatgpt"].primary is None
    ui.checkbox(key=primary_choice).check().run()
    assert ui.button[-1].disabled
    ui.text_input(key="session_api_key_chatgpt_primary").input("replacement-primary").run()
    assert not ui.button[-1].disabled
    assert ui.session_state.test_credentials["chatgpt"].primary == "replacement-primary"
    ui.checkbox(key=fallback_choice).uncheck().run()
    assert ui.session_state.test_credentials["chatgpt"].fallback is None
    next(button for button in ui.button if button.label == "Clear session API keys").click().run()
    assert not ui.exception
    assert not ui.checkbox(key=primary_choice).value
    assert not ui.checkbox(key=fallback_choice).value
    assert "chatgpt" not in ui.session_state.test_credentials
    assert not ui.text_input
    assert not ui.button[-1].disabled


def test_replacement_choice_and_value_survive_switching_away_from_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-primary")
    ui = app(tmp_path)
    choice = "session_api_key_override_chatgpt_primary"
    ui.checkbox(key=choice).check().run()
    ui.text_input(key="session_api_key_chatgpt_primary").input("replacement-primary").run()
    ui.multiselect(key="providers").set_value(["local"]).run()
    assert choice not in [item.key for item in ui.checkbox]
    ui.multiselect(key="providers").set_value(["chatgpt"]).run()
    assert not ui.exception
    assert ui.checkbox(key=choice).value
    assert ui.session_state.test_credentials["chatgpt"].primary == "replacement-primary"


def test_keys_survive_provider_switch_and_can_be_cleared_without_entering_exports(tmp_path):
    ui = app(tmp_path)
    ui.text_input(key="session_api_key_chatgpt_primary").input("private-session-value").run()
    ui.multiselect(key="providers").set_value(["deepseek"]).run()
    assert "session_api_key_chatgpt_primary" not in [item.key for item in ui.text_input]
    ui.multiselect(key="providers").set_value(["chatgpt"]).run()
    assert ui.text_input(key="session_api_key_chatgpt_primary").value == "private-session-value"
    assert "private-session-value" not in json.dumps(ui.session_state.graph_export)
    next(button for button in ui.button if button.label == "Clear session API keys").click().run()
    assert not ui.exception
    assert ui.text_input(key="session_api_key_chatgpt_primary").value == ""
    assert "chatgpt" not in ui.session_state.test_credentials


def test_local_and_disabled_vad_do_not_request_keys(tmp_path):
    ui = app(tmp_path)
    ui.multiselect(key="providers").set_value(["local"]).run()
    ui.selectbox(key="extraction").set_value("local").run()
    ui.selectbox(key="method").set_value("model").run()
    assert not ui.exception
    assert not ui.text_input
    assert ui.session_state.test_missing == []
    assert not ui.button[-1].disabled


def test_multiple_graphs_models_and_evaluation_roles_are_collected_once_per_provider():
    graphs = [
        {
            "name": "First",
            "nodes": [
                {"provider": "chatgpt", "label": "A", "model": "gpt-one"},
                {"provider": "chatgpt", "label": "B", "model": "gpt-two"},
            ],
        },
        {"name": "Queued", "nodes": [{"provider": "openrouter", "model": "third"}]},
    ]
    uses = required_provider_uses(
        graphs,
        {
            "topic_drift_provider": "gemini",
            "topic_drift_model": "extract",
            "stdi_comparison_method": "cluster",
            "vad_method": "dual",
            "vad_llm_provider": "grok",
            "vad_llm_model": "evaluate",
        },
    )
    assert set(uses) == {"chatgpt", "gemini", "openrouter", "grok"}
    assert len(uses["chatgpt"]) == 2
    assert "gpt-one" in uses["chatgpt"][0]
    assert "gpt-two" in uses["chatgpt"][1]
    assert uses["gemini"] == ["Topic extraction: extract"]
    assert uses["grok"] == ["VAD evaluator: evaluate"]


def test_run_controls_collect_queue_credentials_separately_from_exportable_settings(
    tmp_path,
    monkeypatch,
):
    from misinformation_simulation.apps import interaction_graph_sections as sections

    captured = []
    monkeypatch.setattr(sections, "start_graph_run_job", lambda **kwargs: captured.append(kwargs))
    path = tmp_path / "run_controls_app.py"
    path.write_text(
        """
import pandas as pd
import streamlit as st
from misinformation_simulation.apps.interaction_graph_sections import _render_run_controls
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form

if "graph_nodes" not in st.session_state:
    st.session_state.graph_nodes = [create_default_node_form(1)]
    queued = create_default_node_form(1)
    queued.update(provider="openrouter", model="queued-model")
    st.session_state.graph_queue = [{"name": "Queued", "nodes": [queued]}]
settings = {"graph_name": "Editor", "text_column": "description", "title_column": "title",
    "topic_drift_provider": "local", "topic_drift_model": "local-extractor",
    "output_prefix": "test", "vad_method": "model"}
_render_run_controls(pd.DataFrame([{"title": "Title", "description": "News"}]), settings)
""",
        encoding="utf-8",
    )
    ui = AppTest.from_file(str(path)).run(timeout=25)
    assert not ui.exception
    assert [item.key for item in ui.text_input] == ["session_api_key_openrouter_primary"]
    assert next(button for button in ui.button if button.label == "Run graph queue").disabled
    ui.text_input[0].input("queue-private-value").run()
    next(button for button in ui.button if button.label == "Run graph queue").click().run()
    assert not ui.exception
    assert len(captured) == 1
    assert captured[0]["api_credentials"]["openrouter"].primary == "queue-private-value"
    assert "queue-private-value" not in json.dumps(captured[0]["settings"])
    assert "queue-private-value" not in json.dumps(captured[0]["graphs"])
