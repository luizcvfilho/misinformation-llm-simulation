from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps.interaction_graph_sections import _validate_run_inputs
from misinformation_simulation.apps.interaction_graph_ui import (
    CUSTOM_OPTION,
    apply_bulk_node_updates,
    create_default_node_form,
    validate_node_forms,
)
from misinformation_simulation.enums import Models, Provider
from misinformation_simulation.enums.models import (
    MODEL_PROVIDERS,
    default_model_for_provider,
    models_for_provider,
    validate_provider_model,
)
from misinformation_simulation.llm.pricing import MODEL_PRICING
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.simulation.generation import generate_branching_graphs


def settings_app():
    return AppTest.from_string(
        "import pandas as pd\n"
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_sections "
        "import _render_execution_settings\n"
        "from misinformation_simulation.apps.interaction_graph_workspace "
        "import retain_workspace_widgets\n"
        "retain_workspace_widgets()\n"
        "df = pd.DataFrame([{'title': 'Title', 'description': 'News'}])\n"
        "st.session_state.settings = _render_execution_settings(df, list(df.columns))"
    ).run(timeout=30)


def test_catalog_covers_every_model_and_agrees_with_pricing():
    assert set(MODEL_PROVIDERS) == set(Models)
    assert all(MODEL_PROVIDERS[model] == price.provider for model, price in MODEL_PRICING.items())
    assert models_for_provider("deepseek") == []


@pytest.mark.parametrize("provider", list(Provider))
def test_settings_filter_models_and_reset_provider_changes(provider):
    app = settings_app()
    app.selectbox(key="vad_llm_provider").set_value(provider.value).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="vad_llm_model_option").options == (
        models_for_provider(provider) + ["Custom model"]
    )
    assert app.session_state.settings["vad_llm_model"] == default_model_for_provider(provider)
    app.selectbox(key="simulation_advanced_settings_topic_drift_provider").set_value(
        provider.value
    ).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="topic_drift_model_option").options == (
        models_for_provider(provider) + ["Custom model"]
    )
    assert app.session_state.settings["topic_drift_model"] == default_model_for_provider(provider)
    # The VAD selection remains independent when the extraction provider changes.
    assert app.session_state.settings["vad_llm_provider"] == provider.value
    app.run(timeout=30)
    assert app.session_state.settings["vad_llm_model"] == default_model_for_provider(provider)


def test_vad_custom_model_survives_reruns_and_is_cleared_on_provider_change():
    app = settings_app()
    app.selectbox(key="vad_llm_model_option").set_value(CUSTOM_OPTION).run(timeout=30)
    app.text_input(key="vad_llm_model").set_value("custom-evaluator").run(timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state.settings["vad_llm_model"] == "custom-evaluator"
    app.selectbox(key="vad_llm_provider").set_value("gemini").run(timeout=30)
    assert app.session_state.settings["vad_llm_model"] == Models.GEMINI35FlashLite
    app.selectbox(key="vad_llm_provider").set_value("chatgpt").run(timeout=30)
    assert app.session_state.settings["vad_llm_model"] == Models.GPT6Luna


def test_node_provider_change_filters_models_and_updates_resolved_display():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_components "
        "import render_node_editor\n"
        "from misinformation_simulation.apps.interaction_graph_workspace "
        "import retain_workspace_widgets\n"
        "retain_workspace_widgets()\n"
        "render_node_editor(0, st.session_state.graph_nodes[0])"
    )
    node = create_default_node_form(1)
    uid = node["uid"]
    app.session_state.graph_nodes = [node]
    app.run(timeout=30)
    app.selectbox(key=f"provider_{uid}").set_value("gemini").run(timeout=30)
    assert not app.exception
    assert app.selectbox(key=f"model_option_{uid}").options == [
        Models.GEMINI35FlashLite.value,
        "Custom model",
    ]
    assert app.session_state.graph_nodes[0]["model"] == Models.GEMINI35FlashLite
    assert app.text_input(key=f"model_resolved_{uid}").value == Models.GEMINI35FlashLite
    app.run(timeout=30)
    assert app.session_state.graph_nodes[0]["model"] == Models.GEMINI35FlashLite


def test_generator_filters_models_and_generates_compatible_nodes():
    app = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_state import initialize_state\n"
        "from misinformation_simulation.apps.interaction_graph_generator "
        "import render_graph_generator\n"
        "initialize_state()\nrender_graph_generator()"
    ).run(timeout=30)
    app.selectbox(key="graph_generator_provider").set_value("gemini").run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="graph_generator_model_option").options == [
        Models.GEMINI35FlashLite.value,
        "Custom model",
    ]
    app.button(key="generate_editor_tree").click().run(timeout=30)
    assert not app.exception
    assert all(node["provider"] == "gemini" for node in app.session_state.graph_nodes)
    assert all(node["model"] == Models.GEMINI35FlashLite for node in app.session_state.graph_nodes)


def test_stale_model_selection_is_repaired_when_restoring_session():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.model_selector import render_model_selector\n"
        "st.session_state.result = render_model_selector('gemini', label='Model', "
        "option_key='model_option', custom_key='model_custom', default='gpt-6-luna')"
    )
    app.session_state.model_option = Models.GPT6Luna.value
    app.session_state.model_custom = Models.GPT6Luna.value
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state.result == Models.GEMINI35FlashLite
    assert app.selectbox(key="model_option").value == Models.GEMINI35FlashLite


@pytest.mark.parametrize("model", ["", Models.GEMINI35FlashLite.value])
def test_run_validation_rejects_invalid_vad_model(model):
    app = settings_app()
    settings = dict(app.session_state.settings)
    settings.update(vad_llm_provider="chatgpt", vad_llm_model=model)
    with patch("misinformation_simulation.apps.interaction_graph_sections.st") as streamlit:
        streamlit.session_state.graph_queue = ["queued"]
        errors = _validate_run_inputs(None, settings, [])
    assert any("VAD evaluator:" in error for error in errors)


def test_bulk_provider_only_updates_incompatible_models():
    nodes = [create_default_node_form(1), create_default_node_form(2)]
    updated = apply_bulk_node_updates(nodes, [nodes[0]["uid"]], provider="gemini")
    assert updated[0]["provider"] == "gemini"
    assert updated[0]["model"] == Models.GEMINI35FlashLite
    assert updated[1] == nodes[1]


def test_bulk_provider_change_does_not_reuse_a_custom_id_from_another_provider():
    node = create_default_node_form(1)
    node["model"] = "custom-openai-model"
    updated = apply_bulk_node_updates([node], [node["uid"]], provider="gemini")
    assert updated[0]["model"] == Models.GEMINI35FlashLite
    unchanged = apply_bulk_node_updates([node], [node["uid"]], provider="chatgpt")
    assert unchanged[0]["model"] == "custom-openai-model"


def test_bulk_selector_requires_a_common_provider_for_mixed_nodes():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_bulk_editor "
        "import _render_bulk_fields\n"
        "st.session_state.changes = _render_bulk_fields(st.session_state.graph_nodes[0])"
    )
    nodes = [create_default_node_form(1), create_default_node_form(2)]
    nodes[1].update(provider="gemini", model=Models.GEMINI35FlashLite.value)
    app.session_state.graph_nodes = nodes
    app.session_state.graph_bulk_selected_uids = [node["uid"] for node in nodes]
    app.session_state.graph_bulk_change_model = True
    app.run(timeout=30)
    assert not app.exception
    assert "model" not in app.session_state.changes
    assert any("same provider" in message.value for message in app.info)
    app.checkbox(key="graph_bulk_change_provider").check().run(timeout=30)
    app.selectbox(key="graph_bulk_provider").set_value("gemini").run(timeout=30)
    assert not app.exception
    assert app.session_state.changes == {
        "provider": "gemini",
        "model": Models.GEMINI35FlashLite.value,
    }


def test_mixed_provider_bulk_model_update_rejects_mismatches_atomically():
    nodes = [create_default_node_form(1), create_default_node_form(2)]
    nodes[1].update(provider="gemini", model=Models.GEMINI35FlashLite.value)
    original = deepcopy(nodes)
    with pytest.raises(ValueError, match="belongs to provider"):
        apply_bulk_node_updates(nodes, [node["uid"] for node in nodes], model=Models.GPT6Luna)
    assert nodes == original


def test_invalid_provider_model_is_rejected_in_forms_generation_and_custom_validation():
    node = create_default_node_form(1)
    node["provider"] = "gemini"
    assert any("belongs to provider" in error for error in validate_node_forms([node]))
    with pytest.raises(ValueError, match="belongs to provider"):
        generate_branching_graphs("CCCC", provider="gemini", model=Models.GPT6Luna)
    with pytest.raises(ValueError, match="belongs to provider"):
        validate_provider_model("gemini", Models.GPT6Luna)
    validate_provider_model("deepseek", "custom-deepseek-model")


@pytest.mark.parametrize("provider", ["gemini", "chatgpt"])
def test_incompatible_model_is_rejected_before_api_request(provider):
    client = SimpleNamespace(base_url="https://api.openai.com/v1", models=Mock(), chat=Mock())
    generate = (
        generate_gemini_text_with_retry if provider == "gemini" else generate_openai_text_with_retry
    )
    model = Models.GPT6Luna if provider == "gemini" else Models.GEMINI35FlashLite
    with pytest.raises(ValueError, match="belongs to provider"):
        generate(client, model=model, prompt="Test", system_instruction="Test", temperature=0)
    assert not client.models.mock_calls
    assert not client.chat.mock_calls
