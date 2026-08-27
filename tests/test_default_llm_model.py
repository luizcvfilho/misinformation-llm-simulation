from __future__ import annotations

import inspect

from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form
from misinformation_simulation.enums import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    Models,
    Provider,
)
from misinformation_simulation.llm.false_to_true import rewrite_false_news_as_true
from misinformation_simulation.llm.rewrite import rewrite_news_with_personality
from misinformation_simulation.simulation.graph import run_news_interaction_graph
from misinformation_simulation.topic_drift.comparison_workflow import run_comparison_workflow
from misinformation_simulation.topic_drift.extraction import (
    DEFAULT_TOPIC_DRIFT_MODEL,
    DEFAULT_TOPIC_DRIFT_PROVIDER,
    extract_topic_structure,
)


def test_default_llm_model_is_gpt_56_luna() -> None:
    assert DEFAULT_LLM_MODEL == Models.GPT56Luna
    assert DEFAULT_LLM_PROVIDER == Provider.CHATGPT


def test_llm_entry_points_use_the_shared_default_model() -> None:
    assert inspect.signature(rewrite_news_with_personality).parameters["model"].default == (
        DEFAULT_LLM_MODEL
    )
    assert inspect.signature(rewrite_false_news_as_true).parameters["model"].default == (
        DEFAULT_LLM_MODEL
    )
    assert DEFAULT_TOPIC_DRIFT_MODEL == DEFAULT_LLM_MODEL
    assert inspect.signature(extract_topic_structure).parameters["model"].default == (
        DEFAULT_LLM_MODEL
    )
    assert inspect.signature(run_comparison_workflow).parameters["extraction_model"].default == (
        DEFAULT_LLM_MODEL
    )
    graph_model_default = (
        inspect.signature(run_news_interaction_graph).parameters["topic_drift_model"].default
    )
    assert graph_model_default == DEFAULT_LLM_MODEL


def test_default_llm_provider_matches_the_shared_model() -> None:
    assert inspect.signature(rewrite_news_with_personality).parameters["provider"].default == (
        DEFAULT_LLM_PROVIDER
    )
    assert inspect.signature(rewrite_false_news_as_true).parameters["provider"].default == (
        DEFAULT_LLM_PROVIDER
    )
    assert DEFAULT_TOPIC_DRIFT_PROVIDER == DEFAULT_LLM_PROVIDER
    assert create_default_node_form(1)["provider"] == DEFAULT_LLM_PROVIDER.value
    assert create_default_node_form(1)["model"] == DEFAULT_LLM_MODEL.value
