from __future__ import annotations

import pytest

from misinformation_simulation.apps.interaction_graph_state import graph_nodes_to_forms
from misinformation_simulation.apps.interaction_graph_ui import (
    build_linear_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    create_default_node_form,
    steps_to_dataframe,
)
from misinformation_simulation.enums import Provider
from misinformation_simulation.simulation.graph import (
    SimulationEdge,
    SimulationNode,
    SimulationStepResult,
)


def test_build_simulation_nodes_resolves_preset_personality() -> None:
    node_form = create_default_node_form(1)

    nodes = build_simulation_nodes([node_form])

    assert len(nodes) == 1
    assert nodes[0].node_id == "node_1"
    assert nodes[0].personality.startswith("You are")


def test_build_linear_graph_payload_creates_chain_edges() -> None:
    node_a = create_default_node_form(1)
    node_b = create_default_node_form(2)

    payload = build_linear_graph_payload([node_a, node_b])

    assert payload["start_node_id"] == node_a["node_id"]
    assert payload["edges"] == [{"source": "node_1", "target": "node_2"}]


def test_graph_nodes_to_forms_orders_nodes_from_edges() -> None:
    nodes = [
        SimulationNode(
            node_id="node_a",
            model="gemini-3.1-flash-lite-preview",
            provider=Provider.GEMINI,
            personality="persona-a",
        ),
        SimulationNode(
            node_id="node_b",
            model="gpt-4.1-mini",
            provider=Provider.CHATGPT,
            personality="persona-b",
        ),
        SimulationNode(
            node_id="node_c",
            model="gemini-3.1-flash-lite-preview",
            provider=Provider.GEMINI,
            personality="persona-c",
        ),
    ]
    edges = [
        SimulationEdge(source="node_b", target="node_c"),
        SimulationEdge(source="node_c", target="node_a"),
    ]

    forms = graph_nodes_to_forms(nodes, edges, start_node_id="node_b")

    assert [node["node_id"] for node in forms] == ["node_b", "node_c", "node_a"]


def test_graph_nodes_to_forms_rejects_branching_graphs() -> None:
    nodes = [
        SimulationNode(
            node_id="node_a",
            model="gemini-3.1-flash-lite-preview",
            provider=Provider.GEMINI,
            personality="persona-a",
        ),
        SimulationNode(
            node_id="node_b",
            model="gpt-4.1-mini",
            provider=Provider.CHATGPT,
            personality="persona-b",
        ),
        SimulationNode(
            node_id="node_c",
            model="gemini-3.1-flash-lite-preview",
            provider=Provider.GEMINI,
            personality="persona-c",
        ),
    ]
    edges = [
        SimulationEdge(source="node_a", target="node_b"),
        SimulationEdge(source="node_a", target="node_c"),
    ]

    with pytest.raises(ValueError, match="single chain path"):
        graph_nodes_to_forms(nodes, edges, start_node_id="node_a")


def test_result_summaries_group_steps() -> None:
    steps = [
        SimulationStepResult(
            news_id="news-1",
            step_index=1,
            node_id="node_1",
            node_label="Node 1",
            source_node_id="original",
            source_node_label="description",
            provider="gemini",
            model="gemini-3.1-flash-lite-preview",
            personality="persona",
            source_text="source",
            rewritten_text="rewrite",
            target_language="en",
            target_language_source="row.language",
            rewrite_status="success",
            rewrite_error=None,
            stdi_vs_original=0.2,
            stdi_incremental=0.2,
            metadata={"title": "Title 1"},
        ),
        SimulationStepResult(
            news_id="news-1",
            step_index=2,
            node_id="node_2",
            node_label="Node 2",
            source_node_id="node_1",
            source_node_label="Node 1",
            provider="chatgpt",
            model="gpt-4.1-mini",
            personality="persona",
            source_text="rewrite",
            rewritten_text=None,
            target_language="en",
            target_language_source="row.language",
            rewrite_status="error",
            rewrite_error="failure",
            stdi_vs_original=None,
            stdi_incremental=None,
            metadata={"title": "Title 1"},
        ),
    ]

    steps_df = steps_to_dataframe(steps)
    node_summary_df = build_node_summary_dataframe(steps_df)
    news_summary_df = build_news_summary_dataframe(steps_df)

    assert list(steps_df["node_label"]) == ["Node 1", "Node 2"]
    assert list(node_summary_df["runs"]) == [1, 1]
    assert int(news_summary_df.iloc[0]["errors"]) == 1
