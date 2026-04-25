from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.simulation.graph import (
    SimulationEdge,
    SimulationNode,
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
    _topological_path,
)


def test_normalize_nodes_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate node_id"):
        _normalize_nodes(
            [
                SimulationNode(
                    node_id="agent-1",
                    model="gpt-4.1-mini",
                    provider="chatgpt",
                    personality="persona a",
                ),
                SimulationNode(
                    node_id="agent-1",
                    model="gpt-4.1-mini",
                    provider="chatgpt",
                    personality="persona b",
                ),
            ]
        )


def test_normalize_edges_builds_default_chain() -> None:
    nodes_by_id = _normalize_nodes(
        [
            SimulationNode("n1", "m1", "gemini", "p1"),
            SimulationNode("n2", "m2", "gemini", "p2"),
            SimulationNode("n3", "m3", "gemini", "p3"),
        ]
    )
    edges = _normalize_edges(nodes_by_id, None)

    assert [(edge.source, edge.target) for edge in edges] == [("n1", "n2"), ("n2", "n3")]


def test_path_resolution_returns_single_chain() -> None:
    nodes_by_id = _normalize_nodes(
        [
            SimulationNode("n1", "m1", "gemini", "p1"),
            SimulationNode("n2", "m2", "gemini", "p2"),
            SimulationNode("n3", "m3", "gemini", "p3"),
        ]
    )
    edges = [
        SimulationEdge("n1", "n2"),
        SimulationEdge("n2", "n3"),
    ]

    start = _resolve_start_node(nodes_by_id, edges, None)
    path = _topological_path(nodes_by_id, edges, start)

    assert start == "n1"
    assert path == ["n1", "n2", "n3"]


def test_dataframe_fixture_is_valid() -> None:
    df = pd.DataFrame([{"title": "t", "description": "news text"}])
    assert list(df.columns) == ["title", "description"]
