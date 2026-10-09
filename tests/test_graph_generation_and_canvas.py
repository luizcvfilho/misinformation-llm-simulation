from __future__ import annotations

import pytest

from misinformation_simulation.apps.interaction_graph_state import graph_nodes_to_forms
from misinformation_simulation.apps.interaction_graph_ui import (
    validate_node_forms,
)
from misinformation_simulation.simulation.generation import generate_branching_graphs
from misinformation_simulation.simulation.io import graph_config_from_payload
from misinformation_simulation.simulation.topology import (
    _normalize_nodes,
    _root_to_leaf_paths,
    _topological_path,
)


def test_generator_shares_exact_prefixes_and_keeps_two_linear_paths():
    (payload,) = generate_branching_graphs("cccc, ccpp; CCCC")
    nodes, edges, root = graph_config_from_payload(payload)
    order = _topological_path(_normalize_nodes(nodes), edges, root)
    paths = _root_to_leaf_paths(order, edges)
    assert len(nodes) == 6
    assert [[node.removeprefix("prefix_") for node in path] for path in paths] == [
        ["C", "CC", "CCC", "CCCC"],
        ["C", "CC", "CCP", "CCPP"],
    ]
    assert len(payload["edges"]) == 5


def test_generator_groups_different_initial_personas_into_separate_trees():
    payloads = generate_branching_graphs("NNNN CCCC PPPP DDDD CCPP PPCC DDNN NNDD")
    assert [payload["start_node_id"] for payload in payloads] == [
        "prefix_N",
        "prefix_C",
        "prefix_P",
        "prefix_D",
    ]
    assert sum(len(payload["nodes"]) for payload in payloads) == 24
    assert all(
        validate_node_forms(graph_nodes_to_forms(*graph_config_from_payload(payload))) == []
        for payload in payloads
    )


@pytest.mark.parametrize("sequences", ["", "CCXP", "CC CCPP"])
def test_generator_rejects_invalid_codes_and_ambiguous_endpoints(sequences):
    with pytest.raises(ValueError):
        generate_branching_graphs(sequences)
