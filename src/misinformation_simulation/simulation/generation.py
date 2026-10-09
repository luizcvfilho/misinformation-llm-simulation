from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import networkx as nx

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.simulation.io import build_graph_config_payload
from misinformation_simulation.simulation.paths import PERSONALITY_CODES
from misinformation_simulation.simulation.types import SimulationEdge, SimulationNode

PERSONALITIES_BY_CODE = {code: personality for personality, code in PERSONALITY_CODES.items()}


def parse_chain_sequences(sequences: str | Iterable[str]) -> list[str]:
    values = re.split(r"[\s,;]+", sequences.strip()) if isinstance(sequences, str) else sequences
    chains = list(
        dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip())
    )
    if not chains:
        raise ValueError("Enter at least one persona sequence, such as CCCC or CCPP.")
    for chain in chains:
        unknown = set(chain) - PERSONALITIES_BY_CODE.keys()
        if unknown:
            raise ValueError(
                f"Unknown persona codes in '{chain}': {', '.join(sorted(unknown))}. "
                f"Use {', '.join(PERSONALITIES_BY_CODE)}."
            )
    for chain in chains:
        longer = next(
            (other for other in chains if other != chain and other.startswith(chain)), None
        )
        if longer:
            raise ValueError(
                f"'{chain}' ends inside '{longer}'. Enter complete root-to-leaf sequences; "
                "generate these two lengths as separate graphs to keep both endpoints."
            )
    return chains


def generate_branching_graphs(
    sequences: str | Iterable[str],
    *,
    model: str = DEFAULT_LLM_MODEL.value,
    provider: str = DEFAULT_LLM_PROVIDER.value,
) -> list[dict[str, Any]]:
    """Build one prefix-sharing tree per initial persona, without model calls."""
    chains = parse_chain_sequences(sequences)
    if not model.strip() or not provider.strip():
        raise ValueError("Select a provider and a model for generated nodes.")
    forest = nx.DiGraph()
    for chain in chains:
        for length in range(1, len(chain) + 1):
            prefix = chain[:length]
            forest.add_node(prefix, personality=PERSONALITIES_BY_CODE[prefix[-1]])
            if length > 1:
                forest.add_edge(prefix[:-1], prefix)
    graphs = []
    for root in dict.fromkeys(chain[0] for chain in chains):
        order = list(nx.dfs_preorder_nodes(forest, root))
        nodes = [
            SimulationNode(
                node_id=f"prefix_{prefix}",
                model=model.strip(),
                provider=provider.strip(),
                personality=forest.nodes[prefix]["personality"],
                label=f"{len(prefix)}. {forest.nodes[prefix]['personality'].name}",
            )
            for prefix in order
        ]
        subtree = forest.subgraph(order)
        edges = [
            SimulationEdge(f"prefix_{source}", f"prefix_{target}")
            for source, target in subtree.edges
        ]
        graphs.append(
            build_graph_config_payload(nodes, start_node_id=f"prefix_{root}", edges=edges)
        )
    return graphs
