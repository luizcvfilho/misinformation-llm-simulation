from __future__ import annotations

from misinformation_simulation.simulation.types import SimulationEdge, SimulationNode


def _node_label(node: SimulationNode) -> str:
    return node.label or node.node_id


def _normalize_nodes(nodes: list[SimulationNode]) -> dict[str, SimulationNode]:
    if not nodes:
        raise ValueError("Provide at least one simulation node.")

    nodes_by_id: dict[str, SimulationNode] = {}
    for node in nodes:
        if not node.node_id.strip():
            raise ValueError("Each node must have a non-empty 'node_id'.")
        if node.node_id in nodes_by_id:
            raise ValueError(f"Duplicate node_id detected: '{node.node_id}'.")
        if not node.personality.strip():
            raise ValueError(f"Node '{node.node_id}' must define a personality.")
        nodes_by_id[node.node_id] = node
    return nodes_by_id


def _normalize_edges(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge] | None,
) -> list[SimulationEdge]:
    if edges is None:
        ordered_nodes = list(nodes_by_id.values())
        return [
            SimulationEdge(
                source=ordered_nodes[index].node_id,
                target=ordered_nodes[index + 1].node_id,
            )
            for index in range(len(ordered_nodes) - 1)
        ]

    normalized_edges: list[SimulationEdge] = []
    seen_targets: set[str] = set()
    for edge in edges:
        if edge.source not in nodes_by_id:
            raise ValueError(f"Edge source '{edge.source}' is not a known node.")
        if edge.target not in nodes_by_id:
            raise ValueError(f"Edge target '{edge.target}' is not a known node.")
        if edge.target in seen_targets:
            raise ValueError(
                "Each node can receive text from only one previous node in this simulation."
            )
        seen_targets.add(edge.target)
        normalized_edges.append(edge)
    return normalized_edges


def _resolve_start_node(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge],
    start_node_id: str | None,
) -> str:
    if start_node_id is not None:
        if start_node_id not in nodes_by_id:
            raise ValueError(f"Unknown start node '{start_node_id}'.")
        return start_node_id

    inbound_counts = {node_id: 0 for node_id in nodes_by_id}
    for edge in edges:
        inbound_counts[edge.target] += 1

    candidates = [node_id for node_id, count in inbound_counts.items() if count == 0]
    if len(candidates) != 1:
        raise ValueError("Could not infer a unique start node. Provide 'start_node_id' explicitly.")
    return candidates[0]


def _topological_path(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge],
    start_node_id: str,
) -> list[str]:
    """Return parent-before-child order for a connected rooted tree."""
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes_by_id}
    for edge in edges:
        children[edge.source].append(edge.target)
    if any(edge.target == start_node_id for edge in edges):
        raise ValueError("The start node cannot receive text from another node (cycle detected).")

    path: list[str] = []
    seen: set[str] = set()
    pending = [start_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            raise ValueError("Cycle detected in the interaction graph.")
        seen.add(node_id)
        path.append(node_id)
        pending.extend(reversed(children[node_id]))

    if len(path) != len(nodes_by_id):
        missing = [node_id for node_id in nodes_by_id if node_id not in seen]
        raise ValueError(
            "The interaction graph must form a connected rooted tree without cycles. "
            f"Unreachable nodes: {', '.join(missing)}."
        )
    return path


def _root_to_leaf_paths(
    ordered_node_ids: list[str], edges: list[SimulationEdge]
) -> list[list[str]]:
    parents = {edge.target: edge.source for edge in edges}
    sources = {edge.source for edge in edges}
    paths_by_node: dict[str, list[str]] = {}
    paths: list[list[str]] = []
    for node_id in ordered_node_ids:
        parent = parents.get(node_id)
        path = [*paths_by_node.get(parent, []), node_id]
        paths_by_node[node_id] = path
        if node_id not in sources:
            paths.append(path)
    return paths
