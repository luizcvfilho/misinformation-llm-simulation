from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from misinformation_simulation.apps.interaction_graph_ui import validate_node_forms


def add_graph(queue: list[dict[str, Any]], name: str, node_forms: list[dict[str, str]]) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Enter a name for the graph.")
    errors = validate_node_forms(node_forms)
    if errors:
        raise ValueError(" ".join(errors))
    queue.append({"id": uuid4().hex, "name": name, "nodes": deepcopy(node_forms)})


def move_graph(queue: list[dict[str, Any]], index: int, direction: int) -> None:
    target = index + direction
    if 0 <= index < len(queue) and 0 <= target < len(queue):
        queue[index], queue[target] = queue[target], queue[index]


def output_prefix_for_graph(base_prefix: str, index: int, name: str) -> str:
    base = Path(base_prefix.strip()).name
    if not base or base in {".", ".."}:
        raise ValueError("Enter an output prefix before running.")
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name)
    slug = "_".join(part for part in slug.split("_") if part) or "graph"
    return f"{base}_{index:02d}_{slug}"
