from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from misinformation_simulation.apps.interaction_graph_state import graph_nodes_to_forms
from misinformation_simulation.apps.interaction_graph_ui import validate_node_forms
from misinformation_simulation.simulation.io import graph_config_from_payload, resolve_project_path


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


def add_graphs_from_directory(
    queue: list[dict[str, Any]], directory: Path | str
) -> tuple[list[str], list[str]]:
    if not str(directory).strip():
        raise ValueError("Select a graph folder first.")
    folder = resolve_project_path(directory).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Graph folder not found: {folder}")
    paths = sorted(folder.glob("*.json"), key=lambda path: path.name.lower())
    if not paths:
        raise ValueError(f"No JSON graph files found in: {folder}")

    added: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Graph JSON must contain an object.")
            nodes, edges, start_node_id = graph_config_from_payload(payload)
            if not nodes:
                raise ValueError("Graph config does not contain nodes.")
            forms = graph_nodes_to_forms(nodes, edges, start_node_id)
            add_graph(queue, path.stem, forms)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
        else:
            added.append(path.name)
    return added, errors
