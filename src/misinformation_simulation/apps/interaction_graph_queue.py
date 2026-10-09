from __future__ import annotations

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from misinformation_simulation.apps.interaction_graph_state import graph_nodes_to_forms
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    restore_editor_layout,
    set_tree_mode,
    validate_node_forms,
)
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


def output_prefix_for_graph(
    base_prefix: str, index: int, name: str, *, include_base: bool = True
) -> str:
    base = Path(base_prefix.strip()).name
    if not base or base in {".", ".."}:
        raise ValueError("Enter an output prefix before running.")
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name)
    slug = "_".join(part for part in slug.split("_") if part)[:32].rstrip("_") or "graph"
    graph_prefix = f"{index:02d}_{slug}"
    return f"{base}_{graph_prefix}" if include_base else graph_prefix


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
            restore_editor_layout(forms, payload)
            if payload.get("layout"):
                set_tree_mode(forms, True)
            add_graph(queue, path.stem, forms)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
        else:
            added.append(path.name)
    return added, errors


def build_graph_queue_archive(queue: list[dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    position_width = max(2, len(str(len(queue))))
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for index, graph in enumerate(queue, start=1):
            slug = output_prefix_for_graph("queue", index, graph["name"], include_base=False).split(
                "_", 1
            )[1]
            file_name = f"{index:0{position_width}d}_{slug}.json"
            payload = build_editor_graph_payload(graph["nodes"])
            archive.writestr(
                file_name, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            )
    return buffer.getvalue()
