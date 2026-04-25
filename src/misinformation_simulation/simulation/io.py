from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from misinformation_simulation.datasets.newsdata import QUERY_METADATA_ROW_ID
from misinformation_simulation.simulation.graph import SimulationEdge, SimulationNode

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path: Path | str, project_root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root or DEFAULT_PROJECT_ROOT) / candidate


def filter_query_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "article_id" not in df.columns:
        return df

    article_ids = df["article_id"].fillna("").astype(str).str.strip().str.lower()
    filtered_df = df[article_ids != QUERY_METADATA_ROW_ID].copy()
    return filtered_df


def load_news_dataframe(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> pd.DataFrame:
    resolved_path = resolve_project_path(path, project_root)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {resolved_path}. Pass a valid path in --input/INPUT."
        )

    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(resolved_path)
        return filter_query_metadata_rows(df)
    if suffix in {".json", ".jsonl"}:
        return filter_query_metadata_rows(pd.read_json(resolved_path, lines=suffix == ".jsonl"))
    raise ValueError("Unsupported input file. Use CSV, JSON, or JSONL.")


def load_graph_config(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> tuple[list[SimulationNode], list[SimulationEdge] | None, str | None]:
    payload = load_graph_config_payload(path, project_root=project_root)
    return graph_config_from_payload(payload)


def load_graph_config_payload(
    path: Path | str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    resolved_path = resolve_project_path(path, project_root)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Graph config file not found: {resolved_path}. "
            "Pass a valid path in --graph-config/GRAPH_CONFIG."
        )
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def graph_config_from_payload(
    payload: dict[str, Any],
) -> tuple[list[SimulationNode], list[SimulationEdge] | None, str | None]:
    nodes = [SimulationNode(**node_payload) for node_payload in payload.get("nodes", [])]
    raw_edges = payload.get("edges")
    edges = None
    if raw_edges is not None:
        edges = [SimulationEdge(**edge_payload) for edge_payload in raw_edges]
    return nodes, edges, payload.get("start_node_id")


def build_graph_config_payload(
    nodes: list[SimulationNode],
    *,
    start_node_id: str | None = None,
    edges: list[SimulationEdge] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": [asdict(node) for node in nodes],
    }
    if start_node_id is not None:
        payload["start_node_id"] = start_node_id

    resolved_edges = edges
    if resolved_edges is None and len(nodes) > 1:
        resolved_edges = [
            SimulationEdge(source=nodes[index].node_id, target=nodes[index + 1].node_id)
            for index in range(len(nodes) - 1)
        ]
    if resolved_edges is not None:
        payload["edges"] = [asdict(edge) for edge in resolved_edges]
    return payload
