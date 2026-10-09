from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd

from misinformation_simulation.enums import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DefaultPersonality,
    Models,
    Provider,
)
from misinformation_simulation.simulation import (
    SimulationEdge,
    SimulationNode,
    SimulationStepResult,
)
from misinformation_simulation.simulation.io import build_graph_config_payload
from misinformation_simulation.simulation.topology import (
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
    _topological_path,
)

CUSTOM_OPTION = "__custom__"
PREDEFINED_PERSONALITIES = {
    personality.name: personality.value for personality in DefaultPersonality
}
AVAILABLE_MODELS = [model.value for model in Models]
AVAILABLE_PROVIDERS = [provider.value for provider in Provider]


def branch_score_columns(steps_df: pd.DataFrame) -> list[str]:
    if "stdi_evaluation" in steps_df.columns:
        return []
    return [
        f"stdi_{branch}_{suffix}"
        for branch in ("cluster", "llm_judge")
        if f"stdi_{branch}_vs_original" in steps_df.columns
        for suffix in ("vs_original", "incremental", "cumulative")
        if f"stdi_{branch}_{suffix}" in steps_df.columns
    ]


def create_default_node_form(position: int) -> dict[str, str]:
    preset_names = list(PREDEFINED_PERSONALITIES)
    preset_name = preset_names[(position - 1) % len(preset_names)]
    return {
        "uid": uuid4().hex,
        "node_id": f"node_{position}",
        "label": f"Node {position}",
        "provider": DEFAULT_LLM_PROVIDER.value,
        "model": DEFAULT_LLM_MODEL.value,
        "personality_mode": "preset",
        "personality_preset": preset_name,
        "personality_custom": "",
    }


def append_editor_node(
    forms: list[dict[str, str]],
    *,
    parent_uid: str | None = None,
) -> dict[str, str]:
    position = len(forms) + 1
    used_ids = {form["node_id"] for form in forms}
    while f"node_{position}" in used_ids:
        position += 1
    form = create_default_node_form(position)
    if parent_uid is not None or any("parent_uid" in item for item in forms):
        form["parent_uid"] = (
            parent_uid if parent_uid is not None else (forms[-1]["uid"] if forms else "")
        )
    forms.append(form)
    return form


def normalize_node_form(node: SimulationNode, position: int) -> dict[str, str]:
    form = create_default_node_form(position)
    matched_preset = detect_matching_personality_preset(node.personality)
    form.update(
        {
            "node_id": node.node_id,
            "label": node.label or f"Node {position}",
            "provider": str(node.provider),
            "model": node.model,
            "personality_mode": "preset" if matched_preset else "custom",
            "personality_preset": matched_preset or form["personality_preset"],
            "personality_custom": "" if matched_preset else node.personality,
        }
    )
    return form


def detect_matching_personality_preset(personality_text: str) -> str | None:
    normalized_candidate = normalize_text(personality_text)
    if not normalized_candidate:
        return None

    for preset_name, preset_text in PREDEFINED_PERSONALITIES.items():
        normalized_preset = normalize_text(preset_text)
        if (
            normalized_candidate == normalized_preset
            or normalized_candidate in normalized_preset
            or normalized_preset in normalized_candidate
        ):
            return preset_name
    return None


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def resolve_personality_text(node_form: dict[str, str]) -> str:
    if node_form.get("personality_mode") == "custom":
        return node_form.get("personality_custom", "").strip()
    preset_name = node_form.get("personality_preset", "")
    return PREDEFINED_PERSONALITIES.get(preset_name, "").strip()


def build_simulation_nodes(node_forms: list[dict[str, str]]) -> list[SimulationNode]:
    nodes: list[SimulationNode] = []
    for node_form in node_forms:
        nodes.append(
            SimulationNode(
                node_id=node_form.get("node_id", "").strip(),
                label=node_form.get("label", "").strip() or None,
                provider=node_form.get("provider", "").strip(),
                model=node_form.get("model", "").strip(),
                personality=resolve_personality_text(node_form),
            )
        )
    return nodes


def build_linear_graph_payload(node_forms: list[dict[str, str]]) -> dict[str, Any]:
    nodes = build_simulation_nodes(node_forms)
    start_node_id = nodes[0].node_id if nodes else None
    return build_graph_config_payload(nodes, start_node_id=start_node_id)


def build_editor_graph_payload(node_forms: list[dict[str, str]]) -> dict[str, Any]:
    if not any("parent_uid" in form for form in node_forms):
        return build_linear_graph_payload(node_forms)
    nodes = build_simulation_nodes(node_forms)
    ids_by_uid = {form["uid"]: node.node_id for form, node in zip(node_forms, nodes, strict=True)}
    edges = [
        SimulationEdge(ids_by_uid[form["parent_uid"]], node.node_id)
        for form, node in zip(node_forms, nodes, strict=True)
        if form.get("parent_uid")
    ]
    roots = [
        node.node_id
        for form, node in zip(node_forms, nodes, strict=True)
        if not form.get("parent_uid")
    ]
    payload = build_graph_config_payload(
        nodes,
        start_node_id=roots[0] if len(roots) == 1 else None,
        edges=edges,
    )
    positions = {
        form["node_id"]: {"x": float(form["flow_x"]), "y": float(form["flow_y"])}
        for form in node_forms
        if "flow_x" in form and "flow_y" in form
    }
    if positions:
        payload["layout"] = {"positions": positions}
    return payload


def restore_editor_layout(forms: list[dict[str, str]], payload: dict[str, Any]) -> None:
    from math import isfinite

    positions = payload.get("layout", {}).get("positions", {})
    for form in forms:
        position = positions.get(form["node_id"])
        if position is not None:
            x, y = float(position["x"]), float(position["y"])
            if not isfinite(x) or not isfinite(y):
                raise ValueError("Graph layout coordinates must be finite numbers.")
            form.update(flow_x=str(x), flow_y=str(y))


def set_tree_mode(node_forms: list[dict[str, str]], enabled: bool) -> None:
    for index, form in enumerate(node_forms):
        if enabled:
            form.setdefault("parent_uid", node_forms[index - 1]["uid"] if index else "")
        else:
            form.pop("parent_uid", None)


def validate_node_forms(node_forms: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not node_forms:
        return ["Add at least one node to run the interaction graph."]

    seen_node_ids: set[str] = set()
    for index, node_form in enumerate(node_forms, start=1):
        node_id = node_form.get("node_id", "").strip()
        label = node_form.get("label", "").strip()
        model = node_form.get("model", "").strip()
        provider = node_form.get("provider", "").strip()
        personality = resolve_personality_text(node_form)

        if not node_id:
            errors.append(f"Node {index} must define a node_id.")
        elif node_id in seen_node_ids:
            errors.append(f"Duplicate node_id detected: '{node_id}'.")
        seen_node_ids.add(node_id)

        if not label:
            errors.append(f"Node {index} must define a label.")
        if not provider:
            errors.append(f"Node {index} must define a provider.")
        if not model:
            errors.append(f"Node {index} must define a model.")
        if not personality:
            errors.append(f"Node {index} must define a personality.")
    if not errors:
        try:
            payload = build_editor_graph_payload(node_forms)
            nodes_by_id = _normalize_nodes(build_simulation_nodes(node_forms))
            edges = _normalize_edges(
                nodes_by_id, [SimulationEdge(**edge) for edge in payload.get("edges", [])]
            )
            start = _resolve_start_node(nodes_by_id, edges, payload.get("start_node_id"))
            _topological_path(nodes_by_id, edges, start)
        except (ValueError, KeyError) as exc:
            errors.append(f"Invalid graph topology: {exc}")
    return errors


def steps_to_dataframe(step_results: list[SimulationStepResult]) -> pd.DataFrame:
    if not step_results:
        return pd.DataFrame()
    df = pd.DataFrame([step.to_record() for step in step_results])
    sort_columns = [column for column in ("news_id", "step_index") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


def build_node_summary_dataframe(steps_df: pd.DataFrame) -> pd.DataFrame:
    if steps_df.empty:
        return pd.DataFrame()

    grouped = (
        steps_df.groupby(["step_index", "node_id", "node_label", "provider", "model"], dropna=False)
        .agg(
            runs=("rewrite_status", "size"),
            successes=("rewrite_status", lambda values: int((values == "success").sum())),
            errors=("rewrite_status", lambda values: int((values == "error").sum())),
            mean_stdi_vs_original=("stdi_vs_original", "mean"),
            mean_stdi_incremental=("stdi_incremental", "mean"),
            mean_stdi_cumulative=("stdi_cumulative", "mean"),
            mean_vad_drift_vs_original=("vad_drift_vs_original", "mean"),
            mean_contradiction_drift_vs_original=("contradiction_drift_vs_original", "mean"),
            **{f"mean_{column}": (column, "mean") for column in branch_score_columns(steps_df)},
        )
        .reset_index()
        .sort_values("step_index")
    )
    grouped["success_rate"] = grouped["successes"] / grouped["runs"]
    return grouped


def build_news_summary_dataframe(steps_df: pd.DataFrame) -> pd.DataFrame:
    if steps_df.empty:
        return pd.DataFrame()

    aggregations = {"title": ("metadata_title", "first")}
    if "metadata_category" in steps_df.columns:
        aggregations["category"] = ("metadata_category", "first")
    aggregations.update(
        {f"max_{column}": (column, "max") for column in branch_score_columns(steps_df)}
    )
    grouped = (
        steps_df.groupby("news_id", dropna=False)
        .agg(
            **aggregations,
            steps=("rewrite_status", "size"),
            successes=("rewrite_status", lambda values: int((values == "success").sum())),
            errors=("rewrite_status", lambda values: int((values == "error").sum())),
            max_stdi_vs_original=("stdi_vs_original", "max"),
            max_stdi_incremental=("stdi_incremental", "max"),
            max_stdi_cumulative=("stdi_cumulative", "max"),
        )
        .reset_index()
        .sort_values(["errors", "max_stdi_vs_original", "news_id"], ascending=[False, False, True])
    )
    return grouped
