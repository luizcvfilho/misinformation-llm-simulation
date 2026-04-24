from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd

from misinformation_simulation.enums import DefaultPersonality, Models, Provider
from misinformation_simulation.simulation import SimulationNode, SimulationStepResult
from misinformation_simulation.simulation.io import build_graph_config_payload

CUSTOM_OPTION = "__custom__"
PREDEFINED_PERSONALITIES = {
    personality.name: personality.value for personality in DefaultPersonality
}
AVAILABLE_MODELS = [model.value for model in Models]
AVAILABLE_PROVIDERS = [provider.value for provider in Provider]


def create_default_node_form(position: int) -> dict[str, str]:
    preset_names = list(PREDEFINED_PERSONALITIES)
    preset_name = preset_names[(position - 1) % len(preset_names)]
    return {
        "uid": uuid4().hex,
        "node_id": f"node_{position}",
        "label": f"Node {position}",
        "provider": Provider.GEMINI.value,
        "model": Models.GEMINI31FlashLite.value,
        "personality_mode": "preset",
        "personality_preset": preset_name,
        "personality_custom": "",
    }


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
        )
        .reset_index()
        .sort_values("step_index")
    )
    grouped["success_rate"] = grouped["successes"] / grouped["runs"]
    return grouped


def build_news_summary_dataframe(steps_df: pd.DataFrame) -> pd.DataFrame:
    if steps_df.empty:
        return pd.DataFrame()

    grouped = (
        steps_df.groupby("news_id", dropna=False)
        .agg(
            title=("metadata_title", "first"),
            steps=("rewrite_status", "size"),
            successes=("rewrite_status", lambda values: int((values == "success").sum())),
            errors=("rewrite_status", lambda values: int((values == "error").sum())),
            max_stdi_vs_original=("stdi_vs_original", "max"),
            max_stdi_incremental=("stdi_incremental", "max"),
        )
        .reset_index()
        .sort_values(["errors", "max_stdi_vs_original", "news_id"], ascending=[False, False, True])
    )
    return grouped
