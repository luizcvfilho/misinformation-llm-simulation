from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from misinformation_simulation.topic_drift.extraction import _normalize_relation, _normalize_scalar
from misinformation_simulation.topic_drift.models import TopicStructure


def _jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    intersection = left & right
    return 1.0 - (len(intersection) / len(union))


def _normalized_non_empty_scalars(values: Sequence[str]) -> set[str]:
    return {normalized for item in values if (normalized := _normalize_scalar(item))}


def calculate_stdi(
    original_structure: TopicStructure,
    compared_structure: TopicStructure,
) -> dict[str, float]:
    original_main_topic = _normalize_scalar(original_structure.main_topic)
    compared_main_topic = _normalize_scalar(compared_structure.main_topic)

    theme_drift = 0.0 if original_main_topic == compared_main_topic else 1.0
    subtopic_drift = _jaccard_distance(
        _normalized_non_empty_scalars(original_structure.subtopics),
        _normalized_non_empty_scalars(compared_structure.subtopics),
    )
    entity_drift = _jaccard_distance(
        {
            _normalize_scalar(item)
            for item in original_structure.central_entities
            if _normalize_scalar(item)
        },
        {
            _normalize_scalar(item)
            for item in compared_structure.central_entities
            if _normalize_scalar(item)
        },
    )
    relation_drift = _jaccard_distance(
        {
            _normalize_relation(item.subject, item.action, item.object)
            for item in original_structure.central_relations
            if all(_normalize_relation(item.subject, item.action, item.object))
        },
        {
            _normalize_relation(item.subject, item.action, item.object)
            for item in compared_structure.central_relations
            if all(_normalize_relation(item.subject, item.action, item.object))
        },
    )

    stdi = 0.2 * theme_drift + 0.2 * subtopic_drift + 0.2 * entity_drift + 0.4 * relation_drift

    return {
        "theme_drift": round(theme_drift, 6),
        "subtopic_drift": round(subtopic_drift, 6),
        "entity_drift": round(entity_drift, 6),
        "relation_drift": round(relation_drift, 6),
        "stdi": round(stdi, 6),
    }


def calculate_stdi_chain_metrics(
    structures: Sequence[TopicStructure],
    *,
    version_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if len(structures) < 2:
        return []

    labels = (
        list(version_labels)
        if version_labels is not None
        else [f"version_{i}" for i in range(len(structures))]
    )
    if len(labels) != len(structures):
        raise ValueError("'version_labels' must have the same length as 'structures'.")

    original_structure = structures[0]
    previous_structure = structures[0]
    cumulative_stdi = 0.0
    metrics: list[dict[str, Any]] = []

    for index in range(1, len(structures)):
        current_structure = structures[index]
        vs_original = calculate_stdi(original_structure, current_structure)
        incremental = calculate_stdi(previous_structure, current_structure)
        cumulative_stdi += incremental["stdi"]

        metrics.append(
            {
                "version_label": labels[index],
                "reference_original_label": labels[0],
                "reference_incremental_label": labels[index - 1],
                "stdi_vs_original": vs_original["stdi"],
                "theme_drift_vs_original": vs_original["theme_drift"],
                "subtopic_drift_vs_original": vs_original["subtopic_drift"],
                "entity_drift_vs_original": vs_original["entity_drift"],
                "relation_drift_vs_original": vs_original["relation_drift"],
                "stdi_incremental": incremental["stdi"],
                "theme_drift_incremental": incremental["theme_drift"],
                "subtopic_drift_incremental": incremental["subtopic_drift"],
                "entity_drift_incremental": incremental["entity_drift"],
                "relation_drift_incremental": incremental["relation_drift"],
                "stdi_cumulative": round(cumulative_stdi, 6),
            }
        )
        previous_structure = current_structure

    return metrics
