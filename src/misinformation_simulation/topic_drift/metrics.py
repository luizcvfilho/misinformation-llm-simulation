from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.extraction import _normalize_relation, _normalize_scalar
from misinformation_simulation.topic_drift.models import TopicStructure

DEFAULT_STDI_WEIGHTS = {
    "theme_drift": 0.2,
    "subtopic_drift": 0.2,
    "entity_drift": 0.2,
    "relation_drift": 0.4,
}
DEFAULT_STDI_WITH_VAD_WEIGHTS = {
    "theme_drift": 0.15,
    "subtopic_drift": 0.15,
    "entity_drift": 0.15,
    "relation_drift": 0.3,
    "vad_drift": 0.25,
}
DEFAULT_VAD_SCORE_RANGE = 4.0


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


def calculate_vad_drift(
    original_vad: VADScore | None,
    compared_vad: VADScore | None,
    *,
    score_range: float = DEFAULT_VAD_SCORE_RANGE,
) -> dict[str, float]:
    if score_range <= 0:
        raise ValueError("'score_range' must be greater than zero.")

    if original_vad is None or compared_vad is None:
        return {
            "valence_drift": 0.0,
            "arousal_drift": 0.0,
            "dominance_drift": 0.0,
            "vad_drift": 0.0,
        }

    drifts: dict[str, float] = {}
    for dimension in ("valence", "arousal", "dominance"):
        original_value = getattr(original_vad, dimension)
        compared_value = getattr(compared_vad, dimension)
        if original_value is None or compared_value is None:
            drifts[f"{dimension}_drift"] = 0.0
            continue

        normalized_drift = min(
            abs(float(compared_value) - float(original_value)) / score_range,
            1.0,
        )
        drifts[f"{dimension}_drift"] = round(normalized_drift, 6)

    drifts["vad_drift"] = round(
        (drifts["valence_drift"] + drifts["arousal_drift"] + drifts["dominance_drift"]) / 3.0,
        6,
    )
    return drifts


def calculate_stdi(
    original_structure: TopicStructure,
    compared_structure: TopicStructure,
    *,
    original_vad: VADScore | None = None,
    compared_vad: VADScore | None = None,
    score_range: float = DEFAULT_VAD_SCORE_RANGE,
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

    vad_metrics = calculate_vad_drift(
        original_vad,
        compared_vad,
        score_range=score_range,
    )
    weights = (
        DEFAULT_STDI_WITH_VAD_WEIGHTS
        if original_vad is not None and compared_vad is not None
        else DEFAULT_STDI_WEIGHTS
    )
    stdi = (
        weights["theme_drift"] * theme_drift
        + weights["subtopic_drift"] * subtopic_drift
        + weights["entity_drift"] * entity_drift
        + weights["relation_drift"] * relation_drift
        + weights.get("vad_drift", 0.0) * vad_metrics["vad_drift"]
    )

    return {
        "theme_drift": round(theme_drift, 6),
        "subtopic_drift": round(subtopic_drift, 6),
        "entity_drift": round(entity_drift, 6),
        "relation_drift": round(relation_drift, 6),
        "valence_drift": vad_metrics["valence_drift"],
        "arousal_drift": vad_metrics["arousal_drift"],
        "dominance_drift": vad_metrics["dominance_drift"],
        "vad_drift": vad_metrics["vad_drift"],
        "stdi": round(stdi, 6),
    }


def calculate_stdi_chain_metrics(
    structures: Sequence[TopicStructure],
    *,
    version_labels: Sequence[str] | None = None,
    vad_scores: Sequence[VADScore | None] | None = None,
    score_range: float = DEFAULT_VAD_SCORE_RANGE,
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
    if vad_scores is not None and len(vad_scores) != len(structures):
        raise ValueError("'vad_scores' must have the same length as 'structures'.")

    original_structure = structures[0]
    previous_structure = structures[0]
    original_vad = vad_scores[0] if vad_scores is not None else None
    previous_vad = vad_scores[0] if vad_scores is not None else None
    cumulative_stdi = 0.0
    metrics: list[dict[str, Any]] = []

    for index in range(1, len(structures)):
        current_structure = structures[index]
        current_vad = vad_scores[index] if vad_scores is not None else None
        vs_original = calculate_stdi(
            original_structure,
            current_structure,
            original_vad=original_vad,
            compared_vad=current_vad,
            score_range=score_range,
        )
        incremental = calculate_stdi(
            previous_structure,
            current_structure,
            original_vad=previous_vad,
            compared_vad=current_vad,
            score_range=score_range,
        )
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
                "valence_drift_vs_original": vs_original["valence_drift"],
                "arousal_drift_vs_original": vs_original["arousal_drift"],
                "dominance_drift_vs_original": vs_original["dominance_drift"],
                "vad_drift_vs_original": vs_original["vad_drift"],
                "stdi_incremental": incremental["stdi"],
                "theme_drift_incremental": incremental["theme_drift"],
                "subtopic_drift_incremental": incremental["subtopic_drift"],
                "entity_drift_incremental": incremental["entity_drift"],
                "relation_drift_incremental": incremental["relation_drift"],
                "valence_drift_incremental": incremental["valence_drift"],
                "arousal_drift_incremental": incremental["arousal_drift"],
                "dominance_drift_incremental": incremental["dominance_drift"],
                "vad_drift_incremental": incremental["vad_drift"],
                "stdi_cumulative": round(cumulative_stdi, 6),
            }
        )
        previous_structure = current_structure
        previous_vad = current_vad

    return metrics
