from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict
from typing import Any

from misinformation_simulation.topic_drift.models import TopicRelation

POLARITY_WEIGHT = 0.20
DURATION_WEIGHT = 0.20
_DURATION_UNITS = {
    "second": 1,
    "seconds": 1,
    "segundo": 1,
    "segundos": 1,
    "s": 1,
    "minute": 60,
    "minutes": 60,
    "minuto": 60,
    "minutos": 60,
    "min": 60,
    "hour": 3600,
    "hours": 3600,
    "hora": 3600,
    "horas": 3600,
    "h": 3600,
    "day": 86400,
    "days": 86400,
    "dia": 86400,
    "dias": 86400,
    "week": 604800,
    "weeks": 604800,
    "semana": 604800,
    "semanas": 604800,
}


def normalize_duration(relation: TopicRelation) -> float | None:
    unit = unicodedata.normalize("NFKD", relation.duration_unit or "").casefold().strip()
    factor = _DURATION_UNITS.get(unit)
    value = relation.duration_value
    if factor is None or value is None or isinstance(value, bool):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    normalized = value * factor
    return normalized if math.isfinite(normalized) else None


def compare_duration(left: TopicRelation, right: TopicRelation) -> dict[str, Any]:
    statuses = (left.duration_status, right.duration_status)
    result: dict[str, Any] = {"status": "valid", "distance": 0.0, "percentage": None}
    if statuses == ("absent", "absent"):
        result["change"] = "absent_both"
        return result
    if any(status not in {"absent", "exact"} for status in statuses):
        return {**result, "status": "partial", "distance": None, "change": "unknown"}
    values = (normalize_duration(left), normalize_duration(right))
    if any(
        status == "exact" and value is None for status, value in zip(statuses, values, strict=True)
    ):
        return {**result, "status": "partial", "distance": None, "change": "unsupported"}
    if "absent" in statuses:
        return {
            **result,
            "distance": 1.0,
            "change": "addition" if statuses[0] == "absent" else "omission",
        }
    reference, rewrite = values
    if reference == 0:
        return {**result, "distance": 0.0 if rewrite == 0 else 1.0, "change": "zero_reference"}
    relative = abs(rewrite - reference) / reference
    return {
        **result,
        "distance": min(relative, 1.0),
        "percentage": 100 * relative,
        "change": "numeric",
    }


def adjust_relation_distance(
    semantic_distance: float,
    left: TopicRelation,
    right: TopicRelation,
    *,
    polarity_weight: float = POLARITY_WEIGHT,
    duration_weight: float = DURATION_WEIGHT,
) -> dict[str, Any]:
    if not all(
        math.isfinite(value) and 0 <= value <= 1
        for value in (semantic_distance, polarity_weight, duration_weight)
    ):
        raise ValueError("Distances and contribution weights must be in [0, 1].")
    duration = compare_duration(left, right)
    known = left.polarity in {"affirmed", "negated"} and right.polarity in {"affirmed", "negated"}
    delta = int(left.polarity != right.polarity) if known else None
    adjusted = (
        (1 - polarity_weight) * semantic_distance + polarity_weight * delta if known else None
    )
    distance = (
        adjusted + (1 - adjusted) * duration_weight * duration["distance"]
        if adjusted is not None and duration["status"] == "valid"
        else None
    )
    return {
        "semantic_distance": semantic_distance,
        "delta_p": delta,
        "polarity_adjusted_distance": adjusted,
        "polarity_adjustment": (adjusted - semantic_distance if adjusted is not None else None),
        "duration": duration,
        "distance": distance,
        "status": "valid" if distance is not None else "partial",
        "reference": asdict(left),
        "rewrite": asdict(right),
    }
