from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import date
from typing import Any

from misinformation_simulation.topic_drift.models import NumericValue, TopicRelation


def _normalize(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "").casefold().strip()
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split())


def _role(value: NumericValue) -> tuple[str, ...]:
    return tuple(sorted(re.findall(r"\w+", _normalize(value.role).replace("_", " "))))


_UNITS: dict[str, tuple[str, float]] = {}
for dimension, scale, aliases in (
    ("time", 1.0, "s|second|seconds|segundo|segundos"),
    ("time", 60.0, "min|minute|minutes|minuto|minutos"),
    ("time", 3600.0, "h|hour|hours|hora|horas"),
    ("time", 86400.0, "day|days|dia|dias"),
    ("time", 604800.0, "week|weeks|semana|semanas"),
    ("length", 1.0, "m|meter|meters|metre|metres|metro|metros"),
    ("length", 0.01, "cm|centimeter|centimeters|centimetro|centimetros"),
    ("length", 0.001, "mm|millimeter|millimeters|milimetro|milimetros"),
    ("length", 1000.0, "km|kilometer|kilometers|quilometro|quilometros"),
    ("mass", 1.0, "kg|kilogram|kilograms|quilograma|quilogramas"),
    ("mass", 0.001, "g|gram|grams|grama|gramas"),
    ("mass", 0.000001, "mg|milligram|milligrams|miligrama|miligramas"),
    ("volume", 1.0, "l|liter|liters|litre|litres|litro|litros"),
    ("volume", 0.001, "ml|milliliter|milliliters|mililitro|mililitros"),
    ("proportion", 0.01, "%|percent|percentage|por cento|porcentagem"),
    ("proportion", 1.0, "fraction|ratio|proportion|fracao|proporcao"),
    ("age", 1.0, "year|years|ano|anos"),
    ("age", 1 / 12, "month|months|mes|meses"),
    ("currency:usd", 1.0, "usd|us dollar|us dollars"),
    ("currency:brl", 1.0, "brl|r$|real|reais"),
    ("currency:eur", 1.0, "eur|euro|euros|€"),
):
    for alias in aliases.split("|"):
        _UNITS[alias] = dimension, scale


def _scalar(value: NumericValue) -> tuple[float, str] | None:
    if isinstance(value.value, bool) or not isinstance(value.value, (float, int)):
        return None
    try:
        number = float(value.value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    unit = _normalize(value.unit)
    kind = _normalize(value.kind)
    if kind == "year" and not unit:
        unit = "year"
    dimension, scale = _UNITS.get(unit, (f"unit:{unit}", 1.0))
    if dimension == "age" and kind != "age":
        # Calendar years/months are not fixed elapsed-time units.
        dimension, scale = "calendar:year" if scale == 1 else "calendar:month", 1.0
    if not unit:
        if kind in {"money", "percentage", "duration", "measurement"}:
            return None
        dimension = "unitless"
    if kind == "money" and not dimension.startswith("currency:"):
        dimension = f"currency:{unit}"
    number *= scale
    return (number, dimension) if math.isfinite(number) else None


def _categorical(value: NumericValue) -> str | None:
    if value.value is None or isinstance(value.value, bool):
        return None
    if isinstance(value.value, float) and not math.isfinite(value.value):
        return None
    text = str(value.value).strip()
    if value.kind == "date":
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    return text or None


def _record(value: NumericValue | None) -> dict[str, Any] | None:
    if value is None:
        return None
    record = asdict(value)
    if isinstance(value.value, float) and not math.isfinite(value.value):
        record["value"] = None
        record["invalid_value"] = str(value.value)
    return record


def _pair_distance(left: NumericValue, right: NumericValue) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "partial", "distance": None, "percentage": None}
    if left.status != "exact" or right.status != "exact":
        return {**result, "change": "inexact"}
    if left.kind in {"date", "identifier"} or right.kind in {"date", "identifier"}:
        if left.kind != right.kind:
            return {**result, "change": "incompatible_kind"}
        a, b = _categorical(left), _categorical(right)
        if a is None or b is None:
            return {**result, "change": "unavailable_categorical_value"}
        return {**result, "status": "valid", "distance": float(a != b), "change": "categorical"}
    a, b = _scalar(left), _scalar(right)
    if a is None or b is None:
        return {**result, "change": "unavailable_value_or_unit"}
    if a[1] != b[1]:
        return {**result, "change": "incompatible_units"}
    reference, rewrite = a[0], b[0]
    if reference == 0:
        return {
            **result,
            "status": "valid",
            "distance": float(rewrite != 0),
            "change": "zero_reference",
            "normalized_values": [reference, rewrite],
        }
    relative = abs(rewrite / abs(reference) - math.copysign(1.0, reference))
    percentage = 100 * relative
    return {
        **result,
        "status": "valid",
        "distance": min(relative, 1.0),
        "percentage": percentage if math.isfinite(percentage) else None,
        "percentage_overflow": not math.isfinite(percentage),
        "change": "numeric",
        "normalized_values": [reference, rewrite],
    }


def compare_numeric_values(left: TopicRelation, right: TopicRelation) -> dict[str, Any]:
    """Compare contextual numeric slots within an already aligned relation pair."""
    a, b = left.numeric_values, right.numeric_values
    result: dict[str, Any] = {
        "status": "partial",
        "distance": None,
        "aggregation": "maximum",
        "mode": "numeric_values",
        "comparisons": [],
    }
    if a is None or b is None:
        return {**result, "warnings": ["Numeric values unavailable in one or both extractions"]}
    keys = [_role(value) for value in a + b]
    if any(not key for key in keys) or any(
        count > 1 for values in (a, b) for count in Counter(_role(v) for v in values).values()
    ):
        return {**result, "warnings": ["Missing or duplicate numeric roles prevent alignment"]}
    slots_a, slots_b = {_role(v): v for v in a}, {_role(v): v for v in b}
    details = []
    for key in sorted(slots_a.keys() | slots_b.keys()):
        original, modified = slots_a.get(key), slots_b.get(key)
        if original is None or modified is None:
            present = original or modified
            assert present is not None
            complete = present.status == "exact" and (
                _scalar(present) is not None
                or present.kind in {"date", "identifier"}
                and _categorical(present) is not None
            )
            comparison = {
                "status": "valid" if complete else "partial",
                "distance": 1.0 if complete else None,
                "percentage": None,
                "change": "addition" if original is None else "omission",
            }
        else:
            comparison = _pair_distance(original, modified)
        details.append(
            {
                "role": " ".join(key),
                **comparison,
                "reference": _record(original),
                "rewrite": _record(modified),
            }
        )
    incomplete = [d for d in details if d["status"] != "valid"]
    return {
        **result,
        "status": "partial" if incomplete else "valid",
        "distance": None if incomplete else max((d["distance"] for d in details), default=0.0),
        "comparisons": details,
        "warnings": [
            f"Numeric adjustment unavailable for {d['role']}: {d['change']}" for d in incomplete
        ],
    }
