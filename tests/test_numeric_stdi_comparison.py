from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
from misinformation_simulation.topic_drift.extraction import _build_topic_structure
from misinformation_simulation.topic_drift.models import (
    NumericValue,
    TopicRelation,
    TopicStructure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.numeric_comparison import compare_numeric_values
from misinformation_simulation.topic_drift.qualifiers import adjust_relation_distance
from misinformation_simulation.topic_drift.structured_comparison import (
    StructuredEmbeddingComparator,
    complete_stdi,
)


def relation(*values, **changes):
    return TopicRelation(
        "commission",
        "removed",
        "names",
        polarity="affirmed",
        duration_status="absent",
        numeric_values=list(values),
        **changes,
    )


def quantity(value, **changes):
    return NumericValue("removed_names", value, unit="name", kind="count", **changes)


@pytest.mark.parametrize(
    "a,b,distance,percentage",
    [
        (120, 1200, 1, 900),
        (1200, 120, 0.9, 90),
        (100, 110, 0.1, 10),
        (0, 0, 0, None),
        (0, 5, 1, None),
        (-10, -5, 0.5, 50),
        (-10, 10, 1, 200),
        (0.2, 0.3, 0.5, 50),
    ],
)
def test_general_numeric_relative_distance(a, b, distance, percentage):
    result = compare_numeric_values(relation(quantity(a)), relation(quantity(b)))
    assert result["status"] == "valid"
    assert result["distance"] == pytest.approx(distance)
    actual = result["comparisons"][0]["percentage"]
    assert actual is None if percentage is None else actual == pytest.approx(percentage)


@pytest.mark.parametrize(
    "left,right",
    [
        (
            NumericValue("delay", 2, "dias", "duration"),
            NumericValue("delay", 48, "hours", "duration"),
        ),
        (
            NumericValue("length", 2, "km", "measurement"),
            NumericValue("length", 2000, "m", "measurement"),
        ),
        (
            NumericValue("rate", 5, "%", "percentage"),
            NumericValue("rate", 0.05, "fraction", "percentage"),
        ),
        (NumericValue("age", 2, "years", "age"), NumericValue("age", 24, "months", "age")),
        (
            NumericValue("price", 100, "USD", "money"),
            NumericValue("price", 100, "US dollars", "money"),
        ),
    ],
)
def test_equivalent_units_are_neutral(left, right):
    assert compare_numeric_values(relation(left), relation(right))["distance"] == pytest.approx(0)


def test_maximum_change_uses_roles_instead_of_array_position():
    population = NumericValue("registered_voters", 5000, "person", "count")
    left = relation(quantity(120), population)
    right = relation(population, replace(quantity(1200), role="names_removed"))
    result = compare_numeric_values(left, right)
    assert result["distance"] == 1
    assert len(result["comparisons"]) == 2
    assert result["aggregation"] == "maximum"
    assert adjust_relation_distance(0, left, right)["distance"] == pytest.approx(0.2)


def test_duration_is_not_applied_twice():
    left = relation(
        NumericValue("closure_duration", 2, "days", "duration"),
        duration_value=2,
        duration_unit="days",
    )
    right = relation(
        NumericValue("closure_duration", 20, "days", "duration"),
        duration_value=20,
        duration_unit="days",
    )
    left.duration_status = right.duration_status = "exact"
    result = adjust_relation_distance(0, left, right)
    assert result["duration"]["distance"] == result["numeric"]["distance"] == 1
    assert result["distance"] == pytest.approx(0.2)


def test_historical_duration_fallback_does_not_invent_counts():
    left = replace(
        relation(),
        numeric_values=None,
        duration_status="exact",
        duration_value=2,
        duration_unit="days",
    )
    right = replace(left, duration_value=20)
    result = adjust_relation_distance(0, left, right)
    assert result["numeric"]["mode"] == "legacy_duration"
    assert result["distance"] == pytest.approx(0.2)
    result = compare_numeric_values(left, relation(quantity(120)))
    assert result["distance"] is None and result["status"] == "partial"


def test_missing_numbers_are_distinct_from_explicit_absence():
    assert compare_numeric_values(relation(), relation())["distance"] == 0
    left = replace(relation(), numeric_values=None)
    result = adjust_relation_distance(0.5, left, relation())
    assert result["numeric"]["distance"] is None
    assert result["distance"] == pytest.approx(0.4)
    assert result["warnings"]


@pytest.mark.parametrize(
    "left,right", [(relation(), relation(quantity(10))), (relation(quantity(10)), relation())]
)
def test_explicit_numeric_addition_and_omission(left, right):
    assert compare_numeric_values(left, right)["distance"] == 1


@pytest.mark.parametrize(
    "a,b,kind,expected",
    [
        ("2026-10-05", "2026-10-05", "date", 0),
        ("2026-10-05", "2026-10-06", "date", 1),
        ("00120", "00120", "identifier", 0),
        ("00120", "01200", "identifier", 1),
        ("0", "0", "identifier", 0),
    ],
)
def test_dates_and_identifiers_use_equality_without_percentage(a, b, kind, expected):
    left, right = NumericValue("event_code", a, kind=kind), NumericValue("event_code", b, kind=kind)
    result = compare_numeric_values(relation(left), relation(right))
    assert result["distance"] == expected
    assert result["comparisons"][0]["percentage"] is None


@pytest.mark.parametrize(
    "left,right",
    [
        (NumericValue("price", 10, "USD", "money"), NumericValue("price", 10, "EUR", "money")),
        (quantity(10, status="approximate"), quantity(20)),
        (quantity(float("nan")), quantity(20)),
        (quantity(True), quantity(20)),
        (
            NumericValue("event_date", "2026-99-99", kind="date"),
            NumericValue("event_date", "2026-10-05", kind="date"),
        ),
    ],
)
def test_unusable_numeric_qualifiers_remain_unavailable(left, right):
    result = adjust_relation_distance(0.5, relation(left), relation(right))
    assert result["numeric"]["distance"] is None
    assert result["distance"] == pytest.approx(0.4)
    assert result["warnings"]
    json.dumps(result["numeric"], allow_nan=False)


def test_duplicate_numeric_roles_are_not_arbitrarily_paired():
    result = compare_numeric_values(relation(quantity(10), quantity(20)), relation(quantity(30)))
    assert result["distance"] is None
    assert result["warnings"]


def test_numeric_overflow_keeps_bounded_distance_and_serializable_details():
    result = compare_numeric_values(relation(quantity(1e-308)), relation(quantity(1e308)))
    assert result["distance"] == 1
    assert result["comparisons"][0]["percentage_overflow"]
    json.dumps(result, allow_nan=False)


def test_numeric_schema_round_trip_and_invalid_values():
    structure = TopicStructure(
        "register", [], ["commission"], [relation(quantity(120))], schema_version=2
    )
    payload = topic_structure_to_dict(structure)
    restored = _build_topic_structure(json.loads(json.dumps(payload)))
    assert restored.central_relations[0].numeric_values == [quantity(120)]
    values = payload["central_relations"][0]["numeric_values"]
    values[0]["value"] = True
    assert _build_topic_structure(payload).central_relations[0].numeric_values[0].value is None
    values[0]["value"] = 10**1000
    assert _build_topic_structure(payload).central_relations[0].numeric_values[0].value is None
    assert structure.central_relations[0].numeric_values == [quantity(120)]


def test_count_change_reaches_the_embedding_branch_and_final_stdi():
    class Embedder:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    unchanged = TopicRelation(
        "voters", "requested", "appeal", polarity="affirmed", numeric_values=[]
    )
    left = TopicStructure(
        "register",
        [],
        ["commission", "voters"],
        [relation(quantity(120)), unchanged],
        schema_version=2,
    )
    right = replace(left, central_relations=[relation(quantity(1200)), unchanged])
    comparator = StructuredEmbeddingComparator(embedder=Embedder()).fit(
        [TopicStructurePair("count", left, right)]
    )
    comparison = comparator.compare_structured(left, right)
    assert comparison["status"] == "valid"
    assert comparison["components"]["relation_drift"] == pytest.approx(0.1)
    metrics = complete_stdi(comparison["components"], {"status": "valid", "vad_drift": 0})
    assert metrics["stdi"] == pytest.approx(0.025)
    assert left.central_relations[0].numeric_values[0].value == 120
