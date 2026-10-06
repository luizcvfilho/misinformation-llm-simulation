from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NumericValue:
    role: str
    value: float | str | None
    unit: str | None = None
    kind: str = "number"
    status: str = "exact"
    expression: str | None = None


@dataclass(slots=True)
class TopicRelation:
    subject: str
    action: str
    object: str
    polarity: str | None = None
    predicate: str | None = None
    negation_scope: str | None = None
    signed_action: str | None = None
    base_action: str | None = None
    duration_status: str = "unknown"
    duration_value: float | None = None
    duration_unit: str | None = None
    duration_expression: str | None = None
    assertion_type: str | None = None
    numeric_values: list[NumericValue] | None = None


@dataclass(slots=True)
class TopicStructure:
    main_topic: str | None
    subtopics: list[str]
    central_entities: list[str]
    central_relations: list[TopicRelation]
    narrative_frame: str | None = None
    has_internal_contradiction: bool = False
    internal_contradiction_score: float = 0.0
    schema_version: int = 1
    extraction_status: str = "unavailable"
    extraction_issues: list[str] = field(default_factory=list)
    opinions: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


def empty_topic_structure() -> TopicStructure:
    return TopicStructure(
        main_topic=None,
        subtopics=[],
        central_entities=[],
        central_relations=[],
        narrative_frame=None,
        has_internal_contradiction=False,
        internal_contradiction_score=0.0,
    )


def serialize_relations(relations: list[TopicRelation]) -> str:
    return json.dumps([asdict(item) for item in relations], ensure_ascii=False)


def topic_structure_to_dict(structure: TopicStructure) -> dict[str, Any]:
    return {
        "main_topic": structure.main_topic,
        "subtopics": list(structure.subtopics),
        "central_entities": list(structure.central_entities),
        "central_relations": [asdict(item) for item in structure.central_relations],
        "narrative_frame": structure.narrative_frame,
        "has_internal_contradiction": structure.has_internal_contradiction,
        "internal_contradiction_score": structure.internal_contradiction_score,
        "schema_version": structure.schema_version,
        "extraction_status": structure.extraction_status,
        "extraction_issues": list(structure.extraction_issues),
        "opinions": list(structure.opinions),
        "provenance": dict(structure.provenance),
    }


def flatten_topic_structure(structure: TopicStructure, *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_main_topic": structure.main_topic,
        f"{prefix}_subtopics": json.dumps(structure.subtopics, ensure_ascii=False),
        f"{prefix}_central_entities": json.dumps(
            structure.central_entities,
            ensure_ascii=False,
        ),
        f"{prefix}_central_relations": serialize_relations(structure.central_relations),
        f"{prefix}_narrative_frame": structure.narrative_frame,
        f"{prefix}_has_internal_contradiction": structure.has_internal_contradiction,
        f"{prefix}_internal_contradiction_score": structure.internal_contradiction_score,
        f"{prefix}_json": json.dumps(topic_structure_to_dict(structure), ensure_ascii=False),
    }
