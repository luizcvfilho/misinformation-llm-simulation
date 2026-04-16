from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TopicRelation:
    subject: str
    action: str
    object: str


@dataclass(slots=True)
class TopicStructure:
    main_topic: str | None
    subtopics: list[str]
    central_entities: list[str]
    central_relations: list[TopicRelation]
    narrative_frame: str | None = None


def empty_topic_structure() -> TopicStructure:
    return TopicStructure(
        main_topic=None,
        subtopics=[],
        central_entities=[],
        central_relations=[],
        narrative_frame=None,
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
        f"{prefix}_json": json.dumps(topic_structure_to_dict(structure), ensure_ascii=False),
    }
