from __future__ import annotations

from dataclasses import asdict

from misinformation_simulation.config.prompts import (
    STRUCTURED_JUDGE_PROMPT_TEMPLATE,
    STRUCTURED_TOPIC_PROMPT_TEMPLATE,
)
from misinformation_simulation.topic_drift.extraction import _build_topic_structure
from misinformation_simulation.topic_drift.models import topic_structure_to_dict


def test_legacy_passages_are_ignored_during_deserialization_and_deduplication():
    relation = {
        "subject": "Agency",
        "action": "helps",
        "object": "residents",
        "polarity": "affirmed",
        "duration_status": "absent",
    }
    structure = _build_topic_structure(
        {
            "schema_version": 2,
            "main_topic": "Public services",
            "subtopics": [],
            "central_entities": ["Agency", "residents"],
            "central_relations": [
                {**relation, "evidence": "First legacy passage"},
                {**relation, "evidence": "Second legacy passage"},
            ],
            "has_internal_contradiction": False,
            "internal_contradiction_score": 0,
            "opinions": [],
        }
    )
    assert len(structure.central_relations) == 1
    assert "evidence" not in asdict(structure.central_relations[0])
    assert "evidence" not in topic_structure_to_dict(structure)["central_relations"][0]


def test_prompts_do_not_request_supporting_passage_fields():
    assert "evidence" not in STRUCTURED_TOPIC_PROMPT_TEMPLATE
    assert "- evidence:" not in STRUCTURED_JUDGE_PROMPT_TEMPLATE
    assert "Also return evidence" not in STRUCTURED_JUDGE_PROMPT_TEMPLATE
