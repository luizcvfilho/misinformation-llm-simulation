from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.config.prompts import (
    STRUCTURED_JUDGE_PROMPT_TEMPLATE,
    STRUCTURED_TOPIC_PROMPT_TEMPLATE,
)
from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
from misinformation_simulation.topic_drift.comparison_workflow import run_comparison_workflow
from misinformation_simulation.topic_drift.extraction import _build_topic_structure
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.qualifiers import adjust_relation_distance
from misinformation_simulation.topic_drift.semantic_comparison import (
    parse_structured_semantic_comparison,
)
from misinformation_simulation.topic_drift.structured_comparison import (
    CONTENT_COMPONENTS,
    StructuredEmbeddingComparator,
    structure_issues,
)


class ConstantEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 2))


def payload(action="postponed", **changes):
    return {
        "schema_version": 2,
        "main_topic": "Negotiations",
        "subtopics": [],
        "central_entities": ["government", "union"],
        "central_relations": [
            {
                "subject": "government",
                "action": action,
                "object": "talks",
                "base_action": "postponed",
                "signed_action": action,
                "polarity": "unknown",
                "duration_status": "absent",
                **changes,
            }
        ],
        "has_internal_contradiction": False,
        "internal_contradiction_score": 0,
        "opinions": [],
    }


@pytest.mark.parametrize(
    "action,expected",
    [
        ("may have postponed", "affirmed"),
        ("may not have postponed", "negated"),
        ("should negotiate", "affirmed"),
        ("refused", "affirmed"),
        ("didn't postpone", "negated"),
        ("pode ter adiado", "affirmed"),
        ("pode não ter adiado", "negated"),
    ],
)
def test_uncertainty_is_not_unknown_polarity(action, expected):
    result = _build_topic_structure(payload(action))
    assert result.central_relations[0].polarity == expected
    assert result.extraction_status == "valid"
    assert result.extraction_issues
    assert not structure_issues(result)


def test_optional_metadata_and_intransitive_actions_do_not_block_cluster():
    original = _build_topic_structure(payload("should negotiate", object="", polarity="affirmed"))
    original.extraction_status = "partial"
    original.extraction_issues = ["Missing optional annotation"]
    modified = _build_topic_structure(payload("should negotiate", object="", polarity="affirmed"))
    comparator = StructuredEmbeddingComparator(embedder=ConstantEmbedder()).fit(
        [TopicStructurePair("pair", original, modified)]
    )
    result = comparator.compare_structured(original, modified)
    assert result["status"] == "valid"
    assert result["components"]["relation_drift"] == 0
    assert result["warnings"] == original.extraction_issues


def test_ambiguous_duration_keeps_distance_and_diagnostic_without_fabricating_quantity():
    left = TopicRelation("Agency", "helps", "residents", polarity="affirmed")
    right = replace(left, duration_status="ambiguous", duration_expression="about two days")
    result = adjust_relation_distance(0.5, left, right)
    assert result["distance"] == 0.4
    assert result["duration"]["distance"] is None
    assert result["warnings"]


def test_judge_needs_scores_but_no_supporting_passage_fields():
    raw = json.dumps(
        {
            **dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 0.5),
            "rationales": {},
        }
    )
    result = parse_structured_semantic_comparison(raw)
    assert all(value == 0.5 for value in result.component_drifts.values())
    assert result.warnings


def test_removed_field_is_not_requested_stored_or_exported():
    historical = payload(polarity="affirmed")
    historical["central_relations"][0]["evidence"] = "Legacy passage"
    historical["central_relations"].append(
        {
            **historical["central_relations"][0],
            "evidence": "Different legacy passage",
        }
    )
    structure = _build_topic_structure(historical)
    assert len(structure.central_relations) == 1
    assert not hasattr(structure.central_relations[0], "evidence")
    exported = topic_structure_to_dict(structure)
    assert "evidence" not in exported["central_relations"][0]
    assert "evidence" not in STRUCTURED_TOPIC_PROMPT_TEMPLATE
    assert "- evidence:" not in STRUCTURED_JUDGE_PROMPT_TEMPLATE
    judgment = parse_structured_semantic_comparison(
        json.dumps(
            {
                **dict.fromkeys((*CONTENT_COMPONENTS, "contradiction_drift"), 0.5),
                "rationales": dict.fromkeys(
                    (*CONTENT_COMPONENTS, "contradiction_drift"), "Changed"
                ),
                "evidence": {"theme_drift": "Legacy passage"},
            }
        )
    )
    assert "evidence" not in asdict(judgment)
    assert not judgment.warnings


def test_failed_extraction_still_cannot_be_scored_as_an_empty_structure():
    failed = TopicStructure("topic", [], [], [], schema_version=2, extraction_status="failed")
    assert "Extraction failed" in structure_issues(failed)


def test_standalone_cluster_uses_binary_polarity_without_a_judge():
    calls = []

    def extractor(**kwargs):
        calls.append(kwargs)
        return _build_topic_structure(payload(kwargs["text"]))

    def judge(**kwargs):
        pytest.fail("Cluster-only evaluation must not call a judge")

    result = run_comparison_workflow(
        pd.DataFrame([{"original_text": "postponed", "modified_text": "did not postpone"}]),
        method="cluster",
        extraction_fn=extractor,
        llm_comparison_fn=judge,
        embedder=ConstantEmbedder(),
        reuse_structures=False,
    )
    assert all(call["structured"] for call in calls)
    assert result.results.iloc[0]["relation_drift"] == pytest.approx(0.2)
    assert result.results.iloc[0]["comparison_status"] == "success"
    assert result.manifest["cluster"]["comparison_version"] == "cluster_v4"
