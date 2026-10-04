from __future__ import annotations

import numpy as np

from misinformation_simulation.topic_drift.cluster_comparison import (
    ClusterSTDIComparator,
    TopicStructurePair,
)
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure


class KeywordEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [
                [
                    float("econom" in text.casefold()),
                    float("health" in text.casefold()),
                    float("bank" in text.casefold()),
                    float("minister" in text.casefold()),
                    1.0,
                ]
                for text in texts
            ]
        )


def _structure(
    topic: str,
    entity: str,
    relation_action: str,
) -> TopicStructure:
    return TopicStructure(
        main_topic=topic,
        subtopics=[topic],
        central_entities=[entity],
        central_relations=[TopicRelation(entity, relation_action, "policy")],
    )


def test_cluster_comparison_preserves_identical_structures() -> None:
    structure = _structure("economy policy", "Central Bank", "announces")
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder(), random_state=1).fit(
        [TopicStructurePair("pair_1", structure, structure)]
    )

    result = comparator.compare(structure, structure)

    assert result.component_drifts == {
        "theme_drift": 0.0,
        "subtopic_drift": 0.0,
        "entity_drift": 0.0,
        "relation_drift": 0.0,
    }
    assert result.details["theme"]["original_cluster"] == 0


def test_cluster_comparison_detects_different_structures() -> None:
    original = _structure("economy policy", "Central Bank", "announces")
    modified = _structure("health policy", "Health Minister", "cancels")
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder(), random_state=1).fit(
        [TopicStructurePair("pair_1", original, modified)]
    )

    result = comparator.compare(original, modified)

    assert result.component_drifts["theme_drift"] > 0.0
    assert result.component_drifts["entity_drift"] > 0.0
    assert result.component_drifts["relation_drift"] > 0.0
    assert {row["component"] for row in comparator.artifact_rows()} == {
        "theme",
        "subtopic",
        "entity",
        "relation",
    }


def test_cluster_comparison_preserves_similar_topics_from_legacy_domain_payloads() -> None:
    from misinformation_simulation.topic_drift.extraction import _build_topic_structure

    original = _build_topic_structure(
        {
            "main_topic": "ICE arrest of a gang member",
            "topic_domain": "crime_law_and_justice",
        }
    )
    modified = _build_topic_structure(
        {
            "main_topic": "ICE arrest of a gang member and DHS criticism",
            "topic_domain": "politics",
        }
    )
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder()).fit(
        [TopicStructurePair("legacy_pair", original, modified)]
    )

    result = comparator.compare(original, modified)

    assert result.component_drifts["theme_drift"] == 0.0
    assert result.details["theme"]["embedding_similarity"] == 1.0
    assert "domain_gate_applied" not in result.details["theme"]
    assert "domain_match" not in result.details["theme"]


def test_cluster_comparison_uses_direct_theme_similarity() -> None:
    original = _structure(
        "economy policy",
        "Central Bank",
        "announces",
    )
    modified = _structure(
        "health policy",
        "Health Minister",
        "cancels",
    )
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder(), random_state=1).fit(
        [TopicStructurePair("pair_1", original, modified)]
    )

    result = comparator.compare(original, modified)

    assert result.component_drifts["theme_drift"] == 0.5
    assert result.details["theme"]["embedding_similarity"] == 0.5
