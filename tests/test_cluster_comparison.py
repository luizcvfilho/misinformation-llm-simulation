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
    *,
    domain: str | None = None,
) -> TopicStructure:
    return TopicStructure(
        main_topic=topic,
        subtopics=[topic],
        central_entities=[entity],
        central_relations=[TopicRelation(entity, relation_action, "policy")],
        topic_domain=domain,
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


def test_cluster_comparison_gates_theme_when_domains_differ() -> None:
    original = _structure(
        "Calgary Flames season outlook",
        "Calgary Flames",
        "evaluate",
        domain="sports",
    )
    modified = _structure(
        "Calgary municipal budget process",
        "Calgary City Council",
        "evaluate",
        domain="government_and_public_policy",
    )
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder(), random_state=1).fit(
        [TopicStructurePair("pair_1", original, modified)]
    )

    result = comparator.compare(original, modified)

    assert result.component_drifts["theme_drift"] == 1.0
    assert result.details["theme"]["domain_gate_applied"] == 1
    assert result.details["theme"]["domain_match"] == 0


def test_cluster_comparison_uses_direct_theme_similarity_when_domains_match() -> None:
    original = _structure(
        "economy policy",
        "Central Bank",
        "announces",
        domain="business_and_economy",
    )
    modified = _structure(
        "health policy",
        "Health Minister",
        "cancels",
        domain="business_and_economy",
    )
    comparator = ClusterSTDIComparator(embedder=KeywordEmbedder(), random_state=1).fit(
        [TopicStructurePair("pair_1", original, modified)]
    )

    result = comparator.compare(original, modified)

    assert result.component_drifts["theme_drift"] == 0.5
    assert result.details["theme"]["embedding_similarity"] == 0.5
    assert result.details["theme"]["domain_gate_applied"] == 0
