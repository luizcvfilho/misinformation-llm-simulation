from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.audits import VADScore
from misinformation_simulation.topic_drift import (
    TopicRelation,
    TopicStructure,
    annotate_stdi_for_rewrites,
    calculate_stdi,
    calculate_stdi_chain_metrics,
    calculate_vad_drift,
)
from misinformation_simulation.topic_drift.extraction import _build_topic_structure


def test_calculate_vad_drift_returns_normalized_dimension_scores() -> None:
    drift = calculate_vad_drift(
        VADScore(valence=3.0, arousal=2.5, dominance=2.0),
        VADScore(valence=2.0, arousal=3.5, dominance=3.0),
    )

    assert drift["valence_drift"] == 0.25
    assert drift["arousal_drift"] == 0.25
    assert drift["dominance_drift"] == 0.25
    assert drift["vad_drift"] == 0.25


def test_calculate_stdi_uses_vad_drift_when_topic_structure_is_identical() -> None:
    structure = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
        narrative_frame="economic update",
    )
    contradictory_structure = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
        narrative_frame="economic update",
        has_internal_contradiction=True,
        internal_contradiction_score=1.0,
    )

    metrics_without_vad = calculate_stdi(structure, structure)
    metrics_with_vad = calculate_stdi(
        structure,
        contradictory_structure,
        original_vad=VADScore(valence=3.0, arousal=3.0, dominance=3.0),
        compared_vad=VADScore(valence=2.0, arousal=4.0, dominance=3.5),
    )

    assert metrics_without_vad["stdi"] == 0.0
    assert metrics_with_vad["theme_drift"] == 0.0
    assert metrics_with_vad["contradiction_drift"] == 1.0
    assert metrics_with_vad["vad_drift"] > 0.0
    assert metrics_with_vad["stdi"] > 0.0


def test_calculate_stdi_uses_vad_only_for_remaining_semantic_distance() -> None:
    structure = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
    )

    metrics = calculate_stdi(
        structure,
        structure,
        original_vad=VADScore(valence=3.0, arousal=3.0, dominance=3.0),
        compared_vad=VADScore(valence=2.0, arousal=3.0, dominance=3.0),
    )

    assert metrics["content_drift"] == 0.0
    assert metrics["vad_drift"] == 0.083333
    assert metrics["stdi"] == 0.016667


def test_calculate_stdi_does_not_promote_theme_drift_above_other_components() -> None:
    original = TopicStructure(
        main_topic="government shutdown",
        subtopics=["federal workers"],
        central_entities=["TSA"],
        central_relations=[TopicRelation("TSA", "delays", "payments")],
    )
    compared = TopicStructure(
        main_topic="TSA payroll",
        subtopics=["federal workers"],
        central_entities=["TSA"],
        central_relations=[TopicRelation("TSA", "delays", "payments")],
    )

    metrics = calculate_stdi(
        original,
        compared,
        original_vad=VADScore(valence=3.0, arousal=3.0, dominance=3.0),
        compared_vad=VADScore(valence=3.1, arousal=3.0, dominance=3.0),
    )

    assert metrics["theme_drift"] == 1.0
    assert metrics["content_drift"] == 0.1875
    assert metrics["stdi"] == 0.188854


def test_calculate_stdi_uses_contradiction_as_an_extra_component() -> None:
    original = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
        narrative_frame="economic update",
    )
    compared = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
        narrative_frame="economic update",
        has_internal_contradiction=True,
        internal_contradiction_score=0.5,
    )

    metrics = calculate_stdi(original, compared)

    assert metrics["contradiction_drift"] == 0.5
    assert metrics["stdi"] == 0.1


def test_calculate_stdi_does_not_require_contradiction_for_complete_content_drift() -> None:
    original = TopicStructure(
        main_topic="economy",
        subtopics=["inflation"],
        central_entities=["central bank"],
        central_relations=[TopicRelation("central bank", "raises", "rates")],
    )
    compared = TopicStructure(
        main_topic="football",
        subtopics=["championship"],
        central_entities=["club"],
        central_relations=[TopicRelation("club", "wins", "match")],
    )

    metrics = calculate_stdi(original, compared)

    assert metrics["content_drift"] == 1.0
    assert metrics["contradiction_drift"] == 0.0
    assert metrics["stdi"] == 1.0


def test_calculate_stdi_accepts_semantic_component_overrides() -> None:
    original = TopicStructure(
        main_topic="football transfer",
        subtopics=["player sale"],
        central_entities=["Tottenham"],
        central_relations=[TopicRelation("Tottenham", "sells", "player")],
    )
    compared = TopicStructure(
        main_topic="agreement to sell Tottenham player",
        subtopics=["transfer agreement"],
        central_entities=["Tottenham Hotspur"],
        central_relations=[TopicRelation("Tottenham", "agrees to sell", "player")],
    )

    lexical_metrics = calculate_stdi(original, compared)
    semantic_metrics = calculate_stdi(
        original,
        compared,
        component_overrides={
            "theme_drift": 0.0,
            "subtopic_drift": 0.25,
            "entity_drift": 0.0,
            "relation_drift": 0.0,
        },
    )

    assert lexical_metrics["theme_drift"] == 1.0
    assert semantic_metrics["theme_drift"] == 0.0
    assert semantic_metrics["subtopic_drift"] == 0.25
    assert semantic_metrics["stdi"] < lexical_metrics["stdi"]


def test_calculate_stdi_chain_metrics_propagates_vad_columns() -> None:
    original = TopicStructure(
        main_topic="health",
        subtopics=["vaccines"],
        central_entities=["ministry"],
        central_relations=[TopicRelation("ministry", "announces", "campaign")],
        narrative_frame="public health",
    )
    rewritten = TopicStructure(
        main_topic="health",
        subtopics=["vaccines"],
        central_entities=["ministry"],
        central_relations=[TopicRelation("ministry", "announces", "campaign")],
        narrative_frame="public health",
        has_internal_contradiction=True,
        internal_contradiction_score=0.75,
    )

    metrics = calculate_stdi_chain_metrics(
        [original, rewritten],
        version_labels=["original", "rewrite_1"],
        vad_scores=[
            VADScore(valence=3.2, arousal=2.8, dominance=3.0),
            VADScore(valence=2.6, arousal=3.5, dominance=3.4),
        ],
    )

    assert len(metrics) == 1
    metric_row = metrics[0]
    assert "vad_drift_vs_original" in metric_row
    assert "vad_drift_incremental" in metric_row
    assert "contradiction_drift_vs_original" in metric_row
    assert "contradiction_drift_incremental" in metric_row
    assert metric_row["contradiction_drift_vs_original"] == 0.75
    assert metric_row["contradiction_drift_incremental"] == 0.75
    assert metric_row["vad_drift_vs_original"] > 0.0
    assert metric_row["stdi_vs_original"] > 0.0


def test_build_topic_structure_defaults_missing_contradiction_to_false() -> None:
    structure = _build_topic_structure(
        {
            "main_topic": "economy",
            "subtopics": ["inflation"],
            "central_entities": ["central bank"],
            "central_relations": [
                {"subject": "central bank", "action": "raises", "object": "rates"}
            ],
            "narrative_frame": "economic update",
        }
    )

    assert not structure.has_internal_contradiction
    assert structure.internal_contradiction_score == 0.0


def test_annotate_stdi_for_rewrites_adds_contradiction_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic_structures = iter(
        [
            TopicStructure(
                main_topic="economy",
                subtopics=["inflation"],
                central_entities=["central bank"],
                central_relations=[TopicRelation("central bank", "raises", "rates")],
                narrative_frame="economic update",
            ),
            TopicStructure(
                main_topic="economy",
                subtopics=["inflation"],
                central_entities=["central bank"],
                central_relations=[TopicRelation("central bank", "raises", "rates")],
                narrative_frame="economic update",
                has_internal_contradiction=True,
                internal_contradiction_score=0.25,
            ),
        ]
    )

    def fake_extract_topic_structure(*_args, **_kwargs) -> TopicStructure:
        return next(topic_structures)

    monkeypatch.setattr(
        "misinformation_simulation.topic_drift.annotation.extract_topic_structure",
        fake_extract_topic_structure,
    )
    monkeypatch.setattr(
        "misinformation_simulation.topic_drift.annotation.predict_text_vad",
        lambda *_args, **_kwargs: None,
    )

    df = pd.DataFrame(
        [
            {
                "title": "Economy",
                "description": "Original article text",
                "rewritten_news": "Rewritten text with contradiction",
            }
        ]
    )

    result = annotate_stdi_for_rewrites(df, sleep_seconds=0.0)

    assert "original_has_internal_contradiction" in result.columns
    assert "rewritten_news_has_internal_contradiction" in result.columns
    assert "rewritten_news_internal_contradiction_score" in result.columns
    assert "rewritten_news_contradiction_drift_vs_original" in result.columns
    assert "rewritten_news_contradiction_drift_incremental" in result.columns
    assert not result.at[0, "original_has_internal_contradiction"]
    assert result.at[0, "rewritten_news_has_internal_contradiction"]
    assert result.at[0, "rewritten_news_internal_contradiction_score"] == 0.25
    assert result.at[0, "rewritten_news_contradiction_drift_vs_original"] == 0.25
    assert result.at[0, "rewritten_news_contradiction_drift_incremental"] == 0.25
