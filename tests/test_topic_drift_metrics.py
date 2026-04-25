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


def test_calculate_stdi_includes_contradiction_weight_in_base_formula() -> None:
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
    )

    metrics = calculate_stdi(original, compared)

    assert metrics["contradiction_drift"] == 1.0
    assert metrics["stdi"] == 0.2


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
    assert metric_row["contradiction_drift_vs_original"] == 1.0
    assert metric_row["contradiction_drift_incremental"] == 1.0
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
    assert "rewritten_news_contradiction_drift_vs_original" in result.columns
    assert "rewritten_news_contradiction_drift_incremental" in result.columns
    assert not result.at[0, "original_has_internal_contradiction"]
    assert result.at[0, "rewritten_news_has_internal_contradiction"]
    assert result.at[0, "rewritten_news_contradiction_drift_vs_original"] == 1.0
    assert result.at[0, "rewritten_news_contradiction_drift_incremental"] == 1.0
