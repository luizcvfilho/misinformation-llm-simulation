from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

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


class TopicDriftVADMetricsTest(unittest.TestCase):
    def test_calculate_vad_drift_returns_normalized_dimension_scores(self) -> None:
        drift = calculate_vad_drift(
            VADScore(valence=3.0, arousal=2.5, dominance=2.0),
            VADScore(valence=2.0, arousal=3.5, dominance=3.0),
        )

        self.assertEqual(drift["valence_drift"], 0.25)
        self.assertEqual(drift["arousal_drift"], 0.25)
        self.assertEqual(drift["dominance_drift"], 0.25)
        self.assertEqual(drift["vad_drift"], 0.25)

    def test_calculate_stdi_uses_vad_drift_when_topic_structure_is_identical(self) -> None:
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

        self.assertEqual(metrics_without_vad["stdi"], 0.0)
        self.assertEqual(metrics_with_vad["theme_drift"], 0.0)
        self.assertEqual(metrics_with_vad["contradiction_drift"], 1.0)
        self.assertGreater(metrics_with_vad["vad_drift"], 0.0)
        self.assertGreater(metrics_with_vad["stdi"], 0.0)

    def test_calculate_stdi_includes_contradiction_weight_in_base_formula(self) -> None:
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

        self.assertEqual(metrics["contradiction_drift"], 1.0)
        self.assertEqual(metrics["stdi"], 0.2)

    def test_calculate_stdi_chain_metrics_propagates_vad_columns(self) -> None:
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

        self.assertEqual(len(metrics), 1)
        metric_row = metrics[0]
        self.assertIn("vad_drift_vs_original", metric_row)
        self.assertIn("vad_drift_incremental", metric_row)
        self.assertIn("contradiction_drift_vs_original", metric_row)
        self.assertIn("contradiction_drift_incremental", metric_row)
        self.assertEqual(metric_row["contradiction_drift_vs_original"], 1.0)
        self.assertEqual(metric_row["contradiction_drift_incremental"], 1.0)
        self.assertGreater(metric_row["vad_drift_vs_original"], 0.0)
        self.assertGreater(metric_row["stdi_vs_original"], 0.0)

    def test_build_topic_structure_defaults_missing_contradiction_to_false(self) -> None:
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

        self.assertFalse(structure.has_internal_contradiction)

    @patch("misinformation_simulation.topic_drift.annotation.predict_text_vad", return_value=None)
    @patch("misinformation_simulation.topic_drift.annotation.extract_topic_structure")
    def test_annotate_stdi_for_rewrites_adds_contradiction_columns(
        self,
        extract_topic_structure_mock,
        _predict_text_vad_mock,
    ) -> None:
        extract_topic_structure_mock.side_effect = [
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

        self.assertIn("original_has_internal_contradiction", result.columns)
        self.assertIn("rewritten_news_has_internal_contradiction", result.columns)
        self.assertIn("rewritten_news_contradiction_drift_vs_original", result.columns)
        self.assertIn("rewritten_news_contradiction_drift_incremental", result.columns)
        self.assertFalse(result.at[0, "original_has_internal_contradiction"])
        self.assertTrue(result.at[0, "rewritten_news_has_internal_contradiction"])
        self.assertEqual(result.at[0, "rewritten_news_contradiction_drift_vs_original"], 1.0)
        self.assertEqual(result.at[0, "rewritten_news_contradiction_drift_incremental"], 1.0)


if __name__ == "__main__":
    unittest.main()
