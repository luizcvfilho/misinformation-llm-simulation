from __future__ import annotations

import unittest

from misinformation_simulation.audits import VADScore
from misinformation_simulation.topic_drift import (
    TopicRelation,
    TopicStructure,
    calculate_stdi,
    calculate_stdi_chain_metrics,
    calculate_vad_drift,
)


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

        metrics_without_vad = calculate_stdi(structure, structure)
        metrics_with_vad = calculate_stdi(
            structure,
            structure,
            original_vad=VADScore(valence=3.0, arousal=3.0, dominance=3.0),
            compared_vad=VADScore(valence=2.0, arousal=4.0, dominance=3.5),
        )

        self.assertEqual(metrics_without_vad["stdi"], 0.0)
        self.assertEqual(metrics_with_vad["theme_drift"], 0.0)
        self.assertGreater(metrics_with_vad["vad_drift"], 0.0)
        self.assertGreater(metrics_with_vad["stdi"], 0.0)

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
        self.assertGreater(metric_row["vad_drift_vs_original"], 0.0)
        self.assertGreater(metric_row["stdi_vs_original"], 0.0)


if __name__ == "__main__":
    unittest.main()
