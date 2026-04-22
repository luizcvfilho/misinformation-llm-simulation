from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from misinformation_simulation.audits import (
    DEFAULT_VAD_MODEL_NAME,
    VADScore,
    annotate_vad_scores,
    compare_vad_groups,
    load_huggingface_vad_model,
    predict_text_vad,
    summarize_vad_by_group,
)
from misinformation_simulation.datasets import load_fake_true_news_dataset


class FakeTrueDatasetLoaderTest(unittest.TestCase):
    def test_load_fake_true_news_dataset_builds_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame(
                [
                    {
                        "title": "Explosive claim",
                        "text": "Shocking crisis spreads quickly.",
                        "subject": "politics",
                        "date": "2026-04-17",
                    }
                ]
            ).to_csv(root / "Fake.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "title": "Official report",
                        "text": "Calm officials confirm stable conditions.",
                        "subject": "politics",
                        "date": "2026-04-17",
                    }
                ]
            ).to_csv(root / "True.csv", index=False)

            loaded = load_fake_true_news_dataset(root)

        self.assertEqual(set(loaded["label"]), {"fake", "true"})
        self.assertEqual(list(loaded["dataset_name"].unique()), ["FakeVsTrueVAD"])
        self.assertIn("article_text", loaded.columns)
        self.assertEqual(
            loaded.loc[0, "article_text"],
            "Explosive claim\n\nShocking crisis spreads quickly.",
        )
        self.assertEqual(loaded.loc[0, "source_file"], "Fake.csv")
        self.assertEqual(loaded.loc[1, "source_file"], "True.csv")


class HuggingFaceVADTest(unittest.TestCase):
    @patch("misinformation_simulation.text_metrics.vad.AutoModelForSequenceClassification")
    @patch("misinformation_simulation.text_metrics.vad.AutoTokenizer")
    def test_load_huggingface_vad_model_uses_expected_defaults(
        self,
        tokenizer_cls: Mock,
        model_cls: Mock,
    ) -> None:
        tokenizer = Mock()
        model = Mock()
        tokenizer_cls.from_pretrained.return_value = tokenizer
        model_cls.from_pretrained.return_value = model
        model.to.return_value = model

        bundle = load_huggingface_vad_model()

        tokenizer_cls.from_pretrained.assert_called_once_with(DEFAULT_VAD_MODEL_NAME)
        model_cls.from_pretrained.assert_called_once_with(DEFAULT_VAD_MODEL_NAME)
        model.to.assert_called_once()
        model.eval.assert_called_once()
        self.assertEqual(bundle.model_name, DEFAULT_VAD_MODEL_NAME)
        self.assertEqual(bundle.tokenizer, tokenizer)
        self.assertEqual(bundle.model, model)

    def test_predict_text_vad_allows_injected_scorer(self) -> None:
        score = predict_text_vad(
            "breaking news",
            scorer=lambda text: VADScore(
                valence=float(len(text)),
                arousal=0.5,
                dominance=-0.5,
            ),
        )

        self.assertEqual(score.valence, 13.0)
        self.assertEqual(score.arousal, 0.5)
        self.assertEqual(score.dominance, -0.5)

    def test_dataframe_annotation_summary_and_comparison_are_consistent(self) -> None:
        frame = pd.DataFrame(
            [
                {"label": "fake", "article_text": "sensational fake claim"},
                {"label": "fake", "article_text": "outrage and panic"},
                {"label": "true", "article_text": "calm official bulletin"},
                {"label": "true", "article_text": "measured report"},
            ]
        )

        score_map = {
            "sensational fake claim": VADScore(0.10, 0.90, 0.30),
            "outrage and panic": VADScore(0.12, 0.85, 0.28),
            "calm official bulletin": VADScore(0.35, 0.40, 0.42),
            "measured report": VADScore(0.31, 0.38, 0.40),
        }

        annotated = annotate_vad_scores(
            frame,
            text_column="article_text",
            scorer=lambda text: score_map[text],
        )
        summary = summarize_vad_by_group(annotated, group_column="label")
        comparison = compare_vad_groups(annotated)

        fake_arousal = float(summary.loc[summary["label"] == "fake", "arousal_mean"].iloc[0])
        true_arousal = float(summary.loc[summary["label"] == "true", "arousal_mean"].iloc[0])
        arousal_diff = float(comparison.loc[comparison["metric"] == "arousal", "mean_diff"].iloc[0])

        self.assertGreater(fake_arousal, true_arousal)
        self.assertGreater(arousal_diff, 0.0)
        self.assertIn("vad_valence", annotated.columns)
        self.assertIn("vad_arousal", annotated.columns)
        self.assertIn("vad_dominance", annotated.columns)

    def test_annotate_vad_scores_reports_progress(self) -> None:
        frame = pd.DataFrame(
            [
                {"article_text": "news a"},
                {"article_text": "news b"},
                {"article_text": "news c"},
            ]
        )
        updates: list[tuple[int, int]] = []

        annotated = annotate_vad_scores(
            frame,
            text_column="article_text",
            scorer=lambda _: VADScore(0.1, 0.2, 0.3),
            progress_callback=lambda done, total: updates.append((done, total)),
        )

        self.assertEqual(updates, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(len(annotated), 3)


if __name__ == "__main__":
    unittest.main()
