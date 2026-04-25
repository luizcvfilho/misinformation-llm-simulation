from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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


class FakeModel:
    def __init__(self) -> None:
        self.to_called = False
        self.eval_called = False

    def to(self, *_args) -> FakeModel:
        self.to_called = True
        return self

    def eval(self) -> None:
        self.eval_called = True


class FakePretrainedFactory:
    def __init__(self, return_value: object) -> None:
        self.return_value = return_value
        self.model_name: str | None = None

    def from_pretrained(self, model_name: str) -> object:
        self.model_name = model_name
        return self.return_value


def test_load_fake_true_news_dataset_builds_expected_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "title": "Explosive claim",
                "text": "Shocking crisis spreads quickly.",
                "subject": "politics",
                "date": "2026-04-17",
            }
        ]
    ).to_csv(tmp_path / "Fake.csv", index=False)
    pd.DataFrame(
        [
            {
                "title": "Official report",
                "text": "Calm officials confirm stable conditions.",
                "subject": "politics",
                "date": "2026-04-17",
            }
        ]
    ).to_csv(tmp_path / "True.csv", index=False)

    loaded = load_fake_true_news_dataset(tmp_path)

    assert set(loaded["label"]) == {"fake", "true"}
    assert list(loaded["dataset_name"].unique()) == ["FakeVsTrueVAD"]
    assert "article_text" in loaded.columns
    assert loaded.loc[0, "article_text"] == "Explosive claim\n\nShocking crisis spreads quickly."
    assert loaded.loc[0, "source_file"] == "Fake.csv"
    assert loaded.loc[1, "source_file"] == "True.csv"


def test_load_huggingface_vad_model_uses_expected_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = object()
    model = FakeModel()
    tokenizer_factory = FakePretrainedFactory(tokenizer)
    model_factory = FakePretrainedFactory(model)

    monkeypatch.setattr(
        "misinformation_simulation.text_metrics.vad.AutoTokenizer",
        tokenizer_factory,
    )
    monkeypatch.setattr(
        "misinformation_simulation.text_metrics.vad.AutoModelForSequenceClassification",
        model_factory,
    )

    bundle = load_huggingface_vad_model()

    assert tokenizer_factory.model_name == DEFAULT_VAD_MODEL_NAME
    assert model_factory.model_name == DEFAULT_VAD_MODEL_NAME
    assert model.to_called
    assert model.eval_called
    assert bundle.model_name == DEFAULT_VAD_MODEL_NAME
    assert bundle.tokenizer == tokenizer
    assert bundle.model == model


def test_predict_text_vad_allows_injected_scorer() -> None:
    score = predict_text_vad(
        "breaking news",
        scorer=lambda text: VADScore(
            valence=float(len(text)),
            arousal=0.5,
            dominance=-0.5,
        ),
    )

    assert score.valence == 13.0
    assert score.arousal == 0.5
    assert score.dominance == -0.5


def test_dataframe_annotation_summary_and_comparison_are_consistent() -> None:
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

    assert fake_arousal > true_arousal
    assert arousal_diff > 0.0
    assert "vad_valence" in annotated.columns
    assert "vad_arousal" in annotated.columns
    assert "vad_dominance" in annotated.columns


def test_annotate_vad_scores_reports_progress() -> None:
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

    assert updates == [(1, 3), (2, 3), (3, 3)]
    assert len(annotated) == 3
