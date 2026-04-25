from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from misinformation_simulation.audits.bert_nli import consistency_flag, nli_pair_scores
from misinformation_simulation.audits.fake_news_detector import (
    pretrained_fake_news_detector_prediction,
)


class FakeTokenizer:
    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2]])}


class FakeModel:
    config = SimpleNamespace(id2label={0: "fake", 1: "real"})

    def __call__(self, **_encoded):
        return SimpleNamespace(logits=torch.tensor([[1.0, 3.0]]))


class FakeNLIModel:
    def __call__(self, **_encoded):
        return SimpleNamespace(logits=torch.tensor([[0.5, 2.5, 0.0]]))


def test_consistency_flag_marks_large_contradiction() -> None:
    assert (
        consistency_flag(
            entailment=0.1,
            contradiction=0.9,
            contradiction_threshold=0.7,
            delta_threshold=0.5,
        )
        == "potentially_false_after_rewrite"
    )
    assert (
        consistency_flag(
            entailment=0.6,
            contradiction=0.7,
            contradiction_threshold=0.7,
            delta_threshold=0.5,
        )
        == "consistent_with_original"
    )


def test_nli_pair_scores_maps_probabilities_by_label() -> None:
    scores = nli_pair_scores(
        FakeTokenizer(),
        FakeNLIModel(),
        {0: "entailment", 1: "contradiction", 2: "neutral"},
        premise="original",
        hypothesis="rewrite",
    )

    assert scores["contradiction"] > scores["entailment"]
    assert scores["neutral"] > 0.0


def test_fake_news_detector_prediction_returns_label_probabilities() -> None:
    result = pretrained_fake_news_detector_prediction(FakeTokenizer(), FakeModel(), "news text")

    assert result["prediction_id"] == 1
    assert result["prediction_label"] == "real"
    assert result["prediction_confidence"] == result["label_probabilities"]["real"]
    assert set(result["label_probabilities"]) == {"fake", "real"}


def test_fake_news_detector_prediction_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="Invalid text"):
        pretrained_fake_news_detector_prediction(FakeTokenizer(), FakeModel(), " ")
