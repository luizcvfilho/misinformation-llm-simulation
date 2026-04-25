from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from misinformation_simulation.text_metrics import vad
from misinformation_simulation.text_metrics.vad import (
    VADModelBundle,
    VADScore,
    _resolve_device,
    _vad_score_from_row,
    analyze_text_vad,
    annotate_vad_scores,
    predict_text_vad,
    predict_vad_batch,
)


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2] for _text in texts])}


class FakeModel:
    def __call__(self, **_encoded):
        rows = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        return SimpleNamespace(logits=torch.tensor(rows[: len(_encoded["input_ids"])]))


def make_bundle() -> VADModelBundle:
    return VADModelBundle(
        model_name="fake",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        max_length=16,
    )


def test_predict_text_vad_returns_empty_score_for_blank_text() -> None:
    assert predict_text_vad(" ") == VADScore(None, None, None)


def test_analyze_text_vad_delegates_to_injected_scorer() -> None:
    score = analyze_text_vad("text", scorer=lambda _text: VADScore(1.0, 2.0, 3.0))

    assert score == VADScore(1.0, 2.0, 3.0)


def test_predict_vad_batch_scores_valid_texts_and_preserves_blank_positions() -> None:
    updates: list[tuple[int, int]] = []

    scores = predict_vad_batch(
        ["first", "", "second"],
        model_bundle=make_bundle(),
        batch_size=1,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert scores[0].valence == pytest.approx(0.1)
    assert scores[0].arousal == pytest.approx(0.2)
    assert scores[0].dominance == pytest.approx(0.3)
    assert scores[1] == VADScore(None, None, None)
    assert scores[2].valence == pytest.approx(0.1)
    assert scores[2].arousal == pytest.approx(0.2)
    assert scores[2].dominance == pytest.approx(0.3)
    assert updates == [(1, 2), (2, 2)]


def test_predict_vad_batch_returns_empty_scores_when_no_valid_texts() -> None:
    assert predict_vad_batch(["", None], model_bundle=make_bundle()) == [
        VADScore(None, None, None),
        VADScore(None, None, None),
    ]


def test_annotate_vad_scores_uses_batch_prediction_when_no_scorer() -> None:
    df = pd.DataFrame([{"text": "first"}, {"text": ""}])

    result = annotate_vad_scores(df, text_column="text", model_bundle=make_bundle(), batch_size=2)

    assert result.at[0, "vad_valence"] == pytest.approx(0.1)
    assert result.at[0, "vad_arousal"] == pytest.approx(0.2)
    assert result.at[0, "vad_dominance"] == pytest.approx(0.3)
    assert pd.isna(result.at[1, "vad_valence"])
    assert pd.isna(result.at[1, "vad_arousal"])
    assert pd.isna(result.at[1, "vad_dominance"])


def test_annotate_vad_scores_rejects_missing_text_column() -> None:
    with pytest.raises(ValueError, match="Text column not found"):
        annotate_vad_scores(pd.DataFrame([{"body": "text"}]), text_column="text")


def test_vad_score_from_row_validates_output_width() -> None:
    assert _vad_score_from_row([1, 2, 3]) == VADScore(1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="Expected 3"):
        _vad_score_from_row([1, 2])


def test_resolve_device_prefers_explicit_device(monkeypatch) -> None:
    assert _resolve_device("cpu") == "cpu"
    monkeypatch.setattr(vad.torch.cuda, "is_available", lambda: True)
    assert _resolve_device(None) == "cuda"
