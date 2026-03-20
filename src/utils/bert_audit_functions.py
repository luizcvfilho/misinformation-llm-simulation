from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn.functional import softmax


def read_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Formato nao suportado: {file_path}")


def validate_pair_columns(df: pd.DataFrame, original_col: str, rewritten_col: str) -> None:
    if original_col not in df.columns:
        raise ValueError(f"Coluna original nao encontrada: {original_col}")
    if rewritten_col not in df.columns:
        raise ValueError(f"Coluna reescrita nao encontrada: {rewritten_col}")


def consistency_flag(
    entailment: float,
    contradiction: float,
    contradiction_threshold: float,
    delta_threshold: float,
) -> str:
    delta = contradiction - entailment
    if contradiction >= contradiction_threshold and delta >= delta_threshold:
        return "potencialmente_falsa_apos_reescrita"
    return "consistente_com_original"


def nli_pair_scores(
    tokenizer,
    model,
    id2label: dict[int, str],
    premise: str,
    hypothesis: str,
) -> dict[str, float]:
    encoded = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    probs = softmax(logits, dim=-1).cpu().numpy()

    label_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    entailment = label_probs.get("entailment", label_probs.get("entails", 0.0))
    contradiction = label_probs.get("contradiction", label_probs.get("contradicts", 0.0))
    neutral = label_probs.get("neutral", 0.0)

    return {
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
    }


def pretrained_fake_news_detector_prediction(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Texto invalido para predicao.")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**encoded).logits[0]

    probs = softmax(logits, dim=-1).cpu().numpy()
    pred_id = int(probs.argmax())

    model_id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    id2label = {int(k): str(v) for k, v in model_id2label.items()} if model_id2label else {}

    label_probs = {
        id2label.get(i, str(i)): float(probs[i])
        for i in range(len(probs))
    }

    return {
        "prediction_id": pred_id,
        "prediction_label": id2label.get(pred_id, str(pred_id)),
        "prediction_confidence": float(probs[pred_id]),
        "label_probabilities": label_probs,
    }
