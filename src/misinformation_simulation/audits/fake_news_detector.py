from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import softmax


def pretrained_fake_news_detector_prediction(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Invalid text for prediction.")

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

    label_probs = {id2label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

    return {
        "prediction_id": pred_id,
        "prediction_label": id2label.get(pred_id, str(pred_id)),
        "prediction_confidence": float(probs[pred_id]),
        "label_probabilities": label_probs,
    }
