from __future__ import annotations

import torch
from torch.nn.functional import softmax


def consistency_flag(
    entailment: float,
    contradiction: float,
    contradiction_threshold: float,
    delta_threshold: float,
) -> str:
    delta = contradiction - entailment
    if contradiction >= contradiction_threshold and delta >= delta_threshold:
        return "potentially_false_after_rewrite"
    return "consistent_with_original"


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
