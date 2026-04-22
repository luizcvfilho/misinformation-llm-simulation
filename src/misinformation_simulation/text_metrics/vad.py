from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

VAD_DIMENSIONS = ("valence", "arousal", "dominance")
DEFAULT_VAD_MODEL_NAME = "RobroKools/vad-bert"


@dataclass(slots=True)
class VADScore:
    valence: float | None
    arousal: float | None
    dominance: float | None


@dataclass(slots=True)
class VADModelBundle:
    model_name: str
    tokenizer: Any
    model: Any
    device: str
    max_length: int


def load_huggingface_vad_model(
    model_name: str = DEFAULT_VAD_MODEL_NAME,
    *,
    device: str | None = None,
    max_length: int = 512,
) -> VADModelBundle:
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()
    return VADModelBundle(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        max_length=max_length,
    )


@lru_cache(maxsize=2)
def get_default_vad_model_bundle(
    model_name: str = DEFAULT_VAD_MODEL_NAME,
    device: str | None = None,
    max_length: int = 512,
) -> VADModelBundle:
    return load_huggingface_vad_model(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )


def predict_text_vad(
    text: str,
    *,
    model_bundle: VADModelBundle | None = None,
    scorer: Callable[[str], VADScore] | None = None,
) -> VADScore:
    if not isinstance(text, str) or not text.strip():
        return VADScore(valence=None, arousal=None, dominance=None)
    if scorer is not None:
        return scorer(text)

    active_bundle = model_bundle or get_default_vad_model_bundle()
    return predict_vad_batch([text], model_bundle=active_bundle)[0]


def analyze_text_vad(
    text: str,
    *,
    model_bundle: VADModelBundle | None = None,
    scorer: Callable[[str], VADScore] | None = None,
) -> VADScore:
    return predict_text_vad(
        text,
        model_bundle=model_bundle,
        scorer=scorer,
    )


def predict_vad_batch(
    texts: list[str],
    *,
    model_bundle: VADModelBundle | None = None,
    batch_size: int = 16,
    show_progress: bool = False,
    progress_description: str = "VAD batches",
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[VADScore]:
    active_bundle = model_bundle or get_default_vad_model_bundle()
    valid_items = [
        (index, text) for index, text in enumerate(texts) if isinstance(text, str) and text.strip()
    ]
    scores: list[VADScore] = [VADScore(None, None, None) for _ in texts]

    if not valid_items:
        return scores

    total_items = len(valid_items)
    processed_items = 0
    progress_bar = (
        tqdm(total=total_items, desc=progress_description, unit="news") if show_progress else None
    )

    for batch_start in range(0, len(valid_items), batch_size):
        batch_items = valid_items[batch_start : batch_start + batch_size]
        batch_texts = [text for _, text in batch_items]
        encoded = active_bundle.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=active_bundle.max_length,
        )
        encoded = {key: value.to(active_bundle.device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = active_bundle.model(**encoded).logits.detach().cpu()

        for (original_index, _), row in zip(batch_items, logits.tolist(), strict=False):
            scores[original_index] = _vad_score_from_row(row)

        processed_items += len(batch_items)
        if progress_bar is not None:
            progress_bar.update(len(batch_items))
        if progress_callback is not None:
            progress_callback(processed_items, total_items)

    if progress_bar is not None:
        progress_bar.close()

    return scores


def annotate_vad_scores(
    df: pd.DataFrame,
    *,
    text_column: str,
    prefix: str = "vad",
    model_bundle: VADModelBundle | None = None,
    scorer: Callable[[str], VADScore] | None = None,
    batch_size: int = 16,
    show_progress: bool = False,
    progress_description: str = "VAD news scoring",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Text column not found: {text_column}")

    annotated = df.copy()
    texts = annotated[text_column].tolist()
    if scorer is not None:
        total_items = len(texts)
        scores = []
        progress_bar = (
            tqdm(total=total_items, desc=progress_description, unit="news")
            if show_progress
            else None
        )
        for index, text in enumerate(texts, start=1):
            scores.append(predict_text_vad(text, scorer=scorer))
            if progress_bar is not None:
                progress_bar.update(1)
            if progress_callback is not None:
                progress_callback(index, total_items)
        if progress_bar is not None:
            progress_bar.close()
    else:
        scores = predict_vad_batch(
            texts,
            model_bundle=model_bundle,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_description=progress_description,
            progress_callback=progress_callback,
        )

    annotated[f"{prefix}_valence"] = [score.valence for score in scores]
    annotated[f"{prefix}_arousal"] = [score.arousal for score in scores]
    annotated[f"{prefix}_dominance"] = [score.dominance for score in scores]
    return annotated


def _vad_score_from_row(row: list[float]) -> VADScore:
    if len(row) != 3:
        raise ValueError(f"Expected 3 regression outputs for VAD, received {len(row)}")
    return VADScore(
        valence=float(row[0]),
        arousal=float(row[1]),
        dominance=float(row[2]),
    )


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
