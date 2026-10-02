from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
    VADScore,
    load_huggingface_vad_model,
    predict_vad_batch,
)


def score_current_bert_texts(
    texts: list[str], cache_path: Path
) -> tuple[dict[str, VADScore], dict]:
    """Reevaluate saved texts locally; cache by model revision, runtime, and full input text."""
    unique = sorted({text for text in texts if isinstance(text, str) and text.strip()})
    snapshot = Path(snapshot_download(DEFAULT_VAD_MODEL_NAME, local_files_only=True))
    identity = dict(
        model_id=DEFAULT_VAD_MODEL_NAME,
        revision=snapshot.name,
        max_length=512,
        torch_version=version("torch"),
        transformers_version=version("transformers"),
        device="cpu",
        batch_size=8,
        torch_threads=4,
    )
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    cached = cache.get("texts", {}) if cache.get("identity") == identity else {}
    missing = [text for text in unique if text not in cached]
    if missing:
        torch.set_num_threads(identity["torch_threads"])
        bundle = load_huggingface_vad_model(str(snapshot), device="cpu", max_length=512)
        scores = predict_vad_batch(missing, model_bundle=bundle, batch_size=8, show_progress=True)
        for text, score in zip(missing, scores, strict=True):
            tokens = bundle.tokenizer(text, truncation=False)["input_ids"]
            cached[text] = dict(
                score=[score.valence, score.arousal, score.dominance],
                wordpiece_count=len(tokens),
                input_truncated=len(tokens) > bundle.max_length,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(dict(identity=identity, texts=cached)) + "\n", encoding="utf-8"
        )
    scores = {text: VADScore(*cached[text]["score"]) for text in unique}
    metadata = dict(
        **identity,
        rerun=True,
        unique_texts=len(unique),
        newly_scored_texts=len(missing),
        cache_path=cache_path.as_posix(),
        cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        truncated_texts=sum(cached[text]["input_truncated"] for text in unique),
        inference="Existing project inference functions; raw 1-5 logits; no clipping.",
    )
    return scores, metadata
