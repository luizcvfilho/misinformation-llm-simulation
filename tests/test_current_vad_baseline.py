from types import SimpleNamespace

from misinformation_simulation.analysis import current_vad_baseline as baseline
from misinformation_simulation.text_metrics.vad import VADScore


def test_current_bert_cache_reuses_exact_inputs_and_invalidates_changed_revision(
    tmp_path, monkeypatch
):
    revision = "revision-one"
    calls = []
    monkeypatch.setattr(baseline, "snapshot_download", lambda *a, **k: tmp_path / revision)
    monkeypatch.setattr(baseline.torch, "set_num_threads", lambda value: None)
    bundle = SimpleNamespace(
        max_length=512,
        tokenizer=lambda text, **kwargs: {"input_ids": list(range(513 if text == "long" else 3))},
    )
    monkeypatch.setattr(baseline, "load_huggingface_vad_model", lambda *a, **k: bundle)

    def predict(texts, **kwargs):
        calls.append(texts)
        assert kwargs["model_bundle"] is bundle
        return [VADScore(5.5, 2, 3) for _ in texts]

    monkeypatch.setattr(baseline, "predict_vad_batch", predict)
    cache_path = tmp_path / "bert.json"
    scores, metadata = baseline.score_current_bert_texts(["short", "long", "short"], cache_path)
    assert calls == [["long", "short"]]
    assert scores["long"].valence == 5.5
    assert metadata["truncated_texts"] == 1
    assert metadata["newly_scored_texts"] == 2
    cached, cached_metadata = baseline.score_current_bert_texts(["short"], cache_path)
    assert cached["short"] == scores["short"]
    assert cached_metadata["newly_scored_texts"] == 0
    assert cached_metadata["truncated_texts"] == 0
    assert len(calls) == 1
    revision = "revision-two"
    _, changed_metadata = baseline.score_current_bert_texts(["short"], cache_path)
    assert len(calls) == 2
    assert changed_metadata["revision"] == revision
    assert changed_metadata["newly_scored_texts"] == 1
