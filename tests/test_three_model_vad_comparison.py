import json

import pandas as pd
import pytest

from misinformation_simulation.analysis import three_model_vad_comparison as three
from misinformation_simulation.analysis import vad_model_comparison as pairwise
from misinformation_simulation.audits import memolon_context
from misinformation_simulation.text_metrics.vad import VADScore


@pytest.fixture
def resources(tmp_path, monkeypatch):
    mem_path, nrc_path = tmp_path / "mem.tsv", tmp_path / "nrc.tsv"
    mem_path.write_text("word\tvalence\tarousal\tdominance\ngood\t7\t7\t7\nbad\t3\t3\t3\n")
    nrc_path.write_text(
        "term\tvalence\tarousal\tdominance\ngood\t0.5\t0.5\t0.5\nbad\t-0.5\t-0.5\t-0.5\n"
    )

    def score_current(texts, cache_path):
        unique = set(texts)
        scores = {
            text: VADScore(*(3 * [4 if "good" in text else 2 if "bad" in text else 3]))
            for text in unique
        }
        return scores, dict(
            model_id="test-current-bert",
            revision="current-revision",
            unique_texts=len(unique),
            truncated_texts=0,
        )

    monkeypatch.setattr(three, "score_current_bert_texts", score_current)
    monkeypatch.setattr(pairwise, "score_current_bert_texts", score_current)
    monkeypatch.setattr(memolon_context, "score_current_bert_texts", score_current)
    return mem_path, nrc_path, tmp_path / "bert.json"


def test_context_uses_same_current_bert_and_joint_pairs_without_imputation(tmp_path, resources):
    mem_path, nrc_path, cache = resources
    frame = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2],
            "context_polarity": ["positive", "negative", "positive", "negative"],
            "topic": ["test"] * 4,
            "focal_event": ["event one", "event one", "event two", "event two"],
            "article_text": ["good unknown", "bad unknown", "unknown", "unknown"],
        }
    )
    input_path = tmp_path / "input.csv"
    frame.to_csv(input_path, index=False)
    scored, pairs, summary, manifest = three.run_three_model_context_comparison(
        input_path,
        mem_path,
        nrc_path,
        tmp_path / "output",
        bert_cache_path=cache,
    )
    assert len(scored) == 12
    assert len(pairs) == 6
    assert manifest["joint_valid_pairs"] == 1
    assert (summary.valid_pairs == 1).all()
    assert (summary.mean_abs_delta == 2).all()
    assert (summary.normalized_mean_abs_delta == 0.5).all()
    assert set(summary.model) == set(three.MODEL_LABELS.values())
    missing = pairs.loc[(pairs.pair_id == 2) & (pairs.model != "Current BERT")]
    assert missing.vad_drift.isna().all()
    assert manifest["bert_baseline"]["revision"] == "current-revision"
    assert manifest["lexicons"]["nrc"]["sha256"]
    assert manifest["lexicons"]["memolon"]["sha256"]
    assert {path.name for path in (tmp_path / "output").iterdir()} == {
        "scored_texts.csv",
        "model_pair_deltas.csv",
        "model_summary.csv",
        "manifest.json",
        "findings.md",
    }


def test_simulation_retains_three_native_scales_and_historical_stdi(tmp_path, resources):
    mem_path, nrc_path, cache = resources
    record = dict(
        news_id="one",
        step_index=1,
        source_node_id="original",
        source_text="good",
        rewritten_text="bad",
        rewrite_status="success",
        target_language="en",
        metadata_vad_model="historical-bert",
    )
    for role, value in [("original", 3), ("source", 3), ("rewritten", 3.2)]:
        for dim in ("valence", "arousal", "dominance"):
            record[f"metadata_{role}_vad_{dim}"] = value
    for relation in ("vs_original", "incremental"):
        record[f"vad_drift_{relation}"] = 0.05
        record[f"content_drift_{relation}"] = 0.2
        record[f"contradiction_drift_{relation}"] = 0.1
        record[f"stdi_{relation}"] = 0.22384
    (tmp_path / "simulation_01_01_ssss_steps.jsonl").write_text(json.dumps(record) + "\n")
    combined, summary, manifest = three.run_three_model_simulation_comparison(
        tmp_path,
        mem_path,
        nrc_path,
        tmp_path / "output",
        bert_cache_path=cache,
        chain_codes=("SSSS",),
    )
    assert combined.iloc[0].nrc_native_original_valence == 0.5
    assert combined.iloc[0].memolon_native_original_valence == 7
    assert combined.iloc[0].bert_original_valence == 4
    for model in three.MODEL_LABELS:
        assert combined.iloc[0][f"{model}_vad_drift_vs_original"] == 0.5
        assert combined.iloc[0][f"{model}_stdi_vs_original"] == pytest.approx(0.2944)
    assert combined.iloc[0].saved_bert_stdi_vs_original == 0.22384
    assert len(summary) == 10
    assert (summary.paired_steps == 1).all()
    assert manifest["bert_baseline"]["revision"] == "current-revision"
    assert {path.name for path in (tmp_path / "output").iterdir()} == {
        "step_comparison.csv",
        "chain_step_summary.csv",
        "run_inventory.csv",
        "manifest.json",
        "findings.md",
    }


def test_joint_simulation_summary_keeps_missing_candidates_excluded():
    frame = pd.DataFrame({"run_id": ["run"], "chain_code": ["SSSS"], "step_index": [1]})
    for relation in three.COMPARISON_RELATIONS:
        for metric in ("vad_drift", "valence_drift", "arousal_drift", "dominance_drift", "stdi"):
            for model in three.MODEL_LABELS:
                frame[f"{model}_{metric}_{relation}"] = None if model == "nrc" else 0.2
    summary = three.summarize_three_model_steps(frame)
    assert (summary.paired_steps == 0).all()
    assert summary[["bert_mean", "nrc_mean", "memolon_mean"]].isna().all().all()
