from __future__ import annotations

import json

import pandas as pd
import pytest

from misinformation_simulation.analysis.vad_model_comparison import (
    compare_saved_vad_steps,
    load_vad_comparison_steps,
    run_vad_model_comparison,
    summarize_vad_model_comparison,
)
from misinformation_simulation.text_metrics.memolon import MEmoLonLexicon
from misinformation_simulation.text_metrics.vad import VADScore


@pytest.fixture
def lexicon_path(tmp_path):
    path = tmp_path / "lexicon.tsv"
    path.write_text(
        "word\tvalence\tarousal\tdominance\ngood\t7\t7\t7\nbad\t3\t3\t3\n",
        encoding="utf-8",
    )
    return path


def make_record(step, source, rewritten, source_score, rewritten_score):
    record = {
        "news_id": "news-1",
        "step_index": step,
        "source_node_id": "original" if step == 1 else "p1",
        "source_text": source,
        "rewritten_text": rewritten,
        "rewrite_status": "success",
        "target_language": "en",
        "metadata_title": "Test news",
        "metadata_vad_model": "RobroKools/vad-bert",
        "run_id": "run",
        "chain_code": "SSSS",
    }
    for role, score in (
        ("original", 3.0),
        ("source", source_score),
        ("rewritten", rewritten_score),
    ):
        for dim in ("valence", "arousal", "dominance"):
            record[f"metadata_{role}_vad_{dim}"] = score
    for relation, reference in (("vs_original", 3.0), ("incremental", source_score)):
        drift = abs(rewritten_score - reference) / 4
        record[f"vad_drift_{relation}"] = round(drift, 6)
        record[f"content_drift_{relation}"] = 0.2
        record[f"contradiction_drift_{relation}"] = 0.1
        record[f"stdi_{relation}"] = round(0.216 + 0.784 * 0.2 * drift, 6)
    return record


@pytest.fixture
def steps():
    return pd.DataFrame(
        [
            make_record(1, "good", "bad", 3.0, 3.2),
            make_record(2, "bad", "good good", 3.2, 4.0),
        ]
    )


def test_normalization_and_references_use_original_and_actual_source(steps, lexicon_path):
    result = compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path))
    assert result.iloc[0].memolon_original_valence_normalized == 0.75
    assert result.iloc[0].bert_original_valence_normalized == 0.5
    assert result.iloc[1].memolon_vad_drift_vs_original == 0
    assert result.iloc[1].memolon_vad_drift_incremental == 0.5
    assert result.iloc[0].memolon_valence_delta_vs_original == -0.5
    assert result.iloc[0].memolon_stdi_vs_original == pytest.approx(0.2944)
    assert result.iloc[1].memolon_stdi_vs_original == pytest.approx(0.216)
    assert result.iloc[1].memolon_stdi_cumulative == pytest.approx(0.5888)


def test_historical_formula_mismatch_blocks_counterfactual_stdi(steps, lexicon_path):
    steps.loc[0, "stdi_vs_original"] = 0.9
    result = compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path))
    assert result.iloc[0].formula_check_vs_original == "missing_or_inconsistent"
    assert pd.isna(result.iloc[0].memolon_stdi_vs_original)
    assert pd.notna(result.iloc[0].memolon_vad_drift_vs_original)


def test_fresh_bert_replaces_vad_but_formula_validation_uses_saved_scores(steps, lexicon_path):
    fresh = {
        "good": VADScore(4, 4, 4),
        "bad": VADScore(2, 2, 2),
        "good good": VADScore(4, 4, 4),
    }
    result = compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path), bert_scores=fresh)
    assert (result.formula_check_vs_original == "passed").all()
    assert result.iloc[0].bert_vad_drift_vs_original == 0.5
    assert result.iloc[0].bert_stdi_vs_original == pytest.approx(0.2944)
    assert result.iloc[0].saved_bert_stdi_vs_original == steps.iloc[0].stdi_vs_original
    assert result.iloc[0].bert_stdi_vs_original != result.iloc[0].saved_bert_stdi_vs_original
    assert result.iloc[1].bert_vad_drift_vs_original == 0
    assert result.iloc[1].bert_vad_drift_incremental == 0.5
    assert result.iloc[1].bert_stdi_cumulative == pytest.approx(0.5888)


def test_no_lexical_matches_do_not_become_zero_drift(steps, lexicon_path):
    steps = steps.iloc[:1].copy()
    steps.loc[0, "rewritten_text"] = "unknown"
    result = compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path))
    assert result.iloc[0].memolon_rewritten_token_coverage == 0
    assert pd.isna(result.iloc[0].memolon_vad_drift_vs_original)
    summary = summarize_vad_model_comparison(result)
    assert (summary.paired_steps == 0).all()


def test_failed_step_does_not_change_source_for_later_step(steps, lexicon_path):
    failed = make_record(2, "bad", "good", 3.2, 4.0)
    failed["rewrite_status"] = "error"
    third = make_record(3, "bad", "good good", 3.2, 4.0)
    result = compare_saved_vad_steps(
        pd.DataFrame([steps.iloc[0].to_dict(), failed, third]), MEmoLonLexicon(lexicon_path)
    )
    assert pd.isna(result.iloc[1].memolon_vad_drift_incremental)
    assert result.iloc[2].memolon_vad_drift_incremental == 0.5
    assert result.iloc[2].memolon_stdi_cumulative == pytest.approx(0.5888)


def test_invalid_language_is_excluded_without_neutral_imputation(steps, lexicon_path):
    steps.loc[0, "target_language"] = "pt"
    result = compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path))
    assert result.iloc[0].comparison_status == "excluded_status_or_language"
    assert pd.isna(result.iloc[0].memolon_vad_drift_vs_original)


def test_broken_text_continuity_and_missing_original_are_rejected(steps, lexicon_path):
    steps.loc[1, "source_text"] = "different text"
    with pytest.raises(ValueError, match="previous successful"):
        compare_saved_vad_steps(steps, MEmoLonLexicon(lexicon_path))
    with pytest.raises(ValueError, match="step 1"):
        compare_saved_vad_steps(steps.iloc[1:], MEmoLonLexicon(lexicon_path))


def test_loader_filters_chains_and_rejects_duplicate_steps(tmp_path, steps):
    path = tmp_path / "simulation_01_01_ssss_steps.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in steps.to_dict("records")), encoding="utf-8"
    )
    loaded, paths = load_vad_comparison_steps(tmp_path, ("SSSS",))
    assert len(loaded) == 2
    assert paths == [path]
    with pytest.raises(ValueError, match="Missing requested"):
        load_vad_comparison_steps(tmp_path, ("PPPP", "SSSS"))
    path.write_text(
        json.dumps(steps.iloc[0].to_dict()) + "\n" + json.dumps(steps.iloc[0].to_dict()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_vad_comparison_steps(tmp_path, ("SSSS",))


def test_audit_exports_paired_results_and_provenance(tmp_path, steps, lexicon_path):
    path = tmp_path / "simulation_01_01_ssss_steps.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in steps.to_dict("records")), encoding="utf-8"
    )
    comparison, summary, manifest = run_vad_model_comparison(
        tmp_path, lexicon_path, tmp_path / "output", chain_codes=("SSSS",)
    )
    assert len(comparison) == 2
    assert len(summary) == 20
    assert manifest["rewrites_regenerated"] is False
    assert manifest["unique_texts_scored"] == 3
    assert len(manifest["source_files"][0]["sha256"]) == 64
    assert (tmp_path / "output/step_comparison.csv").exists()
