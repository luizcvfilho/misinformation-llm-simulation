from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from misinformation_simulation.audits.nrc_vad_context import (
    compute_context_pair_deltas,
    run_nrc_context_audit,
    summarize_context_deltas,
    validate_context_pairs,
    verify_baseline,
)
from misinformation_simulation.text_metrics.nrc_vad import NRC_VAD_ARCHIVE_MEMBER, NRCVADLexicon
from misinformation_simulation.text_metrics.vad import VADScore, annotate_vad_scores


@pytest.fixture
def lexicon_path(tmp_path):
    path = tmp_path / "lexicon.tsv"
    path.write_text(
        "term\tvalence\tarousal\tdominance\n"
        "good\t0.5\t-0.5\t0\n"
        "bad\t-0.5\t0.5\t-1\n"
        "very good\t1\t0\t1\n"
        "soda/pop\t0\t0\t0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def context_frame():
    return pd.DataFrame(
        [
            {
                "pair_id": 1,
                "context_polarity": "positive",
                "topic": "test",
                "focal_event": "same event",
                "article_text": "good unknown",
            },
            {
                "pair_id": 1,
                "context_polarity": "negative",
                "topic": "test",
                "focal_event": "same event",
                "article_text": "bad unknown",
            },
        ]
    )


def test_longest_phrase_consumes_words_and_repeated_occurrences_count(lexicon_path):
    scorer = NRCVADLexicon(lexicon_path)
    result = scorer.analyze("VERY good, bad good unknown")
    assert result.native_score.valence == pytest.approx(1 / 3)
    assert result.score.valence == pytest.approx(3 + 2 / 3)
    assert result.matched_term_count == 3
    assert result.multiword_match_count == 1
    assert result.token_coverage == pytest.approx(4 / 5)
    assert scorer.analyze("good good bad").native_score.valence == pytest.approx(1 / 6)


def test_phrase_matching_stops_at_punctuation_and_word_boundaries(lexicon_path):
    scorer = NRCVADLexicon(lexicon_path)
    assert scorer.analyze("very. good").multiword_match_count == 0
    assert scorer.analyze("goodness").score == VADScore(None, None, None)
    assert scorer.analyze("soda/pop").matched_term_count == 1


@pytest.mark.parametrize("text", ["unknown", "", None, "!!!"])
def test_no_matches_are_missing_not_neutral(lexicon_path, text):
    result = NRCVADLexicon(lexicon_path).analyze(text)
    assert result.native_score == VADScore(None, None, None)
    assert result.score == VADScore(None, None, None)
    assert result.token_coverage == 0


def test_scorer_works_with_existing_annotation_api(lexicon_path):
    frame = annotate_vad_scores(
        pd.DataFrame({"text": ["good", "bad"]}),
        text_column="text",
        scorer=NRCVADLexicon(lexicon_path),
    )
    assert frame.vad_valence.tolist() == [4, 2]
    assert frame.vad_dominance.tolist() == [3, 1]


def test_official_zip_member_is_loaded_without_extraction(tmp_path, lexicon_path):
    path = tmp_path / "lexicon.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(lexicon_path, NRC_VAD_ARCHIVE_MEMBER)
    assert NRCVADLexicon(path)("good") == VADScore(4, 2, 3)


@pytest.mark.parametrize("values", ["nan\t0\t0", "2\t0\t0", "0\tinf\t0"])
def test_loader_rejects_invalid_native_scale(tmp_path, values):
    path = tmp_path / "invalid.tsv"
    path.write_text("term\tvalence\tarousal\tdominance\ngood\t" + values)
    with pytest.raises(ValueError, match="finite"):
        NRCVADLexicon(path)


def test_pair_validation_rejects_duplicates_and_mismatched_events(context_frame):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_context_pairs(pd.concat([context_frame, context_frame]))
    context_frame.loc[1, "focal_event"] = "another event"
    with pytest.raises(ValueError, match="same focal event"):
        validate_context_pairs(context_frame)


def test_baseline_requires_identical_text_not_just_same_pair(context_frame):
    baseline = context_frame.copy()
    baseline.loc[1, "article_text"] = "different text"
    with pytest.raises(ValueError, match="exactly the same texts"):
        verify_baseline(context_frame, baseline)


def test_pair_deltas_align_by_id_and_keep_missing_pairs(context_frame):
    frame = context_frame.iloc[::-1].copy()
    for dimension in ("valence", "arousal", "dominance"):
        frame[f"vad_{dimension}"] = [2.0, 4.0]
    deltas = compute_context_pair_deltas(frame)
    assert deltas.iloc[0].positive_minus_negative_valence == 2
    assert deltas.iloc[0].vad_drift == 0.5
    frame.loc[1, "vad_valence"] = float("nan")
    assert pd.isna(compute_context_pair_deltas(frame).iloc[0].vad_drift)
    summary = summarize_context_deltas(compute_context_pair_deltas(frame))
    valence = summary.loc[summary.dimension == "valence"].iloc[0]
    assert valence.valid_pairs == 0
    assert pd.isna(valence.positive_mean)


def test_audit_handles_punctuation_only_input(tmp_path, lexicon_path, context_frame):
    context_frame["article_text"] = "!!!"
    input_path = tmp_path / "input.csv"
    context_frame.to_csv(input_path, index=False)
    output = tmp_path / "output"
    manifest = run_nrc_context_audit(input_path, lexicon_path, output)
    assert manifest["weighted_token_coverage"] == 0
    summary = pd.read_csv(output / "vad_context_contrast_paired_summary.csv")
    assert (summary.valid_pairs == 0).all()


def test_audit_exports_results_and_checked_baseline(tmp_path, lexicon_path, context_frame):
    input_path = tmp_path / "input.csv"
    baseline_path = tmp_path / "baseline.csv"
    context_frame.to_csv(input_path, index=False)
    baseline = context_frame.copy()
    for dimension in ("valence", "arousal", "dominance"):
        baseline[f"vad_{dimension}"] = [3.2, 2.8]
    baseline.to_csv(baseline_path, index=False)
    output = tmp_path / "output"
    run_nrc_context_audit(input_path, lexicon_path, output, baseline_path=baseline_path)
    comparison = pd.read_csv(output / "vad_context_contrast_model_comparison.csv")
    valence = comparison.loc[comparison.dimension == "valence"].iloc[0]
    assert valence.mean_delta_nrc == 2
    assert valence.mean_abs_delta_ratio_nrc_to_bert == pytest.approx(5)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["weighted_token_coverage"] == 0.5
    assert manifest["baseline_sha256"]
    assert "does not validate Portuguese" in (
        output / "vad_context_contrast_findings.md"
    ).read_text(encoding="utf-8")
