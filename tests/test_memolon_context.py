from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.audits import memolon_context
from misinformation_simulation.text_metrics.memolon import MEmoLonLexicon
from misinformation_simulation.text_metrics.vad import VADScore, annotate_vad_scores


@pytest.fixture
def lexicon_path(tmp_path):
    path = tmp_path / "memolon.tsv"
    path.write_text(
        "word\tvalence\tarousal\tdominance\n"
        "good\t5\t5\t5\nGOOD\t9\t9\t9\nbad\t3\t3\t3\n"
        "very good\t9\t9\t9\nother\t5\t5\t5\n"
        "symbol_1\t5\t5\t5\noverflow\t10\t10\t10\n",
        encoding="utf-8",
    )
    return path


def test_duplicates_phrases_and_occurrences_have_explicit_weights(lexicon_path):
    scorer = MEmoLonLexicon(lexicon_path)
    result = scorer.analyze("VERY good, bad good unknown")
    assert result.native_score == VADScore(19 / 3, 19 / 3, 19 / 3)
    assert result.score.valence == pytest.approx(11 / 3)
    assert result.token_coverage == 4 / 5
    assert result.matched_term_count == 3
    assert result.multiword_match_count == 1
    assert scorer.retained_duplicate_rows == 1
    assert scorer.unsupported_row_count == 1
    assert scorer.analyze("good good bad").native_score.valence == pytest.approx(17 / 3)
    assert scorer.analyze("very. good").multiword_match_count == 0


@pytest.mark.parametrize("text", ["unknown", "", None, "!!!", "goodness"])
def test_no_matches_remain_missing(lexicon_path, text):
    result = MEmoLonLexicon(lexicon_path).analyze(text)
    assert result.score == VADScore(None, None, None)
    assert result.token_coverage == 0


def test_vocabulary_filter_preserves_results_and_rejects_out_of_scope_text(lexicon_path):
    texts = ["very good", "good bad unknown"]
    full = MEmoLonLexicon(lexicon_path)
    scoped = MEmoLonLexicon(lexicon_path, texts=texts)
    assert scoped.entry_count < full.entry_count
    assert scoped.source_row_count == full.source_row_count == 7
    for text in texts:
        assert scoped.analyze(text) == full.analyze(text)
    with pytest.raises(ValueError, match="outside the declared"):
        scoped.analyze("other")


def test_predictions_are_not_clipped_and_existing_annotation_api_works(lexicon_path):
    scorer = MEmoLonLexicon(lexicon_path)
    assert scorer.out_of_range_row_count == 1
    assert scorer("overflow").valence == 5.5
    frame = annotate_vad_scores(
        pd.DataFrame({"text": ["good", "bad"]}), text_column="text", scorer=scorer
    )
    assert frame.vad_valence.tolist() == [4, 2]


def test_nonfinite_predictions_are_rejected(tmp_path):
    path = tmp_path / "invalid.tsv"
    path.write_text("word\tvalence\tarousal\tdominance\ngood\tnan\t5\t5\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        MEmoLonLexicon(path)


def test_context_audit_uses_fresh_bert_and_exports_its_provenance(
    tmp_path, lexicon_path, monkeypatch
):
    frame = pd.DataFrame(
        {
            "pair_id": [1, 1],
            "context_polarity": ["positive", "negative"],
            "topic": ["test", "test"],
            "focal_event": ["same", "same"],
            "article_text": ["good unknown", "bad unknown"],
        }
    )
    input_path, baseline_path = tmp_path / "input.csv", tmp_path / "historical.csv"
    frame.to_csv(input_path, index=False)
    historical = frame.copy()
    for dim in ("valence", "arousal", "dominance"):
        historical[f"vad_{dim}"] = [3.2, 2.8]
    historical.to_csv(baseline_path, index=False)

    def score_current(texts, cache_path):
        assert texts == frame.article_text.tolist()
        return {
            texts[0]: VADScore(4, 4, 4),
            texts[1]: VADScore(2, 2, 2),
        }, {"model_id": "current-bert", "revision": "test-revision"}

    monkeypatch.setattr(memolon_context, "score_current_bert_texts", score_current)
    output = tmp_path / "output"
    manifest = memolon_context.run_memolon_context_audit(
        input_path,
        lexicon_path,
        output,
        baseline_path=baseline_path,
        rerun_bert=True,
        bert_cache_path=tmp_path / "cache.json",
    )
    comparison = pd.read_csv(output / "vad_context_contrast_model_comparison.csv")
    assert comparison.mean_delta_bert.tolist() == [2, 2, 2]
    assert comparison.mean_delta_memolon.tolist() == [2, 2, 2]
    assert manifest["baseline_model"] == "current-bert"
    assert manifest["bert_baseline"]["revision"] == "test-revision"
    assert "current_bert_scored.csv" in manifest["outputs"]
    assert manifest["weighted_token_coverage"] == 0.5
    assert manifest["baseline_sha256"]
