import json

import pandas as pd
import pytest

from misinformation_simulation.analysis import llm_vad_comparison as comparison
from misinformation_simulation.text_metrics.llm_vad import (
    parse_llm_vad_assessment,
)
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.metrics import calculate_vad_drift


def payload():
    return dict(
        valence=3.2,
        arousal=1.5,
        dominance=None,
        rationales=dict(valence="Balanced.", arousal="Calm.", dominance="Ambiguous."),
        evidence=dict(valence=["calm"], arousal=[], dominance=[]),
        ambiguities=[],
    )


@pytest.mark.parametrize("value", [True, "3", 0, 6, float("nan"), float("inf")])
def test_invalid_llm_scores_are_rejected(value):
    response = payload()
    response["valence"] = value
    with pytest.raises(ValueError):
        parse_llm_vad_assessment(json.dumps(response), "A calm report.")


def test_quotes_are_grounded_and_missing_scores_are_preserved():
    response = payload()
    result = parse_llm_vad_assessment(json.dumps(response), "A calm report.")
    assert result.score.dominance is None
    assert result.score.arousal == 1.5
    response["evidence"]["valence"] = ["A frantic report."]
    with pytest.raises(ValueError, match="exact"):
        parse_llm_vad_assessment(json.dumps(response), "A calm report.")


def test_comparison_matches_current_formula_and_does_not_impute_missing():
    pairs = pd.DataFrame(
        [
            dict(
                pair_id="context_1",
                dataset="context",
                chain_code="",
                reference_id="a",
                compared_id="b",
            ),
            dict(
                pair_id="context_2",
                dataset="context",
                chain_code="",
                reference_id="a",
                compared_id="c",
            ),
        ]
    )
    records = []
    for text_id, values in [("a", (2, 3, 4)), ("b", (3, 4, 2)), ("c", (None, 3, 4))]:
        record = dict(text_id=text_id, bert_truncated=False)
        for model in ("bert", "llm"):
            for dimension, value in zip(("valence", "arousal", "dominance"), values, strict=True):
                record[f"{model}_{dimension}"] = float("nan") if value is None else value
        records.append(record)
    compared = comparison.compare_pair_scores(pairs, pd.DataFrame(records))
    expected = calculate_vad_drift(VADScore(2, 3, 4), VADScore(3, 4, 2))["vad_drift"]
    assert compared.iloc[0].llm_vad_drift == pytest.approx(expected, abs=1e-6)
    assert compared.iloc[0].llm_dominance_delta == -2
    assert pd.isna(compared.iloc[1].llm_vad_drift)
    summary = comparison.summarize_pairs(compared)
    assert summary.loc[summary.metric == "vad_drift", "paired_pairs"].iloc[0] == 1
    assert summary.loc[summary.metric == "arousal_delta", "paired_pairs"].iloc[0] == 2


def test_cached_assessments_resume_and_configuration_changes_are_rejected(tmp_path):
    pairs = pd.DataFrame([dict(pair_id="p", reference_text="calm", compared_text="calm")])
    comparison.prepare_comparison(pairs, tmp_path, model="test", provider="local", sources=[])

    class Scorer:
        last_attempts = []
        calls = []

        def assess(self, text):
            self.calls.append(text)
            return parse_llm_vad_assessment(json.dumps(payload()), text)

    scorer = Scorer()
    comparison.evaluate_comparison(tmp_path, scorer=scorer)
    comparison.evaluate_comparison(tmp_path, scorer=scorer)
    assert scorer.calls == ["calm"]
    saved = json.loads(next((tmp_path / "llm_responses").glob("*.json")).read_text("utf-8"))
    assert saved["assessment"]["score"]["dominance"] is None
    with pytest.raises(ValueError, match="configuration"):
        comparison.prepare_comparison(
            pairs, tmp_path, model="changed", provider="local", sources=[]
        )


def test_failure_is_checkpointed_without_fabricating_a_score(tmp_path):
    pairs = pd.DataFrame([dict(pair_id="p", reference_text="a", compared_text="b")])
    comparison.prepare_comparison(pairs, tmp_path, model="test", provider="local", sources=[])

    class FailingScorer:
        last_attempts = [{"attempt": 1, "raw_response": "invalid"}]

        def assess(self, text):
            raise ValueError("Invalid response")

    with pytest.raises(RuntimeError):
        comparison.evaluate_comparison(tmp_path, scorer=FailingScorer())
    saved = json.loads(next((tmp_path / "llm_responses").glob("*.json")).read_text("utf-8"))
    assert saved["status"] == "failed"
    assert "assessment" not in saved
    assert saved["attempts"][0]["raw_response"] == "invalid"


def test_paired_news_selection_uses_complete_shared_chains(tmp_path):
    context = tmp_path / "unused.csv"
    simulation = tmp_path / "steps.csv"
    rows = []
    for chain in ("SSSS", "CCCC"):
        for news in ("1", "2", "3"):
            for step in (1, 2, 3, 4):
                rows.append(
                    dict(
                        chain_code=chain,
                        run_id=chain,
                        news_id=news,
                        step_index=step,
                        rewrite_status="success",
                        target_language="en",
                        original_text=f"original {news}",
                        rewritten_text=f"{chain} {step}",
                    )
                )
    pd.DataFrame(rows).sample(frac=1, random_state=42).to_csv(simulation, index=False)
    pairs = comparison.load_comparison_pairs(
        context, simulation, scope="simulation", news_count=2, chain_codes=("SSSS", "CCCC")
    )
    assert len(pairs) == 4
    assert pairs.groupby("news_id").chain_code.nunique().eq(2).all()
    assert pairs.compared_text.str.endswith("4").all()
    assert pairs.groupby("news_id").reference_text.nunique().eq(1).all()
    with pytest.raises(ValueError, match="Not enough"):
        comparison.load_comparison_pairs(
            context, simulation, scope="simulation", news_count=4, chain_codes=("SSSS", "CCCC")
        )


def test_report_keeps_pending_and_partial_dimensions_missing(tmp_path, monkeypatch):
    context = tmp_path / "context.csv"
    pd.DataFrame(
        [
            dict(pair_id="1", context_polarity="negative", article_text="calm", topic="test"),
            dict(pair_id="1", context_polarity="positive", article_text="bright", topic="test"),
        ]
    ).to_csv(context, index=False)
    pairs = comparison.load_comparison_pairs(context, tmp_path / "unused", scope="context")
    comparison.prepare_comparison(pairs, tmp_path, model="test", provider="local", sources=[])
    cache = tmp_path / "bert.json"
    cache.write_text(
        json.dumps(
            dict(
                texts={
                    text: dict(wordpiece_count=3, input_truncated=False)
                    for text in ("calm", "bright")
                }
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        comparison,
        "score_current_bert_texts",
        lambda texts, path: ({text: VADScore(3, 3, 3) for text in texts}, dict(truncated_texts=0)),
    )
    manifest = comparison.write_comparison_report(tmp_path, bert_cache_path=cache)
    assert manifest["llm_status_counts"] == {"pending": 2}
    result = pd.read_csv(tmp_path / "pair_comparison.csv")
    assert result.llm_vad_drift.isna().all()
    assert result.bert_vad_drift.eq(0).all()

    class Scorer:
        last_attempts = []

        def assess(self, text):
            response = payload()
            response["evidence"]["valence"] = []
            return parse_llm_vad_assessment(json.dumps(response), text)

    comparison.evaluate_comparison(tmp_path, scorer=Scorer())
    comparison.write_comparison_report(tmp_path, bert_cache_path=cache)
    result = pd.read_csv(tmp_path / "pair_comparison.csv")
    assert result.llm_vad_drift.isna().all()
    assert result.llm_valence_delta.eq(0).all()
    summary = pd.read_csv(tmp_path / "pair_summary.csv")
    assert summary.loc[summary.metric == "vad_drift", "paired_pairs"].iloc[0] == 0
    assert summary.loc[summary.metric == "valence_delta", "paired_pairs"].iloc[0] == 1


def test_truncation_sensitivity_uses_same_untruncated_pairs_for_both_estimators():
    rows = []
    for truncated, bert, llm in [(True, 0.1, 0.3), (False, 0.2, 0.4)]:
        record = dict(
            dataset="simulation",
            chain_code="SSSS",
            bert_reference_truncated=False,
            bert_compared_truncated=truncated,
        )
        for model, value in [("bert", bert), ("llm", llm)]:
            record[f"{model}_vad_drift"] = value
            for dimension in ("valence", "arousal", "dominance"):
                record[f"{model}_{dimension}_delta"] = value
                record[f"{model}_{dimension}_drift"] = value
        rows.append(record)
    summary = comparison.summarize_pairs(pd.DataFrame(rows))
    selected = summary[
        summary.chain_code.eq("all_untruncated") & summary.metric.eq("vad_drift")
    ].iloc[0]
    assert selected.paired_pairs == 1
    assert selected.bert_mean == 0.2
    assert selected.llm_mean == 0.4
