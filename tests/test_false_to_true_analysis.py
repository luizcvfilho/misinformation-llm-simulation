from __future__ import annotations

from pathlib import Path

import pandas as pd

from misinformation_simulation.topic_drift.stdi_logistic_regression import (
    run_stdi_logistic_regression_analysis,
)
from misinformation_simulation.topic_drift.stdi_regression_features import (
    FEATURE_COLUMNS,
    build_stdi_pair_dataset,
    build_stdi_regression_dataset,
    fit_stdi_logistic_regression,
    fit_stdi_tfidf_comparison,
)


def _annotated_frame(frame: pd.DataFrame, *, rewrite: bool) -> pd.DataFrame:
    result = frame.copy()
    result["original_topic_structure_status"] = "success"
    result["original_main_topic"] = "news"
    result["original_subtopics"] = '["one", "two"]'
    result["original_central_entities"] = '["entity"]'
    result["original_central_relations"] = '[{"subject": "a", "action": "does", "object": "b"}]'
    result["original_has_internal_contradiction"] = False
    result["original_internal_contradiction_score"] = 0.1
    result["original_vad_valence"] = 2.0
    result["original_vad_arousal"] = 2.0
    result["original_vad_dominance"] = 2.0
    if rewrite:
        prefix = "rewritten_article_text"
        result[f"{prefix}_topic_structure_status"] = "success"
        result[f"{prefix}_main_topic"] = "news"
        result[f"{prefix}_subtopics"] = '["one"]'
        result[f"{prefix}_central_entities"] = '["entity", "source"]'
        result[f"{prefix}_central_relations"] = (
            '[{"subject": "a", "action": "reports", "object": "b"}]'
        )
        result[f"{prefix}_has_internal_contradiction"] = False
        result[f"{prefix}_internal_contradiction_score"] = 0.0
        result[f"{prefix}_vad_valence"] = 2.5
        result[f"{prefix}_vad_arousal"] = 1.5
        result[f"{prefix}_vad_dominance"] = 2.2
        for name, value in {
            "stdi_vs_original": 0.3,
            "theme_drift_vs_original": 0.0,
            "subtopic_drift_vs_original": 0.5,
            "entity_drift_vs_original": 0.5,
            "relation_drift_vs_original": 1.0,
            "contradiction_drift_vs_original": 0.0,
            "vad_drift_vs_original": 0.1,
            "content_drift_vs_original": 0.5,
            "valence_drift_vs_original": 0.1,
            "arousal_drift_vs_original": 0.1,
            "dominance_drift_vs_original": 0.1,
        }.items():
            result[f"{prefix}_{name}"] = value
    return result


def test_regression_dataset_uses_false_and_truthified_pairs() -> None:
    false = _annotated_frame(
        pd.DataFrame(
            {
                "workflow_row_id": ["f1"],
                "original_article_text": ["False article with unsupported claim."],
                "rewritten_article_text": ["Neutral corrected article."],
                "verification_status": ["unverified_generated"],
            }
        ),
        rewrite=True,
    )
    dataset = build_stdi_regression_dataset(false)

    assert dataset["reference_class_label"].tolist() == [0, 1]
    assert dataset["reference_group"].tolist() == ["false_reference", "truthified_reference"]
    assert dataset.at[0, "subtopic_count"] == 2
    assert dataset.at[0, "relation_count"] == 1


def test_pair_dataset_contains_stdi_components_and_numeric_deltas() -> None:
    audit = _annotated_frame(
        pd.DataFrame(
            {
                "workflow_row_id": ["f1"],
                "original_article_text": ["one two three four"],
                "rewritten_article_text": ["one two"],
            }
        ),
        rewrite=True,
    )

    dataset = build_stdi_pair_dataset(audit)

    assert dataset.at[0, "stdi_vs_original"] == 0.3
    assert dataset.at[0, "entity_count_delta"] == 1
    assert dataset.at[0, "word_count_ratio"] == 0.5


def test_fit_stdi_logistic_regression_returns_interpretable_importance() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "workflow_row_id": f"f-{index}",
                "reference_class_label": 0,
                **{feature: float(index % 2) for feature in FEATURE_COLUMNS},
            }
        )
        rows.append(
            {
                "workflow_row_id": f"t-{index}",
                "reference_class_label": 1,
                **{feature: float(10 + (index % 2)) for feature in FEATURE_COLUMNS},
            }
        )

    importance, metrics = fit_stdi_logistic_regression(pd.DataFrame(rows))

    assert metrics["fitted"] is True
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert "odds_ratio" in importance.columns


def test_tfidf_comparison_uses_unigrams_and_bigrams_without_replacing_stdi() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "workflow_row_id": f"f-{index}",
                "reference_class_label": 0,
                "document_text": f"unsupported false claim {index}",
                **{feature: float(index % 2) for feature in FEATURE_COLUMNS},
            }
        )
        rows.append(
            {
                "workflow_row_id": f"t-{index}",
                "reference_class_label": 1,
                "document_text": f"neutral verified report {index}",
                **{feature: float(10 + (index % 2)) for feature in FEATURE_COLUMNS},
            }
        )

    comparison, ngrams, metrics = fit_stdi_tfidf_comparison(
        pd.DataFrame(rows), max_features=100, min_df=2
    )

    assert metrics["fitted"] is True
    assert set(comparison["model"]) == {"stdi_only", "tfidf_only", "stdi_plus_tfidf"}
    assert not ngrams.empty
    assert {"ngram", "coefficient", "odds_ratio"}.issubset(ngrams.columns)


def test_workflow_reuses_successful_rewrites_and_audits(tmp_path: Path) -> None:
    calls = {"rewriter": 0, "false_annotator": 0, "true_annotator": 0}

    def rewriter(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        calls["rewriter"] += 1
        result = frame.copy()
        if "rewrite_status" not in result:
            result["rewrite_status"] = "not_requested"
        pending = ~result["rewrite_status"].eq("success")
        result.loc[pending, "rewritten_article_text"] = "Corrected neutral article."
        result.loc[pending, "rewrite_status"] = "success"
        return result

    def false_annotator(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        calls["false_annotator"] += 1
        return _annotated_frame(frame, rewrite=True)

    def true_annotator(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        calls["true_annotator"] += 1
        return _annotated_frame(frame, rewrite=False)

    false = pd.DataFrame(
        {"original_article_text": [f"false article {index}" for index in range(8)]}
    )
    kwargs = {
        "output_dir": tmp_path,
        "rewriter": rewriter,
        "false_annotator": false_annotator,
        "true_annotator": true_annotator,
    }

    outputs = run_stdi_logistic_regression_analysis(false, **kwargs)
    run_stdi_logistic_regression_analysis(false, **kwargs)

    assert calls == {"rewriter": 2, "false_annotator": 1, "true_annotator": 0}
    assert outputs["report"].exists()
    assert outputs["tfidf_comparison"].exists()
    assert outputs["tfidf_ngrams"].exists()
    rewrites = pd.read_csv(outputs["rewrites"])
    assert rewrites["rewrite_status"].eq("success").all()
    assert rewrites["verification_status"].eq("unverified_generated").all()
    assert '"fitted": true' in outputs["metrics"].read_text(encoding="utf-8")


def test_skip_rewrite_generates_rewrites_when_checkpoint_does_not_exist(tmp_path: Path) -> None:
    calls = {"rewriter": 0}

    def rewriter(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        calls["rewriter"] += 1
        result = frame.copy()
        result["rewritten_article_text"] = "Corrected neutral article."
        result["rewrite_status"] = "success"
        return result

    def false_annotator(frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        return _annotated_frame(frame, rewrite=True)

    false = pd.DataFrame(
        {"original_article_text": [f"false article {index}" for index in range(8)]}
    )
    outputs = run_stdi_logistic_regression_analysis(
        false,
        output_dir=tmp_path,
        skip_rewrite=True,
        rewriter=rewriter,
        false_annotator=false_annotator,
    )

    assert calls["rewriter"] == 1
    assert outputs["rewrites"].exists()
