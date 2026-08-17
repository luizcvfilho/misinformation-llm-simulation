from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.topic_drift import (
    CALCULATED_COMPONENT_COLUMNS,
    METRIC_REWRITE_PROMPTS,
    SemanticSTDIComparison,
    TopicRelation,
    TopicStructure,
    build_manual_stdi_evaluation_dataset,
    fit_manual_stdi_regression,
    score_manual_stdi_evaluation_pairs,
    summarize_manual_stdi_evaluation,
)


def _source_news(count: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["__query_metadata__", *[f"article-{index}" for index in range(count)]],
            "title": ["metadata", *[f"Title {index}" for index in range(count)]],
            "content": ["metadata", *[f"News report {index}" for index in range(count)]],
        }
    )


def test_build_manual_dataset_samples_fifty_real_rows_and_balances_prompts() -> None:
    dataset = build_manual_stdi_evaluation_dataset(_source_news())

    counts = dataset["target_metric"].value_counts()
    assert len(dataset) == 50
    assert set(counts.index) == {prompt.metric for prompt in METRIC_REWRITE_PROMPTS}
    assert counts.max() - counts.min() <= 1
    assert dataset["article_id"].ne("__query_metadata__").all()
    assert dataset["manual_expected_stdi"].isna().all()


def test_score_manual_pairs_calculates_stdi_and_summarizes_target_metric() -> None:
    dataset = build_manual_stdi_evaluation_dataset(_source_news(), sample_size=2)
    dataset["modified_text"] = ["Rewritten report 1", "Rewritten report 2"]

    def fake_extract_topic_structure(**kwargs: object) -> TopicStructure:
        text = str(kwargs["text"])
        if text.startswith("Rewritten"):
            return TopicStructure(
                main_topic="economy",
                subtopics=["inflation"],
                central_entities=["central bank"],
                central_relations=[TopicRelation("bank", "raises", "rates")],
            )
        return TopicStructure(
            main_topic="economy",
            subtopics=["inflation"],
            central_entities=["government"],
            central_relations=[TopicRelation("government", "reports", "inflation")],
        )

    def fake_semantic_comparison(**_kwargs: object) -> SemanticSTDIComparison:
        return SemanticSTDIComparison(
            component_drifts={
                "theme_drift": 0.0,
                "subtopic_drift": 0.25,
                "entity_drift": 0.0,
                "relation_drift": 0.0,
            },
            rationales={
                "theme_drift": "Same economic story.",
                "subtopic_drift": "Slightly different angle.",
                "entity_drift": "Same actor.",
                "relation_drift": "Same factual relation.",
            },
        )

    scored = score_manual_stdi_evaluation_pairs(
        dataset,
        model="test-model",
        provider="test-provider",
        extract_topic_structure_fn=fake_extract_topic_structure,
        semantic_comparison_fn=fake_semantic_comparison,
        include_vad=False,
    )
    summary = summarize_manual_stdi_evaluation(scored)

    assert scored["scoring_status"].eq("success").all()
    assert scored["calculated_stdi"].notna().all()
    assert scored["calculated_theme_drift"].eq(0.0).all()
    assert scored["lexical_theme_drift"].eq(0.0).all()
    assert scored["semantic_subtopic_drift"].eq(0.25).all()
    assert scored["semantic_relation_drift_rationale"].eq("Same factual relation.").all()
    assert summary["pair_count"].sum() == 2


def test_fit_manual_regression_uses_reviewed_expected_stdi() -> None:
    rows = []
    for index in range(24):
        value = index / 23
        rows.append(
            {
                "manual_expected_stdi": value,
                "calculated_theme_drift": value,
                "calculated_subtopic_drift": value * 0.8,
                "calculated_entity_drift": value * 0.6,
                "calculated_relation_drift": value * 0.4,
                "calculated_contradiction_drift": float(value > 0.5),
                "calculated_vad_drift": value * 0.2,
            }
        )

    result = fit_manual_stdi_regression(pd.DataFrame(rows), random_state=1)

    assert result.n_samples == 24
    assert set(result.feature_columns) == set(CALCULATED_COMPONENT_COLUMNS)
    assert sum(result.normalized_weights.values()) == pytest.approx(1.0)
