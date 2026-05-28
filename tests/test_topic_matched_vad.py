from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.audits import (
    build_rewrite_prompt,
    build_topic_matched_seed_dataset,
    compute_paired_vad_deltas,
    prepare_long_vad_frame,
    summarize_paired_vad_deltas,
    validate_rewritten_pairs,
)


def test_build_topic_matched_seed_dataset_preserves_pair_ids() -> None:
    frame = pd.DataFrame(
        [
            {
                "source_file": "Fake.csv",
                "source_row_index": 4,
                "label": "fake",
                "subject": "politics",
                "title": "False claim",
                "text": "Unsupported allegation.",
                "article_text": "False claim\n\nUnsupported allegation.",
            },
        ]
    )

    seed_df = build_topic_matched_seed_dataset(frame)

    assert len(seed_df) == 1
    assert seed_df.loc[0, "pair_id"] == "Fake.csv:4"
    assert seed_df.loc[0, "original_id"] == "Fake.csv:4"
    assert seed_df.loc[0, "original_label"] == "fake"
    assert seed_df.loc[0, "original_article_text"] == "False claim\n\nUnsupported allegation."


def test_build_rewrite_prompt_keeps_topic_and_article_context() -> None:
    row = pd.Series(
        {
            "subject": "worldnews",
            "title": "False headline",
            "original_article_text": "False article body.",
        }
    )

    prompt = build_rewrite_prompt(row)

    assert "same topic" in prompt
    assert "worldnews" in prompt
    assert "False headline" in prompt
    assert "False article body." in prompt


def test_prepare_long_vad_frame_and_compute_deltas() -> None:
    paired_df = pd.DataFrame(
        [
            {
                "pair_id": "Fake.csv:1",
                "original_id": "Fake.csv:1",
                "subject": "politics",
                "title": "Claim",
                "original_article_text": "False text",
                "rewritten_article_text": "Corrected text",
            }
        ]
    )

    long_df = prepare_long_vad_frame(paired_df)

    assert set(long_df["variant"]) == {"false_original", "rewritten_true"}
    assert long_df["pair_id"].tolist() == ["Fake.csv:1", "Fake.csv:1"]

    scored_df = long_df.copy()
    scored_df["vad_valence"] = [0.1, 0.4]
    scored_df["vad_arousal"] = [0.8, 0.5]
    scored_df["vad_dominance"] = [0.2, 0.3]

    deltas = compute_paired_vad_deltas(scored_df)

    assert deltas.loc[0, "delta_valence"] == pytest.approx(0.3)
    assert deltas.loc[0, "delta_arousal"] == pytest.approx(-0.3)
    assert deltas.loc[0, "delta_dominance"] == pytest.approx(0.1)


def test_summarize_paired_vad_deltas_supports_topic_groups() -> None:
    pair_delta_df = pd.DataFrame(
        [
            {
                "subject": "politics",
                "false_valence": 0.1,
                "rewritten_valence": 0.3,
                "delta_valence": 0.2,
                "false_arousal": 0.8,
                "rewritten_arousal": 0.6,
                "delta_arousal": -0.2,
                "false_dominance": 0.2,
                "rewritten_dominance": 0.4,
                "delta_dominance": 0.2,
            },
            {
                "subject": "politics",
                "false_valence": 0.2,
                "rewritten_valence": 0.5,
                "delta_valence": 0.3,
                "false_arousal": 0.7,
                "rewritten_arousal": 0.4,
                "delta_arousal": -0.3,
                "false_dominance": 0.3,
                "rewritten_dominance": 0.5,
                "delta_dominance": 0.2,
            },
        ]
    )

    summary = summarize_paired_vad_deltas(pair_delta_df, group_column="subject")

    assert summary.loc[0, "group"] == "politics"
    assert summary.loc[0, "pair_count"] == 2
    assert summary.loc[0, "delta_valence_mean"] == pytest.approx(0.25)


def test_validate_rewritten_pairs_rejects_duplicated_pairs() -> None:
    frame = pd.DataFrame(
        [
            {
                "pair_id": "same",
                "original_article_text": "Original A",
                "rewritten_article_text": "Rewrite A",
            },
            {
                "pair_id": "same",
                "original_article_text": "Original B",
                "rewritten_article_text": "Rewrite B",
            },
        ]
    )

    with pytest.raises(ValueError, match="Duplicated pair ids"):
        validate_rewritten_pairs(frame)
