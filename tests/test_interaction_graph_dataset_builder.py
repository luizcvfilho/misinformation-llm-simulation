from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.apps.interaction_graph_dataset_builder import (
    UNCLASSIFIED_TOPIC,
    build_sample_manifest,
    build_topic_availability,
    extract_primary_topics,
    sample_rows_by_topic,
)


@pytest.fixture
def news_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["a", "b", "c", "d", "e"],
            "category": ["politics; top", "politics", "science; top", "science", None],
            "title": ["A", "B", "C", "D", "E"],
        }
    )


def test_extract_primary_topics_uses_first_label_and_marks_missing_values(
    news_dataframe: pd.DataFrame,
) -> None:
    topics = extract_primary_topics(news_dataframe, topic_column="category")

    assert topics.tolist() == ["politics", "politics", "science", "science", UNCLASSIFIED_TOPIC]


def test_build_topic_availability_counts_primary_topics(news_dataframe: pd.DataFrame) -> None:
    availability = build_topic_availability(news_dataframe, topic_column="category")

    assert availability.set_index("topic")["available"].to_dict() == {
        "politics": 2,
        "science": 2,
        UNCLASSIFIED_TOPIC: 1,
    }


def test_sample_rows_by_topic_is_reproducible_and_preserves_columns(
    news_dataframe: pd.DataFrame,
) -> None:
    first = sample_rows_by_topic(
        news_dataframe,
        topic_column="category",
        requested_counts={"politics": 2, "science": 1},
        seed=123,
    )
    second = sample_rows_by_topic(
        news_dataframe,
        topic_column="category",
        requested_counts={"politics": 2, "science": 1},
        seed=123,
    )

    assert list(first.dataframe.columns) == list(news_dataframe.columns)
    assert len(first.dataframe) == 3
    assert first.dataframe["article_id"].is_unique
    assert first.dataframe.equals(second.dataframe)


def test_sample_rows_by_topic_rejects_counts_above_available(news_dataframe: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="only 2 are available"):
        sample_rows_by_topic(
            news_dataframe,
            topic_column="category",
            requested_counts={"politics": 3},
            seed=42,
        )


def test_sample_manifest_records_reproducibility_inputs(news_dataframe: pd.DataFrame) -> None:
    sample = sample_rows_by_topic(
        news_dataframe,
        topic_column="category",
        requested_counts={"politics": 1},
        seed=9,
    )

    manifest = build_sample_manifest(
        source_dataset="data/graphs/graph_news.csv",
        topic_column="category",
        separator=";",
        seed=9,
        sample=sample,
    )

    assert manifest["requested_counts"] == {"politics": 1}
    assert manifest["selected_rows"] == 1
    assert manifest["topic_interpretation"] == "first label split by separator"
