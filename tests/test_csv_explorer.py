from __future__ import annotations

import pandas as pd

from misinformation_simulation.apps.csv_explorer import (
    build_csv_comparison,
    filter_dataframe,
    infer_csv_delimiter,
    read_csv_bytes,
)


def test_read_csv_bytes_detects_semicolon_delimiter() -> None:
    raw_data = b"name;score\nAna;0.5\nBruno;0.8\n"

    dataframe = read_csv_bytes(raw_data)

    assert infer_csv_delimiter(raw_data) == ";"
    assert dataframe.to_dict(orient="records") == [
        {"name": "Ana", "score": 0.5},
        {"name": "Bruno", "score": 0.8},
    ]


def test_filter_dataframe_combines_text_category_and_numeric_filters() -> None:
    dataframe = pd.DataFrame(
        {
            "title": ["Economic outlook", "Sports update", "Economic report"],
            "category": ["business", "sports", "business"],
            "score": [0.3, 0.7, 0.9],
        }
    )

    filtered = filter_dataframe(
        dataframe,
        search_text="economic",
        search_columns=["title"],
        categorical_filters={"category": ["business"]},
        numeric_ranges={"score": (0.5, 1.0)},
    )

    assert filtered.to_dict(orient="records") == [
        {"title": "Economic report", "category": "business", "score": 0.9}
    ]


def test_build_csv_comparison_keeps_sources_and_numeric_values_separate() -> None:
    left_dataframe = pd.DataFrame(
        {
            "article_id": [1, 2],
            "title": ["First version", "Second version"],
            "score": [0.2, 0.8],
        }
    )
    right_dataframe = pd.DataFrame(
        {
            "article_id": [2, 3],
            "title": ["Rewritten version", "Third version"],
            "score": [0.7, 0.4],
        }
    )

    comparison = build_csv_comparison(
        left_dataframe,
        right_dataframe,
        left_join_column="article_id",
        right_join_column="article_id",
        left_columns=["title", "score"],
        right_columns=["title", "score"],
    )

    assert comparison.left_columns == ["CSV 1: title", "CSV 1: score"]
    assert comparison.right_columns == ["CSV 2: title", "CSV 2: score"]
    assert comparison.dataframe.to_dict(orient="records") == [
        {
            "Join key: article_id = article_id": 2,
            "CSV 1: title": "Second version",
            "CSV 1: score": 0.8,
            "CSV 2: title": "Rewritten version",
            "CSV 2: score": 0.7,
        }
    ]
