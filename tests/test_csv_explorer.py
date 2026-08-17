from __future__ import annotations

import pandas as pd

from misinformation_simulation.apps.csv_explorer import (
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
