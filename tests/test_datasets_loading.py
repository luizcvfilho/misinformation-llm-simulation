from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.datasets.loading import read_dataset, validate_pair_columns


def test_read_dataset_loads_csv_and_json(tmp_path) -> None:
    csv_path = tmp_path / "news.csv"
    json_path = tmp_path / "news.json"
    frame = pd.DataFrame([{"original": "a", "rewritten": "b"}])
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records")

    assert read_dataset(csv_path).to_dict("records") == frame.to_dict("records")
    assert read_dataset(json_path).to_dict("records") == frame.to_dict("records")


def test_read_dataset_rejects_unsupported_file_type(tmp_path) -> None:
    text_path = tmp_path / "news.txt"
    text_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported format"):
        read_dataset(text_path)


def test_validate_pair_columns_accepts_present_columns() -> None:
    df = pd.DataFrame([{"original": "a", "rewritten": "b"}])

    validate_pair_columns(df, "original", "rewritten")


def test_validate_pair_columns_reports_missing_columns() -> None:
    df = pd.DataFrame([{"original": "a"}])

    with pytest.raises(ValueError, match="Rewritten column not found"):
        validate_pair_columns(df, "original", "rewritten")
