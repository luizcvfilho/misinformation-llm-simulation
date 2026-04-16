from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported format: {file_path}")


def validate_pair_columns(df: pd.DataFrame, original_col: str, rewritten_col: str) -> None:
    if original_col not in df.columns:
        raise ValueError(f"Original column not found: {original_col}")
    if rewritten_col not in df.columns:
        raise ValueError(f"Rewritten column not found: {rewritten_col}")
