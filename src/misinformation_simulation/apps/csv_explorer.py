from __future__ import annotations

import csv
from collections.abc import Iterable
from io import BytesIO

import pandas as pd

DELIMITER_OPTIONS = {
    "Auto": None,
    "Comma": ",",
    "Semicolon": ";",
    "Tab": "\t",
    "Pipe": "|",
}


def _decode_csv_bytes(raw_data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The CSV could not be decoded as UTF-8 or Latin-1.")


def infer_csv_delimiter(raw_data: bytes) -> str:
    sample = _decode_csv_bytes(raw_data)[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_bytes(raw_data: bytes, *, delimiter: str | None = None) -> pd.DataFrame:
    resolved_delimiter = delimiter or infer_csv_delimiter(raw_data)
    return pd.read_csv(BytesIO(raw_data), sep=resolved_delimiter)


def filter_dataframe(
    dataframe: pd.DataFrame,
    *,
    search_text: str = "",
    search_columns: Iterable[str] | None = None,
    categorical_filters: dict[str, list[object]] | None = None,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    filtered = dataframe.copy()
    if search_text.strip():
        columns = list(search_columns or dataframe.columns)
        valid_columns = [column for column in columns if column in dataframe.columns]
        if valid_columns:
            match_mask = pd.Series(False, index=dataframe.index)
            for column in valid_columns:
                match_mask |= (
                    dataframe[column]
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        search_text.strip(),
                        case=False,
                        regex=False,
                    )
                )
            filtered = filtered.loc[match_mask]

    for column, values in (categorical_filters or {}).items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]

    for column, bounds in (numeric_ranges or {}).items():
        if column in filtered.columns:
            lower_bound, upper_bound = bounds
            numeric_values = pd.to_numeric(filtered[column], errors="coerce")
            filtered = filtered[numeric_values.between(lower_bound, upper_bound)]

    return filtered
