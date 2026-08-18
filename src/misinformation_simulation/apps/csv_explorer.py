from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO

import pandas as pd

DELIMITER_OPTIONS = {
    "Auto": None,
    "Comma": ",",
    "Semicolon": ";",
    "Tab": "\t",
    "Pipe": "|",
}


@dataclass(frozen=True)
class CsvComparison:
    dataframe: pd.DataFrame
    left_columns: list[str]
    right_columns: list[str]


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


def build_csv_comparison(
    left_dataframe: pd.DataFrame,
    right_dataframe: pd.DataFrame,
    *,
    left_join_column: str,
    right_join_column: str,
    left_columns: Iterable[str],
    right_columns: Iterable[str],
    how: str = "inner",
) -> CsvComparison:
    if left_join_column not in left_dataframe.columns:
        raise ValueError(f"Unknown CSV 1 join column: {left_join_column}")
    if right_join_column not in right_dataframe.columns:
        raise ValueError(f"Unknown CSV 2 join column: {right_join_column}")
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError(f"Unsupported join type: {how}")

    selected_left_columns = _select_comparison_columns(
        left_columns,
        left_dataframe.columns,
        excluded_column=left_join_column,
    )
    selected_right_columns = _select_comparison_columns(
        right_columns,
        right_dataframe.columns,
        excluded_column=right_join_column,
    )
    left_key = "__csv_explorer_left_join_key__"
    right_key = "__csv_explorer_right_join_key__"
    left_payload, left_output_columns = _build_comparison_payload(
        left_dataframe,
        join_column=left_join_column,
        selected_columns=selected_left_columns,
        key_column=left_key,
        source_label="CSV 1",
    )
    right_payload, right_output_columns = _build_comparison_payload(
        right_dataframe,
        join_column=right_join_column,
        selected_columns=selected_right_columns,
        key_column=right_key,
        source_label="CSV 2",
    )
    merged = left_payload.merge(right_payload, how=how, left_on=left_key, right_on=right_key)
    join_label = f"Join key: {left_join_column} = {right_join_column}"
    result = pd.DataFrame({join_label: merged[left_key].combine_first(merged[right_key])})
    for internal_column, output_column in left_output_columns.items():
        result[output_column] = merged[internal_column]
    for internal_column, output_column in right_output_columns.items():
        result[output_column] = merged[internal_column]

    return CsvComparison(
        dataframe=result,
        left_columns=list(left_output_columns.values()),
        right_columns=list(right_output_columns.values()),
    )


def _select_comparison_columns(
    requested_columns: Iterable[str],
    available_columns: Iterable[str],
    *,
    excluded_column: str,
) -> list[str]:
    available = set(available_columns)
    return list(
        dict.fromkeys(
            column
            for column in requested_columns
            if column in available and column != excluded_column
        )
    )


def _build_comparison_payload(
    dataframe: pd.DataFrame,
    *,
    join_column: str,
    selected_columns: list[str],
    key_column: str,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    payload = pd.DataFrame({key_column: dataframe[join_column]})
    output_columns: dict[str, str] = {}
    for position, column in enumerate(selected_columns):
        internal_column = f"{key_column}_{position}"
        output_column = f"{source_label}: {column}"
        payload[internal_column] = dataframe[column]
        output_columns[internal_column] = output_column
    return payload, output_columns


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
