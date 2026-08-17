from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from misinformation_simulation.apps.csv_explorer import (  # noqa: E402
    DELIMITER_OPTIONS,
    filter_dataframe,
    read_csv_bytes,
)


def _read_uploaded_csv(uploaded_file: Any, delimiter: str | None) -> pd.DataFrame:
    return read_csv_bytes(uploaded_file.getvalue(), delimiter=delimiter)


def _render_filters(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    st.sidebar.header("Filters")
    search_text = st.sidebar.text_input("Search text")
    search_columns = st.sidebar.multiselect(
        "Search in columns",
        options=list(dataframe.columns),
        default=list(dataframe.columns),
    )

    categorical_filters: dict[str, list[object]] = {}
    categorical_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) <= 40
        and not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    selected_categorical_columns = st.sidebar.multiselect(
        "Categorical filters",
        options=categorical_columns,
    )
    for column in selected_categorical_columns:
        values = dataframe[column].dropna().unique().tolist()
        categorical_filters[column] = st.sidebar.multiselect(column, options=values)

    numeric_ranges: dict[str, tuple[float, float]] = {}
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    selected_numeric_columns = st.sidebar.multiselect("Numeric filters", options=numeric_columns)
    for column in selected_numeric_columns:
        numeric_values = dataframe[column].dropna()
        if numeric_values.empty:
            continue
        minimum = float(numeric_values.min())
        maximum = float(numeric_values.max())
        if minimum == maximum:
            continue
        numeric_ranges[column] = st.sidebar.slider(
            column,
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
        )

    filtered = filter_dataframe(
        dataframe,
        search_text=search_text,
        search_columns=search_columns,
        categorical_filters=categorical_filters,
        numeric_ranges=numeric_ranges,
    )
    display_columns = st.sidebar.multiselect(
        "Visible columns",
        options=list(dataframe.columns),
        default=list(dataframe.columns[: min(12, len(dataframe.columns))]),
    )
    return filtered, display_columns


def _render_chart(dataframe: pd.DataFrame) -> None:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns are available for a chart.")
        return

    metric_column = st.selectbox("Metric", options=numeric_columns)
    categorical_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) <= 40 and column != metric_column
    ]
    group_column = st.selectbox("Group by", options=["None", *categorical_columns])
    if group_column == "None":
        counts, bin_edges = np.histogram(dataframe[metric_column].dropna(), bins=20)
        chart_data = pd.DataFrame(
            {metric_column: counts},
            index=[f"{edge:.3g}" for edge in bin_edges[:-1]],
        )
        st.bar_chart(chart_data)
        return

    chart_data = dataframe.groupby(group_column, dropna=False)[metric_column].mean().sort_values()
    st.bar_chart(chart_data)


def _render_row_details(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        st.info("No rows match the current filters.")
        return

    row_index = st.selectbox("Row", options=dataframe.index, format_func=lambda value: str(value))
    row = dataframe.loc[row_index]
    details = pd.DataFrame({"column": row.index, "value": row.values})
    st.dataframe(details, use_container_width=True, height=560, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="CSV Explorer", layout="wide")
    st.title("CSV Explorer")
    st.sidebar.header("Source")
    uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
    delimiter_label = st.sidebar.selectbox("Delimiter", options=list(DELIMITER_OPTIONS))
    if uploaded_file is None:
        st.info("Select a CSV file to begin.")
        return

    try:
        dataframe = _read_uploaded_csv(uploaded_file, DELIMITER_OPTIONS[delimiter_label])
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to load CSV: {error}")
        return

    filtered, display_columns = _render_filters(dataframe)
    left, right, third = st.columns(3)
    left.metric("Rows", f"{len(filtered):,}")
    right.metric("Columns", len(dataframe.columns))
    third.metric("File", uploaded_file.name)

    table_tab, chart_tab, details_tab = st.tabs(["Table", "Chart", "Row details"])
    with table_tab:
        visible_data = filtered[display_columns] if display_columns else filtered
        st.dataframe(visible_data, use_container_width=True, height=560)
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"filtered_{uploaded_file.name}",
            mime="text/csv",
        )
    with chart_tab:
        _render_chart(filtered)
    with details_tab:
        _render_row_details(filtered)


if __name__ == "__main__":
    main()
