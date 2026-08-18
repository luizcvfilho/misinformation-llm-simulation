from __future__ import annotations

import sys
from math import ceil
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
    build_csv_comparison,
    filter_dataframe,
    read_csv_bytes,
)

CSV_1_COLOR = "#173f46"
CSV_2_COLOR = "#4d2a38"
COMPARISON_TEXT_COLOR = "#f4f7f9"


def _read_uploaded_csv(uploaded_file: Any, delimiter: str | None) -> pd.DataFrame:
    return read_csv_bytes(uploaded_file.getvalue(), delimiter=delimiter)


def _render_filters(dataframe: pd.DataFrame, *, key_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    st.sidebar.header("Filters")
    search_text = st.sidebar.text_input("Search text", key=f"{key_prefix}-search-text")
    search_columns = st.sidebar.multiselect(
        "Search in columns",
        options=list(dataframe.columns),
        default=list(dataframe.columns),
        key=f"{key_prefix}-search-columns",
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
        key=f"{key_prefix}-categorical-columns",
    )
    for column in selected_categorical_columns:
        values = dataframe[column].dropna().unique().tolist()
        categorical_filters[column] = st.sidebar.multiselect(
            column,
            options=values,
            key=f"{key_prefix}-category-{column}",
        )

    numeric_ranges: dict[str, tuple[float, float]] = {}
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    selected_numeric_columns = st.sidebar.multiselect(
        "Numeric filters",
        options=numeric_columns,
        key=f"{key_prefix}-numeric-columns",
    )
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
            key=f"{key_prefix}-numeric-{column}",
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
        key=f"{key_prefix}-visible-columns",
    )
    return filtered, display_columns


def _render_chart(dataframe: pd.DataFrame, *, key_prefix: str) -> None:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns are available for a chart.")
        return

    metric_column = st.selectbox("Metric", options=numeric_columns, key=f"{key_prefix}-metric")
    categorical_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=True) <= 40 and column != metric_column
    ]
    group_column = st.selectbox(
        "Group by",
        options=["None", *categorical_columns],
        key=f"{key_prefix}-group-by",
    )
    if group_column == "None":
        counts, bin_edges = np.histogram(dataframe[metric_column].dropna(), bins=20)
        chart_data = pd.DataFrame(
            {"count": counts},
            index=[f"{edge:.3g}" for edge in bin_edges[:-1]],
        )
        chart_data.index.name = "bin"
        st.bar_chart(chart_data)
        return

    chart_data = (
        dataframe.groupby(group_column, dropna=False)[metric_column]
        .mean()
        .sort_values()
        .rename("mean")
        .to_frame()
    )
    chart_data.index.name = "group"
    st.bar_chart(chart_data)


def _render_row_details(dataframe: pd.DataFrame, *, key_prefix: str) -> None:
    if dataframe.empty:
        st.info("No rows match the current filters.")
        return

    row_index = st.selectbox(
        "Row",
        options=dataframe.index,
        format_func=lambda value: str(value),
        key=f"{key_prefix}-row",
    )
    row = dataframe.loc[row_index]
    for column, value in row.items():
        text_value = "" if pd.isna(value) else str(value)
        estimated_lines = sum(max(1, ceil(len(line) / 100)) for line in text_value.splitlines())
        height = min(max(68, 24 * max(estimated_lines, 1) + 16), 800)
        st.text_area(
            label=column,
            value=text_value,
            height=height,
            disabled=True,
            key=f"{key_prefix}-row-detail-{row_index}-{column}",
        )


def _style_comparison_dataframe(
    dataframe: pd.DataFrame,
    *,
    left_columns: list[str],
    right_columns: list[str],
) -> pd.io.formats.style.Styler:
    styled = dataframe.style
    visible_left_columns = [column for column in left_columns if column in dataframe.columns]
    visible_right_columns = [column for column in right_columns if column in dataframe.columns]
    if visible_left_columns:
        styled = styled.set_properties(
            subset=visible_left_columns,
            **{"background-color": CSV_1_COLOR, "color": COMPARISON_TEXT_COLOR},
        )
    if visible_right_columns:
        styled = styled.set_properties(
            subset=visible_right_columns,
            **{"background-color": CSV_2_COLOR, "color": COMPARISON_TEXT_COLOR},
        )
    return styled


def _render_data_views(
    dataframe: pd.DataFrame,
    *,
    display_columns: list[str],
    file_name: str,
    key_prefix: str,
    left_columns: list[str] | None = None,
    right_columns: list[str] | None = None,
) -> None:
    left, right, third = st.columns(3)
    left.metric("Rows", f"{len(dataframe):,}")
    right.metric("Columns", len(dataframe.columns))
    third.metric("File", file_name)

    table_tab, chart_tab, details_tab = st.tabs(["Table", "Chart", "Row details"])
    with table_tab:
        visible_data = dataframe[display_columns] if display_columns else dataframe
        if left_columns is not None and right_columns is not None:
            visible_data = _style_comparison_dataframe(
                visible_data,
                left_columns=left_columns,
                right_columns=right_columns,
            )
        st.dataframe(visible_data, use_container_width=True, height=560)
        st.download_button(
            "Download filtered CSV",
            data=dataframe.to_csv(index=False).encode("utf-8"),
            file_name=f"filtered_{file_name}",
            mime="text/csv",
            key=f"{key_prefix}-download",
        )
    with chart_tab:
        _render_chart(dataframe, key_prefix=key_prefix)
    with details_tab:
        _render_row_details(dataframe, key_prefix=key_prefix)


def _render_single_csv(dataframe: pd.DataFrame, *, file_name: str) -> None:
    filtered, display_columns = _render_filters(dataframe, key_prefix="single")
    _render_data_views(
        filtered,
        display_columns=display_columns,
        file_name=file_name,
        key_prefix="single",
    )


def _render_csv_comparison(
    left_dataframe: pd.DataFrame,
    right_dataframe: pd.DataFrame,
    *,
    left_file_name: str,
    right_file_name: str,
) -> None:
    st.sidebar.header("Join")
    left_join_column = st.sidebar.selectbox(
        "CSV 1 join column",
        options=list(left_dataframe.columns),
        key="comparison-left-join",
    )
    right_join_column = st.sidebar.selectbox(
        "CSV 2 join column",
        options=list(right_dataframe.columns),
        key="comparison-right-join",
    )
    join_types = {
        "Matching rows": "inner",
        "All CSV 1 rows": "left",
        "All CSV 2 rows": "right",
        "All rows": "outer",
    }
    join_label = st.sidebar.selectbox(
        "Join type",
        options=list(join_types),
        key="comparison-join-type",
    )
    left_columns = st.sidebar.multiselect(
        "CSV 1 columns",
        options=[column for column in left_dataframe.columns if column != left_join_column],
        default=[column for column in left_dataframe.columns if column != left_join_column],
        key="comparison-left-columns",
    )
    right_columns = st.sidebar.multiselect(
        "CSV 2 columns",
        options=[column for column in right_dataframe.columns if column != right_join_column],
        default=[column for column in right_dataframe.columns if column != right_join_column],
        key="comparison-right-columns",
    )
    try:
        comparison = build_csv_comparison(
            left_dataframe,
            right_dataframe,
            left_join_column=left_join_column,
            right_join_column=right_join_column,
            left_columns=left_columns,
            right_columns=right_columns,
            how=join_types[join_label],
        )
    except ValueError as error:
        st.error(f"Unable to compare CSVs: {error}")
        return

    filtered, display_columns = _render_filters(comparison.dataframe, key_prefix="comparison")
    _render_data_views(
        filtered,
        display_columns=display_columns,
        file_name=f"comparison_{left_file_name}_{right_file_name}",
        key_prefix="comparison",
        left_columns=comparison.left_columns,
        right_columns=comparison.right_columns,
    )


def main() -> None:
    st.set_page_config(page_title="CSV Explorer", layout="wide")
    st.title("CSV Explorer")
    st.sidebar.header("Source")
    mode = st.sidebar.radio(
        "Mode",
        options=["Explore CSV", "Compare CSVs"],
        key="csv-explorer-mode",
    )
    if mode == "Explore CSV":
        uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"], key="single-file")
        delimiter_label = st.sidebar.selectbox(
            "Delimiter",
            options=list(DELIMITER_OPTIONS),
            key="single-delimiter",
        )
        if uploaded_file is None:
            st.info("Select a CSV file to begin.")
            return
        try:
            dataframe = _read_uploaded_csv(uploaded_file, DELIMITER_OPTIONS[delimiter_label])
        except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Unable to load CSV: {error}")
            return
        _render_single_csv(dataframe, file_name=uploaded_file.name)
        return

    left_uploaded_file = st.sidebar.file_uploader("CSV 1", type=["csv"], key="comparison-left-file")
    left_delimiter_label = st.sidebar.selectbox(
        "CSV 1 delimiter",
        options=list(DELIMITER_OPTIONS),
        key="comparison-left-delimiter",
    )
    right_uploaded_file = st.sidebar.file_uploader(
        "CSV 2",
        type=["csv"],
        key="comparison-right-file",
    )
    right_delimiter_label = st.sidebar.selectbox(
        "CSV 2 delimiter",
        options=list(DELIMITER_OPTIONS),
        key="comparison-right-delimiter",
    )
    if left_uploaded_file is None or right_uploaded_file is None:
        st.info("Select two CSV files to compare.")
        return
    try:
        left_dataframe = _read_uploaded_csv(
            left_uploaded_file,
            DELIMITER_OPTIONS[left_delimiter_label],
        )
        right_dataframe = _read_uploaded_csv(
            right_uploaded_file,
            DELIMITER_OPTIONS[right_delimiter_label],
        )
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to load CSV: {error}")
        return
    _render_csv_comparison(
        left_dataframe,
        right_dataframe,
        left_file_name=left_uploaded_file.name,
        right_file_name=right_uploaded_file.name,
    )


if __name__ == "__main__":
    main()
