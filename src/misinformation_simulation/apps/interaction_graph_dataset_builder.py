from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

UNCLASSIFIED_TOPIC = "Unclassified"


@dataclass(frozen=True)
class TopicSample:
    dataframe: pd.DataFrame
    availability: pd.DataFrame
    requested_counts: dict[str, int]


def extract_primary_topics(
    dataframe: pd.DataFrame,
    *,
    topic_column: str,
    separator: str = ";",
) -> pd.Series:
    """Return one stable, primary topic label for every dataset row."""
    if topic_column not in dataframe.columns:
        raise ValueError(f"Unknown topic column: {topic_column}")

    topics = dataframe[topic_column].fillna("").astype(str).str.split(separator).str[0]
    topics = topics.str.strip()
    return topics.mask(topics.eq(""), UNCLASSIFIED_TOPIC)


def build_topic_availability(
    dataframe: pd.DataFrame,
    *,
    topic_column: str,
    separator: str = ";",
) -> pd.DataFrame:
    topics = extract_primary_topics(dataframe, topic_column=topic_column, separator=separator)
    availability = topics.value_counts().rename_axis("topic").reset_index(name="available")
    return availability.sort_values(["available", "topic"], ascending=[False, True]).reset_index(
        drop=True
    )


def sample_rows_by_topic(
    dataframe: pd.DataFrame,
    *,
    topic_column: str,
    requested_counts: Mapping[str, int],
    seed: int,
    separator: str = ";",
) -> TopicSample:
    """Sample requested row counts per primary topic without changing source columns."""
    availability = build_topic_availability(
        dataframe,
        topic_column=topic_column,
        separator=separator,
    )
    available_by_topic = availability.set_index("topic")["available"].to_dict()
    normalized_requests = {
        str(topic): int(count) for topic, count in requested_counts.items() if int(count) > 0
    }

    for topic, count in normalized_requests.items():
        if topic not in available_by_topic:
            raise ValueError(f"Unknown topic requested: {topic}")
        if count > available_by_topic[topic]:
            raise ValueError(
                f"Requested {count} row(s) for '{topic}', but only "
                f"{available_by_topic[topic]} are available."
            )

    primary_topics = extract_primary_topics(
        dataframe,
        topic_column=topic_column,
        separator=separator,
    )
    selected_frames: list[pd.DataFrame] = []
    for topic in sorted(normalized_requests):
        topic_rows = dataframe.loc[primary_topics.eq(topic)]
        selected_frames.append(
            topic_rows.sample(n=normalized_requests[topic], random_state=_topic_seed(seed, topic))
        )

    if not selected_frames:
        sampled = dataframe.iloc[0:0].copy()
    else:
        sampled = (
            pd.concat(selected_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
        )

    return TopicSample(
        dataframe=sampled,
        availability=availability,
        requested_counts=normalized_requests,
    )


def build_sample_manifest(
    *,
    source_dataset: str,
    topic_column: str,
    separator: str,
    seed: int,
    sample: TopicSample,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": source_dataset,
        "topic_column": topic_column,
        "topic_interpretation": "first label split by separator",
        "topic_separator": separator,
        "seed": seed,
        "requested_counts": sample.requested_counts,
        "available_counts": {
            row.topic: int(row.available) for row in sample.availability.itertuples(index=False)
        },
        "selected_rows": len(sample.dataframe),
    }


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def manifest_to_json_bytes(manifest: dict[str, object]) -> bytes:
    return json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")


def render_dataset_builder_tab(df: pd.DataFrame | None, dataset_label: str) -> None:
    st.subheader("Create a topic-balanced dataset")
    st.caption(
        "Use the dataset selected in the sidebar as the source. The generated CSV preserves its "
        "columns and contains only the sampled news rows."
    )
    if df is None or df.empty:
        st.info("Load a non-empty dataset from the sidebar to create a custom CSV.")
        return

    text_columns = [
        column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if not text_columns:
        st.error("The selected dataset does not contain a text column that can represent topics.")
        return
    default_topic_column = "category" if "category" in df.columns else text_columns[0]
    topic_column = st.selectbox(
        "Topic column",
        options=text_columns,
        index=text_columns.index(default_topic_column),
        help="Column used to identify the main topic of each news item.",
    )
    separator = st.text_input(
        "Multi-topic separator",
        value=";",
        max_chars=1,
        help=(
            "Only the first label is used as the main topic. "
            "For 'business; top', the topic is 'business'."
        ),
    )
    if not separator:
        st.error("Enter a separator character.")
        return

    availability = build_topic_availability(
        df,
        topic_column=topic_column,
        separator=separator,
    )
    request_table = availability.rename(columns={"topic": "Topic", "available": "Available"})
    request_table["Requested"] = 0
    edited_requests = st.data_editor(
        request_table,
        column_config={
            "Topic": st.column_config.TextColumn(disabled=True),
            "Available": st.column_config.NumberColumn(disabled=True, format="%d"),
            "Requested": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
        },
        disabled=["Topic", "Available"],
        hide_index=True,
        key=f"dataset_builder_requests_{topic_column}_{separator}_{len(df)}",
        width="stretch",
    )
    requested_counts = {
        str(row.Topic): int(row.Requested)
        for row in edited_requests.itertuples(index=False)
        if pd.notna(row.Requested) and int(row.Requested) > 0
    }

    seed, output_name = _render_dataset_builder_options(dataset_label)
    requested_total = sum(requested_counts.values())
    st.caption(f"Requested total: {requested_total} news item(s).")
    if requested_total == 0:
        st.info("Set a quantity above zero for at least one topic to prepare the download.")
        return

    try:
        sample = sample_rows_by_topic(
            df,
            topic_column=topic_column,
            requested_counts=requested_counts,
            seed=seed,
            separator=separator,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Custom dataset ready: {len(sample.dataframe)} news item(s).")
    preview_columns = list(sample.dataframe.columns[:8])
    st.dataframe(sample.dataframe[preview_columns].head(10), width="stretch")
    manifest = build_sample_manifest(
        source_dataset=dataset_label,
        topic_column=topic_column,
        separator=separator,
        seed=seed,
        sample=sample,
    )
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download custom CSV",
        data=dataframe_to_csv_bytes(sample.dataframe),
        file_name=output_name,
        mime="text/csv",
        width="stretch",
    )
    manifest_name = f"{output_name.rsplit('.', maxsplit=1)[0]}_manifest.json"
    download_columns[1].download_button(
        "Download sampling manifest",
        data=manifest_to_json_bytes(manifest),
        file_name=manifest_name,
        mime="application/json",
        width="stretch",
    )


def _render_dataset_builder_options(dataset_label: str) -> tuple[int, str]:
    options = st.columns(2)
    seed = options[0].number_input(
        "Sampling seed",
        min_value=0,
        value=42,
        step=1,
        help="Use the same seed and quantities to reproduce the exact sampled dataset.",
    )
    output_name = options[1].text_input(
        "Output CSV filename",
        value=_default_output_name(dataset_label),
    )
    if not output_name.lower().endswith(".csv"):
        output_name = f"{output_name}.csv"
    return int(seed), output_name


def _default_output_name(dataset_label: str) -> str:
    source_name = dataset_label.rsplit("/", maxsplit=1)[-1].rsplit("\\", maxsplit=1)[-1]
    source_stem = source_name.rsplit(".", maxsplit=1)[0] or "news"
    return f"{source_stem}_custom.csv"


def _topic_seed(seed: int, topic: str) -> int:
    digest = hashlib.sha256(f"{seed}:{topic}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")
