from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
from scipy.stats import wilcoxon

from misinformation_simulation.llm.false_to_true import build_false_to_true_prompt

from .vad import VAD_DIMENSIONS

DEFAULT_PAIR_COLUMN = "pair_id"
DEFAULT_TOPIC_COLUMN = "subject"
DEFAULT_ORIGINAL_TEXT_COLUMN = "original_article_text"
DEFAULT_REWRITTEN_TEXT_COLUMN = "rewritten_article_text"


def build_topic_matched_seed_dataset(
    df: pd.DataFrame,
    *,
    text_column: str = "article_text",
    topic_column: str = DEFAULT_TOPIC_COLUMN,
    pair_column: str = DEFAULT_PAIR_COLUMN,
) -> pd.DataFrame:
    """Build the seed dataset used by the topic-matched VAD workflow from Fake.csv."""
    _require_columns(df, [text_column])
    seed_df = df.copy()
    if seed_df.empty:
        raise ValueError("The fake-news seed DataFrame is empty.")

    if "source_row_index" in seed_df.columns:
        source_ids = seed_df["source_row_index"].astype(str)
    else:
        source_ids = seed_df.index.astype(str)

    source_file = (
        seed_df["source_file"].astype(str)
        if "source_file" in seed_df.columns
        else pd.Series("Fake.csv", index=seed_df.index)
    )
    seed_df[pair_column] = source_file + ":" + source_ids
    seed_df["original_id"] = seed_df[pair_column]
    seed_df["original_label"] = "fake"
    seed_df[DEFAULT_ORIGINAL_TEXT_COLUMN] = seed_df[text_column]

    preferred_columns = [
        pair_column,
        "original_id",
        "dataset_name",
        "source_file",
        "source_row_index",
        "original_label",
        topic_column,
        "title",
        "text",
        DEFAULT_ORIGINAL_TEXT_COLUMN,
        "article_char_count",
        "article_word_count",
        "date",
    ]
    existing_preferred = [column for column in preferred_columns if column in seed_df.columns]
    remaining = [column for column in seed_df.columns if column not in existing_preferred]
    return seed_df[existing_preferred + remaining].reset_index(drop=True)


def build_rewrite_prompt(
    row: pd.Series,
    *,
    topic_column: str = DEFAULT_TOPIC_COLUMN,
    title_column: str = "title",
    text_column: str = DEFAULT_ORIGINAL_TEXT_COLUMN,
) -> str:
    """Create a rewriting prompt that preserves topic while correcting false claims."""
    return build_false_to_true_prompt(
        row.get(text_column, ""),
        topic=row.get(topic_column, "unknown"),
        title=row.get(title_column, ""),
    )


def validate_rewritten_pairs(
    df: pd.DataFrame,
    *,
    pair_column: str = DEFAULT_PAIR_COLUMN,
    original_text_column: str = DEFAULT_ORIGINAL_TEXT_COLUMN,
    rewritten_text_column: str = DEFAULT_REWRITTEN_TEXT_COLUMN,
) -> None:
    _require_columns(df, [pair_column, original_text_column, rewritten_text_column])
    duplicated_pairs = df[pair_column].duplicated(keep=False)
    if duplicated_pairs.any():
        examples = df.loc[duplicated_pairs, pair_column].head(5).tolist()
        raise ValueError(f"Duplicated pair ids found: {examples}")


def prepare_long_vad_frame(
    paired_df: pd.DataFrame,
    *,
    pair_column: str = DEFAULT_PAIR_COLUMN,
    topic_column: str = DEFAULT_TOPIC_COLUMN,
    original_text_column: str = DEFAULT_ORIGINAL_TEXT_COLUMN,
    rewritten_text_column: str = DEFAULT_REWRITTEN_TEXT_COLUMN,
) -> pd.DataFrame:
    """Convert paired false/rewritten rows into a long frame for VAD scoring."""
    validate_rewritten_pairs(
        paired_df,
        pair_column=pair_column,
        original_text_column=original_text_column,
        rewritten_text_column=rewritten_text_column,
    )

    metadata_columns = [
        column
        for column in [pair_column, "original_id", topic_column, "title", "date"]
        if column in paired_df.columns
    ]
    original_frame = paired_df[metadata_columns].copy()
    original_frame["variant"] = "false_original"
    original_frame["article_text"] = paired_df[original_text_column]

    rewritten_frame = paired_df[metadata_columns].copy()
    rewritten_frame["variant"] = "rewritten_true"
    rewritten_frame["article_text"] = paired_df[rewritten_text_column]

    return pd.concat([original_frame, rewritten_frame], ignore_index=True)


def compute_paired_vad_deltas(
    scored_long_df: pd.DataFrame,
    *,
    pair_column: str = DEFAULT_PAIR_COLUMN,
    variant_column: str = "variant",
    original_variant: str = "false_original",
    rewritten_variant: str = "rewritten_true",
    prefix: str = "vad",
    metadata_columns: Iterable[str] = ("original_id", DEFAULT_TOPIC_COLUMN, "title", "date"),
) -> pd.DataFrame:
    """Compute rewritten-minus-original VAD deltas for each pair."""
    required_columns = [pair_column, variant_column, *[f"{prefix}_{dim}" for dim in VAD_DIMENSIONS]]
    _require_columns(scored_long_df, required_columns)

    score_columns = required_columns[2:]
    pivot = scored_long_df.pivot(
        index=pair_column,
        columns=variant_column,
        values=score_columns,
    )
    metadata = (
        scored_long_df[[pair_column, *[c for c in metadata_columns if c in scored_long_df.columns]]]
        .drop_duplicates(subset=[pair_column])
        .sort_values(pair_column)
        .reset_index(drop=True)
    )

    deltas = metadata.copy()
    for dimension in VAD_DIMENSIONS:
        score_column = f"{prefix}_{dimension}"
        false_column = f"false_{dimension}"
        rewritten_column = f"rewritten_{dimension}"
        delta_column = f"delta_{dimension}"
        deltas[false_column] = (
            pivot[(score_column, original_variant)].reindex(deltas[pair_column]).to_numpy()
        )
        deltas[rewritten_column] = (
            pivot[(score_column, rewritten_variant)].reindex(deltas[pair_column]).to_numpy()
        )
        deltas[delta_column] = deltas[rewritten_column] - deltas[false_column]

    delta_columns = [f"delta_{dimension}" for dimension in VAD_DIMENSIONS]
    deltas["absolute_delta_sum"] = deltas[delta_columns].abs().sum(axis=1)
    return deltas.sort_values("absolute_delta_sum", ascending=False).reset_index(drop=True)


def summarize_paired_vad_deltas(
    pair_delta_df: pd.DataFrame,
    *,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Summarize paired rewritten-minus-original VAD deltas globally or by topic."""
    if group_column is not None:
        _require_columns(pair_delta_df, [group_column])
        groups = pair_delta_df.groupby(group_column, dropna=False)
    else:
        groups = [("all", pair_delta_df)]

    rows: list[dict[str, float | int | str]] = []
    for group_value, group_frame in groups:
        row: dict[str, float | int | str] = {
            "group": str(group_value),
            "pair_count": int(len(group_frame)),
        }
        for dimension in VAD_DIMENSIONS:
            false_values = group_frame[f"false_{dimension}"].dropna()
            rewritten_values = group_frame[f"rewritten_{dimension}"].dropna()
            delta_values = group_frame[f"delta_{dimension}"].dropna()
            row[f"false_{dimension}_mean"] = _mean_or_nan(false_values)
            row[f"rewritten_{dimension}_mean"] = _mean_or_nan(rewritten_values)
            row[f"delta_{dimension}_mean"] = _mean_or_nan(delta_values)
            row[f"delta_{dimension}_median"] = _median_or_nan(delta_values)
            row[f"wilcoxon_{dimension}_pvalue"] = _wilcoxon_pvalue(
                false_values,
                rewritten_values,
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pair_count", ascending=False).reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else math.nan


def _median_or_nan(series: pd.Series) -> float:
    return float(series.median()) if not series.empty else math.nan


def _wilcoxon_pvalue(original_values: pd.Series, rewritten_values: pd.Series) -> float:
    aligned = pd.concat([original_values, rewritten_values], axis=1).dropna()
    if len(aligned) < 2:
        return math.nan
    try:
        return float(wilcoxon(aligned.iloc[:, 0], aligned.iloc[:, 1]).pvalue)
    except ValueError:
        return math.nan
