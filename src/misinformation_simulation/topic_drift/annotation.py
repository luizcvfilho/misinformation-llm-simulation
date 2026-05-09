from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import pandas as pd

from misinformation_simulation.datasets.selection import choose_news_text_column, resolve_row_text
from misinformation_simulation.enums import Provider
from misinformation_simulation.llm.rewrite import (
    DEFAULT_ALLOW_TITLE_FALLBACK,
    DEFAULT_MAX_REQUESTS_PER_MINUTE,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TEXT_COLUMN,
)
from misinformation_simulation.text_metrics.vad import (
    VADModelBundle,
    VADScore,
    predict_text_vad,
)
from misinformation_simulation.topic_drift.annotation_columns import (
    _apply_metric_defaults,
    _apply_topic_structure_defaults,
    _apply_vad_defaults,
    _ensure_expected_chain_output_columns,
    _initialize_chain_metric_columns,
    _sanitize_column_token,
    _write_vad_score,
)
from misinformation_simulation.topic_drift.extraction import (
    DEFAULT_REWRITTEN_COLUMN,
    DEFAULT_TOPIC_DRIFT_MODEL,
    DEFAULT_TOPIC_DRIFT_PROVIDER,
    extract_topic_structure,
)
from misinformation_simulation.topic_drift.metrics import calculate_stdi_chain_metrics
from misinformation_simulation.topic_drift.models import flatten_topic_structure


def annotate_stdi_for_version_chain(
    df: pd.DataFrame,
    *,
    version_columns: Sequence[str],
    text_column: str | None = DEFAULT_TEXT_COLUMN,
    title_column: str = "title",
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_requests_per_minute: int | None = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    allow_title_fallback: bool = DEFAULT_ALLOW_TITLE_FALLBACK,
    vad_model_bundle: VADModelBundle | None = None,
    vad_scorer: Callable[[str], VADScore] | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if not version_columns:
        raise ValueError("Provide at least one version column.")

    resolved_text_column = text_column or choose_news_text_column(df)
    for column in version_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    result_df = df.copy()

    version_tokens = {column: _sanitize_column_token(column) for column in version_columns}
    _initialize_chain_metric_columns(
        result_df,
        version_tokens=list(version_tokens.values()),
    )

    target_indexes = list(result_df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]

    for row_index in target_indexes:
        row = result_df.loc[row_index]

        _apply_topic_structure_defaults(result_df, row_index=row_index, prefix="original")
        _apply_vad_defaults(result_df, row_index=row_index, prefix="original")
        for version_column in version_columns:
            token = version_tokens[version_column]
            _apply_topic_structure_defaults(result_df, row_index=row_index, prefix=token)
            _apply_vad_defaults(result_df, row_index=row_index, prefix=token)
            _apply_metric_defaults(result_df, row_index=row_index, token=token)

        try:
            source_column, original_text = resolve_row_text(
                row=row,
                preferred_column=resolved_text_column,
                allow_title_fallback=allow_title_fallback,
            )
            result_df.at[row_index, "source_text_column"] = source_column
        except ValueError as exc:
            result_df.at[row_index, "original_topic_structure_status"] = "skipped"
            result_df.at[row_index, "original_topic_structure_error"] = str(exc)
            for version_column in version_columns:
                token = version_tokens[version_column]
                result_df.at[row_index, f"{token}_topic_structure_status"] = "blocked"
                result_df.at[row_index, f"{token}_topic_structure_error"] = (
                    "Skipped because the original text could not be resolved."
                )
            continue

        title = ""
        if title_column in result_df.columns and pd.notna(result_df.at[row_index, title_column]):
            title = str(result_df.at[row_index, title_column]).strip()

        try:
            original_structure = extract_topic_structure(
                text=original_text,
                title=title,
                model=model,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                max_requests_per_minute=max_requests_per_minute,
                retry_attempts=retry_attempts,
            )
            result_df.at[row_index, "original_topic_structure_status"] = "success"
            result_df.at[row_index, "original_topic_structure_error"] = pd.NA
            for column_name, value in flatten_topic_structure(
                original_structure,
                prefix="original",
            ).items():
                result_df.at[row_index, column_name] = value
        except Exception as exc:
            result_df.at[row_index, "original_topic_structure_status"] = "error"
            result_df.at[row_index, "original_topic_structure_error"] = str(exc)
            for version_column in version_columns:
                token = version_tokens[version_column]
                result_df.at[row_index, f"{token}_topic_structure_status"] = "blocked"
                result_df.at[row_index, f"{token}_topic_structure_error"] = (
                    "Skipped because original topic extraction failed."
                )
            continue

        chain_structures = [original_structure]
        chain_labels = ["original"]
        original_vad = predict_text_vad(
            original_text,
            model_bundle=vad_model_bundle,
            scorer=vad_scorer,
        )
        _write_vad_score(
            result_df,
            row_index=row_index,
            prefix="original",
            vad_score=original_vad,
        )
        chain_vad_scores: list[VADScore | None] = [original_vad]

        for version_column in version_columns:
            token = version_tokens[version_column]
            version_text_value = result_df.at[row_index, version_column]
            version_text = "" if pd.isna(version_text_value) else str(version_text_value).strip()

            if not version_text:
                result_df.at[row_index, f"{token}_topic_structure_status"] = "skipped"
                result_df.at[row_index, f"{token}_topic_structure_error"] = (
                    f"Column '{version_column}' has no usable text."
                )
                _apply_metric_defaults(
                    result_df,
                    row_index=row_index,
                    token=token,
                    reference_original_label="original",
                    reference_incremental_label=chain_labels[-1],
                )
                continue

            try:
                version_structure = extract_topic_structure(
                    text=version_text,
                    title=title,
                    model=model,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    max_requests_per_minute=max_requests_per_minute,
                    retry_attempts=retry_attempts,
                )
                result_df.at[row_index, f"{token}_topic_structure_status"] = "success"
                result_df.at[row_index, f"{token}_topic_structure_error"] = pd.NA
                for column_name, value in flatten_topic_structure(
                    version_structure,
                    prefix=token,
                ).items():
                    result_df.at[row_index, column_name] = value
                version_vad = predict_text_vad(
                    version_text,
                    model_bundle=vad_model_bundle,
                    scorer=vad_scorer,
                )
                _write_vad_score(
                    result_df,
                    row_index=row_index,
                    prefix=token,
                    vad_score=version_vad,
                )

                chain_structures.append(version_structure)
                chain_labels.append(token)
                chain_vad_scores.append(version_vad)
            except Exception as exc:
                result_df.at[row_index, f"{token}_topic_structure_status"] = "error"
                result_df.at[row_index, f"{token}_topic_structure_error"] = str(exc)
                _apply_metric_defaults(
                    result_df,
                    row_index=row_index,
                    token=token,
                    reference_original_label="original",
                    reference_incremental_label=chain_labels[-1],
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        chain_metrics = calculate_stdi_chain_metrics(
            chain_structures,
            version_labels=chain_labels,
            vad_scores=chain_vad_scores,
        )
        for metric_row in chain_metrics:
            token = metric_row["version_label"]
            for key, value in metric_row.items():
                if key == "version_label":
                    continue
                result_df.at[row_index, f"{token}_{key}"] = value

    return _ensure_expected_chain_output_columns(
        result_df,
        version_tokens=list(version_tokens.values()),
    )


def annotate_stdi_for_rewrites(
    df: pd.DataFrame,
    *,
    rewritten_column: str = DEFAULT_REWRITTEN_COLUMN,
    text_column: str | None = DEFAULT_TEXT_COLUMN,
    title_column: str = "title",
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_requests_per_minute: int | None = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    allow_title_fallback: bool = DEFAULT_ALLOW_TITLE_FALLBACK,
    vad_model_bundle: VADModelBundle | None = None,
    vad_scorer: Callable[[str], VADScore] | None = None,
) -> pd.DataFrame:
    return annotate_stdi_for_version_chain(
        df=df,
        version_columns=[rewritten_column],
        text_column=text_column,
        title_column=title_column,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_rows=max_rows,
        sleep_seconds=sleep_seconds,
        max_requests_per_minute=max_requests_per_minute,
        retry_attempts=retry_attempts,
        allow_title_fallback=allow_title_fallback,
        vad_model_bundle=vad_model_bundle,
        vad_scorer=vad_scorer,
    )
