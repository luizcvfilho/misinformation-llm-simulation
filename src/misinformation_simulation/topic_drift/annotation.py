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
    DEFAULT_VAD_MODEL_NAME,
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
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    return _ChainAnnotator(
        df=df,
        version_columns=version_columns,
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
        progress_callback=progress_callback,
    ).run()


class _ChainAnnotator:
    """Annotate versions while retaining the last successful reference."""

    def __init__(
        self,
        *,
        df,
        version_columns,
        text_column,
        title_column,
        provider,
        model,
        api_key,
        base_url,
        max_rows,
        sleep_seconds,
        max_requests_per_minute,
        retry_attempts,
        allow_title_fallback,
        vad_model_bundle,
        vad_scorer,
        progress_callback,
    ):
        self.df = df
        self.version_columns = version_columns
        self.text_column = text_column
        self.title_column = title_column
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_rows = max_rows
        self.sleep_seconds = sleep_seconds
        self.max_requests_per_minute = max_requests_per_minute
        self.retry_attempts = retry_attempts
        self.allow_title_fallback = allow_title_fallback
        self.vad_model_bundle = vad_model_bundle
        self.vad_scorer = vad_scorer
        self.progress_callback = progress_callback

    def run(self):
        self._validate_input()

        self._prepare_result()

        self.total_rows = len(self.target_indexes)
        for completed_rows, row_index in enumerate(self.target_indexes, start=1):
            self.completed_rows, self.row_index = (completed_rows, row_index)
            self._annotate_row()

        return _ensure_expected_chain_output_columns(
            self.result_df, version_tokens=list(self.version_tokens.values())
        )

    def _validate_input(self):
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("'df' must be a pandas.DataFrame.")
        if self.df.empty:
            raise ValueError("The DataFrame is empty.")
        if not self.version_columns:
            raise ValueError("Provide at least one version column.")
        self.resolved_text_column = self.text_column or choose_news_text_column(self.df)
        for column in self.version_columns:
            self.column = column
            if self.column not in self.df.columns:
                raise ValueError(f"Column '{self.column}' does not exist in the DataFrame.")

    def _prepare_result(self):
        self.result_df = self.df.copy()
        self.version_tokens = {
            column: _sanitize_column_token(column) for column in self.version_columns
        }
        _initialize_chain_metric_columns(
            self.result_df, version_tokens=list(self.version_tokens.values())
        )
        self.vad_model_name = (
            "custom_scorer"
            if self.vad_scorer is not None
            else self.vad_model_bundle.model_name
            if self.vad_model_bundle is not None
            else DEFAULT_VAD_MODEL_NAME
        )
        for prefix in ("original", *self.version_tokens.values()):
            self.prefix = prefix
            for suffix in ("extraction_model", "extraction_provider", "vad_model"):
                self.suffix = suffix
                self.column = f"{self.prefix}_{self.suffix}"
                if self.column not in self.result_df:
                    self.result_df[self.column] = pd.NA
                self.result_df[self.column] = self.result_df[self.column].astype("object")
        self.target_indexes = list(self.result_df.index)
        if self.max_rows is not None:
            self.target_indexes = self.target_indexes[: self.max_rows]

    def _annotate_version(self):
        self.token = self.version_tokens[self.version_column]
        version_text_value = self.result_df.at[self.row_index, self.version_column]
        version_text = "" if pd.isna(version_text_value) else str(version_text_value).strip()
        if not version_text:
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_status"] = "skipped"
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_error"] = (
                f"Column '{self.version_column}' has no usable text."
            )
            _apply_metric_defaults(
                self.result_df,
                row_index=self.row_index,
                token=self.token,
                reference_original_label="original",
                reference_incremental_label=self.chain_labels[-1],
            )
            return
        try:
            self.result_df.at[self.row_index, f"{self.token}_extraction_model"] = str(self.model)
            self.result_df.at[self.row_index, f"{self.token}_extraction_provider"] = str(
                self.provider
            )
            version_structure = extract_topic_structure(
                text=version_text,
                title=self.title,
                model=self.model,
                provider=self.provider,
                api_key=self.api_key,
                base_url=self.base_url,
                max_requests_per_minute=self.max_requests_per_minute,
                retry_attempts=self.retry_attempts,
            )
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_status"] = "success"
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_error"] = pd.NA
            for column_name, value in flatten_topic_structure(
                version_structure, prefix=self.token
            ).items():
                self.column_name, self.value = (column_name, value)
                self.result_df.at[self.row_index, self.column_name] = self.value
            self.result_df.at[self.row_index, f"{self.token}_vad_model"] = self.vad_model_name
            version_vad = predict_text_vad(
                version_text,
                model_bundle=self.vad_model_bundle,
                scorer=self.vad_scorer,
            )
            _write_vad_score(
                self.result_df,
                row_index=self.row_index,
                prefix=self.token,
                vad_score=version_vad,
            )
            self.chain_structures.append(version_structure)
            self.chain_labels.append(self.token)
            self.chain_vad_scores.append(version_vad)
        except Exception as exc:
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_status"] = "error"
            self.result_df.at[self.row_index, f"{self.token}_topic_structure_error"] = str(exc)
            _apply_metric_defaults(
                self.result_df,
                row_index=self.row_index,
                token=self.token,
                reference_original_label="original",
                reference_incremental_label=self.chain_labels[-1],
            )
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _annotate_row(self):
        self._reset_row()

        try:
            source_column, self.original_text = resolve_row_text(
                row=self.row,
                preferred_column=self.resolved_text_column,
                allow_title_fallback=self.allow_title_fallback,
            )
            self.result_df.at[self.row_index, "source_text_column"] = source_column
        except ValueError as exc:
            self.result_df.at[self.row_index, "original_topic_structure_status"] = "skipped"
            self.result_df.at[self.row_index, "original_topic_structure_error"] = str(exc)
            for version_column in self.version_columns:
                self.version_column = version_column
                self.token = self.version_tokens[self.version_column]
                self.result_df.at[self.row_index, f"{self.token}_topic_structure_status"] = (
                    "blocked"
                )
                self.result_df.at[self.row_index, f"{self.token}_topic_structure_error"] = (
                    "Skipped because the original text could not be resolved."
                )
            _notify_progress(self.progress_callback, self.completed_rows, self.total_rows)
            return
        self.title = ""
        if self.title_column in self.result_df.columns and pd.notna(
            self.result_df.at[self.row_index, self.title_column]
        ):
            self.title = str(self.result_df.at[self.row_index, self.title_column]).strip()
        try:
            self._extract_original()

        except Exception as exc:
            self.result_df.at[self.row_index, "original_topic_structure_status"] = "error"
            self.result_df.at[self.row_index, "original_topic_structure_error"] = str(exc)
            for version_column in self.version_columns:
                self.version_column = version_column
                self.token = self.version_tokens[self.version_column]
                self.result_df.at[self.row_index, f"{self.token}_topic_structure_status"] = (
                    "blocked"
                )
                self.result_df.at[self.row_index, f"{self.token}_topic_structure_error"] = (
                    "Skipped because original topic extraction failed."
                )
            _notify_progress(self.progress_callback, self.completed_rows, self.total_rows)
            return
        self.chain_structures = [self.original_structure]
        self.chain_labels = ["original"]
        self.result_df.at[self.row_index, "original_vad_model"] = self.vad_model_name
        original_vad = predict_text_vad(
            self.original_text, model_bundle=self.vad_model_bundle, scorer=self.vad_scorer
        )
        _write_vad_score(
            self.result_df,
            row_index=self.row_index,
            prefix="original",
            vad_score=original_vad,
        )
        self.chain_vad_scores: list[VADScore | None] = [original_vad]
        for version_column in self.version_columns:
            self.version_column = version_column
            self._annotate_version()

        chain_metrics = calculate_stdi_chain_metrics(
            self.chain_structures,
            version_labels=self.chain_labels,
            vad_scores=self.chain_vad_scores,
        )
        for metric_row in chain_metrics:
            self.token = metric_row["version_label"]
            for key, value in metric_row.items():
                key, self.value = (key, value)
                if key == "version_label":
                    continue
                self.result_df.at[self.row_index, f"{self.token}_{key}"] = self.value
        _notify_progress(self.progress_callback, self.completed_rows, self.total_rows)

    def _reset_row(self):
        self.row = self.result_df.loc[self.row_index]
        for prefix in ("original", *self.version_tokens.values()):
            self.prefix = prefix
            for suffix in ("extraction_model", "extraction_provider", "vad_model"):
                self.suffix = suffix
                self.result_df.at[self.row_index, f"{self.prefix}_{self.suffix}"] = pd.NA
        _apply_topic_structure_defaults(self.result_df, row_index=self.row_index, prefix="original")
        _apply_vad_defaults(self.result_df, row_index=self.row_index, prefix="original")
        for version_column in self.version_columns:
            self.version_column = version_column
            self.token = self.version_tokens[self.version_column]
            _apply_topic_structure_defaults(
                self.result_df, row_index=self.row_index, prefix=self.token
            )
            _apply_vad_defaults(self.result_df, row_index=self.row_index, prefix=self.token)
            _apply_metric_defaults(self.result_df, row_index=self.row_index, token=self.token)

    def _extract_original(self):
        self.result_df.at[self.row_index, "original_extraction_model"] = str(self.model)
        self.result_df.at[self.row_index, "original_extraction_provider"] = str(self.provider)
        self.original_structure = extract_topic_structure(
            text=self.original_text,
            title=self.title,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            max_requests_per_minute=self.max_requests_per_minute,
            retry_attempts=self.retry_attempts,
        )
        self.result_df.at[self.row_index, "original_topic_structure_status"] = "success"
        self.result_df.at[self.row_index, "original_topic_structure_error"] = pd.NA
        for column_name, value in flatten_topic_structure(
            self.original_structure, prefix="original"
        ).items():
            self.column_name, self.value = (column_name, value)
            self.result_df.at[self.row_index, self.column_name] = self.value


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
    progress_callback: Callable[[int, int], None] | None = None,
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
        progress_callback=progress_callback,
    )


def _notify_progress(
    progress_callback: Callable[[int, int], None] | None,
    completed_rows: int,
    total_rows: int,
) -> None:
    if progress_callback is not None:
        progress_callback(completed_rows, total_rows)
