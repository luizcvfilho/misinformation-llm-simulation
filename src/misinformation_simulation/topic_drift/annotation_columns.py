from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.models import (
    empty_topic_structure,
    flatten_topic_structure,
)


def _sanitize_column_token(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").lower()
    return normalized or "version"


def _topic_structure_field_suffixes() -> tuple[str, ...]:
    return (
        "main_topic",
        "subtopics",
        "central_entities",
        "central_relations",
        "narrative_frame",
        "has_internal_contradiction",
        "json",
    )


def _topic_drift_metric_suffixes() -> tuple[str, ...]:
    return (
        "reference_original_label",
        "reference_incremental_label",
        "stdi_vs_original",
        "theme_drift_vs_original",
        "subtopic_drift_vs_original",
        "entity_drift_vs_original",
        "relation_drift_vs_original",
        "contradiction_drift_vs_original",
        "valence_drift_vs_original",
        "arousal_drift_vs_original",
        "dominance_drift_vs_original",
        "vad_drift_vs_original",
        "stdi_incremental",
        "theme_drift_incremental",
        "subtopic_drift_incremental",
        "entity_drift_incremental",
        "relation_drift_incremental",
        "contradiction_drift_incremental",
        "valence_drift_incremental",
        "arousal_drift_incremental",
        "dominance_drift_incremental",
        "vad_drift_incremental",
        "stdi_cumulative",
    )


def _topic_drift_vad_suffixes() -> tuple[str, ...]:
    return ("vad_valence", "vad_arousal", "vad_dominance")


def _expected_chain_output_columns(version_tokens: Sequence[str]) -> dict[str, Any]:
    expected_columns: dict[str, Any] = {
        "source_text_column": pd.NA,
        "original_topic_structure_status": "not_requested",
        "original_topic_structure_error": pd.NA,
    }

    for suffix in _topic_structure_field_suffixes():
        expected_columns[f"original_{suffix}"] = pd.NA
    for suffix in _topic_drift_vad_suffixes():
        expected_columns[f"original_{suffix}"] = pd.NA

    for token in version_tokens:
        expected_columns[f"{token}_topic_structure_status"] = "not_requested"
        expected_columns[f"{token}_topic_structure_error"] = pd.NA

        for suffix in _topic_structure_field_suffixes():
            expected_columns[f"{token}_{suffix}"] = pd.NA
        for suffix in _topic_drift_vad_suffixes():
            expected_columns[f"{token}_{suffix}"] = pd.NA

        for suffix in _topic_drift_metric_suffixes():
            expected_columns[f"{token}_{suffix}"] = pd.NA

    return expected_columns


def _default_metric_values(
    *,
    reference_original_label: str = "original",
    reference_incremental_label: str = "original",
) -> dict[str, Any]:
    return {
        "reference_original_label": reference_original_label,
        "reference_incremental_label": reference_incremental_label,
        "stdi_vs_original": 0.0,
        "theme_drift_vs_original": 0.0,
        "subtopic_drift_vs_original": 0.0,
        "entity_drift_vs_original": 0.0,
        "relation_drift_vs_original": 0.0,
        "contradiction_drift_vs_original": 0.0,
        "valence_drift_vs_original": 0.0,
        "arousal_drift_vs_original": 0.0,
        "dominance_drift_vs_original": 0.0,
        "vad_drift_vs_original": 0.0,
        "stdi_incremental": 0.0,
        "theme_drift_incremental": 0.0,
        "subtopic_drift_incremental": 0.0,
        "entity_drift_incremental": 0.0,
        "relation_drift_incremental": 0.0,
        "contradiction_drift_incremental": 0.0,
        "valence_drift_incremental": 0.0,
        "arousal_drift_incremental": 0.0,
        "dominance_drift_incremental": 0.0,
        "vad_drift_incremental": 0.0,
        "stdi_cumulative": 0.0,
    }


def _apply_topic_structure_defaults(
    df: pd.DataFrame,
    *,
    row_index: Any,
    prefix: str,
) -> None:
    empty_structure_columns = flatten_topic_structure(
        empty_topic_structure(),
        prefix=prefix,
    )
    for column_name, value in empty_structure_columns.items():
        df.at[row_index, column_name] = value


def _apply_vad_defaults(
    df: pd.DataFrame,
    *,
    row_index: Any,
    prefix: str,
) -> None:
    for suffix in _topic_drift_vad_suffixes():
        df.at[row_index, f"{prefix}_{suffix}"] = pd.NA


def _write_vad_score(
    df: pd.DataFrame,
    *,
    row_index: Any,
    prefix: str,
    vad_score: VADScore | None,
) -> None:
    if vad_score is None:
        _apply_vad_defaults(df, row_index=row_index, prefix=prefix)
        return

    df.at[row_index, f"{prefix}_vad_valence"] = vad_score.valence
    df.at[row_index, f"{prefix}_vad_arousal"] = vad_score.arousal
    df.at[row_index, f"{prefix}_vad_dominance"] = vad_score.dominance


def _apply_metric_defaults(
    df: pd.DataFrame,
    *,
    row_index: Any,
    token: str,
    reference_original_label: str = "original",
    reference_incremental_label: str = "original",
) -> None:
    for suffix, value in _default_metric_values(
        reference_original_label=reference_original_label,
        reference_incremental_label=reference_incremental_label,
    ).items():
        df.at[row_index, f"{token}_{suffix}"] = value


def _initialize_chain_metric_columns(
    df: pd.DataFrame,
    *,
    version_tokens: Sequence[str],
) -> None:
    for column_name, default_value in _expected_chain_output_columns(version_tokens).items():
        df[column_name] = default_value


def _ensure_expected_chain_output_columns(
    df: pd.DataFrame,
    *,
    version_tokens: Sequence[str],
) -> pd.DataFrame:
    for column_name, default_value in _expected_chain_output_columns(version_tokens).items():
        if column_name not in df.columns:
            df[column_name] = default_value
    return df
