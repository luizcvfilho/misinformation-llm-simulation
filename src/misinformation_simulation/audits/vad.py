from __future__ import annotations

import math

import pandas as pd
from scipy.stats import mannwhitneyu

from misinformation_simulation.text_metrics import vad as vad_metrics

DEFAULT_VAD_MODEL_NAME = vad_metrics.DEFAULT_VAD_MODEL_NAME
VAD_DIMENSIONS = vad_metrics.VAD_DIMENSIONS
VADModelBundle = vad_metrics.VADModelBundle
VADScore = vad_metrics.VADScore
load_huggingface_vad_model = vad_metrics.load_huggingface_vad_model
get_default_vad_model_bundle = vad_metrics.get_default_vad_model_bundle
predict_text_vad = vad_metrics.predict_text_vad
predict_vad_batch = vad_metrics.predict_vad_batch
analyze_text_vad = vad_metrics.analyze_text_vad
annotate_vad_scores = vad_metrics.annotate_vad_scores

__all__ = [
    "VAD_DIMENSIONS",
    "DEFAULT_VAD_MODEL_NAME",
    "VADScore",
    "VADModelBundle",
    "load_huggingface_vad_model",
    "get_default_vad_model_bundle",
    "predict_text_vad",
    "predict_vad_batch",
    "analyze_text_vad",
    "annotate_vad_scores",
    "summarize_vad_by_group",
    "compare_vad_groups",
]


def summarize_vad_by_group(
    df: pd.DataFrame,
    *,
    group_column: str,
    prefix: str = "vad",
) -> pd.DataFrame:
    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    summaries: list[dict[str, float | int | str]] = []
    for group_value, group_frame in df.groupby(group_column, dropna=False):
        summary: dict[str, float | int | str] = {
            group_column: str(group_value),
            "document_count": int(len(group_frame)),
        }
        for dimension in VAD_DIMENSIONS:
            series = group_frame[f"{prefix}_{dimension}"].dropna()
            summary[f"{dimension}_mean"] = float(series.mean()) if not series.empty else math.nan
            summary[f"{dimension}_median"] = (
                float(series.median()) if not series.empty else math.nan
            )
            summary[f"{dimension}_std"] = float(series.std(ddof=0)) if len(series) > 1 else math.nan
        summaries.append(summary)

    return pd.DataFrame(summaries).sort_values(group_column).reset_index(drop=True)


def compare_vad_groups(
    df: pd.DataFrame,
    *,
    group_column: str = "label",
    baseline_group: str = "true",
    comparison_group: str = "fake",
    prefix: str = "vad",
) -> pd.DataFrame:
    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    baseline_frame = df.loc[df[group_column] == baseline_group]
    comparison_frame = df.loc[df[group_column] == comparison_group]
    if baseline_frame.empty or comparison_frame.empty:
        raise ValueError("Both comparison groups must contain at least one row.")

    comparisons: list[dict[str, float | str]] = []
    for metric in VAD_DIMENSIONS:
        baseline_values = baseline_frame[f"{prefix}_{metric}"].dropna()
        comparison_values = comparison_frame[f"{prefix}_{metric}"].dropna()
        comparisons.append(
            {
                "metric": metric,
                "baseline_group": baseline_group,
                "comparison_group": comparison_group,
                "baseline_mean": float(baseline_values.mean()),
                "comparison_mean": float(comparison_values.mean()),
                "mean_diff": float(comparison_values.mean() - baseline_values.mean()),
                "baseline_median": float(baseline_values.median()),
                "comparison_median": float(comparison_values.median()),
                "median_diff": float(comparison_values.median() - baseline_values.median()),
                "effect_size_cohens_d": _cohens_d(baseline_values, comparison_values),
                "mannwhitney_pvalue": _mannwhitney_pvalue(baseline_values, comparison_values),
            }
        )

    return pd.DataFrame(comparisons)


def _cohens_d(baseline: pd.Series, comparison: pd.Series) -> float:
    if len(baseline) < 2 or len(comparison) < 2:
        return math.nan

    baseline_std = float(baseline.std(ddof=1))
    comparison_std = float(comparison.std(ddof=1))
    pooled_variance = (
        ((len(baseline) - 1) * baseline_std**2) + ((len(comparison) - 1) * comparison_std**2)
    ) / (len(baseline) + len(comparison) - 2)
    if pooled_variance <= 0:
        return math.nan

    pooled_std = math.sqrt(pooled_variance)
    return float((comparison.mean() - baseline.mean()) / pooled_std)


def _mannwhitney_pvalue(baseline: pd.Series, comparison: pd.Series) -> float:
    if baseline.empty or comparison.empty:
        return math.nan
    try:
        return float(mannwhitneyu(baseline, comparison, alternative="two-sided").pvalue)
    except ValueError:
        return math.nan
