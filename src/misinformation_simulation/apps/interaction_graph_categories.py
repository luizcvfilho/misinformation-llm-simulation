from __future__ import annotations

from typing import Any

import pandas as pd

CATEGORY_METRICS = (
    "stdi_vs_original",
    "stdi_incremental",
    "stdi_cumulative",
    "vad_drift_vs_original",
    "contradiction_drift_vs_original",
)


def split_news_categories(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(part.strip().casefold() for part in value.split(";") if part.strip()))


def build_category_summary_dataframe(steps_df: pd.DataFrame) -> pd.DataFrame:
    if steps_df.empty or "metadata_category" not in steps_df.columns:
        return pd.DataFrame()

    expanded = steps_df.copy()
    expanded["category"] = expanded["metadata_category"].map(split_news_categories)
    expanded = expanded.explode("category").dropna(subset=["category"])
    if expanded.empty:
        return pd.DataFrame()

    for metric in CATEGORY_METRICS:
        if metric not in expanded.columns:
            expanded[metric] = pd.NA
        expanded[metric] = pd.to_numeric(expanded[metric], errors="coerce")

    grouped = (
        expanded.groupby(["category", "step_index", "node_label"], dropna=False)
        .agg(
            news_count=("news_id", "nunique"),
            runs=("rewrite_status", "size"),
            successes=("rewrite_status", lambda values: int((values == "success").sum())),
            errors=("rewrite_status", lambda values: int((values == "error").sum())),
            blocked=("rewrite_status", lambda values: int((values == "blocked").sum())),
            mean_stdi_vs_original=("stdi_vs_original", "mean"),
            mean_stdi_incremental=("stdi_incremental", "mean"),
            mean_stdi_cumulative=("stdi_cumulative", "mean"),
            mean_vad_drift_vs_original=("vad_drift_vs_original", "mean"),
            mean_contradiction_drift_vs_original=("contradiction_drift_vs_original", "mean"),
        )
        .reset_index()
        .sort_values(["category", "step_index", "node_label"])
    )
    grouped["success_rate"] = grouped["successes"] / grouped["runs"]
    return grouped


def build_category_comparison_dataframe(bundles: list[dict[str, Any]]) -> pd.DataFrame:
    comparisons = []
    for index, bundle in enumerate(bundles, start=1):
        steps_df = bundle.get("steps_df")
        if not isinstance(steps_df, pd.DataFrame):
            continue
        summary = build_category_summary_dataframe(steps_df)
        if summary.empty:
            continue
        summary.insert(0, "graph", bundle["name"])
        summary.insert(0, "run", f"{index}. {bundle['name']}")
        comparisons.append(summary)
    if not comparisons:
        return pd.DataFrame()
    return pd.concat(comparisons, ignore_index=True)
