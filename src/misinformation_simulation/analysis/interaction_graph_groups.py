from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from misinformation_simulation.analysis.interaction_graph_visualization import available_metrics

ORIGINAL_CATEGORY_GROUPING = "original_category"
GROUPING_LABELS = {
    ORIGINAL_CATEGORY_GROUPING: "Original news classification",
}
PROXIMITY_MEASURE_LABELS = {
    "range_between_chains": "Range across chains",
    "sd_between_chains": "Standard deviation across chains",
    "mean_pairwise_abs_diff": "Mean pairwise absolute difference",
}


def available_news_groupings(steps: pd.DataFrame) -> dict[str, str]:
    """Return persisted news groupings that contain at least one usable value."""
    available: dict[str, str] = {}
    if _has_text_values(steps, "metadata_category"):
        available[ORIGINAL_CATEGORY_GROUPING] = GROUPING_LABELS[ORIGINAL_CATEGORY_GROUPING]
    return available


def news_chain_proximity(
    steps: pd.DataFrame,
    metric: str,
    grouping: str,
    *,
    minimum_chains: int = 2,
) -> pd.DataFrame:
    """Measure final-score dispersion across chains for every news item and group."""
    final = _final_metric_steps(steps, metric)
    if final.empty:
        return _empty_news_proximity()

    chain_column = _chain_instance_column(final)
    wide = final.pivot_table(
        index="news_id",
        columns=chain_column,
        values=metric,
        aggfunc="mean",
    )
    proximity = pd.DataFrame(
        {
            "chains_observed": wide.notna().sum(axis=1),
            "mean_metric": wide.mean(axis=1),
            "sd_between_chains": wide.std(axis=1, ddof=0),
            "min_metric": wide.min(axis=1),
            "max_metric": wide.max(axis=1),
            "range_between_chains": wide.max(axis=1) - wide.min(axis=1),
            "mean_pairwise_abs_diff": wide.apply(_mean_pairwise_absolute_difference, axis=1),
        }
    ).reset_index()
    proximity = proximity.loc[proximity["chains_observed"].ge(minimum_chains)]

    titles = (
        final.groupby("news_id", dropna=False)["metadata_title"].first().rename("metadata_title")
        if "metadata_title" in final.columns
        else pd.Series(dtype=object, name="metadata_title")
    )
    memberships = _group_memberships(final, grouping)
    return (
        proximity.merge(titles, on="news_id", how="left")
        .merge(memberships, on="news_id", how="inner")
        .sort_values(["group_label", "range_between_chains", "news_id"])
        .reset_index(drop=True)
    )


def summarize_group_proximity(
    news_proximity: pd.DataFrame,
    *,
    bootstrap_iterations: int = 4_000,
    random_seed: int = 20260930,
) -> pd.DataFrame:
    """Summarize news-level chain proximity by original news category."""
    if news_proximity.empty:
        return _empty_group_summary()

    rows = []
    for group_index, ((group_value, group_label), group) in enumerate(
        news_proximity.groupby(["group_value", "group_label"], sort=True)
    ):
        ci_low, ci_high = _bootstrap_mean_ci(
            group["range_between_chains"].to_numpy(dtype=float),
            iterations=bootstrap_iterations,
            seed=random_seed + group_index,
        )
        rows.append(
            {
                "group_value": group_value,
                "group_label": group_label,
                "news_items": int(group["news_id"].nunique()),
                "mean_news_metric": float(group["mean_metric"].mean()),
                "mean_range": float(group["range_between_chains"].mean()),
                "median_range": float(group["range_between_chains"].median()),
                "range_ci_low": ci_low,
                "range_ci_high": ci_high,
                "mean_sd": float(group["sd_between_chains"].mean()),
                "mean_pairwise_abs_diff": float(group["mean_pairwise_abs_diff"].mean()),
                "minimum_chains": int(group["chains_observed"].min()),
                "maximum_chains": int(group["chains_observed"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_range", "group_label"]).reset_index(drop=True)


def summarize_chain_pairs_by_group(
    steps: pd.DataFrame,
    metric: str,
    grouping: str,
) -> pd.DataFrame:
    """Compare every pair of final chain scores within each news grouping."""
    final = _final_metric_steps(steps, metric)
    if final.empty:
        return _empty_pair_summary()

    chain_column = _chain_instance_column(final)
    wide = final.pivot_table(
        index="news_id",
        columns=chain_column,
        values=metric,
        aggfunc="mean",
    )
    labels = _chain_labels(final, chain_column)
    memberships = _group_memberships(final, grouping)
    rows = []
    for left, right in combinations(sorted(wide.columns), 2):
        differences = (wide[left] - wide[right]).abs().dropna().rename("absolute_difference")
        if differences.empty:
            continue
        pair_news = memberships.merge(differences, on="news_id", how="inner")
        for (group_value, group_label), group in pair_news.groupby(
            ["group_value", "group_label"], sort=True
        ):
            rows.append(
                {
                    "group_value": group_value,
                    "group_label": group_label,
                    "chain_a": labels[left],
                    "chain_b": labels[right],
                    "chain_pair": f"{labels[left]} – {labels[right]}",
                    "paired_news": int(group["news_id"].nunique()),
                    "mean_absolute_difference": float(group["absolute_difference"].mean()),
                    "median_absolute_difference": float(group["absolute_difference"].median()),
                }
            )
    if not rows:
        return _empty_pair_summary()
    return (
        pd.DataFrame(rows)
        .sort_values(["group_label", "mean_absolute_difference", "chain_pair"])
        .reset_index(drop=True)
    )


def _final_metric_steps(steps: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in available_metrics(steps):
        raise ValueError(f"Metric is unavailable: {metric}")
    data = steps.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    run_column = "run_id" if "run_id" in data.columns else "chain_label"
    final_step = data.groupby(run_column, dropna=False)["step_index"].transform("max")
    return data.loc[
        data["step_index"].eq(final_step)
        & data["rewrite_status"].eq("success")
        & data[metric].notna()
    ].copy()


def _group_memberships(final: pd.DataFrame, grouping: str) -> pd.DataFrame:
    if grouping == ORIGINAL_CATEGORY_GROUPING:
        column = "metadata_category"
        if column not in final.columns:
            return pd.DataFrame(columns=["news_id", "group_value", "group_label"])
        membership = final[["news_id", column]].drop_duplicates("news_id").copy()
        membership["group_value"] = membership[column].map(_split_categories)
        membership = membership.explode("group_value").dropna(subset=["group_value"])
        membership["group_label"] = membership["group_value"].map(_humanize_value)
    else:
        raise ValueError(f"Unsupported news grouping: {grouping}")
    return membership[["news_id", "group_value", "group_label"]].drop_duplicates()


def _chain_instance_column(final: pd.DataFrame) -> str:
    return "run_id" if "run_id" in final.columns else "chain_label"


def _chain_labels(final: pd.DataFrame, chain_column: str) -> dict[Any, str]:
    if chain_column == "chain_label":
        return {value: str(value) for value in final[chain_column].dropna().unique()}
    labels = (
        final[[chain_column, "chain_label"]]
        .drop_duplicates(chain_column)
        .set_index(chain_column)["chain_label"]
        .astype(str)
        .to_dict()
    )
    duplicated_labels = pd.Series(labels).duplicated(keep=False)
    return {
        chain: (f"{label} · {chain}" if duplicated_labels.get(chain, False) else label)
        for chain, label in labels.items()
    }


def _mean_pairwise_absolute_difference(row: pd.Series) -> float:
    values = row.dropna().to_numpy(dtype=float)
    differences = [abs(left - right) for left, right in combinations(values, 2)]
    return float(np.mean(differences)) if differences else float("nan")


def _bootstrap_mean_ci(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _split_categories(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(part.strip().casefold() for part in value.split(";") if part.strip()))


def _normalize_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().casefold()
    return normalized or None


def _humanize_value(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _has_text_values(steps: pd.DataFrame, column: str) -> bool:
    return column in steps.columns and steps[column].map(_normalize_text).notna().any()


def _empty_news_proximity() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "news_id",
            "chains_observed",
            "mean_metric",
            "sd_between_chains",
            "min_metric",
            "max_metric",
            "range_between_chains",
            "mean_pairwise_abs_diff",
            "metadata_title",
            "group_value",
            "group_label",
        ]
    )


def _empty_group_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group_value",
            "group_label",
            "news_items",
            "mean_news_metric",
            "mean_range",
            "median_range",
            "range_ci_low",
            "range_ci_high",
            "mean_sd",
            "mean_pairwise_abs_diff",
            "minimum_chains",
            "maximum_chains",
        ]
    )


def _empty_pair_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group_value",
            "group_label",
            "chain_a",
            "chain_b",
            "chain_pair",
            "paired_news",
            "mean_absolute_difference",
            "median_absolute_difference",
        ]
    )
