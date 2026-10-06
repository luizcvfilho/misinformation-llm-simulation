from __future__ import annotations

import numpy as np
import pandas as pd

from misinformation_simulation.simulation.types import (
    EVALUATION_METRICS,
    expand_dual_evaluation_record,
)

EVALUATION_LABELS = {
    "dual": "Dual",
    "cluster": "Cluster",
    "llm_judge": "LLM",
    "saved": "Saved evaluation (legacy)",
}
BRANCH_ANALYSIS_COLUMNS = (
    {
        f"{metric}_{branch}_{suffix}"
        for branch in ("cluster", "llm_judge", "embedding")
        for suffix in ("vs_original", "incremental")
        for metric in EVALUATION_METRICS
    }
    | {
        f"stdi_{branch}_{suffix}"
        for branch in ("cluster", "llm_judge")
        for suffix in ("cumulative", "cumulative_valid_steps", "chain_complete")
    }
    | {
        "stdi_cumulative_valid_steps",
        "stdi_chain_complete",
        "metadata_stdi_comparison_method",
    }
    | {
        f"stdi_{field}_{branch}_{suffix}"
        for field in ("status", "error")
        for branch in ("cluster", "llm_judge")
        for suffix in ("vs_original", "incremental")
    }
)


def _numeric(steps: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(
        steps.get(column, pd.Series(index=steps.index, dtype=float)), errors="coerce"
    )
    return values.replace([np.inf, -np.inf], np.nan)


def prepare_evaluation_steps(steps: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Recover saved branch metrics without changing the source dataframe."""
    if steps.empty:
        return steps.copy()
    data = pd.DataFrame(
        [expand_dual_evaluation_record(record) for record in steps.to_dict("records")],
        index=steps.index,
    )
    methods = data.get("metadata_stdi_comparison_method", pd.Series(index=data.index, dtype=object))
    methods = methods.fillna(method or "")
    versions = data.get("metadata_stdi_comparison_version", pd.Series("", index=data.index)).fillna(
        ""
    )
    for name in ("dual", "cluster", "lexical"):
        methods = methods.mask(methods.eq("") & versions.str.startswith(name), name)
    has_dual = pd.Series(False, index=data.index)
    for branch in ("cluster", "embedding", "llm_judge"):
        for suffix in ("vs_original", "incremental"):
            has_dual |= _numeric(data, f"stdi_{branch}_{suffix}").notna()
    for suffix in ("vs_original", "incremental"):
        column = f"metadata_dual_stdi_{suffix}"
        if column in data:
            has_dual |= data[column].map(lambda value: isinstance(value, dict))
    methods = methods.mask(methods.eq("") & has_dual, "dual")
    data["metadata_stdi_comparison_method"] = methods.replace("", "saved")
    for suffix in ("vs_original", "incremental"):
        for metric in EVALUATION_METRICS:
            column = f"{metric}_cluster_{suffix}"
            values = _numeric(data, column)
            values = values.fillna(_numeric(data, f"{metric}_embedding_{suffix}"))
            # Historical dual component columns explicitly describe the Cluster branch.
            native = methods.eq("cluster") | (methods.eq("dual") if metric != "stdi" else False)
            values = values.fillna(_numeric(data, f"{metric}_{suffix}").where(native))
            data[column] = values
    if "news_id" in data and "step_index" in data:
        ordered = data.sort_values("step_index")
        groupers = [
            ordered[column]
            for column in ("execution_id", "source_path", "run_id", "chain_label", "news_id")
            if column in ordered
        ]
        for branch in ("cluster", "llm_judge"):
            scores = _numeric(ordered, f"stdi_{branch}_incremental")
            valid = scores.notna()
            available = valid.groupby(groupers, dropna=False).transform("any")
            counts = valid.groupby(groupers, dropna=False).cumsum()
            totals = scores.fillna(0).groupby(groupers, dropna=False).cumsum()
            has_saved_cumulative = _numeric(data, f"stdi_{branch}_cumulative").notna()
            for field, values in (
                ("cumulative", totals),
                ("cumulative_valid_steps", counts),
                ("chain_complete", counts.eq(ordered["step_index"])),
            ):
                column = f"stdi_{branch}_{field}"
                existing = data.get(column, pd.Series(index=data.index, dtype=object))
                existing = existing.where(has_saved_cumulative)
                data[column] = existing.combine_first(values.where(available).reindex(data.index))
    return data


def available_evaluations(steps: pd.DataFrame, method: str | None = None) -> list[str]:
    data = prepare_evaluation_steps(steps, method)
    if data.empty:
        return []
    methods = data["metadata_stdi_comparison_method"]
    available = []
    for name in EVALUATION_LABELS:
        has_scores = pd.Series(False, index=data.index)
        for suffix in ("vs_original", "incremental"):
            if name in ("dual", "saved"):
                has_scores |= _numeric(data, f"stdi_{suffix}").notna() & (
                    methods.eq("dual") if name == "dual" else methods.isin(["saved", "lexical"])
                )
            else:
                has_scores |= _numeric(data, f"stdi_{name}_{suffix}").notna()
                has_scores |= _numeric(data, f"stdi_{suffix}").notna() & methods.eq(name)
        if has_scores.any():
            available.append(name)
    return available


def select_evaluation_steps(
    steps: pd.DataFrame, evaluation: str, method: str | None = None
) -> pd.DataFrame:
    """Project one evaluation into the metric columns consumed by existing views."""
    if evaluation not in EVALUATION_LABELS:
        raise ValueError(f"Unknown evaluation: {evaluation}")
    data = prepare_evaluation_steps(steps, method)
    if data.empty:
        return data
    original = data.copy()
    methods = original["metadata_stdi_comparison_method"]
    native = methods.eq(evaluation)
    if evaluation == "saved":
        native = methods.isin(["saved", "lexical"])
    for suffix in ("vs_original", "incremental"):
        for metric in EVALUATION_METRICS:
            column = f"{metric}_{suffix}"
            if evaluation in ("dual", "saved"):
                values = _numeric(original, column).where(native)
                if evaluation == "dual" and metric != "stdi":
                    values = (
                        _numeric(original, f"{metric}_cluster_{suffix}")
                        + _numeric(original, f"{metric}_llm_judge_{suffix}")
                    ) / 2
                    values = values.where(native)
            else:
                values = _numeric(original, f"{metric}_{evaluation}_{suffix}")
                values = values.fillna(_numeric(original, column).where(native))
            data[column] = values
        if evaluation in ("cluster", "llm_judge"):
            for field in ("status", "error"):
                data[f"stdi_{field}_{suffix}"] = original.get(
                    f"stdi_{field}_{evaluation}_{suffix}",
                    pd.Series(index=original.index, dtype=object),
                )
    for field in ("cumulative", "cumulative_valid_steps", "chain_complete"):
        column = f"stdi_{field}"
        values = original.get(column, pd.Series(index=original.index, dtype=object)).where(native)
        if evaluation in ("cluster", "llm_judge"):
            values = original.get(
                f"stdi_{evaluation}_{field}", pd.Series(index=original.index, dtype=object)
            ).combine_first(values)
        data[column] = values
    data["stdi_evaluation"] = evaluation
    return data
