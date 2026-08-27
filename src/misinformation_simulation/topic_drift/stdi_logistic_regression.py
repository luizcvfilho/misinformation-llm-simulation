"""Resumable STDI logistic-regression workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from misinformation_simulation.llm.false_to_true import rewrite_false_news_as_true
from misinformation_simulation.topic_drift.annotation import (
    annotate_stdi_for_rewrites,
    annotate_stdi_for_version_chain,
)
from misinformation_simulation.topic_drift.stdi_regression_features import (
    ROW_ID_COLUMN,
    build_stdi_pair_dataset,
    build_stdi_regression_dataset,
    build_stdi_regression_report,
    fit_stdi_logistic_regression,
)

_FILES = {
    "rewrites": "rewritten_false_as_true.csv",
    "false_audit": "stdi_audit.csv",
    "true_audit": "true_reference_stdi_audit.csv",
    "regression_dataset": "stdi_regression_dataset.csv",
    "pair_dataset": "stdi_pair_metrics.csv",
    "importance": "stdi_regression_feature_importance.csv",
    "metrics": "stdi_logistic_regression_metrics.json",
    "manifest": "stdi_logistic_regression_manifest.json",
    "report": "stdi_logistic_regression_report.md",
}


def run_stdi_logistic_regression_analysis(
    false_news: pd.DataFrame,
    *,
    output_dir: str | Path,
    true_news: pd.DataFrame | None = None,
    text_column: str = "original_article_text",
    title_column: str | None = "title",
    topic_column: str | None = "subject",
    true_text_column: str | None = None,
    verification_column: str = "verification_status",
    rewrite_kwargs: Mapping[str, Any] | None = None,
    annotation_kwargs: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    skip_rewrite: bool = False,
    log: Callable[[str], None] | None = None,
    rewriter: Callable[..., pd.DataFrame] = rewrite_false_news_as_true,
    false_annotator: Callable[..., pd.DataFrame] = annotate_stdi_for_rewrites,
    true_annotator: Callable[..., pd.DataFrame] = annotate_stdi_for_version_chain,
) -> dict[str, Path]:
    """Analyze which STDI features separate false/true reference groups.

    The original false article and its truthified rewrite form the main reference
    pair. The model analyzes STDI features; it is not a factual verifier.
    """
    if false_news.empty or text_column not in false_news:
        raise ValueError(f"false_news must be non-empty and contain '{text_column}'.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {name: output / filename for name, filename in _FILES.items()}
    source = _with_row_ids(false_news, text_column, title_column)
    _log(log, f"Truthification: preparing {len(source)} news item(s).")
    rewritten = _rewrite(
        source,
        files["rewrites"],
        text_column,
        title_column,
        topic_column,
        verification_column,
        max_rows,
        skip_rewrite,
        rewriter,
        rewrite_kwargs,
    )
    false_audit = _reuse_audit(
        rewritten.loc[rewritten["rewrite_status"].eq("success")],
        _read(files["false_audit"]),
        text_column,
        title_column or "title",
        false_annotator,
        annotation_kwargs,
        rewritten_column="rewritten_article_text",
    )
    _save(false_audit, files["false_audit"])
    _log(log, f"STDI audit: {len(false_audit)} pair(s) available.")
    true_text = true_text_column or text_column
    true_audit = _true_audit(
        true_news,
        files["true_audit"],
        true_text,
        title_column or "title",
        true_annotator,
        annotation_kwargs,
    )
    dataset = build_stdi_regression_dataset(
        false_audit,
        true_audit=true_audit,
        text_column=text_column,
        true_text_column=true_text,
    )
    pair_dataset = build_stdi_pair_dataset(false_audit, text_column=text_column)
    importance, metrics = fit_stdi_logistic_regression(dataset)
    _log(log, f"Logistic regression: {len(dataset)} reference sample(s).")
    _save(dataset, files["regression_dataset"])
    _save(pair_dataset, files["pair_dataset"])
    _save(importance, files["importance"])
    _json(metrics, files["metrics"])
    _json(
        {
            "analysis": "stdi_logistic_regression",
            "purpose": "STDI analysis of false news and its truthified counterpart",
            "input_rows": len(source),
            "successful_rewrites": int(rewritten["rewrite_status"].eq("success").sum()),
            "true_reference_rows": len(true_audit),
            "source_hash": _hash(source, text_column),
            "text_column": text_column,
            "true_text_column": true_text,
        },
        files["manifest"],
    )
    files["report"].write_text(build_stdi_regression_report(metrics, importance), encoding="utf-8")
    _log(log, "Completed. Outputs saved to " + str(output))
    return files


def _rewrite(
    source: pd.DataFrame,
    path: Path,
    text_column: str,
    title_column: str | None,
    topic_column: str | None,
    verification_column: str,
    max_rows: int | None,
    skip: bool,
    rewriter: Callable[..., pd.DataFrame],
    kwargs: Mapping[str, Any] | None,
) -> pd.DataFrame:
    checkpoint_exists = path.exists()
    resumed = _merge(source, _read(path))
    result = (
        resumed
        if skip and checkpoint_exists
        else rewriter(
            resumed,
            text_column=text_column,
            title_column=title_column,
            topic_column=topic_column,
            checkpoint_path=path,
            max_rows=max_rows,
            **dict(kwargs or {}),
        )
    )
    if "rewrite_status" not in result:
        result["rewrite_status"] = "not_requested"
    if verification_column not in result:
        result[verification_column] = pd.NA
    result.loc[
        result["rewrite_status"].eq("success") & result[verification_column].isna(),
        verification_column,
    ] = "unverified_generated"
    _save(result, path)
    return result


def _true_audit(
    source: pd.DataFrame | None,
    path: Path,
    text_column: str,
    title_column: str,
    annotator: Callable[..., pd.DataFrame],
    kwargs: Mapping[str, Any] | None,
) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame()
    if text_column not in source:
        raise ValueError(f"true_news must contain '{text_column}'.")
    frame = _with_row_ids(source, text_column, title_column)
    reusable = _reusable(frame, _read(path), ["original_topic_structure_status"])
    pending = frame.loc[~frame[ROW_ID_COLUMN].isin(reusable[ROW_ID_COLUMN])].copy()
    if not pending.empty:
        pending["_same_text"] = pending[text_column]
        generated = annotator(
            pending,
            version_columns=["_same_text"],
            text_column=text_column,
            title_column=title_column,
            **dict(kwargs or {}),
        ).drop(columns="_same_text")
    else:
        generated = pending
    result = _combine(frame, reusable, generated)
    _save(result, path)
    return result


def _reuse_audit(
    source: pd.DataFrame,
    previous: pd.DataFrame | None,
    text_column: str,
    title_column: str,
    annotator: Callable[..., pd.DataFrame],
    kwargs: Mapping[str, Any] | None,
    *,
    rewritten_column: str,
) -> pd.DataFrame:
    statuses = ["original_topic_structure_status", f"{rewritten_column}_topic_structure_status"]
    reusable = _reusable(source, previous, statuses)
    pending = source.loc[~source[ROW_ID_COLUMN].isin(reusable[ROW_ID_COLUMN])]
    generated = (
        annotator(
            pending,
            rewritten_column=rewritten_column,
            text_column=text_column,
            title_column=title_column,
            **dict(kwargs or {}),
        )
        if not pending.empty
        else pending
    )
    return _combine(source, reusable, generated)


def _with_row_ids(frame: pd.DataFrame, text_column: str, title_column: str | None) -> pd.DataFrame:
    result = frame.copy()
    title = result.get(title_column, pd.Series("", index=result.index)) if title_column else ""
    base = (
        result[text_column].fillna("").astype(str)
        + "\n"
        + pd.Series(title, index=result.index).fillna("").astype(str)
    )
    result[ROW_ID_COLUMN] = (
        base + "\n" + base.groupby(base, sort=False).cumcount().astype(str)
    ).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    return result


def _merge(source: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    if previous is None or ROW_ID_COLUMN not in previous:
        return source
    result, old = (
        source.copy().set_index(ROW_ID_COLUMN),
        previous.drop_duplicates(ROW_ID_COLUMN, keep="last").set_index(ROW_ID_COLUMN),
    )
    for column in old:
        result[column] = (
            old[column]
            .reindex(result.index)
            .combine_first(result.get(column, pd.Series(pd.NA, index=result.index)))
        )
    return result.reset_index()


def _reusable(
    source: pd.DataFrame, previous: pd.DataFrame | None, statuses: list[str]
) -> pd.DataFrame:
    if (
        previous is None
        or ROW_ID_COLUMN not in previous
        or not all(status in previous for status in statuses)
    ):
        return source.iloc[0:0].copy()
    candidate = previous.loc[previous[ROW_ID_COLUMN].isin(source[ROW_ID_COLUMN])].copy()
    mask = pd.Series(True, index=candidate.index)
    for status in statuses:
        mask &= candidate[status].eq("success")
    return candidate.loc[mask].drop_duplicates(ROW_ID_COLUMN, keep="last")


def _combine(source: pd.DataFrame, reusable: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    result = pd.concat([reusable, generated], ignore_index=True, sort=False)
    return (
        result.drop_duplicates(ROW_ID_COLUMN, keep="last")
        .set_index(ROW_ID_COLUMN)
        .reindex(source[ROW_ID_COLUMN])
        .reset_index()
        if not result.empty
        else result
    )


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _save(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _json(value: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _hash(frame: pd.DataFrame, text_column: str) -> str:
    return hashlib.sha256(
        frame[text_column].fillna("").astype(str).str.cat(sep="\n").encode()
    ).hexdigest()


def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)
