from __future__ import annotations

import json
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import which

import pandas as pd

from misinformation_simulation.analysis.stdi_evaluation import BRANCH_ANALYSIS_COLUMNS
from misinformation_simulation.simulation.types import expand_dual_evaluation_record

# Accept compact prefixes and historical run names used by VAD analysis.
RUN_ID_PATTERN = re.compile(
    r"(?:^|_)(?:(?P<batch_id>\d+)_)?(?P<graph_id>\d+)_(?P<chain_code>[A-Za-z0-9-]+)$"
)
RUN_ID_PATTERNS = (
    re.compile(r"^(?:(?P<batch_id>\d+)_)?(?P<graph_id>\d+)_(?P<chain_code>.+)$"),
    re.compile(
        r"^simulation_ui_\d{8}_\d{6}_(?:(?P<batch_id>\d+)_)?"
        r"(?P<graph_id>\d+)_(?P<chain_code>.+)$"
    ),
    re.compile(r"^.+?_(?P<batch_id>\d+)_(?P<graph_id>\d+)_(?P<chain_code>.+)$"),
    re.compile(r"^.+?_(?P<graph_id>\d+)_(?P<chain_code>.+)$"),
)
REQUIRED_STEP_COLUMNS = {"news_id", "step_index", "rewrite_status", "stdi_vs_original"}
STDI_COMPONENT_COLUMNS = {
    "theme_drift_vs_original": "Theme",
    "subtopic_drift_vs_original": "Subtopics",
    "entity_drift_vs_original": "Entities",
    "relation_drift_vs_original": "Relations",
    "contradiction_drift_vs_original": "Internal contradiction",
    "vad_drift_vs_original": "VAD",
    "valence_drift_vs_original": "Valence (VAD)",
    "arousal_drift_vs_original": "Arousal (VAD)",
    "dominance_drift_vs_original": "Dominance (VAD)",
}
METRIC_LABELS = {
    "stdi_vs_original": "STDI vs original",
    "stdi_incremental": "STDI incremental",
    "stdi_cumulative": "Cumulative STDI",
    **STDI_COMPONENT_COLUMNS,
}
ANALYSIS_STEP_COLUMNS = {
    *BRANCH_ANALYSIS_COLUMNS,
    *REQUIRED_STEP_COLUMNS,
    *METRIC_LABELS,
    "node_id",
    "node_label",
    "source_node_id",
    "source_node_label",
    "personality",
    "source_text",
    "rewritten_text",
    "metadata_title",
    "metadata_category",
    "metadata_rewrite_mode",
    "metadata_stdi_comparison_version",
    "theme_drift_incremental",
    "subtopic_drift_incremental",
    "entity_drift_incremental",
    "relation_drift_incremental",
    "contradiction_drift_incremental",
    "vad_drift_incremental",
    "valence_drift_incremental",
    "arousal_drift_incremental",
    "dominance_drift_incremental",
}


@dataclass(frozen=True)
class InteractionGraphRuns:
    """Persisted interaction-graph steps and the files that produced them."""

    steps: pd.DataFrame
    source_paths: tuple[Path, ...]


def discover_step_paths(runs_dir: Path) -> list[Path]:
    """Find persisted step files while excluding archived runs case-insensitively."""
    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    return sorted(
        path
        for path in runs_dir.rglob("*_steps.jsonl")
        if "old_runs" not in {part.casefold() for part in path.relative_to(runs_dir).parts}
    )


def load_interaction_graph_runs(runs_dir: Path | Sequence[Path]) -> InteractionGraphRuns:
    """Load all valid interaction-graph runs into one labelled dataframe."""
    directories = [runs_dir] if isinstance(runs_dir, Path) else list(runs_dir)
    if not directories:
        raise ValueError("Select at least one runs directory.")
    discovered: set[Path] = set()
    for directory in directories:
        paths = discover_step_paths(directory)
        if not paths:
            raise ValueError(f"No '*_steps.jsonl' files were found in: {directory}")
        discovered.update(paths)
    source_paths = sorted(discovered)

    frames: list[pd.DataFrame] = []
    for path in source_paths:
        frame = _read_analysis_columns(path)
        if frame.empty:
            continue
        missing = REQUIRED_STEP_COLUMNS - set(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"'{path}' is missing required columns: {missing_text}")

        frame = frame.copy()
        version_column = "metadata_stdi_comparison_version"
        if version_column not in frame.columns:
            frame[version_column] = "legacy"
        else:
            frame[version_column] = frame[version_column].fillna("legacy")
        if "metadata_rewrite_mode" not in frame.columns:
            frame["metadata_rewrite_mode"] = "legacy"
        else:
            frame["metadata_rewrite_mode"] = frame["metadata_rewrite_mode"].fillna("legacy")
        frame["step_index"] = pd.to_numeric(frame["step_index"], errors="coerce")
        if frame["step_index"].isna().any():
            raise ValueError(f"'{path}' contains non-numeric step indexes.")
        frame["step_index"] = frame["step_index"].astype(int)
        for column in METRIC_LABELS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.assign(**_run_metadata(path))
        frames.append(frame)

    if not frames:
        raise ValueError("All discovered step files were empty.")

    steps = pd.concat(frames, ignore_index=True)
    steps = steps.sort_values(["graph_id", "chain_code", "news_id", "step_index"])
    return InteractionGraphRuns(
        steps=steps.reset_index(drop=True),
        source_paths=tuple(source_paths),
    )


def successful_steps(steps: pd.DataFrame) -> pd.DataFrame:
    """Return successful steps only; unsuccessful rewrites do not have comparable scores."""
    return steps.loc[steps["rewrite_status"].eq("success")].copy()


def available_metrics(steps: pd.DataFrame) -> dict[str, str]:
    """Return supported metric columns that contain at least one numeric observation."""
    return {
        column: label
        for column, label in METRIC_LABELS.items()
        if column in steps.columns and steps[column].notna().any()
    }


def summarize_metric(
    steps: pd.DataFrame,
    metric: str,
    *,
    group_columns: tuple[str, ...] = ("step_index",),
) -> pd.DataFrame:
    """Summarize a metric with central tendency and interquartile range."""
    _validate_metric(steps, metric)
    data = successful_steps(steps).dropna(subset=[metric])
    summary = (
        data.groupby(list(group_columns), dropna=False)[metric]
        .agg(
            observations="count",
            mean="mean",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    return summary.sort_values(list(group_columns)).reset_index(drop=True)


def summarize_components(steps: pd.DataFrame) -> pd.DataFrame:
    """Summarize every available STDI component and VAD dimension by iteration."""
    component_columns = [
        column
        for column in STDI_COMPONENT_COLUMNS
        if column in steps.columns and steps[column].notna().any()
    ]
    if not component_columns:
        return pd.DataFrame(
            columns=[
                "step_index",
                "component",
                "component_label",
                "observations",
                "mean",
                "q1",
                "q3",
            ]
        )

    data = successful_steps(steps).melt(
        id_vars=["step_index"],
        value_vars=component_columns,
        var_name="component",
        value_name="value",
    )
    summary = (
        data.dropna(subset=["value"])
        .groupby(["step_index", "component"], dropna=False)["value"]
        .agg(
            observations="count",
            mean="mean",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary["component_label"] = summary["component"].map(STDI_COMPONENT_COLUMNS)
    return summary.sort_values(["component", "step_index"]).reset_index(drop=True)


def create_static_figures(runs: InteractionGraphRuns, output_dir: Path) -> dict[str, Path]:
    """Export presentation-oriented Plotly figures from valid simulation steps."""
    from misinformation_simulation.analysis.interaction_graph_plotly import (
        PLOTLY_CONFIG,
        build_component_figure,
        build_evolution_figure,
        build_iteration_distribution_figure,
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = successful_steps(runs.steps)
    if steps.empty:
        raise ValueError("No successful simulation steps are available for plotting.")

    figures = {
        "stdi_evolution": build_evolution_figure(steps, "stdi_vs_original"),
        "components_evolution": build_component_figure(
            steps,
            [
                component
                for component in STDI_COMPONENT_COLUMNS
                if component in steps.columns and steps[component].notna().any()
            ],
        ),
        "stdi_distribution": build_iteration_distribution_figure(steps, "stdi_vs_original"),
    }
    file_stems = {
        "stdi_evolution": "stdi_evolution_by_iteration",
        "components_evolution": "stdi_components_by_iteration",
        "stdi_distribution": "stdi_distribution_by_iteration",
    }
    paths = {f"{name}_html": output_dir / f"{file_stems[name]}.html" for name in figures}
    for name, figure in figures.items():
        figure.write_html(
            paths[f"{name}_html"],
            include_plotlyjs=True,
            full_html=True,
            config=PLOTLY_CONFIG,
        )

    png_paths = {f"{name}_png": output_dir / f"{file_stems[name]}.png" for name in figures}
    for path in png_paths.values():
        path.unlink(missing_ok=True)

    export_status = {"png_exported": False, "png_error": None}
    if not _is_kaleido_browser_available():
        export_status["png_error"] = (
            "Chrome or Chromium was not found, so Kaleido PNG export was skipped."
        )
        warnings.warn(
            "Plotly HTML figures were generated, but PNG export is unavailable. "
            "Install Chrome or Chromium for Kaleido to enable it.",
            stacklevel=2,
        )
    else:
        try:
            from plotly import io as plotly_io

            plotly_io.write_images(
                list(figures.values()),
                list(png_paths.values()),
                format="png",
            )
        except Exception as error:  # PNG export is an optional convenience output.
            export_status["png_error"] = str(error)
            warnings.warn(
                "Plotly HTML figures were generated, but PNG export is unavailable. "
                "Install Chrome or Chromium for Kaleido to enable it.",
                stacklevel=2,
            )
        else:
            export_status["png_exported"] = True
            paths.update(png_paths)

    status_path = output_dir / "plotly_figure_export.json"
    status_path.write_text(
        json.dumps(export_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["figure_export_status"] = status_path
    return paths


def export_analysis_tables(runs: InteractionGraphRuns, output_dir: Path) -> dict[str, Path]:
    """Export tidy data, chain summaries, and a reproducibility manifest."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = successful_steps(runs.steps)
    metric_columns = [column for column in METRIC_LABELS if column in successful.columns]
    retained_columns = [
        "run_id",
        "execution_id",
        "execution_label",
        "source_path",
        "batch_id",
        "graph_id",
        "chain_code",
        "chain_label",
        "news_id",
        "step_index",
        "node_id",
        "node_label",
        "metadata_title",
        "metadata_category",
        "metadata_rewrite_mode",
        "metadata_stdi_comparison_version",
        "rewrite_status",
        *metric_columns,
    ]
    retained_columns = [column for column in retained_columns if column in successful.columns]

    paths = {
        "tidy_steps": output_dir / "successful_steps.csv",
        "stdi_by_iteration": output_dir / "stdi_by_iteration_summary.csv",
        "final_stdi_by_chain": output_dir / "final_stdi_by_chain.csv",
        "components_by_iteration": output_dir / "stdi_components_by_iteration_summary.csv",
        "manifest": output_dir / "analysis_manifest.json",
    }
    successful.loc[:, retained_columns].to_csv(paths["tidy_steps"], index=False)

    by_iteration = summarize_metric(
        successful,
        "stdi_vs_original",
        group_columns=("graph_id", "chain_code", "chain_label", "step_index"),
    )
    by_iteration.to_csv(paths["stdi_by_iteration"], index=False)

    final_steps = successful.loc[
        successful["step_index"].eq(successful.groupby("run_id")["step_index"].transform("max"))
    ]
    final_summary = summarize_metric(
        final_steps,
        "stdi_vs_original",
        group_columns=("graph_id", "chain_code", "chain_label"),
    )
    final_summary.to_csv(paths["final_stdi_by_chain"], index=False)
    summarize_components(successful).to_csv(paths["components_by_iteration"], index=False)

    manifest = {
        "runs_directory": str(runs.source_paths[0].parent.parent) if runs.source_paths else None,
        "source_files": [str(path) for path in runs.source_paths],
        "runs_loaded": int(successful["run_id"].nunique()),
        "graphs_loaded": int(successful["graph_id"].nunique()),
        "news_items": int(successful["news_id"].nunique()),
        "successful_steps": int(len(successful)),
        "iterations": sorted(int(value) for value in successful["step_index"].unique()),
        "metrics": available_metrics(successful),
        "excluded_directory_name": "OLD_RUNS (case-insensitive)",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths


def _run_metadata(path: Path) -> dict[str, str | None]:
    run_id = path.stem.removesuffix("_steps")
    summary_path = path.with_name(f"{run_id}_summary.json")
    graph_name = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict):
            graph_name = summary.get("graph_name")
            comparison_method = summary.get("stdi_comparison_method")
        else:
            comparison_method = None
    else:
        comparison_method = None
    execution_dir = path.parent
    if execution_dir.name == run_id:
        execution_dir = execution_dir.parent
    execution_metadata = {
        "execution_id": str(execution_dir),
        "execution_label": execution_dir.name,
        "source_path": str(path),
    }
    if comparison_method is not None:
        execution_metadata["metadata_stdi_comparison_method"] = comparison_method
    match = next((match for pattern in RUN_ID_PATTERNS if (match := pattern.match(run_id))), None)
    if match is None:
        return {
            **execution_metadata,
            "run_id": run_id,
            "batch_id": "unknown",
            "graph_id": run_id,
            "chain_code": run_id,
            "chain_label": run_id,
        }

    metadata = match.groupdict()
    graph_id = metadata["graph_id"].zfill(2)
    chain_code = metadata["chain_code"]
    if metadata.get("batch_id") and re.fullmatch(r"[CPDSEMNcpdsemn]+", chain_code):
        chain_code = chain_code.upper()
    return {
        **execution_metadata,
        "run_id": run_id,
        "batch_id": metadata.get("batch_id") or "unknown",
        "graph_id": graph_id,
        "chain_code": chain_code,
        "chain_label": f"{graph_id} · {graph_name or chain_code}",
    }


def _read_analysis_columns(path: Path) -> pd.DataFrame:
    """Read only columns used by the analysis, omitting large text and metadata fields."""
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"'{path}' must contain JSON objects, one per line.")
            record = expand_dual_evaluation_record(record)
            records.append({column: record.get(column) for column in ANALYSIS_STEP_COLUMNS})
    return pd.DataFrame.from_records(records)


def _validate_metric(steps: pd.DataFrame, metric: str) -> None:
    if metric not in available_metrics(steps):
        raise ValueError(f"Metric is unavailable: {metric}")


def _is_kaleido_browser_available() -> bool:
    """Avoid a slow Kaleido browser lookup on Linux environments without Chrome."""
    if not sys.platform.startswith("linux"):
        return True
    return any(
        which(name) is not None for name in ("google-chrome", "chromium", "chromium-browser")
    )
