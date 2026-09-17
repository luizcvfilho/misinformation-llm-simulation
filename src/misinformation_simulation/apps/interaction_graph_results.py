from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from misinformation_simulation.apps.interaction_graph_ui import (
    build_news_summary_dataframe,
    build_node_summary_dataframe,
)


def find_saved_results(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        output_dir.rglob("*_summary.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )


def load_saved_result(summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    if not summary_path.name.endswith("_summary.json"):
        raise ValueError("Select a simulation summary file ending in _summary.json.")
    prefix = summary_path.name.removesuffix("_summary.json")
    steps_path = summary_path.with_name(f"{prefix}_steps.jsonl")
    if not steps_path.is_file():
        raise ValueError(f"Matching steps file was not found: {steps_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not {"rows_processed", "steps_total"} <= summary.keys():
        raise ValueError("The selected file is not a graph simulation summary.")
    records = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("The steps file must contain JSON objects, one per line.")
    steps_df = pd.DataFrame(records)
    if not steps_df.empty:
        required = {
            "news_id",
            "step_index",
            "node_id",
            "node_label",
            "provider",
            "model",
            "rewrite_status",
            "metadata_title",
            "stdi_vs_original",
            "stdi_incremental",
            "stdi_cumulative",
            "vad_drift_vs_original",
            "contradiction_drift_vs_original",
        }
        missing = required - set(steps_df.columns)
        if missing:
            raise ValueError(f"The steps file is missing columns: {', '.join(sorted(missing))}")
        steps_df = steps_df.sort_values(["news_id", "step_index"]).reset_index(drop=True)
    return {
        "name": prefix,
        "source": "imported",
        "status": "cancelled" if summary.get("cancelled") else "completed",
        "summary": summary,
        "summary_path": str(summary_path),
        "steps_path": str(steps_path),
        "steps_df": steps_df,
        "node_summary_df": build_node_summary_dataframe(steps_df),
        "news_summary_df": build_news_summary_dataframe(steps_df),
        "output_prefix": prefix,
        "graph_payload": None,
    }


def remove_imported_result(bundles: list[dict[str, Any]], index: int) -> bool:
    if bundles[index].get("source") != "imported":
        return False
    bundles.pop(index)
    return True


def clear_imported_results(bundles: list[dict[str, Any]]) -> int:
    count = sum(bundle.get("source") == "imported" for bundle in bundles)
    bundles[:] = [bundle for bundle in bundles if bundle.get("source") != "imported"]
    return count
