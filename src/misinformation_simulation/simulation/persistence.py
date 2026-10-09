from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from misinformation_simulation.simulation.types import SimulationResult


def _persist_results(
    *,
    result: SimulationResult,
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{output_prefix}_summary.json"
    steps_path = output_dir / f"{output_prefix}_steps.jsonl"

    if result.path_results:
        descriptors = []
        for path_result in result.path_results:
            path_id = path_result.summary["path_id"]
            path_result.summary["parent_graph_summary_path"] = f"../../{summary_path.name}"
            path_result.summary["parent_graph_steps_path"] = f"../../{steps_path.name}"
            path_result.summary["execution_name"] = result.summary.get(
                "execution_name", output_dir.name
            )
            path_summary, path_steps = _persist_results(
                result=path_result,
                output_dir=output_dir / "paths" / path_id,
                output_prefix=path_id,
            )
            path_result.summary_path = path_summary
            path_result.steps_path = path_steps
            descriptors.append(
                {
                    "path_id": path_id,
                    "chain_code": path_result.summary["chain_code"],
                    "node_ids": path_result.summary["graph"]["ordered_node_ids"],
                    "summary_path": str(path_summary.relative_to(output_dir)),
                    "steps_path": str(path_steps.relative_to(output_dir)),
                }
            )
        result.summary["path_results"] = descriptors

    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if result.summary.get("stdi_comparison_method") == "dual":
        steps_path.write_text(
            "".join(
                json.dumps(step.to_record(), ensure_ascii=False, allow_nan=False) + "\n"
                for step in result.step_results
            ),
            encoding="utf-8",
        )
    else:
        steps_df = pd.DataFrame([step.to_record() for step in result.step_results])
        steps_df.to_json(steps_path, orient="records", lines=True, force_ascii=False)
    return summary_path, steps_path
