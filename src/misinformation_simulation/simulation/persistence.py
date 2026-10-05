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
