from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_execution_report_path(default_output_root: Path = Path("../output")) -> tuple[str | None, Path]:
    run_id = os.getenv("RUN_ID", "").strip() or None
    run_dir_env = os.getenv("RUN_DIR", "").strip()

    if run_dir_env:
        base_dir = Path(run_dir_env)
    elif run_id:
        base_dir = default_output_root / "runs" / run_id
    else:
        base_dir = default_output_root

    report_path = base_dir / "execution_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return run_id, report_path


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def append_execution_report(
    *,
    report_path: Path,
    notebook_name: str,
    section_title: str,
    run_id: str | None,
    details: dict[str, Any],
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    if not report_path.exists():
        lines.extend([
            "# Execution Report",
            "",
            "Auto-generated execution metadata per notebook run.",
            "",
        ])

    lines.extend([
        f"## {timestamp} - {section_title}",
        f"- notebook: {notebook_name}",
        f"- run_id: {run_id or 'manual'}",
    ])

    for key, value in details.items():
        lines.append(f"- {key}: {_format_markdown_value(value)}")

    lines.append("")

    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
