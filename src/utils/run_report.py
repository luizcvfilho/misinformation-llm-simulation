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


def _indent(level: int) -> str:
    return "  " * max(level, 0)


def _escape_markdown_table_cell(value: str) -> str:
    # Evita quebrar tabelas markdown quando houver pipes ou quebras de linha.
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _humanize_key(key: str) -> str:
    normalized = key.replace("_", " ").strip()
    return normalized[:1].upper() + normalized[1:] if normalized else key


def _render_scalar_list(lines: list[str], items: list[Any]) -> None:
    for item in items:
        lines.append(f"- {_format_markdown_value(item)}")


def _render_nested_value(lines: list[str], value: Any, level: int = 0) -> None:
    prefix = f"{_indent(level)}- "

    if isinstance(value, dict):
        if not value:
            lines.append(f"{prefix}n/a")
            return

        for sub_key, sub_value in value.items():
            key_label = _humanize_key(str(sub_key))
            if isinstance(sub_value, (dict, list)):
                lines.append(f"{prefix}**{key_label}**:")
                _render_nested_value(lines, sub_value, level + 1)
            else:
                lines.append(f"{prefix}**{key_label}**: {_format_markdown_value(sub_value)}")
        return

    if isinstance(value, list):
        if not value:
            lines.append(f"{prefix}n/a")
            return

        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}Item:")
                _render_nested_value(lines, item, level + 1)
            else:
                lines.append(f"{prefix}{_format_markdown_value(item)}")
        return

    lines.append(f"{prefix}{_format_markdown_value(value)}")


def _render_dict(lines: list[str], content: dict[str, Any]) -> None:
    if not content:
        lines.append("- n/a")
        return

    _render_nested_value(lines, content, level=0)


def _render_rewrite_metrics_table(lines: list[str], metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        lines.append("- n/a")
        return

    preferred_columns = [
        "dataset",
        "output_name",
        "provider",
        "model",
        "duration_seconds",
        "rows_requested",
        "rows_success",
        "rows_error",
    ]
    seen = set()
    columns: list[str] = []

    for column in preferred_columns:
        if any(column in row for row in metrics):
            columns.append(column)
            seen.add(column)

    for row in metrics:
        for column in row.keys():
            if column not in seen:
                columns.append(column)
                seen.add(column)

    header_columns = [_humanize_key(str(column)) for column in columns]

    lines.append("| " + " | ".join(header_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in metrics:
        values = [_escape_markdown_table_cell(_format_markdown_value(row.get(column))) for column in columns]
        lines.append("| " + " | ".join(values) + " |")


def _render_detail_section(lines: list[str], key: str, value: Any) -> None:
    lines.append(f"### {_humanize_key(key)}")

    if isinstance(value, dict):
        _render_dict(lines, value)
        lines.append("")
        return

    if isinstance(value, list):
        if not value:
            lines.append("- n/a")
            lines.append("")
            return

        if all(isinstance(item, dict) for item in value):
            if key.lower() in {"rewrite_metrics", "rewrite_run_metrics"}:
                _render_rewrite_metrics_table(lines, value)
            else:
                for idx, item in enumerate(value, start=1):
                    lines.append(f"- Item {idx}")
                    _render_dict(lines, item)
        else:
            _render_scalar_list(lines, value)

        lines.append("")
        return

    lines.append(f"- {_format_markdown_value(value)}")
    lines.append("")


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
        "",
        "### Details",
    ])

    for key, value in details.items():
        _render_detail_section(lines, key, value)

    lines.append("---")
    lines.append("")

    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
