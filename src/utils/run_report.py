from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_execution_report_path(
    default_output_root: Path = Path("../output"),
) -> tuple[str | None, Path]:
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
    # Prevent markdown tables from breaking when there are pipes or line breaks.
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _format_range(value: Any) -> str:
    if not isinstance(value, dict):
        return _format_markdown_value(value)

    start = value.get("start")
    end = value.get("end")
    count = value.get("count")

    if start is None and end is None:
        return "n/a"
    if start == end:
        return f"{start} (count: {count})"
    return f"{start}-{end} (count: {count})"


def _extract_date_range_value(summary: dict[str, Any], key: str) -> Any:
    date_range = summary.get("date_range", {})
    if not isinstance(date_range, dict):
        return None
    return date_range.get(key)


def _extract_unique_count(summary: dict[str, Any], key: str) -> Any:
    section = summary.get(key, {})
    if not isinstance(section, dict):
        return None
    return section.get("unique_count")


def _render_newsdata_query_summary(lines: list[str], content: dict[str, Any]) -> None:
    if not content:
        lines.append("- n/a")
        lines.append("")
        return

    lines.append(f"- **Updated at utc**: {_format_markdown_value(content.get('updated_at_utc'))}")
    lines.append(f"- **Total requests**: {_format_markdown_value(content.get('total_requests'))}")

    latest = content.get("latest_request", {})
    if isinstance(latest, dict):
        params = latest.get("query_parameters", {})
        if not isinstance(params, dict):
            params = {}

        lines.append("- **Latest query summary**:")
        lines.append(
            f"  - **Fetched at utc**: {_format_markdown_value(latest.get('fetched_at_utc'))}"
        )
        lines.append(f"  - **Query**: {_format_markdown_value(params.get('query'))}")
        lines.append(f"  - **Language**: {_format_markdown_value(params.get('language'))}")
        lines.append(f"  - **Country**: {_format_markdown_value(params.get('country'))}")
        lines.append(f"  - **Category**: {_format_markdown_value(params.get('category'))}")
        lines.append(f"  - **Max records**: {_format_markdown_value(params.get('max_records'))}")
        lines.append(
            f"  - **Rows fetched**: {_format_markdown_value(latest.get('rows_fetched_in_request'))}"
        )
        lines.append(
            f"  - **Rows appended**: {_format_markdown_value(latest.get('rows_appended_to_file'))}"
        )
        lines.append(
            "  - **Rows skipped as duplicates**: "
            f"{_format_markdown_value(latest.get('rows_skipped_as_duplicates'))}"
        )
        lines.append(
            f"  - **Dataset rows**: {_format_range(latest.get('dataset_row_index_range'))}"
        )
        lines.append(f"  - **Csv lines**: {_format_range(latest.get('csv_line_range'))}")

        summary = latest.get("query_results_summary", {})
        if isinstance(summary, dict):
            lines.append("  - **Result summary**:")
            lines.append(
                f"    - **Rows fetched**: {_format_markdown_value(summary.get('rows_fetched'))}"
            )
            lines.append(
                "    - **Date range min**: "
                f"{_format_markdown_value(_extract_date_range_value(summary, 'min_pubDate'))}"
            )
            lines.append(
                "    - **Date range max**: "
                f"{_format_markdown_value(_extract_date_range_value(summary, 'max_pubDate'))}"
            )
            lines.append(
                "    - **Unique sources**: "
                f"{_format_markdown_value(_extract_unique_count(summary, 'source_name_summary'))}"
            )
            lines.append(
                "    - **Unique keywords**: "
                f"{_format_markdown_value(_extract_unique_count(summary, 'keyword_summary'))}"
            )

    history = content.get("request_history", [])
    if isinstance(history, list) and history:
        lines.append("- **Query history**:")
        headers = [
            "#",
            "fetched_at_utc",
            "query",
            "language",
            "country",
            "category",
            "max_records",
            "rows_fetched",
            "rows_appended",
            "rows_skipped",
            "dataset_rows",
            "csv_lines",
        ]

        human_headers = [_humanize_key(h) for h in headers]
        lines.append("  | " + " | ".join(human_headers) + " |")
        lines.append("  | " + " | ".join(["---"] * len(headers)) + " |")

        for idx, item in enumerate(history, start=1):
            if not isinstance(item, dict):
                continue
            params = item.get("query_parameters", {})
            if not isinstance(params, dict):
                params = {}

            row_values = [
                str(idx),
                _format_markdown_value(item.get("fetched_at_utc")),
                _format_markdown_value(params.get("query")),
                _format_markdown_value(params.get("language")),
                _format_markdown_value(params.get("country")),
                _format_markdown_value(params.get("category")),
                _format_markdown_value(params.get("max_records")),
                _format_markdown_value(item.get("rows_fetched_in_request")),
                _format_markdown_value(item.get("rows_appended_to_file")),
                _format_markdown_value(item.get("rows_skipped_as_duplicates")),
                _format_range(item.get("dataset_row_index_range")),
                _format_range(item.get("csv_line_range")),
            ]
            row_values = [_escape_markdown_table_cell(v) for v in row_values]
            lines.append("  | " + " | ".join(row_values) + " |")

    accumulated = content.get("accumulated_dataset_summary", {})
    if isinstance(accumulated, dict):
        lines.append("- **Accumulated dataset summary**:")
        lines.append(
            f"  - **Rows fetched**: {_format_markdown_value(accumulated.get('rows_fetched'))}"
        )

        date_range = accumulated.get("date_range", {})
        if isinstance(date_range, dict):
            lines.append(
                f"  - **Date range min**: {_format_markdown_value(date_range.get('min_pubDate'))}"
            )
            lines.append(
                f"  - **Date range max**: {_format_markdown_value(date_range.get('max_pubDate'))}"
            )

        lines.append(
            "  - **Unique sources**: "
            f"{_format_markdown_value(_extract_unique_count(accumulated, 'source_name_summary'))}"
        )
        lines.append(
            "  - **Unique countries**: "
            f"{_format_markdown_value(_extract_unique_count(accumulated, 'country_summary'))}"
        )
        lines.append(
            "  - **Unique categories**: "
            f"{_format_markdown_value(_extract_unique_count(accumulated, 'category_summary'))}"
        )
        lines.append(
            "  - **Unique keywords**: "
            f"{_format_markdown_value(_extract_unique_count(accumulated, 'keyword_summary'))}"
        )


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
        values = [
            _escape_markdown_table_cell(_format_markdown_value(row.get(column)))
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")


def _render_detail_section(lines: list[str], key: str, value: Any) -> None:
    lines.append(f"### {_humanize_key(key)}")

    if key.lower() == "newsdata_query_details" and isinstance(value, dict):
        _render_newsdata_query_summary(lines, value)
        lines.append("")
        return

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
        lines.extend(
            [
                "# Execution Report",
                "",
                "Auto-generated execution metadata per notebook run.",
                "",
            ]
        )

    lines.extend(
        [
            f"## {timestamp} - {section_title}",
            f"- notebook: {notebook_name}",
            f"- run_id: {run_id or 'manual'}",
            "",
            "### Details",
        ]
    )

    for key, value in details.items():
        _render_detail_section(lines, key, value)

    lines.append("---")
    lines.append("")

    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
