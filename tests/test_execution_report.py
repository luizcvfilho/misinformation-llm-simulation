from __future__ import annotations

from datetime import datetime

from misinformation_simulation.reporting import execution_report
from misinformation_simulation.reporting.execution_report import (
    _escape_markdown_table_cell,
    _format_markdown_value,
    _format_range,
    _humanize_key,
    append_execution_report,
    resolve_execution_report_path,
)


def test_resolve_execution_report_path_uses_run_dir_env(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "custom-run"
    monkeypatch.setenv("RUN_ID", "run-1")
    monkeypatch.setenv("RUN_DIR", str(run_dir))

    run_id, report_path = resolve_execution_report_path(default_output_root=tmp_path / "output")

    assert run_id == "run-1"
    assert report_path == run_dir / "execution_report.md"
    assert report_path.parent.exists()


def test_resolve_execution_report_path_uses_run_id_when_run_dir_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ID", "run-2")
    monkeypatch.delenv("RUN_DIR", raising=False)

    run_id, report_path = resolve_execution_report_path(default_output_root=tmp_path / "output")

    assert run_id == "run-2"
    assert report_path == tmp_path / "output" / "runs" / "run-2" / "execution_report.md"


def test_formatting_helpers_handle_markdown_values_and_tables() -> None:
    assert _format_markdown_value(None) == "n/a"
    assert _format_markdown_value(True) == "true"
    assert _format_markdown_value({"a": "b"}) == '{"a": "b"}'
    assert _format_range({"start": 3, "end": 3, "count": 1}) == "3 (count: 1)"
    assert _format_range({"start": 3, "end": 5, "count": 3}) == "3-5 (count: 3)"
    assert _format_range({"count": 0}) == "n/a"
    assert _escape_markdown_table_cell("a|b\nc") == "a\\|b c"
    assert _humanize_key("rows_fetched") == "Rows fetched"


def test_append_execution_report_writes_nested_details_and_tables(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "execution_report.md"
    fixed_now = type(
        "FixedDateTime",
        (),
        {"now": staticmethod(lambda: datetime(2026, 4, 25, 12, 0, 0))},
    )
    monkeypatch.setattr(execution_report, "datetime", fixed_now)

    append_execution_report(
        report_path=report_path,
        notebook_name="notebook.ipynb",
        section_title="Simulation",
        run_id=None,
        details={
            "rewrite_metrics": [
                {
                    "dataset": "news",
                    "provider": "gemini",
                    "rows_success": 2,
                    "notes": "a|b",
                }
            ],
            "nested": {"enabled": True, "items": ["one", "two"]},
            "empty_list": [],
        },
    )

    content = report_path.read_text(encoding="utf-8")

    assert "# Execution Report" in content
    assert "## 2026-04-25 12:00:00 - Simulation" in content
    assert "- run_id: manual" in content
    assert "| Dataset | Provider | Rows success | Notes |" in content
    assert "a\\|b" in content
    assert "### Nested" in content
    assert "- **Enabled**: true" in content
    assert "### Empty list" in content
    assert "- n/a" in content


def test_append_execution_report_appends_without_repeating_header(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "execution_report.md"
    fixed_now = type(
        "FixedDateTime",
        (),
        {"now": staticmethod(lambda: datetime(2026, 4, 25, 12, 0, 0))},
    )
    monkeypatch.setattr(
        execution_report,
        "datetime",
        fixed_now,
    )

    append_execution_report(
        report_path=report_path,
        notebook_name="first.ipynb",
        section_title="First",
        run_id="run",
        details={},
    )
    append_execution_report(
        report_path=report_path,
        notebook_name="second.ipynb",
        section_title="Second",
        run_id="run",
        details={},
    )

    content = report_path.read_text(encoding="utf-8")

    assert content.count("# Execution Report") == 1
    assert "first.ipynb" in content
    assert "second.ipynb" in content
