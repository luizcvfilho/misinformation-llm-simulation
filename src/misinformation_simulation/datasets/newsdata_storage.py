from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from misinformation_simulation.datasets.newsdata_schema import CSV_COLUMNS, QUERY_METADATA_ROW_ID
from misinformation_simulation.datasets.newsdata_summary import _to_text, summarize_query_results


def _normalize_row(item: dict[str, Any], columns: list[str]) -> dict[str, str]:
    return {column: _to_text(item.get(column)) for column in columns}


def _row_unique_key(row: dict[str, Any]) -> str:
    article_id = _to_text(row.get("article_id")).strip()
    link = _to_text(row.get("link")).strip()
    if article_id and article_id != QUERY_METADATA_ROW_ID:
        return f"article::{article_id}"
    if link:
        return f"link::{link}"
    return ""


def _load_existing_csv(
    output_path: Path, columns: list[str]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not output_path.exists():
        return [], {}

    rows: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}

    with output_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for raw_row in reader:
            if raw_row is None:
                continue

            row = {column: _to_text(raw_row.get(column, "")) for column in columns}
            if row.get("article_id", "").strip().lower() == QUERY_METADATA_ROW_ID:
                raw_metadata = row.get("description", "").strip()
                if raw_metadata:
                    try:
                        parsed = json.loads(raw_metadata)
                        if isinstance(parsed, dict):
                            metadata = parsed
                    except json.JSONDecodeError:
                        metadata = {"raw_description": raw_metadata}
                continue

            if any(value.strip() for value in row.values()):
                rows.append(row)

    return rows, metadata


def _merge_news_rows(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    merged = list(existing_rows)
    seen_keys = {key for key in (_row_unique_key(row) for row in existing_rows) if key}
    appended_count = 0

    for row in new_rows:
        key = _row_unique_key(row)
        if key and key in seen_keys:
            continue

        merged.append(row)
        if key:
            seen_keys.add(key)
        appended_count += 1

    return merged, appended_count


def _build_merged_metadata(
    *,
    existing_metadata: dict[str, Any],
    latest_query_metadata: dict[str, Any],
    existing_rows_count: int,
    merged_rows: list[dict[str, str]],
    new_rows_fetched: int,
    new_rows_appended: int,
) -> dict[str, Any]:
    now_utc = datetime.now(UTC).isoformat()

    if new_rows_appended > 0:
        dataset_start = existing_rows_count + 1
        dataset_end = existing_rows_count + new_rows_appended
        dataset_indexes = list(range(dataset_start, dataset_end + 1))
        csv_line_start = dataset_start + 2
        csv_line_end = dataset_end + 2
        csv_line_numbers = list(range(csv_line_start, csv_line_end + 1))
    else:
        dataset_start = None
        dataset_end = None
        dataset_indexes = []
        csv_line_start = None
        csv_line_end = None
        csv_line_numbers = []

    latest_entry = {
        "fetched_at_utc": latest_query_metadata.get("fetched_at_utc"),
        "query_parameters": latest_query_metadata.get("query_parameters", {}),
        "query_results_summary": latest_query_metadata.get("query_results_summary", {}),
        "rows_fetched_in_request": int(new_rows_fetched),
        "rows_appended_to_file": int(new_rows_appended),
        "rows_skipped_as_duplicates": int(max(new_rows_fetched - new_rows_appended, 0)),
        "dataset_row_index_range": {
            "start": dataset_start,
            "end": dataset_end,
            "count": int(new_rows_appended),
        },
        "dataset_row_indexes": dataset_indexes,
        "csv_line_range": {
            "start": csv_line_start,
            "end": csv_line_end,
            "count": int(new_rows_appended),
        },
        "csv_line_numbers": csv_line_numbers,
    }

    legacy_entry = None
    if not isinstance(existing_metadata.get("request_history"), list):
        if isinstance(existing_metadata.get("query_parameters"), dict):
            legacy_entry = {
                "fetched_at_utc": existing_metadata.get("fetched_at_utc"),
                "query_parameters": existing_metadata.get("query_parameters", {}),
                "query_results_summary": existing_metadata.get("query_results_summary", {}),
                "rows_fetched_in_request": existing_metadata.get("query_results_summary", {}).get(
                    "rows_fetched"
                ),
                "rows_appended_to_file": None,
                "rows_skipped_as_duplicates": None,
                "dataset_row_index_range": None,
                "dataset_row_indexes": [],
                "csv_line_range": None,
                "csv_line_numbers": [],
            }

    history_raw = existing_metadata.get("request_history", [])
    history: list[dict[str, Any]] = []
    if isinstance(legacy_entry, dict):
        history.append(legacy_entry)
    if isinstance(history_raw, list):
        for entry in history_raw:
            if isinstance(entry, dict):
                history.append(entry)
    history.append(latest_entry)

    return {
        "updated_at_utc": now_utc,
        "latest_request": latest_entry,
        "request_history": history,
        "total_requests": len(history),
        "accumulated_dataset_summary": summarize_query_results(merged_rows),
    }


def save_csv(
    news: list[dict[str, Any]],
    output_path: Path,
    query_metadata: dict[str, Any],
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows, existing_metadata = _load_existing_csv(output_path, CSV_COLUMNS)
    new_rows = [_normalize_row(item, CSV_COLUMNS) for item in news]
    merged_rows, appended_count = _merge_news_rows(existing_rows, new_rows)

    merged_metadata = _build_merged_metadata(
        existing_metadata=existing_metadata,
        latest_query_metadata=query_metadata,
        existing_rows_count=len(existing_rows),
        merged_rows=merged_rows,
        new_rows_fetched=len(new_rows),
        new_rows_appended=appended_count,
    )

    latest_params = query_metadata.get("query_parameters", {})
    if not isinstance(latest_params, dict):
        latest_params = {}

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        metadata_row = {column: "" for column in CSV_COLUMNS}
        metadata_row["article_id"] = QUERY_METADATA_ROW_ID
        metadata_row["title"] = "QUERY_METADATA"
        metadata_row["description"] = json.dumps(merged_metadata, ensure_ascii=False)
        metadata_row["language"] = _to_text(latest_params.get("language"))
        metadata_row["country"] = _to_text(latest_params.get("country"))
        metadata_row["category"] = _to_text(latest_params.get("category"))
        writer.writerow(metadata_row)

        for row in merged_rows:
            writer.writerow(row)

    return len(new_rows), appended_count, len(merged_rows)
