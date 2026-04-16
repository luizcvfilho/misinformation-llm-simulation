from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from misinformation_simulation.config.settings import DEFAULT_RAW_DATA_DIR

API_URL = "https://newsdata.io/api/1/latest"
DEFAULT_OUTPUT = DEFAULT_RAW_DATA_DIR / "newsdata_news.csv"
QUERY_METADATA_ROW_ID = "__query_metadata__"
CSV_COLUMNS = [
    "article_id",
    "title",
    "link",
    "description",
    "content",
    "full_description",
    "pubDate",
    "pubDateTZ",
    "image_url",
    "video_url",
    "source_id",
    "source_name",
    "source_priority",
    "source_url",
    "source_icon",
    "language",
    "country",
    "category",
    "creator",
    "keywords",
    "duplicate",
]


def resolve_output_path(output: Path | None, category: str) -> Path:
    if output is not None:
        return output

    category_value = category.strip()
    if category_value:
        return DEFAULT_OUTPUT.parent / f"{category_value}_news.csv"

    return DEFAULT_OUTPUT


def _request_news(params: dict[str, Any]) -> dict[str, Any]:
    query_string = urlencode(params)
    url = f"{API_URL}?{query_string}"

    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP error {error.code} while querying NewsData.io: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Connection failure while querying NewsData.io: {error.reason}"
        ) from error

    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("API response is not valid JSON.") from error


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _split_multi_value(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value)
        if not text.strip():
            return []
        items = []
        for chunk in text.split(";"):
            items.extend(chunk.split(","))

    return [str(item).strip() for item in items if str(item).strip()]


def _count_values(news: list[dict[str, Any]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in news:
        counter.update(_split_multi_value(item.get(field)))
    return counter


def _summarize_field(news: list[dict[str, Any]], field: str, top_n: int = 10) -> dict[str, Any]:
    counts = _count_values(news, field)
    return {
        "unique_count": len(counts),
        "top_values": [
            {"value": value, "count": count} for value, count in counts.most_common(top_n)
        ],
    }


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return True


def summarize_query_results(news: list[dict[str, Any]]) -> dict[str, Any]:
    pub_dates = [
        str(item.get("pubDate", "")).strip()
        for item in news
        if str(item.get("pubDate", "")).strip()
    ]

    return {
        "rows_fetched": len(news),
        "date_range": {
            "min_pubDate": min(pub_dates) if pub_dates else None,
            "max_pubDate": max(pub_dates) if pub_dates else None,
        },
        "source_name_summary": _summarize_field(news, "source_name"),
        "language_summary": _summarize_field(news, "language"),
        "country_summary": _summarize_field(news, "country"),
        "category_summary": _summarize_field(news, "category"),
        "keyword_summary": _summarize_field(news, "keywords"),
        "content_coverage": {
            "with_title": sum(1 for item in news if _is_non_empty(item.get("title"))),
            "with_description": sum(1 for item in news if _is_non_empty(item.get("description"))),
            "with_content": sum(1 for item in news if _is_non_empty(item.get("content"))),
            "with_full_description": sum(
                1 for item in news if _is_non_empty(item.get("full_description"))
            ),
        },
    }


def fetch_news(
    api_key: str,
    query: str,
    language: str,
    country: str,
    category: str,
    max_records: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_page: str | None = None

    while len(records) < max_records:
        params: dict[str, Any] = {
            "apikey": api_key,
            "language": language,
        }
        if query:
            params["q"] = query
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if next_page:
            params["page"] = next_page

        data = _request_news(params)
        status = data.get("status")
        if status != "success":
            message = data.get("results", data)
            raise RuntimeError(f"API returned unexpected status: {status} | {message}")

        batch = data.get("results", [])
        if not isinstance(batch, list) or not batch:
            break

        for item in batch:
            article_id = _to_text(item.get("article_id")).strip()
            fallback_key = _to_text(item.get("link")).strip()
            unique_key = article_id or fallback_key
            if unique_key and unique_key in seen_ids:
                continue
            if unique_key:
                seen_ids.add(unique_key)

            records.append(item)
            if len(records) >= max_records:
                break

        next_page = data.get("nextPage")
        if not next_page:
            break

    return records


def build_query_metadata(
    *,
    query: str,
    language: str,
    country: str,
    category: str,
    max_records: int,
    news: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query_parameters": {
            "query": query,
            "language": language,
            "country": country,
            "category": category,
            "max_records": max_records,
        },
        "query_results_summary": summarize_query_results(news),
        "fetched_at_utc": datetime.now(UTC).isoformat(),
    }


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
