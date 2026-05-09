from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from misinformation_simulation.config.settings import DEFAULT_RAW_DATA_DIR
from misinformation_simulation.datasets.newsdata_schema import CSV_COLUMNS, QUERY_METADATA_ROW_ID
from misinformation_simulation.datasets.newsdata_storage import (
    _load_existing_csv,
    _merge_news_rows,
    _normalize_row,
    _row_unique_key,
    save_csv,
)
from misinformation_simulation.datasets.newsdata_summary import _to_text, summarize_query_results

API_URL = "https://newsdata.io/api/1/latest"
DEFAULT_OUTPUT = DEFAULT_RAW_DATA_DIR / "newsdata_news.csv"
__all__ = [
    "CSV_COLUMNS",
    "QUERY_METADATA_ROW_ID",
    "_load_existing_csv",
    "_merge_news_rows",
    "_normalize_row",
    "_row_unique_key",
    "build_query_metadata",
    "fetch_news",
    "resolve_output_path",
    "save_csv",
    "summarize_query_results",
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
