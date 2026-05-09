from __future__ import annotations

from collections import Counter
from typing import Any


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
