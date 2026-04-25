from __future__ import annotations

import csv
import json
from urllib.error import URLError

import pytest

from misinformation_simulation.datasets import newsdata
from misinformation_simulation.datasets.newsdata import (
    CSV_COLUMNS,
    QUERY_METADATA_ROW_ID,
    _load_existing_csv,
    _merge_news_rows,
    _row_unique_key,
    build_query_metadata,
    fetch_news,
    resolve_output_path,
    save_csv,
    summarize_query_results,
)


def test_resolve_output_path_uses_category_when_output_is_missing(tmp_path, monkeypatch) -> None:
    default_output = tmp_path / "raw" / "newsdata_news.csv"
    monkeypatch.setattr(newsdata, "DEFAULT_OUTPUT", default_output)

    assert resolve_output_path(None, "politics") == default_output.parent / "politics_news.csv"
    assert resolve_output_path(tmp_path / "custom.csv", "politics") == tmp_path / "custom.csv"


def test_summarize_query_results_counts_multi_value_fields() -> None:
    summary = summarize_query_results(
        [
            {
                "title": "A",
                "description": "Body",
                "content": "",
                "full_description": ["Full"],
                "pubDate": "2026-04-20",
                "source_name": "Source A",
                "language": "en",
                "country": ["us", "br"],
                "category": "politics,world",
                "keywords": "election; vote",
            },
            {
                "title": "B",
                "pubDate": "2026-04-21",
                "source_name": "Source A",
                "language": "pt",
                "country": "br",
                "category": "world",
            },
        ]
    )

    assert summary["rows_fetched"] == 2
    assert summary["date_range"] == {
        "min_pubDate": "2026-04-20",
        "max_pubDate": "2026-04-21",
    }
    assert summary["source_name_summary"]["top_values"][0] == {"value": "Source A", "count": 2}
    assert summary["country_summary"]["unique_count"] == 2
    assert summary["category_summary"]["unique_count"] == 2
    assert summary["content_coverage"]["with_title"] == 2
    assert summary["content_coverage"]["with_description"] == 1
    assert summary["content_coverage"]["with_full_description"] == 1


def test_fetch_news_paginates_deduplicates_and_respects_max_records(monkeypatch) -> None:
    responses = iter(
        [
            {
                "status": "success",
                "results": [
                    {"article_id": "1", "link": "link-1"},
                    {"article_id": "2", "link": "link-2"},
                ],
                "nextPage": "page-2",
            },
            {
                "status": "success",
                "results": [
                    {"article_id": "2", "link": "duplicate"},
                    {"article_id": "3", "link": "link-3"},
                ],
            },
        ]
    )
    requested_params: list[dict[str, object]] = []

    def fake_request_news(params):
        requested_params.append(params)
        return next(responses)

    monkeypatch.setattr(newsdata, "_request_news", fake_request_news)

    records = fetch_news(
        api_key="key",
        query="topic",
        language="en",
        country="us",
        category="politics",
        max_records=3,
    )

    assert [item["article_id"] for item in records] == ["1", "2", "3"]
    assert requested_params[0]["q"] == "topic"
    assert requested_params[1]["page"] == "page-2"


def test_fetch_news_rejects_unexpected_api_status(monkeypatch) -> None:
    monkeypatch.setattr(
        newsdata,
        "_request_news",
        lambda _params: {"status": "error", "results": "invalid key"},
    )

    with pytest.raises(RuntimeError, match="unexpected status"):
        fetch_news("key", "", "en", "", "", 10)


def test_request_news_wraps_connection_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        newsdata, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down"))
    )

    with pytest.raises(RuntimeError, match="Connection failure"):
        newsdata._request_news({"apikey": "key"})


def test_build_query_metadata_includes_parameters_and_summary(monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now(_tz):
            return type("FixedNow", (), {"isoformat": lambda self: "2026-04-25T00:00:00+00:00"})()

    monkeypatch.setattr(newsdata, "datetime", FixedDateTime)

    metadata = build_query_metadata(
        query="topic",
        language="en",
        country="us",
        category="politics",
        max_records=5,
        news=[{"title": "A"}],
    )

    assert metadata["query_parameters"]["query"] == "topic"
    assert metadata["query_results_summary"]["rows_fetched"] == 1
    assert metadata["fetched_at_utc"] == "2026-04-25T00:00:00+00:00"


def test_row_unique_key_prefers_article_id_and_ignores_metadata_row() -> None:
    assert _row_unique_key({"article_id": "abc", "link": "link"}) == "article::abc"
    assert _row_unique_key({"article_id": QUERY_METADATA_ROW_ID, "link": "link"}) == "link::link"
    assert _row_unique_key({"article_id": "", "link": ""}) == ""


def test_load_existing_csv_reads_metadata_and_skips_blank_rows(tmp_path) -> None:
    csv_path = tmp_path / "news.csv"
    metadata = {"query_parameters": {"language": "en"}}
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        metadata_row = {column: "" for column in CSV_COLUMNS}
        metadata_row["article_id"] = QUERY_METADATA_ROW_ID
        metadata_row["description"] = json.dumps(metadata)
        writer.writerow(metadata_row)
        writer.writerow({column: "" for column in CSV_COLUMNS})
        writer.writerow({"article_id": "1", "title": "Title"})

    rows, loaded_metadata = _load_existing_csv(csv_path, CSV_COLUMNS)

    assert loaded_metadata == metadata
    assert rows == [{**{column: "" for column in CSV_COLUMNS}, "article_id": "1", "title": "Title"}]


def test_merge_news_rows_skips_duplicate_keys() -> None:
    merged, appended_count = _merge_news_rows(
        [{"article_id": "1", "link": "link-1"}],
        [{"article_id": "1", "link": "duplicate"}, {"article_id": "2", "link": "link-2"}],
    )

    assert [row["article_id"] for row in merged] == ["1", "2"]
    assert appended_count == 1


def test_save_csv_merges_rows_and_tracks_metadata(tmp_path, monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now(_tz):
            return type("FixedNow", (), {"isoformat": lambda self: "2026-04-25T00:00:00+00:00"})()

    monkeypatch.setattr(newsdata, "datetime", FixedDateTime)
    output_path = tmp_path / "news.csv"
    query_metadata = {
        "fetched_at_utc": "2026-04-25T00:00:00+00:00",
        "query_parameters": {"language": "en"},
        "query_results_summary": {"rows_fetched": 2},
    }

    first_counts = save_csv(
        [{"article_id": "1", "title": "First"}],
        output_path,
        query_metadata,
    )
    second_counts = save_csv(
        [{"article_id": "1", "title": "Duplicate"}, {"article_id": "2", "title": "Second"}],
        output_path,
        query_metadata,
    )

    rows, metadata = _load_existing_csv(output_path, CSV_COLUMNS)

    assert first_counts == (1, 1, 1)
    assert second_counts == (2, 1, 2)
    assert [row["article_id"] for row in rows] == ["1", "2"]
    assert metadata["total_requests"] == 2
    assert metadata["latest_request"]["rows_skipped_as_duplicates"] == 1
