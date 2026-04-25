from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
    infer_text_language,
    normalize_language_code,
    resolve_output_language,
    resolve_output_language_name,
    resolve_row_text,
)


def test_choose_news_text_column_prefers_richest_candidate() -> None:
    df = pd.DataFrame(
        [
            {"title": "Short", "description": "A longer body of text."},
            {"title": "Tiny", "description": "Another useful paragraph."},
        ]
    )

    assert choose_news_text_column(df, candidates=["title", "description"]) == "description"


def test_choose_news_text_column_rejects_empty_candidates() -> None:
    df = pd.DataFrame([{"title": "   ", "description": None}])

    with pytest.raises(ValueError, match="No text column"):
        choose_news_text_column(df, candidates=["title", "description"])


def test_resolve_row_text_uses_preferred_column_before_candidates() -> None:
    row = pd.Series({"title": "Fallback title", "description": "Detailed text"})

    source_column, text = resolve_row_text(row, preferred_column="description")

    assert source_column == "description"
    assert text == "Detailed text"


def test_resolve_row_text_can_disable_title_fallback() -> None:
    row = pd.Series({"title": "Only title"})

    with pytest.raises(ValueError, match="no usable text"):
        resolve_row_text(row, candidates=["title"], allow_title_fallback=False)


@pytest.mark.parametrize(
    ("raw_language", "expected"),
    [
        ("English", "en"),
        ("pt,br", "pt"),
        (" Portuguese ", "pt"),
        ("unknown", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_language_code(raw_language: str | None, expected: str | None) -> None:
    assert normalize_language_code(raw_language) == expected


def test_resolve_output_language_prioritizes_row_language() -> None:
    row = pd.Series({"language": "Portuguese", "country": "us"})

    assert resolve_output_language(row, "the article body") == ("pt", "row.language")


def test_resolve_output_language_uses_country_before_text_heuristic() -> None:
    row = pd.Series({"country": "br"})

    assert resolve_output_language(row, "the article and the report") == ("pt", "row.country")


def test_resolve_output_language_uses_text_heuristic_then_default() -> None:
    assert resolve_output_language(pd.Series({}), "the article and the report") == (
        "en",
        "heuristic",
    )
    assert resolve_output_language(pd.Series({}), "ambiguous") == ("en", "default")


def test_infer_text_language_detects_portuguese_markers() -> None:
    assert infer_text_language("A decisão pública gerou reações na população.") == "pt"


def test_resolve_output_language_name_falls_back_to_uppercase_code() -> None:
    assert resolve_output_language_name("pt") == "Portuguese"
    assert resolve_output_language_name("xx") == "XX"
