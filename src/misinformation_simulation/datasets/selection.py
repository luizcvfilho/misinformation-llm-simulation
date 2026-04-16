from __future__ import annotations

import pandas as pd

from misinformation_simulation.config.datasets import (
    DETAILED_TEXT_COLUMNS,
    LANGUAGE_CODE_TO_NAME,
    TEXT_COLUMN_CANDIDATES,
)


def choose_news_text_column(
    df: pd.DataFrame,
    candidates=TEXT_COLUMN_CANDIDATES,
    prefer_detailed_text: bool = True,
) -> str:
    if prefer_detailed_text:
        priority_candidates = [column for column in DETAILED_TEXT_COLUMNS if column in candidates]
    else:
        priority_candidates = list(candidates)

    fallback_candidates = [column for column in candidates if column not in priority_candidates]
    ordered_candidates = [*priority_candidates, *fallback_candidates]

    best_column = None
    best_total_chars = -1

    for column in ordered_candidates:
        if column not in df.columns:
            continue

        series = df[column].fillna("").astype(str).str.strip()
        non_empty_series = series[series.ne("")]
        if non_empty_series.empty:
            continue

        total_chars = int(non_empty_series.str.len().sum())
        if total_chars > best_total_chars:
            best_total_chars = total_chars
            best_column = column

    if best_column is not None:
        return best_column

    raise ValueError("No text column was found. Provide 'text_column' manually.")


def resolve_row_text(
    row: pd.Series,
    preferred_column: str | None = None,
    candidates=TEXT_COLUMN_CANDIDATES,
    allow_title_fallback: bool = False,
) -> tuple[str, str]:
    ordered_candidates = []
    if preferred_column:
        ordered_candidates.append(preferred_column)

    ordered_candidates.extend(column for column in candidates if column not in ordered_candidates)

    if not allow_title_fallback:
        ordered_candidates = [column for column in ordered_candidates if column != "title"]

    for column in ordered_candidates:
        if column not in row.index:
            continue

        value = row[column]
        text = "" if pd.isna(value) else str(value).strip()
        if text:
            return column, text

    if allow_title_fallback and "title" in row.index:
        title_value = row["title"]
        title_text = "" if pd.isna(title_value) else str(title_value).strip()
        if title_text:
            return "title", title_text

    raise ValueError("The row has no usable text in the candidate columns.")


def normalize_language_code(raw_language: str | None) -> str | None:
    if not raw_language:
        return None

    value = raw_language.strip().lower()
    if not value:
        return None

    aliases = {
        "english": "en",
        "portuguese": "pt",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
    }

    if value in aliases:
        return aliases[value]

    if "," in value:
        first = value.split(",", maxsplit=1)[0].strip()
        if first in aliases:
            return aliases[first]
        if len(first) == 2 and first.isalpha():
            return first

    if len(value) == 2 and value.isalpha():
        return value

    return None


def infer_text_language(text: str) -> str | None:
    sample = f" {text.lower()} "

    english_markers = [
        " the ",
        " and ",
        " of ",
        " to ",
        " in ",
        " is ",
        " for ",
        " on ",
        " with ",
        " that ",
    ]
    portuguese_char_markers = [
        "\u00e1",
        "\u00e0",
        "\u00e2",
        "\u00e3",
        "\u00e9",
        "\u00ea",
        "\u00ed",
        "\u00f3",
        "\u00f4",
        "\u00f5",
        "\u00fa",
        "\u00e7",
    ]
    portuguese_pattern_markers = ["\u00e7\u00e3o", "\u00e7\u00f5es", "nh", "lh"]

    en_score = sum(sample.count(marker) for marker in english_markers)
    pt_score = sum(sample.count(marker) for marker in portuguese_char_markers)
    pt_score += sum(sample.count(marker) for marker in portuguese_pattern_markers)

    if pt_score >= 2 and pt_score > en_score:
        return "pt"
    if en_score >= 2 and en_score >= pt_score:
        return "en"
    return None


def resolve_output_language(row: pd.Series, original_text: str) -> tuple[str, str]:
    if "language" in row.index and pd.notna(row["language"]):
        row_language = normalize_language_code(str(row["language"]))
        if row_language:
            return row_language, "row.language"

    if "country" in row.index and pd.notna(row["country"]):
        countries = str(row["country"]).lower()
        if "br" in countries:
            return "pt", "row.country"

    inferred = infer_text_language(original_text)
    if inferred:
        return inferred, "heuristic"

    return "en", "default"


def resolve_output_language_name(language_code: str) -> str:
    return LANGUAGE_CODE_TO_NAME.get(language_code, language_code.upper())
