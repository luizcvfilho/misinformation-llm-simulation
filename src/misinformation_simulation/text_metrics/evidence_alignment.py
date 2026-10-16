"""Align quote formatting to a unique source span without changing its wording."""

from __future__ import annotations

from dataclasses import dataclass
from unicodedata import category

EVIDENCE_VALIDATION_VERSION = "unique_case_whitespace_v1"


@dataclass(frozen=True, slots=True)
class EvidenceAlignment:
    original_quote: str
    matched_quote: str
    rule: str
    source_start: int
    source_end: int


def _normalize(text: str) -> tuple[str, list[tuple[int, int]]]:
    characters: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character.isspace():
            if characters and characters[-1] == " ":
                spans[-1] = (spans[-1][0], index + 1)
            else:
                characters.append(" ")
                spans.append((index, index + 1))
        else:
            # lower(), unlike casefold(), does not equate German sharp s with "ss".
            for lowered in character.lower():
                characters.append(lowered)
                spans.append((index, index + 1))
    return "".join(characters), spans


def _word_character(character: str) -> bool:
    return character == "_" or category(character)[0] in "LMN"


class EvidenceAligner:
    def __init__(self, text: str):
        self.text = text
        self._normalized: tuple[str, list[tuple[int, int]]] | None = None

    def align(self, quote: str) -> EvidenceAlignment:
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("The quote must be a non-empty string.")
        exact_start = self.text.find(quote)
        if exact_start >= 0:
            return EvidenceAlignment(quote, quote, "exact", exact_start, exact_start + len(quote))
        if self._normalized is None:
            self._normalized = _normalize(self.text)
        normalized, spans = self._normalized
        needle = _normalize(quote)[0].strip()
        matches = []
        position = normalized.find(needle)
        while position >= 0:
            end = position + len(needle)
            start_offset, end_offset = spans[position][0], spans[end - 1][1]
            complete_characters = (position == 0 or spans[position - 1] != spans[position]) and (
                end == len(spans) or spans[end - 1] != spans[end]
            )
            complete_words = not (
                _word_character(needle[0])
                and start_offset > 0
                and _word_character(self.text[start_offset - 1])
            ) and not (
                _word_character(needle[-1])
                and end_offset < len(self.text)
                and _word_character(self.text[end_offset])
            )
            if complete_characters and complete_words:
                matches.append((start_offset, end_offset))
                if len(matches) > 1:
                    raise ValueError(
                        "Case/whitespace alignment is ambiguous: multiple source spans."
                    )
            position = normalized.find(needle, position + 1)
        if not matches:
            raise ValueError("No unique source span differs only in letter case or whitespace.")
        start, end = matches[0]
        matched = self.text[start:end]
        case_changed = " ".join(quote.split()) != " ".join(matched.split())
        whitespace_changed = quote.lower() != matched.lower()
        rule = (
            "case_and_whitespace"
            if case_changed and whitespace_changed
            else ("case" if case_changed else "whitespace")
        )
        return EvidenceAlignment(quote, matched, rule, start, end)
