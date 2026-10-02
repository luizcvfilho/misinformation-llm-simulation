from __future__ import annotations

import csv
import hashlib
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .nrc_vad import TOKEN_PATTERN
from .vad import VADScore

MEMOLON_SOURCE_URL = "https://zenodo.org/records/3756607"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


@dataclass(frozen=True, slots=True)
class MEmoLonAnalysis:
    native_score: VADScore
    score: VADScore
    token_count: int
    matched_token_count: int
    matched_term_count: int
    multiword_match_count: int

    @property
    def token_coverage(self) -> float:
        return self.matched_token_count / self.token_count if self.token_count else 0.0


class MEmoLonLexicon:
    """MTL_grouped VAD: case-insensitive duplicate means and equal occurrence weights."""

    model_name = "MEmoLon-MTL_grouped"

    def __init__(self, path: str | Path, *, texts: Iterable[str] | None = None) -> None:
        self.path = Path(path)
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.vocabulary = (
            {
                token
                for text in texts
                if isinstance(text, str)
                for token in TOKEN_PATTERN.findall(_normalize(text))
            }
            if texts is not None
            else None
        )
        retained = defaultdict(list)
        self.source_row_count = self.unsupported_row_count = self.out_of_range_row_count = 0
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or reader.fieldnames[:4] != [
                "word",
                "valence",
                "arousal",
                "dominance",
            ]:
                raise ValueError("Expected MEmoLon TSV columns: word, valence, arousal, dominance.")
            for row in reader:
                self.source_row_count += 1
                tokens = tuple(_normalize(row["word"]).split())
                if not tokens or any(TOKEN_PATTERN.fullmatch(token) is None for token in tokens):
                    self.unsupported_row_count += 1
                    continue
                values = tuple(float(row[dim]) for dim in ("valence", "arousal", "dominance"))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Nonfinite MEmoLon VAD values: {row['word']!r}")
                self.out_of_range_row_count += any(not 1 <= value <= 9 for value in values)
                if self.vocabulary is None or all(token in self.vocabulary for token in tokens):
                    retained[tokens].append(values)
        self._trie: dict = {}
        self.entry_count = len(retained)
        self.multiword_entry_count = sum(len(tokens) > 1 for tokens in retained)
        self.retained_duplicate_rows = sum(len(values) - 1 for values in retained.values())
        for tokens, values in retained.items():
            node = self._trie
            for token in tokens:
                node = node.setdefault(token, {})
            node[None] = tuple(math.fsum(row[i] for row in values) / len(values) for i in range(3))
        if not self.source_row_count:
            raise ValueError("The MEmoLon lexicon is empty.")

    def analyze(self, text: str) -> MEmoLonAnalysis:
        normalized = _normalize(text) if isinstance(text, str) else ""
        tokens = list(TOKEN_PATTERN.finditer(normalized))
        if self.vocabulary is not None and any(t.group() not in self.vocabulary for t in tokens):
            raise ValueError("Text contains vocabulary outside the declared MEmoLon input scope.")
        values = []
        matched_tokens = multiword_matches = index = 0
        while index < len(tokens):
            node, cursor, best = self._trie, index, None
            while cursor < len(tokens):
                if (
                    cursor > index
                    and not normalized[tokens[cursor - 1].end() : tokens[cursor].start()].isspace()
                ):
                    break
                node = node.get(tokens[cursor].group())
                if node is None:
                    break
                cursor += 1
                if None in node:
                    best = (cursor, node[None])
            if best is None:
                index += 1
                continue
            end, scores = best
            values.append(scores)
            matched_tokens += end - index
            multiword_matches += end - index > 1
            index = end
        native = (
            VADScore(*(math.fsum(row[i] for row in values) / len(values) for i in range(3)))
            if values
            else VADScore(None, None, None)
        )
        scaled = VADScore(
            *(
                (value + 1) / 2 if value is not None else None
                for value in (native.valence, native.arousal, native.dominance)
            )
        )
        return MEmoLonAnalysis(
            native, scaled, len(tokens), matched_tokens, len(values), multiword_matches
        )

    def __call__(self, text: str) -> VADScore:
        return self.analyze(text).score
