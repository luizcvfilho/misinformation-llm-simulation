from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .vad import VADScore

NRC_VAD_DOWNLOAD_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_VAD_ARCHIVE_MEMBER = "NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['/-][^\W_]+)*", re.UNICODE)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


@dataclass(frozen=True, slots=True)
class NRCVADAnalysis:
    native_score: VADScore
    score: VADScore
    token_count: int
    matched_token_count: int
    matched_term_count: int
    multiword_match_count: int

    @property
    def token_coverage(self) -> float:
        return self.matched_token_count / self.token_count if self.token_count else 0.0


class NRCVADLexicon:
    """English NRC v2 scorer; longest non-overlapping matches have equal occurrence weight."""

    model_name = "NRC-VAD-Lexicon-v2.1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if zipfile.is_zipfile(self.path):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                payload = archive.read(NRC_VAD_ARCHIVE_MEMBER)
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
        if reader.fieldnames != ["term", "valence", "arousal", "dominance"]:
            raise ValueError("Expected NRC v2 TSV columns: term, valence, arousal, dominance.")
        self._trie: dict = {}
        self.entry_count = 0
        self.multiword_entry_count = 0
        for row in reader:
            term = _normalize(row["term"]).strip()
            tokens = tuple(term.split())
            if not tokens or any(TOKEN_PATTERN.fullmatch(token) is None for token in tokens):
                raise ValueError(f"Unsupported lexicon term: {row['term']!r}")
            values = tuple(
                float(row[dimension]) for dimension in ("valence", "arousal", "dominance")
            )
            if any(not math.isfinite(value) or not -1 <= value <= 1 for value in values):
                raise ValueError(f"NRC v2 values must be finite and within [-1, 1]: {term}")
            node = self._trie
            for token in tokens:
                node = node.setdefault(token, {})
            if None in node:
                raise ValueError(f"Duplicate normalized lexicon term: {term}")
            node[None] = values
            self.entry_count += 1
            self.multiword_entry_count += len(tokens) > 1
        if not self.entry_count:
            raise ValueError("The NRC VAD lexicon is empty.")

    def analyze(self, text: str) -> NRCVADAnalysis:
        normalized = _normalize(text) if isinstance(text, str) else ""
        tokens = list(TOKEN_PATTERN.finditer(normalized))
        values: list[tuple[float, float, float]] = []
        matched_tokens = multiword_matches = index = 0
        while index < len(tokens):
            node = self._trie
            cursor = index
            best = None
            while cursor < len(tokens):
                if cursor > index:
                    gap = normalized[tokens[cursor - 1].end() : tokens[cursor].start()]
                    if not gap.isspace():
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
            VADScore(*(math.fsum(row[dim] for row in values) / len(values) for dim in range(3)))
            if values
            else VADScore(None, None, None)
        )
        scaled = VADScore(
            *(
                None if value is None else 3 + 2 * value
                for value in (native.valence, native.arousal, native.dominance)
            )
        )
        return NRCVADAnalysis(
            native, scaled, len(tokens), matched_tokens, len(values), multiword_matches
        )

    def __call__(self, text: str) -> VADScore:
        """Return 1–5 values compatible with the project's injected VAD scorer API."""
        return self.analyze(text).score
