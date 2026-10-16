from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "vad_lexicon_pilot", ROOT / "scripts/compare_portuguese_vad_lexicons.py"
)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def write_resource(tmp_path, model, language, rows):
    content = "\n".join("\t".join(map(str, row)) for row in rows) + "\n"
    if model == "nrc_v1":
        member = pilot.NRC_MEMBER_PT if language == "pt" else pilot.NRC_MEMBER_EN
        if language == "pt":
            content = "English Word\tValence\tArousal\tDominance\tPortuguese Word\n" + content
        with zipfile.ZipFile(tmp_path / "NRC-VAD-Lexicon-v1.zip", "w") as archive:
            archive.writestr(member, content)
    elif model == "nrc_v2_1":
        content = "term\tvalence\tarousal\tdominance\n" + content
        with zipfile.ZipFile(tmp_path / "nrc-v2.zip", "w") as archive:
            archive.writestr(pilot.NRC_V2_MEMBER, content)
    else:
        content = "word\tvalence\tarousal\tdominance\n" + content
        (tmp_path / f"memolon_{language}.tsv").write_text(content, encoding="utf-8")


@pytest.mark.parametrize("language", ["pt", "en"])
@pytest.mark.parametrize(
    ("model", "first", "second", "unused", "expected"),
    [
        ("nrc_v1", (0.2, 0.4, 0.6), (0.8, 0.6, 0.4), (1.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ("nrc_v2_1", (-0.4, 0.4, 0.6), (0.4, 0.6, 0.4), (2, 0.5, 0.5), (0, 0.5, 0.5)),
        ("memolon", (2, 4, 6), (8, 6, 4), (10, 5, 5), (5, 5, 5)),
    ],
)
def test_load_lexicon_preserves_duplicates_missing_terms_and_native_scale(
    tmp_path, monkeypatch, language, model, first, second, unused, expected
):
    monkeypatch.setattr(pilot, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pilot, "NRC_V2_PATH", tmp_path / "nrc-v2.zip")
    rows = [("joy", *first), ("JOY", *second), ("unused", *unused), ("bad!", *first)]
    if model == "nrc_v1" and language == "pt":
        rows = [(*row, row[0]) for row in rows]
    write_resource(tmp_path, model, language, rows)

    entries, metadata = pilot.load_lexicon(model, language, {"joy", "missing"})

    assert set(entries) == {"joy"}
    assert entries["joy"]["native"] == pytest.approx(expected)
    assert entries["joy"]["source_rows"] == 2
    assert metadata["source_rows"] == 4
    assert metadata["unsupported_rows"] == 1
    assert metadata["retained_duplicate_rows"] == 1
    assert metadata["out_of_range_supported_rows"] == 1
    assert metadata["retained_out_of_range_source_rows"] == 0


def test_load_lexicon_rejects_invalid_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "CACHE_DIR", tmp_path)
    (tmp_path / "memolon_pt.tsv").write_text("term\tV\tA\tD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected MEmoLon header"):
        pilot.load_lexicon("memolon", "pt", {"joy"})


def test_load_lexicon_rejects_nonfinite_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "CACHE_DIR", tmp_path)
    write_resource(tmp_path, "memolon", "en", [("joy", "nan", 5, 5)])

    with pytest.raises(ValueError, match="Nonfinite values"):
        pilot.load_lexicon("memolon", "en", {"joy"})
