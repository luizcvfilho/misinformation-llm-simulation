from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from misinformation_simulation.text_metrics.evidence_alignment import (
    EVIDENCE_VALIDATION_VERSION,
    EvidenceAligner,
)
from misinformation_simulation.text_metrics.llm_vad import (
    LLM_VAD_RUBRIC,
    LLM_VAD_SYSTEM_INSTRUCTION,
    LLM_VAD_VERSION,
    parse_llm_vad_assessment,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache


def response(quotes):
    return json.dumps(
        dict(
            valence=3.2,
            arousal=1.5,
            dominance=None,
            rationales=dict(valence="Balanced.", arousal="Calm.", dominance="Ambiguous."),
            evidence=dict(valence=quotes, arousal=[], dominance=[]),
            ambiguities=[],
        ),
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "text,quote,expected,rule",
    [
        ("With gas prices spiking today.", "Gas prices spiking", "gas prices spiking", "case"),
        ("Os preços subiram.", "OS PREÇOS SUBIRAM", "Os preços subiram", "case"),
        ("A calm\t\nreport.", "calm report", "calm\t\nreport", "whitespace"),
        ("A calm report.", "  calm  report  ", "calm report", "whitespace"),
        ("São\u00a0Paulo", "SÃO PAULO", "São\u00a0Paulo", "case_and_whitespace"),
        (
            "Before 📰 a quiet\nreport; after.",
            "A QUIET REPORT",
            "a quiet\nreport",
            "case_and_whitespace",
        ),
        ("📰 İ.", "i\u0307", "İ", "case"),
    ],
)
def test_unique_formatting_alignment_keeps_literal_source_span(text, quote, expected, rule):
    result = parse_llm_vad_assessment(response([quote]), text)
    assert asdict(result.score) == dict(valence=3.2, arousal=1.5, dominance=None)
    assert result.evidence == dict(valence=[expected], arousal=[], dominance=[])
    assert result.evidence_validation_version == EVIDENCE_VALIDATION_VERSION
    assert len(result.evidence_alignments) == 1
    aligned = result.evidence_alignments[0]
    assert aligned["original_quote"] == quote
    assert aligned["matched_quote"] == expected
    assert aligned["rule"] == rule
    assert aligned["dimension"] == "valence" and aligned["quote_index"] == 0
    assert text[aligned["source_start"] : aligned["source_end"]] == expected


@pytest.mark.parametrize(
    "text,quote",
    [
        ("Calm report. calm report.", "CALM REPORT"),
        ("One\tword and one word.", "one  word"),
        ("A A A", "a a"),
    ],
)
def test_ambiguous_alignment_is_rejected(text, quote):
    with pytest.raises(ValueError, match="ambiguous"):
        parse_llm_vad_assessment(response([quote]), text)


@pytest.mark.parametrize(
    "text,quote",
    [
        ("A calmer report.", "CALM"),
        ("We paid 50 dollars.", "PAID 5"),
        ("A calm report.", "calmer report"),
        ("Prices did not rise.", "Prices did rise"),
        ("Prices rose 10%.", "Prices rose 20%"),
        ("A calm report.", "calm ... report"),
        ("A calm report.", '"calm"'),
        ("We can't agree.", "We can’t agree"),
        ("Uma notícia calma.", "Uma noticia calma"),
        ("Uma notícia calma.", "Uma noti\u0301cia calma"),
        ("U.S. officials", "US officials"),
        ("Straße", "STRASSE"),
        ("İ.", "I"),
    ],
)
def test_alignment_does_not_change_words_numbers_punctuation_or_accents(text, quote):
    with pytest.raises(ValueError):
        parse_llm_vad_assessment(response([quote]), text)


def test_exact_repeated_quote_preserves_existing_acceptance_and_empty_lists():
    text = "Calm report. Calm report."
    result = parse_llm_vad_assessment(response(["Calm report"]), text)
    assert result.evidence["valence"] == ["Calm report"]
    assert result.evidence_alignments == []
    assert result.evidence["arousal"] == []
    assert parse_llm_vad_assessment(response([]), text).evidence_alignments == []


def test_alignment_does_not_mutate_source_and_records_only_adjusted_quote_indices():
    text = "First calm report. Second clear result."
    result = parse_llm_vad_assessment(response(["First", "CLEAR RESULT"]), text)
    assert text == "First calm report. Second clear result."
    assert result.evidence["valence"] == ["First", "clear result"]
    assert [entry["quote_index"] for entry in result.evidence_alignments] == [1]
    assert EvidenceAligner(text).align("First").rule == "exact"


def test_new_validation_version_excludes_old_cache_and_persists_alignment(tmp_path, monkeypatch):
    from misinformation_simulation.text_metrics import vad_evaluation as vad

    calls = []

    class Scorer:
        last_attempts = []

        def __init__(self, **kwargs):
            pass

        def assess(self, text):
            calls.append(text)
            return parse_llm_vad_assessment(response(["CALM"]), text)

    monkeypatch.setattr(vad, "LLMVADScorer", Scorer)
    cache = EvaluationCache(tmp_path)
    text = "A calm report."
    old_inputs = dict(
        text=text,
        model="test",
        provider="chatgpt",
        base_url=None,
        version="llm_vad_expressed_tone_v1",
        rubric=LLM_VAD_RUBRIC,
        system_instruction=LLM_VAD_SYSTEM_INSTRUCTION,
    )
    cache.put(
        "vad_llm", old_inputs, dict(status="valid", score=dict(valence=1, arousal=1, dominance=1))
    )
    assert LLM_VAD_VERSION != old_inputs["version"]
    result = vad.VADTextEvaluator(
        method="llm", llm_model="test", llm_provider="chatgpt", cache=cache
    ).evaluate(text)["llm"]
    assert not result["cache_hit"]
    assert result["score"]["valence"] == 3.2
    assert result["evidence_alignments"][0]["original_quote"] == "CALM"
    assert result["evidence_alignments"][0]["matched_quote"] == "calm"
    restored = vad.VADTextEvaluator(
        method="llm", llm_model="test", llm_provider="chatgpt", cache=EvaluationCache(tmp_path)
    ).evaluate(text)["llm"]
    assert restored["cache_hit"]
    assert restored["evidence_alignments"] == result["evidence_alignments"]
    assert restored["evidence_validation_version"] == EVIDENCE_VALIDATION_VERSION
    assert calls == [text]
