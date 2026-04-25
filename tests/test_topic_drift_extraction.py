from __future__ import annotations

import pytest

from misinformation_simulation.topic_drift import extraction
from misinformation_simulation.topic_drift.extraction import (
    _build_topic_structure,
    _coerce_bool,
    _coerce_relations,
    _coerce_string_list,
    _deduplicate_preserve_order,
    _extract_json_object,
    extract_topic_structure,
)


def test_extract_json_object_accepts_fenced_and_embedded_json() -> None:
    fenced = '```json\n{"main_topic": "economy"}\n```'
    embedded = 'prefix {"main_topic": "health"} suffix'

    assert _extract_json_object(fenced) == {"main_topic": "economy"}
    assert _extract_json_object(embedded) == {"main_topic": "health"}


def test_extract_json_object_rejects_missing_json_object() -> None:
    with pytest.raises(ValueError, match="did not contain"):
        _extract_json_object("no json here")

    with pytest.raises(ValueError, match="did not contain"):
        _extract_json_object("[1, 2]")


def test_coercion_helpers_normalize_lists_relations_and_booleans() -> None:
    assert _deduplicate_preserve_order([" Economy ", "economy", "Health"]) == [
        "Economy",
        "Health",
    ]
    assert _coerce_string_list([" A ", None, "", "a", "B"]) == ["A", "B"]
    assert _coerce_bool(True)
    assert _coerce_bool("yes")
    assert _coerce_bool(1)
    assert not _coerce_bool("maybe")

    relations = _coerce_relations(
        [
            {"subject": "Agency", "action": "reported", "object": "case"},
            {"subject": " agency ", "action": "reported", "object": "case"},
            {"subject": "", "action": "ignored", "object": "case"},
            "invalid",
        ]
    )

    assert len(relations) == 1
    assert relations[0].subject == "Agency"


def test_build_topic_structure_trims_scalars_and_defaults_missing_values() -> None:
    structure = _build_topic_structure(
        {
            "main_topic": " Economy ",
            "subtopics": ["inflation", "Inflation"],
            "central_entities": ["Central Bank"],
            "central_relations": [
                {"subject": "Bank", "action": "raises", "object": "rates"},
            ],
            "narrative_frame": " Alert ",
            "has_internal_contradiction": "true",
        }
    )

    assert structure.main_topic == "Economy"
    assert structure.subtopics == ["inflation"]
    assert structure.narrative_frame == "Alert"
    assert structure.has_internal_contradiction


def test_extract_topic_structure_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty text"):
        extract_topic_structure(text=" ")

    with pytest.raises(ValueError, match="greater than zero"):
        extract_topic_structure(text="text", max_requests_per_minute=0)


def test_extract_topic_structure_uses_gemini_generator(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(extraction, "create_llm_client", lambda **_kwargs: ("gemini", object()))

    def fake_gemini_generator(_client, **kwargs) -> str:
        calls.append(kwargs)
        return '{"main_topic": "economy", "has_internal_contradiction": false}'

    monkeypatch.setattr(extraction, "generate_gemini_text_with_retry", fake_gemini_generator)

    structure = extract_topic_structure(
        text="Article text",
        title="Title",
        model="gemini-test",
        retry_attempts=2,
    )

    assert structure.main_topic == "economy"
    assert calls[0]["model"] == "gemini-test"
    assert calls[0]["max_attempts"] == 2
    assert "Title: Title" in calls[0]["prompt"]


def test_extract_topic_structure_uses_openai_generator(monkeypatch) -> None:
    monkeypatch.setattr(extraction, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        extraction,
        "generate_openai_text_with_retry",
        lambda _client, **_kwargs: '{"main_topic": "science"}',
    )

    structure = extract_topic_structure(text="Article text", provider="chatgpt")

    assert structure.main_topic == "science"
