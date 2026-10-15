from __future__ import annotations

import json
from copy import deepcopy

import pytest

from misinformation_simulation.config.prompts import (
    SEMANTIC_COMPARISON_PROMPT_TEMPLATE,
    STRUCTURED_JUDGE_PROMPT_TEMPLATE,
    STRUCTURED_JUDGE_VERSION,
)
from misinformation_simulation.topic_drift import semantic_comparison
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache, input_hash
from misinformation_simulation.topic_drift.semantic_comparison import (
    compare_stdi_components_semantically,
)


def _structure(topic: str) -> TopicStructure:
    return TopicStructure(
        main_topic=topic,
        subtopics=["transfer agreement"],
        central_entities=["Tottenham Hotspur"],
        central_relations=[TopicRelation("Tottenham", "sells", "player")],
    )


def test_semantic_comparison_uses_one_structured_openai_request(monkeypatch) -> None:
    captured: dict[str, object] = {}
    hook_calls: list[str] = []
    monkeypatch.setattr(
        semantic_comparison,
        "create_llm_client",
        lambda **_kwargs: ("chatgpt", object()),
    )

    def fake_generator(_client, **kwargs) -> str:
        captured.update(kwargs)
        kwargs["before_request_hook"]()
        return """{
            "theme_drift": 0,
            "subtopic_drift": 0.3,
            "entity_drift": 0,
            "relation_drift": 0.75,
            "contradiction_drift": 0,
            "rationales": {
                "theme_drift": "Same transfer story.",
                "subtopic_drift": "Slight focus shift.",
                "entity_drift": "Same club.",
                "relation_drift": "The actor role changes.",
                "contradiction_drift": "No internal contradiction."
            }
        }"""

    monkeypatch.setattr(semantic_comparison, "generate_openai_text_with_retry", fake_generator)
    result = compare_stdi_components_semantically(
        original_text="Tottenham may sell a player.",
        modified_text="A Tottenham player may be sold under an agreement.",
        title="Transfer report",
        original_structure=_structure("Tottenham player transfer"),
        modified_structure=_structure("Agreement to sell Tottenham player"),
        model="gpt-5-mini",
        provider="chatgpt",
        before_request_hook=lambda: hook_calls.append("called"),
    )

    assert result.component_drifts == {
        "theme_drift": 0.0,
        "subtopic_drift": 0.25,
        "entity_drift": 0.0,
        "relation_drift": 0.75,
        "contradiction_drift": 0.0,
    }
    assert result.rationales["theme_drift"] == "Same transfer story."
    assert hook_calls == ["called"]
    assert "active/passive wording" in str(captured["prompt"])


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("provider", ["chatgpt", "gemini"])
def test_judge_prompt_preserves_content_and_keeps_provenance_local(
    monkeypatch, tmp_path, structured, provider
):
    original = _structure("Tottenham player transfer")
    modified = _structure("Agreement to sell Tottenham player")
    original.provenance = {"raw_response": "original audit response", "input_sha256": "abc"}
    modified.provenance = {"raw_response": "modified audit response", "input_sha256": "def"}
    snapshots = [deepcopy(topic_structure_to_dict(item)) for item in (original, modified)]
    texts = {
        "original_text": "Tottenham may sell a player.",
        "modified_text": "A Tottenham player may be sold under an agreement.",
        "title": "Transfer report",
    }
    raw_response = json.dumps(
        dict.fromkeys(
            (
                "theme_drift",
                "subtopic_drift",
                "entity_drift",
                "relation_drift",
                "contradiction_drift",
            ),
            0.5,
        )
    )
    prompts = []
    monkeypatch.setattr(
        semantic_comparison, "create_llm_client", lambda **kwargs: (provider, object())
    )

    def generate(_client, **kwargs):
        prompts.append(kwargs["prompt"])
        return raw_response

    monkeypatch.setattr(
        semantic_comparison,
        "generate_gemini_text_with_retry"
        if provider == "gemini"
        else "generate_openai_text_with_retry",
        generate,
    )
    result = compare_stdi_components_semantically(
        **texts,
        original_structure=original,
        modified_structure=modified,
        model="test-model",
        provider=provider,
        structured=structured,
    )

    template = (
        STRUCTURED_JUDGE_PROMPT_TEMPLATE if structured else SEMANTIC_COMPARISON_PROMPT_TEMPLATE
    )
    expected = template.format(
        **texts,
        **{
            key: json.dumps(
                {k: v for k, v in snapshot.items() if k != "provenance"}, ensure_ascii=False
            )
            for key, snapshot in zip(
                ("original_structure", "modified_structure"), snapshots, strict=True
            )
        },
    )
    assert prompts == [expected]
    assert "audit response" not in expected
    assert [topic_structure_to_dict(item) for item in (original, modified)] == snapshots
    if structured:
        assert result.provenance["raw_response"] == raw_response
        assert result.provenance["prompt_sha256"] == input_hash(expected)
        assert result.provenance["prompt_version"] == STRUCTURED_JUDGE_VERSION

    cache = EvaluationCache(tmp_path)
    for index, item in enumerate((original, modified)):
        cache.put("extraction", {"index": index}, topic_structure_to_dict(item))
        assert EvaluationCache(tmp_path).get("extraction", {"index": index}) == snapshots[index]
