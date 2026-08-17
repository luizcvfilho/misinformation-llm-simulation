from __future__ import annotations

from misinformation_simulation.topic_drift import semantic_comparison
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure
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
            "rationales": {
                "theme_drift": "Same transfer story.",
                "subtopic_drift": "Slight focus shift.",
                "entity_drift": "Same club.",
                "relation_drift": "The actor role changes."
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
    }
    assert result.rationales["theme_drift"] == "Same transfer story."
    assert hook_calls == ["called"]
    assert "active/passive wording" in str(captured["prompt"])
