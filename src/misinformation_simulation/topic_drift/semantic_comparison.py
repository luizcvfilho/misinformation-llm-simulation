from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.manual_evaluation_definitions import (
    SEMANTIC_COMPARISON_PROMPT_TEMPLATE,
    SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
    SEMANTIC_COMPONENT_COLUMNS,
    SEMANTIC_DRIFT_LEVELS,
)
from misinformation_simulation.topic_drift.models import TopicStructure, topic_structure_to_dict


@dataclass(frozen=True, slots=True)
class SemanticSTDIComparison:
    component_drifts: dict[str, float]
    rationales: dict[str, str]


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("The semantic comparison response must be a JSON object.")
    return payload


def _coerce_score(value: Any, component: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"'{component}' must be a numeric semantic drift score.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{component}' must be a numeric semantic drift score.") from exc
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"'{component}' must be between 0 and 1.")
    return min(SEMANTIC_DRIFT_LEVELS, key=lambda level: abs(level - numeric_value))


def _parse_semantic_comparison(raw_text: str) -> SemanticSTDIComparison:
    payload = _extract_json_object(raw_text)
    missing_components = [
        component for component in SEMANTIC_COMPONENT_COLUMNS if component not in payload
    ]
    if missing_components:
        raise ValueError(
            f"Semantic comparison response is missing: {', '.join(missing_components)}"
        )

    component_drifts = {
        component: _coerce_score(payload[component], component)
        for component in SEMANTIC_COMPONENT_COLUMNS
    }
    rationales_payload = payload.get("rationales", {})
    if not isinstance(rationales_payload, dict):
        rationales_payload = {}
    rationales = {
        component: str(rationales_payload.get(component, "")).strip()
        for component in SEMANTIC_COMPONENT_COLUMNS
    }
    return SemanticSTDIComparison(component_drifts=component_drifts, rationales=rationales)


def compare_stdi_components_semantically(
    *,
    original_text: str,
    modified_text: str,
    title: str | None,
    original_structure: TopicStructure,
    modified_structure: TopicStructure,
    model: str,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    retry_attempts: int = 5,
    before_request_hook: Callable[[], None] | None = None,
) -> SemanticSTDIComparison:
    """Judge semantic drift in a pair once, keeping component judgments consistent."""
    if not original_text or not original_text.strip():
        raise ValueError("Provide non-empty original text for semantic comparison.")
    if not modified_text or not modified_text.strip():
        raise ValueError("Provide non-empty modified text for semantic comparison.")

    provider_normalized, client = create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    prompt = SEMANTIC_COMPARISON_PROMPT_TEMPLATE.format(
        title=title or "Untitled",
        original_text=original_text.strip(),
        modified_text=modified_text.strip(),
        original_structure=json.dumps(
            topic_structure_to_dict(original_structure), ensure_ascii=False
        ),
        modified_structure=json.dumps(
            topic_structure_to_dict(modified_structure), ensure_ascii=False
        ),
    )
    if provider_normalized == "gemini":
        raw_response = generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=before_request_hook,
        )
    else:
        raw_response = generate_openai_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=before_request_hook,
        )
    return _parse_semantic_comparison(raw_response)
