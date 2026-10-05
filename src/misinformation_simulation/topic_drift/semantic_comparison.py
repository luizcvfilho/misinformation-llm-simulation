from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from misinformation_simulation.config.prompts import (
    SEMANTIC_COMPARISON_PROMPT_TEMPLATE,
    SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
    STRUCTURED_JUDGE_PROMPT_TEMPLATE,
    STRUCTURED_JUDGE_VERSION,
)
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.manual_evaluation_definitions import (
    SEMANTIC_COMPONENT_COLUMNS,
    SEMANTIC_DRIFT_LEVELS,
)
from misinformation_simulation.topic_drift.models import TopicStructure, topic_structure_to_dict


@dataclass(frozen=True, slots=True)
class SemanticSTDIComparison:
    component_drifts: dict[str, float]
    rationales: dict[str, str]
    evidence: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


class JudgeValidationError(ValueError):
    def __init__(self, message: str, provenance: dict[str, Any]) -> None:
        super().__init__(message)
        self.provenance = provenance


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
    structured: bool = False,
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
    template = (
        STRUCTURED_JUDGE_PROMPT_TEMPLATE if structured else SEMANTIC_COMPARISON_PROMPT_TEMPLATE
    )
    prompt = template.format(
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
    if not structured:
        return _parse_semantic_comparison(raw_response)
    from misinformation_simulation.topic_drift.provenance import request_provenance

    provenance = request_provenance(
        model=model,
        provider=provider_normalized,
        base_url=base_url,
        version=STRUCTURED_JUDGE_VERSION,
        prompt=prompt,
        raw_response=raw_response,
        inputs={"original_text": original_text, "modified_text": modified_text, "title": title},
    )
    try:
        result = _parse_semantic_comparison(raw_response)
        payload = _extract_json_object(raw_response)
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("The structured judge must provide evidence for every component.")
        for component in SEMANTIC_COMPONENT_COLUMNS:
            if payload[component] not in SEMANTIC_DRIFT_LEVELS:
                raise ValueError(f"'{component}' must use an anchored rubric level.")
            if not result.rationales[component] or not isinstance(evidence.get(component), list):
                raise ValueError(f"Missing rationale/evidence for '{component}'.")
            if result.component_drifts[component] > 0 and not evidence[component]:
                raise ValueError(f"Changed component '{component}' requires supporting passages.")
            for passage in evidence[component]:
                if not isinstance(passage, dict):
                    raise ValueError("Evidence must contain original/modified passage objects.")
                for side, text in (("original", original_text), ("modified", modified_text)):
                    span = passage.get(side)
                    if not isinstance(span, str) or (span and span not in text):
                        raise ValueError(f"Judge evidence is not a verbatim {side} passage.")
                if not passage["original"] and not passage["modified"]:
                    raise ValueError("An evidence pair cannot contain two empty passages.")
        return SemanticSTDIComparison(
            result.component_drifts,
            result.rationales,
            evidence,
            provenance,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise JudgeValidationError(str(exc), provenance) from exc
