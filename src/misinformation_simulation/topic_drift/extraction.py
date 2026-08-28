from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure

DEFAULT_TOPIC_DRIFT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_TOPIC_DRIFT_PROVIDER = DEFAULT_LLM_PROVIDER
DEFAULT_REWRITTEN_COLUMN = "rewritten_news"

TOPIC_DRIFT_SYSTEM_INSTRUCTION = """
You extract the semantic structure of a news report for topic-drift analysis.
Return only valid JSON.
Do not add markdown fences, explanations, or extra keys.
Use concise, factual phrases grounded in the provided text.
If a field is unavailable, use null or an empty array.
""".strip()

TOPIC_DRIFT_PROMPT_TEMPLATE = """
Analyze the following news item and return a JSON object with exactly these keys:
- main_topic: string or null
- topic_domain: string or null
- subtopics: array of strings
- central_entities: array of strings
- central_relations: array of objects with keys subject, action, object
- narrative_frame: string or null
- has_internal_contradiction: boolean
- internal_contradiction_score: number between 0 and 1

Extraction rules:
- main_topic must capture the primary subject of the article.
- topic_domain must be exactly one of: business_and_economy, crime_and_justice,
  culture_and_entertainment, education, environment, government_and_public_policy, health,
  international_affairs, science_and_technology, sports, or other. Select the article's primary
  domain; use null only when no domain can be determined.
- subtopics must list secondary themes or angles.
- central_entities must include the most important people, organizations, places, or groups.
- central_relations must describe core factual relations in (subject, action, object) form.
- narrative_frame is optional and should summarize the dominant framing if present.
- has_internal_contradiction must be true only when the text contradicts itself internally.
- internal_contradiction_score must grade the severity/centrality of internal contradiction:
  0 means none, 0.25 means slight or peripheral tension, 0.5 means partial contradiction,
  0.75 means strong contradiction in an important claim, and 1 means a central contradiction.
- Keep outputs short and normalized.
- Do not invent facts beyond the text.

Title: {title}

Text:
{text}
""".strip()


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        normalized = _normalize_scalar(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value.strip())

    return ordered


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_relation(subject: Any, action: Any, obj: Any) -> tuple[str, str, str]:
    return (
        _normalize_scalar(subject),
        _normalize_scalar(action),
        _normalize_scalar(obj),
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The model response did not contain a JSON object.")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("The model response JSON must be an object.")
    return payload


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return _deduplicate_preserve_order(items)


def _coerce_relations(value: Any) -> list[TopicRelation]:
    if not isinstance(value, list):
        return []

    relations: list[TopicRelation] = []
    seen: set[tuple[str, str, str]] = set()

    for item in value:
        if not isinstance(item, dict):
            continue

        subject = str(item.get("subject", "") or "").strip()
        action = str(item.get("action", "") or "").strip()
        obj = str(item.get("object", "") or "").strip()
        normalized = _normalize_relation(subject, action, obj)

        if not all(normalized) or normalized in seen:
            continue

        seen.add(normalized)
        relations.append(TopicRelation(subject=subject, action=action, object=obj))

    return relations


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return False


def _coerce_unit_score(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(numeric_value, 0.0), 1.0)


def _build_topic_structure(payload: dict[str, Any]) -> TopicStructure:
    main_topic = payload.get("main_topic")
    topic_domain = payload.get("topic_domain")
    narrative_frame = payload.get("narrative_frame")
    has_internal_contradiction = _coerce_bool(payload.get("has_internal_contradiction"))
    internal_contradiction_score = _coerce_unit_score(
        payload.get("internal_contradiction_score"),
        default=1.0 if has_internal_contradiction else 0.0,
    )

    return TopicStructure(
        main_topic=str(main_topic).strip() if main_topic else None,
        subtopics=_coerce_string_list(payload.get("subtopics")),
        central_entities=_coerce_string_list(payload.get("central_entities")),
        central_relations=_coerce_relations(payload.get("central_relations")),
        topic_domain=str(topic_domain).strip() if topic_domain else None,
        narrative_frame=str(narrative_frame).strip() if narrative_frame else None,
        has_internal_contradiction=has_internal_contradiction or internal_contradiction_score > 0.0,
        internal_contradiction_score=internal_contradiction_score,
    )


def extract_topic_structure(
    *,
    text: str,
    title: str | None = None,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    api_key: str | None = None,
    base_url: str | None = None,
    max_requests_per_minute: int | None = None,
    retry_attempts: int = 5,
    before_request_hook: Callable[[], None] | None = None,
) -> TopicStructure:
    if not text or not str(text).strip():
        raise ValueError("Provide a non-empty text to extract the topic structure.")

    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")

    provider_normalized, client = create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    prompt = TOPIC_DRIFT_PROMPT_TEMPLATE.format(title=title or "Untitled", text=text.strip())
    limiter = MinuteRateLimiter(max_requests_per_minute)
    request_hook = before_request_hook or limiter.acquire

    if provider_normalized == "gemini":
        raw_response = generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=request_hook,
        )
    else:
        raw_response = generate_openai_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=request_hook,
        )

    payload = _extract_json_object(raw_response)
    return _build_topic_structure(payload)
