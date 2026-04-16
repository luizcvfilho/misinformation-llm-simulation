from __future__ import annotations

import json
import re
from typing import Any

from misinformation_simulation.enums import Models, Provider
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure

DEFAULT_TOPIC_DRIFT_MODEL = Models.GEMINI31FlashLite
DEFAULT_TOPIC_DRIFT_PROVIDER = Provider.GEMINI
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
- subtopics: array of strings
- central_entities: array of strings
- central_relations: array of objects with keys subject, action, object
- narrative_frame: string or null

Extraction rules:
- main_topic must capture the primary subject of the article.
- subtopics must list secondary themes or angles.
- central_entities must include the most important people, organizations, places, or groups.
- central_relations must describe core factual relations in (subject, action, object) form.
- narrative_frame is optional and should summarize the dominant framing if present.
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


def _build_topic_structure(payload: dict[str, Any]) -> TopicStructure:
    main_topic = payload.get("main_topic")
    narrative_frame = payload.get("narrative_frame")

    return TopicStructure(
        main_topic=str(main_topic).strip() if main_topic else None,
        subtopics=_coerce_string_list(payload.get("subtopics")),
        central_entities=_coerce_string_list(payload.get("central_entities")),
        central_relations=_coerce_relations(payload.get("central_relations")),
        narrative_frame=str(narrative_frame).strip() if narrative_frame else None,
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

    if provider_normalized == "gemini":
        raw_response = generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=limiter.acquire,
        )
    else:
        raw_response = generate_openai_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=retry_attempts,
            before_request_hook=limiter.acquire,
        )

    payload = _extract_json_object(raw_response)
    return _build_topic_structure(payload)
