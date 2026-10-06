from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

from misinformation_simulation.config.prompts import (
    STRUCTURED_EXTRACTION_VERSION,
    STRUCTURED_TOPIC_PROMPT_TEMPLATE,
)
from misinformation_simulation.config.prompts import (
    TOPIC_DRIFT_PROMPT_TEMPLATE as TOPIC_DRIFT_PROMPT_TEMPLATE,
)
from misinformation_simulation.config.prompts import (
    TOPIC_DRIFT_SYSTEM_INSTRUCTION as TOPIC_DRIFT_SYSTEM_INSTRUCTION,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    MAX_EVALUATION_ATTEMPTS,
    generate_and_parse_with_retry,
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.models import NumericValue, TopicRelation, TopicStructure
from misinformation_simulation.topic_drift.qualifiers import resolve_polarity

DEFAULT_TOPIC_DRIFT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_TOPIC_DRIFT_PROVIDER = DEFAULT_LLM_PROVIDER
DEFAULT_REWRITTEN_COLUMN = "rewritten_news"


class ExtractionValidationError(ValueError):
    def __init__(self, message: str, provenance: dict[str, Any]) -> None:
        super().__init__(message)
        self.provenance = provenance


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


def _coerce_numeric_values(value: Any) -> list[NumericValue] | None:
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        if not isinstance(item, dict):
            result.append(NumericValue("", None, status="unknown"))
            continue
        kind = str(item.get("kind") or "number").strip().lower()
        number = item.get("value")
        if kind in {"date", "identifier"}:
            finite = not isinstance(number, float) or math.isfinite(number)
            number = (
                str(number).strip()
                if number is not None and not isinstance(number, bool) and finite
                else None
            )
        else:
            try:
                number = float(number) if not isinstance(number, bool) else float("nan")
                number = number if math.isfinite(number) else None
            except (TypeError, ValueError, OverflowError):
                number = None
        result.append(
            NumericValue(
                role=str(item.get("role") or "").strip(),
                value=number,
                unit=str(item["unit"]).strip() if item.get("unit") else None,
                kind=kind,
                status=str(item.get("status") or "unknown").strip().lower(),
                expression=str(item["expression"]).strip() if item.get("expression") else None,
            )
        )
    return result


def _coerce_relations(value: Any, *, structured: bool = False) -> list[TopicRelation]:
    if not isinstance(value, list):
        return []

    relations: list[TopicRelation] = []
    seen: set[Any] = set()

    for item in value:
        if not isinstance(item, dict):
            continue

        subject = str(item.get("subject", "") or "").strip()
        action = str(item.get("action", "") or "").strip()
        obj = str(item.get("object", "") or "").strip()
        normalized = (
            json.dumps(
                {
                    key: value
                    for key, value in item.items()
                    if key in TopicRelation.__dataclass_fields__
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if structured
            else _normalize_relation(subject, action, obj)
        )

        if (not structured and not all(normalized)) or normalized in seen:
            continue

        seen.add(normalized)
        qualifiers = {}
        if structured:
            for key in (
                "predicate",
                "negation_scope",
                "signed_action",
                "base_action",
                "duration_unit",
                "duration_expression",
                "assertion_type",
            ):
                qualifiers[key] = str(item[key]).strip() if item.get(key) else None
            qualifiers["polarity"] = (
                item.get("polarity") if item.get("polarity") in {"affirmed", "negated"} else None
            )
            qualifiers["duration_status"] = (
                item.get("duration_status")
                if item.get("duration_status") in {"exact", "absent", "ambiguous"}
                else "unknown"
            )
            value = item.get("duration_value")
            try:
                numeric = float(value) if not isinstance(value, bool) else float("nan")
                qualifiers["duration_value"] = (
                    numeric if math.isfinite(numeric) and numeric >= 0 else None
                )
            except (TypeError, ValueError):
                qualifiers["duration_value"] = None
            qualifiers["numeric_values"] = _coerce_numeric_values(item.get("numeric_values"))
        relation = TopicRelation(subject=subject, action=action, object=obj, **qualifiers)
        if structured:
            relation.polarity = resolve_polarity(relation)
        relations.append(relation)

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
    if not math.isfinite(numeric_value):
        return default
    return min(max(numeric_value, 0.0), 1.0)


def _build_topic_structure(payload: dict[str, Any]) -> TopicStructure:
    main_topic = payload.get("main_topic")
    narrative_frame = payload.get("narrative_frame")
    has_internal_contradiction = _coerce_bool(payload.get("has_internal_contradiction"))
    internal_contradiction_score = _coerce_unit_score(
        payload.get("internal_contradiction_score"),
        default=1.0 if has_internal_contradiction else 0.0,
    )

    structured = payload.get("schema_version") == 2
    issues = list(payload.get("extraction_issues", []))
    if structured:
        if not isinstance(payload.get("has_internal_contradiction"), bool):
            issues.append("Invalid contradiction flag")
        required = (
            "main_topic",
            "subtopics",
            "central_entities",
            "central_relations",
            "has_internal_contradiction",
            "internal_contradiction_score",
            "opinions",
        )
        issues.extend(f"Missing required field: {key}" for key in required if key not in payload)
        for key in ("subtopics", "central_entities", "central_relations", "opinions"):
            if not isinstance(payload.get(key), list):
                issues.append(f"Invalid list: {key}")
        if isinstance(payload.get("central_relations"), list) and any(
            not isinstance(item, dict) for item in payload["central_relations"]
        ):
            issues.append("Invalid relation record")
        try:
            score = float(payload.get("internal_contradiction_score"))
            if (
                isinstance(payload.get("internal_contradiction_score"), bool)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                issues.append("Invalid contradiction score")
        except (TypeError, ValueError):
            issues.append("Invalid contradiction score")
    relations = _coerce_relations(payload.get("central_relations"), structured=structured)
    if structured and isinstance(payload.get("central_relations"), list):
        for index, item in enumerate(payload["central_relations"]):
            if isinstance(item, dict) and item.get("polarity") not in {"affirmed", "negated"}:
                issues.append(f"Relation {index} polarity normalized from the signed action")
            if isinstance(item, dict) and "numeric_values" in item:
                if not isinstance(item["numeric_values"], list):
                    issues.append(f"Relation {index} numeric values unavailable")
    blocking = [
        issue
        for issue in issues
        if not issue.endswith("polarity normalized from the signed action")
    ]
    return TopicStructure(
        main_topic=str(main_topic).strip() if main_topic else None,
        subtopics=_coerce_string_list(payload.get("subtopics")),
        central_entities=_coerce_string_list(payload.get("central_entities")),
        central_relations=relations,
        narrative_frame=str(narrative_frame).strip() if narrative_frame else None,
        has_internal_contradiction=has_internal_contradiction or internal_contradiction_score > 0.0,
        internal_contradiction_score=internal_contradiction_score,
        schema_version=2 if structured else 1,
        extraction_status=("partial" if blocking else "valid") if structured else "unavailable",
        extraction_issues=issues,
        opinions=_coerce_string_list(payload.get("opinions")),
        provenance=payload.get("provenance", {}),
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
    retry_attempts: int = MAX_EVALUATION_ATTEMPTS,
    before_request_hook: Callable[[], None] | None = None,
    structured: bool = False,
) -> TopicStructure:
    if not text or not str(text).strip():
        raise ValueError("Provide a non-empty text to extract the topic structure.")

    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")
    if retry_attempts <= 0:
        raise ValueError("'retry_attempts' must be greater than zero.")

    provider_normalized, client = create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )
    prompt_template = (
        STRUCTURED_TOPIC_PROMPT_TEMPLATE if structured else TOPIC_DRIFT_PROMPT_TEMPLATE
    )
    if not title or not title.strip():
        prompt_template = prompt_template.replace("Title: {title}\n\n", "")
    prompt = prompt_template.format(title=title, text=text.strip())
    limiter = MinuteRateLimiter(max_requests_per_minute)
    request_hook = before_request_hook or limiter.acquire

    generate = (
        generate_gemini_text_with_retry
        if provider_normalized == "gemini"
        else generate_openai_text_with_retry
    )

    def request() -> str:
        return generate(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_attempts=1,
            before_request_hook=request_hook,
        )

    def parse(raw_response: str) -> TopicStructure:
        if not structured:
            return _build_topic_structure(_extract_json_object(raw_response))
        from misinformation_simulation.topic_drift.provenance import request_provenance

        provenance = request_provenance(
            model=model,
            provider=provider_normalized,
            base_url=base_url,
            version=STRUCTURED_EXTRACTION_VERSION,
            prompt=prompt,
            raw_response=raw_response,
            inputs={"text": text, "title": title},
        )
        try:
            structure = _build_topic_structure(_extract_json_object(raw_response))
        except (ValueError, TypeError) as exc:
            raise ExtractionValidationError(str(exc), provenance) from exc
        structure.provenance = provenance
        return structure

    def should_retry(structure: TopicStructure) -> bool:
        from misinformation_simulation.topic_drift.structured_comparison import structure_issues

        return bool(structure_issues(structure))

    return generate_and_parse_with_retry(
        request,
        parse,
        max_attempts=retry_attempts,
        should_retry=should_retry if structured else None,
    )
