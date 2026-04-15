from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from google import genai
from google.genai import types
from openai import OpenAI

from enums.providers import Provider
from utils.simulation_functions import (
    DEFAULT_ALLOW_TITLE_FALLBACK,
    DEFAULT_MAX_REQUESTS_PER_MINUTE,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TEXT_COLUMN,
    choose_news_text_column,
    resolve_row_text,
)

DEFAULT_TOPIC_DRIFT_MODEL = "gemini-2.5-flash-lite"
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


@dataclass(slots=True)
class TopicRelation:
    subject: str
    action: str
    object: str


@dataclass(slots=True)
class TopicStructure:
    main_topic: str | None
    subtopics: list[str]
    central_entities: list[str]
    central_relations: list[TopicRelation]
    narrative_frame: str | None = None


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
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_relation(subject: Any, action: Any, obj: Any) -> tuple[str, str, str]:
    return (
        _normalize_scalar(subject),
        _normalize_scalar(action),
        _normalize_scalar(obj),
    )


def _jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    intersection = left & right
    return 1.0 - (len(intersection) / len(union))


def _normalized_non_empty_scalars(values: Sequence[str]) -> set[str]:
    return {normalized for item in values if (normalized := _normalize_scalar(item))}


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


def _serialize_relations(relations: list[TopicRelation]) -> str:
    return json.dumps([asdict(item) for item in relations], ensure_ascii=False)


def topic_structure_to_dict(structure: TopicStructure) -> dict[str, Any]:
    return {
        "main_topic": structure.main_topic,
        "subtopics": list(structure.subtopics),
        "central_entities": list(structure.central_entities),
        "central_relations": [asdict(item) for item in structure.central_relations],
        "narrative_frame": structure.narrative_frame,
    }


def flatten_topic_structure(structure: TopicStructure, *, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_main_topic": structure.main_topic,
        f"{prefix}_subtopics": json.dumps(structure.subtopics, ensure_ascii=False),
        f"{prefix}_central_entities": json.dumps(
            structure.central_entities,
            ensure_ascii=False,
        ),
        f"{prefix}_central_relations": _serialize_relations(structure.central_relations),
        f"{prefix}_narrative_frame": structure.narrative_frame,
        f"{prefix}_json": json.dumps(topic_structure_to_dict(structure), ensure_ascii=False),
    }


def _resolve_client(
    *,
    provider: Provider | str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, genai.Client | OpenAI]:
    provider_normalized = (
        provider.value if isinstance(provider, Provider) else str(provider).strip().lower()
    )
    if provider_normalized not in {item.value for item in Provider}:
        raise ValueError(
            "Invalid provider. Use: gemini, chatgpt, openrouter, deepseek, grok, or local."
        )

    env_key_name = None
    if provider_normalized == "gemini":
        env_key_name = "GEMINI_API_KEY"
    elif provider_normalized == "chatgpt":
        env_key_name = "CHATGPT_API_KEY"
    elif provider_normalized == "openrouter":
        env_key_name = "OPENROUTER_API_KEY"
    elif provider_normalized == "deepseek":
        env_key_name = "DEEPSEEK_API_KEY"
    elif provider_normalized == "grok":
        env_key_name = "GROK_API_KEY"

    if provider_normalized == "local":
        resolved_api_key = api_key or os.getenv("LOCAL_OPENAI_API_KEY") or "ollama"
        resolved_base_url = (
            base_url or os.getenv("LOCAL_OPENAI_BASE_URL") or "http://127.0.0.1:11434/v1"
        )
        return provider_normalized, OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    if provider_normalized == "chatgpt":
        resolved_api_key = api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
    elif provider_normalized == "grok":
        resolved_api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    else:
        resolved_api_key = api_key or os.getenv(env_key_name)

    if not resolved_api_key:
        if provider_normalized == "chatgpt":
            raise ValueError(
                "Set CHATGPT_API_KEY or OPENAI_API_KEY in the environment, "
                "or pass the key in 'api_key'."
            )
        if provider_normalized == "grok":
            raise ValueError(
                "Set GROK_API_KEY or XAI_API_KEY in the environment, or pass the key in 'api_key'."
            )
        raise ValueError(f"Set {env_key_name} in the environment or pass the key in 'api_key'.")

    if provider_normalized == "gemini":
        return provider_normalized, genai.Client(api_key=resolved_api_key)
    if provider_normalized == "chatgpt":
        return provider_normalized, OpenAI(
            api_key=resolved_api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
    if provider_normalized == "openrouter":
        return provider_normalized, OpenAI(
            api_key=resolved_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "misinformation-llm-simulation"),
            },
        )
    if provider_normalized == "deepseek":
        return provider_normalized, OpenAI(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
        )
    return provider_normalized, OpenAI(
        api_key=resolved_api_key,
        base_url=base_url or "https://api.x.ai/v1",
    )


def _generate_structured_analysis(
    client: genai.Client | OpenAI,
    *,
    provider_normalized: str,
    model: str,
    prompt: str,
    max_attempts: int,
    before_request_hook: Any = None,
) -> str:
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if before_request_hook is not None:
                before_request_hook()

            if provider_normalized == "gemini":
                response = client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=TOPIC_DRIFT_SYSTEM_INSTRUCTION,
                        temperature=0.1,
                    ),
                    contents=prompt,
                )
                raw_text = (response.text or "").strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": TOPIC_DRIFT_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_text = (response.choices[0].message.content or "").strip()

            if not raw_text:
                raise ValueError("The API response was empty.")

            return raw_text
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            is_retryable = (
                any(
                    marker in error_text
                    for marker in ("429", "503", "unavailable", "resource_exhausted", "rate limit")
                )
                or "timeout" in error_text
            )

            if not is_retryable or attempt == max_attempts:
                break

            sleep_for = 2.0 * (2 ** (attempt - 1)) + random.uniform(0, 1)
            time.sleep(sleep_for)

    raise last_error


def extract_topic_structure(
    *,
    text: str,
    title: str | None = None,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    api_key: str | None = None,
    base_url: str | None = None,
    max_requests_per_minute: int | None = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> TopicStructure:
    if not text or not str(text).strip():
        raise ValueError("Provide a non-empty text to extract the topic structure.")

    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")

    provider_normalized, client = _resolve_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    prompt = TOPIC_DRIFT_PROMPT_TEMPLATE.format(title=title or "Untitled", text=text.strip())

    current_minute_bucket = int(time.time() // 60)
    requests_in_current_minute = 0

    def wait_for_rate_limit_slot() -> None:
        nonlocal current_minute_bucket, requests_in_current_minute

        if max_requests_per_minute is None:
            return

        now_bucket = int(time.time() // 60)
        if now_bucket != current_minute_bucket:
            current_minute_bucket = now_bucket
            requests_in_current_minute = 0

        if requests_in_current_minute >= max_requests_per_minute:
            seconds_until_next_minute = 60 - (time.time() % 60)
            time.sleep(seconds_until_next_minute)
            current_minute_bucket = int(time.time() // 60)
            requests_in_current_minute = 0

        requests_in_current_minute += 1

    raw_response = _generate_structured_analysis(
        client,
        provider_normalized=provider_normalized,
        model=model,
        prompt=prompt,
        max_attempts=retry_attempts,
        before_request_hook=wait_for_rate_limit_slot,
    )
    payload = _extract_json_object(raw_response)
    return _build_topic_structure(payload)


def calculate_stdi(
    original_structure: TopicStructure,
    compared_structure: TopicStructure,
) -> dict[str, float]:
    original_main_topic = _normalize_scalar(original_structure.main_topic)
    compared_main_topic = _normalize_scalar(compared_structure.main_topic)

    theme_drift = 0.0 if original_main_topic == compared_main_topic else 1.0
    subtopic_drift = _jaccard_distance(
        _normalized_non_empty_scalars(original_structure.subtopics),
        _normalized_non_empty_scalars(compared_structure.subtopics),
    )
    entity_drift = _jaccard_distance(
        {
            _normalize_scalar(item)
            for item in original_structure.central_entities
            if _normalize_scalar(item)
        },
        {
            _normalize_scalar(item)
            for item in compared_structure.central_entities
            if _normalize_scalar(item)
        },
    )
    relation_drift = _jaccard_distance(
        {
            _normalize_relation(item.subject, item.action, item.object)
            for item in original_structure.central_relations
            if all(_normalize_relation(item.subject, item.action, item.object))
        },
        {
            _normalize_relation(item.subject, item.action, item.object)
            for item in compared_structure.central_relations
            if all(_normalize_relation(item.subject, item.action, item.object))
        },
    )

    stdi = 0.2 * theme_drift + 0.2 * subtopic_drift + 0.2 * entity_drift + 0.4 * relation_drift

    return {
        "theme_drift": round(theme_drift, 6),
        "subtopic_drift": round(subtopic_drift, 6),
        "entity_drift": round(entity_drift, 6),
        "relation_drift": round(relation_drift, 6),
        "stdi": round(stdi, 6),
    }


def calculate_stdi_chain_metrics(
    structures: Sequence[TopicStructure],
    *,
    version_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if len(structures) < 2:
        return []

    labels = (
        list(version_labels)
        if version_labels is not None
        else [f"version_{i}" for i in range(len(structures))]
    )
    if len(labels) != len(structures):
        raise ValueError("'version_labels' must have the same length as 'structures'.")

    original_structure = structures[0]
    previous_structure = structures[0]
    cumulative_stdi = 0.0
    metrics: list[dict[str, Any]] = []

    for index in range(1, len(structures)):
        current_structure = structures[index]
        vs_original = calculate_stdi(original_structure, current_structure)
        incremental = calculate_stdi(previous_structure, current_structure)
        cumulative_stdi += incremental["stdi"]

        metrics.append(
            {
                "version_label": labels[index],
                "reference_original_label": labels[0],
                "reference_incremental_label": labels[index - 1],
                "stdi_vs_original": vs_original["stdi"],
                "theme_drift_vs_original": vs_original["theme_drift"],
                "subtopic_drift_vs_original": vs_original["subtopic_drift"],
                "entity_drift_vs_original": vs_original["entity_drift"],
                "relation_drift_vs_original": vs_original["relation_drift"],
                "stdi_incremental": incremental["stdi"],
                "theme_drift_incremental": incremental["theme_drift"],
                "subtopic_drift_incremental": incremental["subtopic_drift"],
                "entity_drift_incremental": incremental["entity_drift"],
                "relation_drift_incremental": incremental["relation_drift"],
                "stdi_cumulative": round(cumulative_stdi, 6),
            }
        )
        previous_structure = current_structure

    return metrics


def _sanitize_column_token(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip()).strip("_").lower()
    return normalized or "version"


def annotate_stdi_for_version_chain(
    df: pd.DataFrame,
    *,
    version_columns: Sequence[str],
    text_column: str | None = DEFAULT_TEXT_COLUMN,
    title_column: str = "title",
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_requests_per_minute: int | None = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    allow_title_fallback: bool = DEFAULT_ALLOW_TITLE_FALLBACK,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if not version_columns:
        raise ValueError("Provide at least one version column.")

    resolved_text_column = text_column or choose_news_text_column(df)
    for column in version_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    result_df = df.copy()

    base_status_columns = {
        "original_topic_structure_status": "not_requested",
        "original_topic_structure_error": pd.NA,
    }
    for column, default_value in base_status_columns.items():
        result_df[column] = default_value

    version_tokens = {column: _sanitize_column_token(column) for column in version_columns}
    for token in version_tokens.values():
        result_df[f"{token}_topic_structure_status"] = "not_requested"
        result_df[f"{token}_topic_structure_error"] = pd.NA

    target_indexes = list(result_df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]

    for row_index in target_indexes:
        row = result_df.loc[row_index]

        try:
            source_column, original_text = resolve_row_text(
                row=row,
                preferred_column=resolved_text_column,
                allow_title_fallback=allow_title_fallback,
            )
            result_df.at[row_index, "source_text_column"] = source_column
        except ValueError as exc:
            result_df.at[row_index, "original_topic_structure_status"] = "skipped"
            result_df.at[row_index, "original_topic_structure_error"] = str(exc)
            continue

        title = ""
        if title_column in result_df.columns and pd.notna(result_df.at[row_index, title_column]):
            title = str(result_df.at[row_index, title_column]).strip()

        try:
            original_structure = extract_topic_structure(
                text=original_text,
                title=title,
                model=model,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                max_requests_per_minute=max_requests_per_minute,
                retry_attempts=retry_attempts,
            )
            result_df.at[row_index, "original_topic_structure_status"] = "success"
            result_df.at[row_index, "original_topic_structure_error"] = pd.NA
            for column_name, value in flatten_topic_structure(
                original_structure,
                prefix="original",
            ).items():
                result_df.at[row_index, column_name] = value
        except Exception as exc:
            result_df.at[row_index, "original_topic_structure_status"] = "error"
            result_df.at[row_index, "original_topic_structure_error"] = str(exc)
            continue

        chain_structures = [original_structure]
        chain_labels = ["original"]

        for version_column in version_columns:
            token = version_tokens[version_column]
            version_text_value = result_df.at[row_index, version_column]
            version_text = "" if pd.isna(version_text_value) else str(version_text_value).strip()

            if not version_text:
                result_df.at[row_index, f"{token}_topic_structure_status"] = "skipped"
                result_df.at[row_index, f"{token}_topic_structure_error"] = (
                    f"Column '{version_column}' has no usable text."
                )
                continue

            try:
                version_structure = extract_topic_structure(
                    text=version_text,
                    title=title,
                    model=model,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    max_requests_per_minute=max_requests_per_minute,
                    retry_attempts=retry_attempts,
                )
                result_df.at[row_index, f"{token}_topic_structure_status"] = "success"
                result_df.at[row_index, f"{token}_topic_structure_error"] = pd.NA
                for column_name, value in flatten_topic_structure(
                    version_structure,
                    prefix=token,
                ).items():
                    result_df.at[row_index, column_name] = value

                chain_structures.append(version_structure)
                chain_labels.append(token)
            except Exception as exc:
                result_df.at[row_index, f"{token}_topic_structure_status"] = "error"
                result_df.at[row_index, f"{token}_topic_structure_error"] = str(exc)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        chain_metrics = calculate_stdi_chain_metrics(chain_structures, version_labels=chain_labels)
        for metric_row in chain_metrics:
            token = metric_row["version_label"]
            for key, value in metric_row.items():
                if key == "version_label":
                    continue
                result_df.at[row_index, f"{token}_{key}"] = value

    return result_df


def annotate_stdi_for_rewrites(
    df: pd.DataFrame,
    *,
    rewritten_column: str = DEFAULT_REWRITTEN_COLUMN,
    text_column: str | None = DEFAULT_TEXT_COLUMN,
    title_column: str = "title",
    provider: Provider | str = DEFAULT_TOPIC_DRIFT_PROVIDER,
    model: str = DEFAULT_TOPIC_DRIFT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_requests_per_minute: int | None = DEFAULT_MAX_REQUESTS_PER_MINUTE,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    allow_title_fallback: bool = DEFAULT_ALLOW_TITLE_FALLBACK,
) -> pd.DataFrame:
    return annotate_stdi_for_version_chain(
        df=df,
        version_columns=[rewritten_column],
        text_column=text_column,
        title_column=title_column,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_rows=max_rows,
        sleep_seconds=sleep_seconds,
        max_requests_per_minute=max_requests_per_minute,
        retry_attempts=retry_attempts,
        allow_title_fallback=allow_title_fallback,
    )
