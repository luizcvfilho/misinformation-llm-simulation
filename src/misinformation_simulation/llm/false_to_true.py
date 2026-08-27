from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import create_llm_client, normalize_provider
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)

DEFAULT_FALSE_TO_TRUE_TEXT_COLUMN = "original_article_text"
DEFAULT_FALSE_TO_TRUE_OUTPUT_COLUMN = "rewritten_article_text"
DEFAULT_FALSE_TO_TRUE_SYSTEM_INSTRUCTION = (
    "You are a careful news verification and rewriting assistant. "
    "Rewrite false or unsupported news text into a truthful, neutral news "
    "article about the same topic. "
    "Preserve the approximate length, structure, and journalistic style. "
    "Correct or remove unsupported claims. Do not invent sources, "
    "quotes, numbers, dates, or events. "
    "Return only the rewritten article text."
)


def build_false_to_true_prompt(
    article_text: object,
    *,
    topic: object = "unknown",
    title: object = "",
) -> str:
    """Build the prompt used to turn a false article into a neutral rewrite."""
    return (
        "Rewrite the following false news article as a truthful news article about the same "
        "topic. Preserve the approximate style, structure, and length, but correct or remove "
        "unsupported claims. Do not add sensational claims. Return only the rewritten article.\n\n"
        f"Topic: {_safe_text(topic) or 'unknown'}\n"
        f"Original title: {_safe_text(title)}\n\n"
        f"False article:\n{_safe_text(article_text)}"
    )


def rewrite_false_news_as_true(
    df: pd.DataFrame,
    *,
    text_column: str = DEFAULT_FALSE_TO_TRUE_TEXT_COLUMN,
    topic_column: str | None = "subject",
    title_column: str | None = "title",
    output_column: str = DEFAULT_FALSE_TO_TRUE_OUTPUT_COLUMN,
    prompt_column: str = "rewrite_prompt",
    status_column: str = "rewrite_status",
    error_column: str = "rewrite_error",
    provider_column: str = "rewrite_provider",
    model_column: str = "rewrite_model",
    model: str = DEFAULT_LLM_MODEL,
    provider: Provider | str = DEFAULT_LLM_PROVIDER,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
    retry_attempts: int = 5,
    sleep_seconds: float = 0.0,
    max_requests_per_minute: int | None = None,
    max_rows: int | None = None,
    skip_successful: bool = True,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Rewrite false-news rows into neutral, truthful versions without VAD processing.

    Failed rows are retained and marked ``error`` so they can be retried in a later call.
    Existing successful rows are left untouched by default. When ``checkpoint_path`` is set,
    the DataFrame is saved before and after each attempted rewrite.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The false-news DataFrame is empty.")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")
    if retry_attempts <= 0:
        raise ValueError("'retry_attempts' must be greater than zero.")
    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")

    provider_name = normalize_provider(provider)
    _, client = create_llm_client(
        provider=provider_name,
        api_key=api_key,
        base_url=base_url,
    )
    limiter = MinuteRateLimiter(max_requests_per_minute)
    rewritten_df = df.copy()
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None

    _ensure_column(rewritten_df, output_column, pd.NA)
    _ensure_column(rewritten_df, prompt_column, pd.NA)
    _ensure_column(rewritten_df, status_column, "not_requested")
    _ensure_column(rewritten_df, error_column, pd.NA)
    rewritten_df[provider_column] = provider_name
    rewritten_df[model_column] = model
    _save_checkpoint(rewritten_df, checkpoint_file)

    target_indexes = list(rewritten_df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]

    for row_index in target_indexes:
        if skip_successful and rewritten_df.at[row_index, status_column] == "success":
            continue

        row = rewritten_df.loc[row_index]
        article_text = _safe_text(row[text_column])
        if not article_text:
            rewritten_df.at[row_index, status_column] = "skipped"
            rewritten_df.at[row_index, error_column] = f"No usable text in '{text_column}'."
            _save_checkpoint(rewritten_df, checkpoint_file)
            continue

        topic = (
            row[topic_column]
            if topic_column and topic_column in rewritten_df.columns
            else "unknown"
        )
        title = row[title_column] if title_column and title_column in rewritten_df.columns else ""
        prompt = build_false_to_true_prompt(article_text, topic=topic, title=title)
        rewritten_df.at[row_index, prompt_column] = prompt
        rewritten_df.at[row_index, output_column] = pd.NA
        rewritten_df.at[row_index, status_column] = "running"
        rewritten_df.at[row_index, error_column] = pd.NA
        _save_checkpoint(rewritten_df, checkpoint_file)

        try:
            rewritten_text = _generate_rewrite(
                client=client,
                provider=provider_name,
                model=model,
                prompt=prompt,
                temperature=temperature,
                retry_attempts=retry_attempts,
                limiter=limiter,
            )
            rewritten_df.at[row_index, output_column] = rewritten_text
            rewritten_df.at[row_index, status_column] = "success"
        except Exception as exc:
            rewritten_df.at[row_index, status_column] = "error"
            rewritten_df.at[row_index, error_column] = str(exc)

        _save_checkpoint(rewritten_df, checkpoint_file)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rewritten_df


def _generate_rewrite(
    *,
    client: object,
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    retry_attempts: int,
    limiter: MinuteRateLimiter,
) -> str:
    kwargs = {
        "model": model,
        "prompt": prompt,
        "system_instruction": DEFAULT_FALSE_TO_TRUE_SYSTEM_INSTRUCTION,
        "temperature": temperature,
        "max_attempts": retry_attempts,
        "before_request_hook": limiter.acquire,
    }
    if provider == "gemini":
        return generate_gemini_text_with_retry(client, **kwargs)  # type: ignore[arg-type]
    return generate_openai_text_with_retry(client, **kwargs)  # type: ignore[arg-type]


def _ensure_column(df: pd.DataFrame, column: str, value: object) -> None:
    if column not in df.columns:
        df[column] = value


def _save_checkpoint(df: pd.DataFrame, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(checkpoint_path, index=False)


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
