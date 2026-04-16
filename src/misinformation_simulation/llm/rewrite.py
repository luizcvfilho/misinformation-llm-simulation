from __future__ import annotations

import time

import pandas as pd

from misinformation_simulation.config.prompts import PROMPT_TEMPLATE, REWRITE_SYSTEM_INSTRUCTION
from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
    resolve_output_language,
    resolve_output_language_name,
    resolve_row_text,
)
from misinformation_simulation.enums import Provider
from misinformation_simulation.llm.clients import create_llm_client, normalize_provider
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)

DEFAULT_TITLE_COLUMN = "title"
DEFAULT_OUTPUT_COLUMN = "rewritten_news"
DEFAULT_TEXT_COLUMN = "description"
DEFAULT_MAX_ROWS: int | None = None
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_MAX_REQUESTS_PER_MINUTE: int | None = None
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_ALLOW_TITLE_FALLBACK = False
DEFAULT_PERSONALITY: str | None = None
DEFAULT_DF: pd.DataFrame | None = None
_UNSET = object()


def rewrite_news_with_personality(
    df: pd.DataFrame | object = _UNSET,
    personality: str | object = _UNSET,
    text_column: str | None | object = _UNSET,
    title_column: str | object = _UNSET,
    model: str = "gemini-2.5-flash-lite",
    provider: Provider | str = Provider.GEMINI,
    api_key: str | None = None,
    base_url: str | None = None,
    output_column: str | object = _UNSET,
    max_rows: int | None | object = _UNSET,
    sleep_seconds: float | object = _UNSET,
    max_requests_per_minute: int | None | object = _UNSET,
    retry_attempts: int | object = _UNSET,
    allow_title_fallback: bool | object = _UNSET,
) -> pd.DataFrame:
    def pick(value, default):
        return default if value is _UNSET else value

    df = pick(df, DEFAULT_DF)
    personality = pick(personality, DEFAULT_PERSONALITY)
    text_column = pick(text_column, DEFAULT_TEXT_COLUMN)
    title_column = pick(title_column, DEFAULT_TITLE_COLUMN)
    output_column = pick(output_column, DEFAULT_OUTPUT_COLUMN)
    max_rows = pick(max_rows, DEFAULT_MAX_ROWS)
    sleep_seconds = pick(sleep_seconds, DEFAULT_SLEEP_SECONDS)
    max_requests_per_minute = pick(max_requests_per_minute, DEFAULT_MAX_REQUESTS_PER_MINUTE)
    retry_attempts = pick(retry_attempts, DEFAULT_RETRY_ATTEMPTS)
    allow_title_fallback = pick(allow_title_fallback, DEFAULT_ALLOW_TITLE_FALLBACK)

    if df is None:
        raise ValueError("Provide 'df' in the call or set llm.rewrite.DEFAULT_DF.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if not personality or not str(personality).strip():
        raise ValueError("Provide a valid personality.")
    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")

    provider_normalized = normalize_provider(provider)
    text_column = text_column or choose_news_text_column(df)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")

    _, client = create_llm_client(
        provider=provider_normalized,
        api_key=api_key,
        base_url=base_url,
    )
    limiter = MinuteRateLimiter(max_requests_per_minute)

    rewritten_df = df.copy()
    rewritten_df["source_text_column"] = pd.NA
    rewritten_df["target_language"] = pd.NA
    rewritten_df["target_language_source"] = pd.NA
    rewritten_df[output_column] = pd.NA
    rewritten_df["rewrite_status"] = "not_requested"
    rewritten_df["rewrite_error"] = pd.NA

    target_indexes = list(rewritten_df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]

    for row_index in target_indexes:
        row = rewritten_df.loc[row_index]

        try:
            source_column, original_text = resolve_row_text(
                row=row,
                preferred_column=text_column,
                allow_title_fallback=allow_title_fallback,
            )
            rewritten_df.at[row_index, "source_text_column"] = source_column
        except ValueError as exc:
            rewritten_df.at[row_index, "rewrite_status"] = "skipped"
            rewritten_df.at[row_index, "rewrite_error"] = str(exc)
            continue

        target_language_code, target_language_source = resolve_output_language(
            row=row,
            original_text=original_text,
        )
        rewritten_df.at[row_index, "target_language"] = target_language_code
        rewritten_df.at[row_index, "target_language_source"] = target_language_source

        title = ""
        if title_column in rewritten_df.columns and pd.notna(
            rewritten_df.at[row_index, title_column]
        ):
            title = str(rewritten_df.at[row_index, title_column]).strip()

        prompt = PROMPT_TEMPLATE.format(
            personality=str(personality),
            target_language_name=resolve_output_language_name(target_language_code),
            target_language_code=target_language_code,
            title=title or "Untitled",
            original_text=original_text,
        )

        try:
            if provider_normalized == "gemini":
                rewritten_text = generate_gemini_text_with_retry(
                    client,
                    model=model,
                    prompt=prompt,
                    system_instruction=REWRITE_SYSTEM_INSTRUCTION,
                    temperature=0.8,
                    max_attempts=retry_attempts,
                    before_request_hook=limiter.acquire,
                )
            else:
                rewritten_text = generate_openai_text_with_retry(
                    client,
                    model=model,
                    prompt=prompt,
                    system_instruction=REWRITE_SYSTEM_INSTRUCTION,
                    temperature=0.8,
                    max_attempts=retry_attempts,
                    before_request_hook=limiter.acquire,
                )

            rewritten_df.at[row_index, output_column] = rewritten_text
            rewritten_df.at[row_index, "rewrite_status"] = "success"
            rewritten_df.at[row_index, "rewrite_error"] = pd.NA
        except Exception as exc:
            rewritten_df.at[row_index, "rewrite_status"] = "error"
            rewritten_df.at[row_index, "rewrite_error"] = str(exc)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rewritten_df
