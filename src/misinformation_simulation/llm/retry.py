from __future__ import annotations

import random
import time
from collections.abc import Callable

from google import genai
from google.genai import types
from openai import OpenAI

from misinformation_simulation.llm.progress import retry_wait
from misinformation_simulation.llm.usage import api_request, provider_for_client, record_response

MAX_EVALUATION_ATTEMPTS = 4


def generate_and_parse_with_retry[T](
    generate: Callable[[], str],
    parse: Callable[[str], T],
    *,
    max_attempts: int = MAX_EVALUATION_ATTEMPTS,
    should_retry: Callable[[T], bool] | None = None,
    base_delay: float = 2.0,
) -> T:
    """Share one budget across API and validation failures: initial call plus three retries."""
    if max_attempts <= 0:
        raise ValueError("'retry_attempts' must be greater than zero.")
    max_attempts = min(max_attempts, MAX_EVALUATION_ATTEMPTS)
    for attempt in range(1, max_attempts + 1):
        reason = "response did not meet validation requirements"
        try:
            result = parse(generate())
            if should_retry is None or not should_retry(result) or attempt == max_attempts:
                return result
        except Exception as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            if attempt == max_attempts:
                raise
        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
        retry_wait(delay, attempt=attempt, max_attempts=max_attempts, reason=reason)
        time.sleep(delay)
    raise AssertionError("The evaluation retry budget was exhausted without a result.")


def _openai_chat_completion_kwargs(
    *,
    model: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
    }
    if not model.strip().lower().startswith(("gpt-5", "gpt-6")):
        kwargs["temperature"] = temperature
    return kwargs


def generate_gemini_text_with_retry(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    before_request_hook: Callable[[], None] | None = None,
) -> str:
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if before_request_hook is not None:
                before_request_hook()

            with api_request(provider="gemini", model=model):
                response = client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ),
                    contents=prompt,
                )
            record_response(response, provider="gemini", model=model)
            text = (response.text or "").strip()
            if not text:
                raise ValueError("The API response was empty.")
            return text
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
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            retry_wait(
                delay,
                attempt=attempt,
                max_attempts=max_attempts,
                reason=type(exc).__name__ + ": " + str(exc),
            )
            time.sleep(delay)

    raise last_error


def generate_openai_text_with_retry(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    before_request_hook: Callable[[], None] | None = None,
) -> str:
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if before_request_hook is not None:
                before_request_hook()

            provider = provider_for_client(client)
            with api_request(provider=provider, model=model):
                response = client.chat.completions.create(
                    **_openai_chat_completion_kwargs(
                        model=model,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )
                )
            record_response(response, provider=provider, model=model)
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("The API response was empty.")
            return text
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
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            retry_wait(
                delay,
                attempt=attempt,
                max_attempts=max_attempts,
                reason=type(exc).__name__ + ": " + str(exc),
            )
            time.sleep(delay)

    raise last_error
