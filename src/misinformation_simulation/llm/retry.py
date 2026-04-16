from __future__ import annotations

import random
import time
from collections.abc import Callable

from google import genai
from google.genai import types
from openai import OpenAI


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

            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
                contents=prompt,
            )
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
            time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1))

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

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
            )
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
            time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1))

    raise last_error
