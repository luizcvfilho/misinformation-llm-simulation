from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from functools import partial

from google import genai
from google.genai import types
from openai import OpenAI

from misinformation_simulation.enums.models import validate_provider_model
from misinformation_simulation.llm.clients import refresh_llm_client
from misinformation_simulation.llm.flex_fallback import get_flex_fallback, is_flex_unavailable
from misinformation_simulation.llm.progress import retry_wait
from misinformation_simulation.llm.recovery import (
    RequestContext,
    automatic_openai_call,
    discard_successful_attempts,
    get_recovery,
    set_request_context,
)
from misinformation_simulation.llm.service_tier import (
    configured_openai_service_tier,
    openai_request_options,
    use_openai_service_tier,
)
from misinformation_simulation.llm.usage import api_request, provider_for_client, record_response

MAX_EVALUATION_ATTEMPTS = 4


def _validate_request_model(provider: str, model: str) -> None:
    if provider != "unknown":
        validate_provider_model(provider, model)


def generate_and_parse_with_retry[T](
    generate: Callable[[], str],
    parse: Callable[[str], T],
    *,
    max_attempts: int = MAX_EVALUATION_ATTEMPTS,
    should_retry: Callable[[T], bool] | None = None,
    base_delay: float = 2.0,
    on_validation_error: Callable[[Exception], None] | None = None,
    validation_error_factory: Callable[[T], Exception] | None = None,
) -> T:
    """Share one budget across API and validation failures: initial call plus three retries."""
    if max_attempts <= 0:
        raise ValueError("'retry_attempts' must be greater than zero.")

    def parse_response(raw):
        return _parse_evaluation_response(raw, parse, on_validation_error)

    if session := get_recovery():

        def evaluate():
            return _evaluate_with_recovery(
                session,
                generate,
                parse_response,
                should_retry,
                max_attempts,
                on_validation_error,
                validation_error_factory,
            )

        return session.run(evaluate)
    max_attempts = min(max_attempts, MAX_EVALUATION_ATTEMPTS)
    for attempt in range(1, max_attempts + 1):
        reason = "response did not meet validation requirements"
        try:
            result = parse_response(generate())
            if should_retry is None or not should_retry(result) or attempt == max_attempts:
                return result
            if on_validation_error is not None:
                on_validation_error(_validation_error(result, validation_error_factory))
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
    _validate_request_model("gemini", model)
    if session := get_recovery():

        def generate():
            set_request_context(RequestContext("gemini", model))
            request_client = refresh_llm_client(client)
            if before_request_hook is not None:
                before_request_hook()
            with api_request(provider="gemini", model=model):
                response = request_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ),
                )
            record_response(response, provider="gemini", model=model)
            text = (response.text or "").strip()
            if not text:
                raise ValueError("The API response was empty.")
            return text

        return session.run(generate, context=RequestContext("gemini", model))
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
    provider = provider_for_client(client)
    _validate_request_model(provider, model)
    session = get_recovery() if provider != "local" else None

    def generate_once(request_client):
        if before_request_hook is not None:
            before_request_hook()
        with api_request(provider=provider, model=model):
            response = request_client.chat.completions.create(
                **_openai_chat_completion_kwargs(
                    model=model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
                **openai_request_options(provider),
            )
            _raise_response_error(response)
        record_response(response, provider=provider, model=model)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("The API response was empty.")
        return text

    generate = partial(_generate_with_flex_fallback, generate_once, provider=provider)

    if session:
        context = RequestContext(provider, model, endpoint=str(getattr(client, "base_url", "")))

        def request():
            if provider == "chatgpt":
                return session.openai_call(client, generate, model=model)
            set_request_context(context)
            return generate(refresh_llm_client(client))

        return session.run(request, context=context)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return _generate_unattended_openai(provider, client, generate, model)

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


def _generate_with_flex_fallback(generate, request_client, provider):
    controller = get_flex_fallback()
    if provider != "chatgpt" or configured_openai_service_tier() != "flex" or not controller:
        return generate(request_client)
    if isinstance(request_client, OpenAI):
        request_client = request_client.with_options(max_retries=0)
    attempt = controller.select()
    succeeded = False
    try:
        with use_openai_service_tier(attempt.tier):
            result = generate(request_client)
        succeeded = True
        return result
    except Exception as error:
        if attempt.tier != "flex" or not is_flex_unavailable(error):
            raise
        controller.unavailable(attempt)
    finally:
        controller.finish(attempt, succeeded=succeeded)
    # Retry only an explicitly rejected Flex request; Standard errors use normal recovery.
    with use_openai_service_tier("default"):
        return generate(request_client)


class ResponseEnvelopeError(Exception):
    def __init__(self, error: dict):
        self.body = error
        self.code = error.get("code")
        self.type = error.get("type")
        self.message = error.get("message") or "The provider returned an error response."
        super().__init__(self.message)


def _raise_response_error(response) -> None:
    error = (
        response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
    )
    if error:
        if hasattr(error, "model_dump"):
            error = error.model_dump()
        raise ResponseEnvelopeError(error if isinstance(error, dict) else {"message": str(error)})


def _evaluate_with_recovery(
    session,
    generate,
    parse_response,
    should_retry,
    max_attempts,
    on_validation_error,
    validation_error_factory,
):
    attempts = min(max_attempts, MAX_EVALUATION_ATTEMPTS) if on_validation_error else 1
    for attempt in range(1, attempts + 1):
        raw = generate()
        try:
            result = parse_response(raw)
            if should_retry is not None and should_retry(result):
                error = _validation_error(result, validation_error_factory)
                if on_validation_error is not None:
                    on_validation_error(error)
                raise error
            return result
        except (ValueError, TypeError, KeyError) as error:
            if attempt == attempts:
                raise
            session.check_cancelled()
            discard_successful_attempts()
            retry_wait(0, attempt=attempt, max_attempts=attempts, reason=str(error))
    raise AssertionError("The validation retry budget was exhausted without a result.")


def _validation_error(result, factory):
    if factory is not None:
        return factory(result)
    return ValueError("The response did not meet validation requirements.")


def _generate_unattended_openai(provider, client, generate, model):
    if provider == "chatgpt" and os.getenv("LLM_ERROR_RECOVERY", "auto").lower() != "off":
        return automatic_openai_call(client, generate, model=model)
    return generate(client)


def _parse_evaluation_response(raw, parse, on_validation_error):
    try:
        return parse(raw)
    except (ValueError, TypeError, KeyError) as error:
        if on_validation_error is not None:
            on_validation_error(error)
        raise
