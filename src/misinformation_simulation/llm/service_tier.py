"""OpenAI processing options scoped to one execution."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from misinformation_simulation.llm.flex_fallback import use_flex_fallback

OPENAI_SERVICE_TIERS = ("auto", "default", "flex")
FLEX_TIMEOUT_SECONDS = 900.0

_service_tier: ContextVar[str | None] = ContextVar("openai_service_tier", default=None)


def configured_openai_service_tier() -> str:
    return normalize_openai_service_tier(
        _service_tier.get() or os.getenv("OPENAI_SERVICE_TIER", "auto")
    )


def normalize_openai_service_tier(tier: str) -> str:
    normalized = tier.strip().lower()
    if normalized not in OPENAI_SERVICE_TIERS:
        raise ValueError("OpenAI service tier must be auto, default, or flex.")
    return normalized


@contextmanager
def use_openai_service_tier(tier: str | None):
    resolved = (
        configured_openai_service_tier() if tier is None else normalize_openai_service_tier(tier)
    )
    token = _service_tier.set(resolved)
    try:
        yield resolved
    finally:
        _service_tier.reset(token)


def with_openai_service_tier(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with (
            use_openai_service_tier(kwargs.get("openai_service_tier")),
            use_flex_fallback(
                kwargs.get("openai_flex_fallback", False),
                kwargs.get("openai_flex_retry_seconds", 300),
            ),
        ):
            return function(*args, **kwargs)

    return wrapped


def openai_request_options(provider: str) -> dict[str, object]:
    if provider != "chatgpt":
        return {}
    tier = configured_openai_service_tier()
    if tier == "auto":
        return {}
    options: dict[str, object] = {"service_tier": tier}
    if tier == "flex":
        options["timeout"] = FLEX_TIMEOUT_SECONDS
    return options
