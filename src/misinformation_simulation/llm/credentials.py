"""API credentials scoped to an execution instead of the process environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

API_KEY_NAMES = {
    "chatgpt": ("CHATGPT_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "grok": ("GROK_API_KEY", "XAI_API_KEY"),
    "local": ("LOCAL_OPENAI_API_KEY",),
}
OPENAI_FALLBACK_NAMES = ("CHATGPT_API_KEY_FALLBACK", "OPENAI_API_KEY_FALLBACK")


@dataclass(frozen=True)
class ProviderAPIKeys:
    primary: str | None = field(default=None, repr=False)
    fallback: str | None = field(default=None, repr=False)


_credentials: ContextVar[Mapping[str, ProviderAPIKeys] | None] = ContextVar(
    "api_credentials",
    default=None,
)


def environment_api_key(provider: str) -> str | None:
    return next((os.getenv(name) for name in API_KEY_NAMES[provider] if os.getenv(name)), None)


def environment_openai_fallback() -> str | None:
    return next((os.getenv(name) for name in OPENAI_FALLBACK_NAMES if os.getenv(name)), None)


def configured_api_key(provider: str) -> str | None:
    keys = (_credentials.get() or {}).get(provider, ProviderAPIKeys())
    return keys.primary or environment_api_key(provider)


def configured_openai_fallback() -> str | None:
    keys = (_credentials.get() or {}).get("chatgpt", ProviderAPIKeys())
    return keys.fallback or environment_openai_fallback()


@contextmanager
def use_api_credentials(credentials: Mapping[str, ProviderAPIKeys] | None):
    from misinformation_simulation.llm.recovery import register_secret

    snapshot = dict(credentials or {})
    for keys in snapshot.values():
        register_secret(keys.primary)
        register_secret(keys.fallback)
    token = _credentials.set(snapshot)
    try:
        yield
    finally:
        _credentials.reset(token)
