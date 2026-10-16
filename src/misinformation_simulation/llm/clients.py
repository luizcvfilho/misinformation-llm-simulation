from __future__ import annotations

import os

from google import genai
from google.genai import types
from openai import OpenAI

from misinformation_simulation.enums import Provider
from misinformation_simulation.llm.credentials import (
    configured_api_key,
    configured_openai_fallback,
)
from misinformation_simulation.llm.recovery import RequestContext, get_recovery, register_secret


def normalize_provider(provider: Provider | str) -> str:
    provider_normalized = (
        provider.value if isinstance(provider, Provider) else str(provider).strip().lower()
    )
    if provider_normalized not in {item.value for item in Provider}:
        raise ValueError(
            "Invalid provider. Use: gemini, chatgpt, openrouter, deepseek, grok, or local."
        )
    return provider_normalized


def create_llm_client(
    *,
    provider: Provider | str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int | None = None,
) -> tuple[str, genai.Client | OpenAI]:
    provider_name = normalize_provider(provider)
    register_secret(api_key)
    session = get_recovery() if provider_name != "local" else None

    def create():
        result = _create_llm_client(
            provider=provider_name,
            api_key=api_key,
            base_url=base_url,
            max_retries=0 if session else max_retries,
        )
        result[1]._recovery_configuration = {
            "provider": provider_name,
            "api_key": api_key,
            "base_url": base_url,
        }
        result[1]._recovery_resolved_key = api_key or configured_api_key(provider_name)
        register_secret(result[1]._recovery_resolved_key)
        return result

    if session:
        return session.run(create, context=RequestContext(provider=provider_name))
    return create()


def refresh_llm_client(client):
    current = getattr(client, "_recovery_replacement", client)
    config = getattr(current, "_recovery_configuration", None)
    if config and config["api_key"] is None:
        key = configured_api_key(config["provider"])
        if key != current._recovery_resolved_key:
            _, current = create_llm_client(**config, max_retries=0)
            client._recovery_replacement = current
    return current


def openai_credentials(client: OpenAI) -> dict[str, str]:
    if hasattr(client, "_recovery_explicit_key"):
        primary = client._recovery_explicit_key or (configured_api_key("chatgpt") or client.api_key)
    else:
        primary = client.api_key
    secondary = configured_openai_fallback()
    keys = {"primary": primary}
    if secondary and secondary != primary:
        keys["secondary"] = secondary
    return keys


def _create_llm_client(
    *,
    provider: Provider | str,
    api_key: str | None,
    base_url: str | None,
    max_retries: int | None,
) -> tuple[str, genai.Client | OpenAI]:
    provider_normalized = normalize_provider(provider)
    retry_kwargs = {} if max_retries is None else {"max_retries": max_retries}

    env_key_name = _provider_environment_key(provider_normalized)

    if provider_normalized == "local":
        resolved_api_key = api_key or configured_api_key("local") or "ollama"
        resolved_base_url = (
            base_url or os.getenv("LOCAL_OPENAI_BASE_URL") or "http://127.0.0.1:11434/v1"
        )
        return provider_normalized, OpenAI(
            api_key=resolved_api_key, base_url=resolved_base_url, **retry_kwargs
        )

    resolved_api_key = api_key or configured_api_key(provider_normalized)

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
        if max_retries is None:
            return provider_normalized, genai.Client(api_key=resolved_api_key)
        return provider_normalized, genai.Client(
            api_key=resolved_api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=max_retries + 1)
            ),
        )
    if provider_normalized == "chatgpt":
        client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url or "https://api.openai.com/v1",
            **retry_kwargs,
        )
        client._recovery_explicit_key = api_key
        return provider_normalized, client
    if provider_normalized == "openrouter":
        return provider_normalized, OpenAI(
            api_key=resolved_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "misinformation-llm-simulation"),
            },
            **retry_kwargs,
        )
    if provider_normalized == "deepseek":
        return provider_normalized, OpenAI(
            api_key=resolved_api_key,
            base_url="https://api.deepseek.com",
            **retry_kwargs,
        )
    return provider_normalized, OpenAI(
        api_key=resolved_api_key,
        base_url=base_url or "https://api.x.ai/v1",
        **retry_kwargs,
    )


def _provider_environment_key(provider_normalized):
    return {
        "gemini": "GEMINI_API_KEY",
        "chatgpt": "CHATGPT_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "grok": "GROK_API_KEY",
    }.get(provider_normalized)
