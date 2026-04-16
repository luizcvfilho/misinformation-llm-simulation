from __future__ import annotations

import os

from google import genai
from openai import OpenAI

from misinformation_simulation.enums import Provider


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
) -> tuple[str, genai.Client | OpenAI]:
    provider_normalized = normalize_provider(provider)

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
