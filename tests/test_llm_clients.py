from __future__ import annotations

import pytest

from misinformation_simulation.enums import Provider
from misinformation_simulation.llm import clients
from misinformation_simulation.llm.clients import create_llm_client, normalize_provider


class FakeGeminiClient:
    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key


class FakeOpenAIClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_normalize_provider_accepts_enum_and_strings() -> None:
    assert normalize_provider(Provider.GEMINI) == "gemini"
    assert normalize_provider(" ChatGPT ") == "chatgpt"


def test_normalize_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Invalid provider"):
        normalize_provider("unknown")


def test_create_llm_client_uses_explicit_gemini_key(monkeypatch) -> None:
    monkeypatch.setattr(clients.genai, "Client", FakeGeminiClient)

    provider, client = create_llm_client(provider="gemini", api_key="secret")

    assert provider == "gemini"
    assert isinstance(client, FakeGeminiClient)
    assert client.api_key == "secret"


def test_create_llm_client_requires_api_key_for_remote_provider(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_llm_client(provider="openrouter")


def test_create_llm_client_configures_local_openai_defaults(monkeypatch) -> None:
    monkeypatch.setattr(clients, "OpenAI", FakeOpenAIClient)
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)

    provider, client = create_llm_client(provider="local")

    assert provider == "local"
    assert client.kwargs == {
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
    }


def test_create_llm_client_uses_openai_fallback_for_chatgpt(monkeypatch) -> None:
    monkeypatch.setattr(clients, "OpenAI", FakeOpenAIClient)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    provider, client = create_llm_client(provider="chatgpt")

    assert provider == "chatgpt"
    assert client.kwargs == {
        "api_key": "openai-key",
        "base_url": "https://api.openai.com/v1",
    }
