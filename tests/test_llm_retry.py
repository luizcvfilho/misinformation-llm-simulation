from __future__ import annotations

from types import SimpleNamespace

import pytest

from misinformation_simulation.llm import retry
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)


class FakeGeminiModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def generate_content(self, **_kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


class FakeOpenAICompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        message = SimpleNamespace(content=response)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_generate_gemini_text_retries_retryable_errors(monkeypatch) -> None:
    sleeps: list[float] = []
    hooks: list[str] = []
    monkeypatch.setattr(retry.random, "uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    client = SimpleNamespace(models=FakeGeminiModels([RuntimeError("429 rate limit"), " done "]))

    text = generate_gemini_text_with_retry(
        client,
        model="gemini",
        prompt="prompt",
        system_instruction="system",
        temperature=0.0,
        base_delay=2.0,
        before_request_hook=lambda: hooks.append("called"),
    )

    assert text == "done"
    assert client.models.calls == 2
    assert sleeps == [2.0]
    assert hooks == ["called", "called"]


def test_generate_gemini_text_stops_on_non_retryable_error(monkeypatch) -> None:
    monkeypatch.setattr(retry.time, "sleep", lambda _seconds: None)
    client = SimpleNamespace(models=FakeGeminiModels([RuntimeError("bad request")]))

    with pytest.raises(RuntimeError, match="bad request"):
        generate_gemini_text_with_retry(
            client,
            model="gemini",
            prompt="prompt",
            system_instruction="system",
            temperature=0.0,
        )


def test_generate_openai_text_retries_and_strips_response(monkeypatch) -> None:
    monkeypatch.setattr(retry.random, "uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr(retry.time, "sleep", lambda _seconds: None)
    completions = FakeOpenAICompletions([TimeoutError("timeout"), " rewritten "])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    text = generate_openai_text_with_retry(
        client,
        model="gpt",
        prompt="prompt",
        system_instruction="system",
        temperature=0.0,
    )

    assert text == "rewritten"
    assert completions.calls == 2


def test_generate_openai_text_rejects_empty_response() -> None:
    completions = FakeOpenAICompletions(["   "])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with pytest.raises(ValueError, match="empty"):
        generate_openai_text_with_retry(
            client,
            model="gpt",
            prompt="prompt",
            system_instruction="system",
            temperature=0.0,
            max_attempts=1,
        )
