from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.llm import rewrite
from misinformation_simulation.llm.rewrite import rewrite_news_with_personality


def test_rewrite_news_validates_required_inputs() -> None:
    with pytest.raises(ValueError, match="Provide 'df'"):
        rewrite_news_with_personality(personality="persona")

    with pytest.raises(ValueError, match="pandas.DataFrame"):
        rewrite_news_with_personality(df=[], personality="persona")

    with pytest.raises(ValueError, match="empty"):
        rewrite_news_with_personality(df=pd.DataFrame(), personality="persona")

    with pytest.raises(ValueError, match="valid personality"):
        rewrite_news_with_personality(df=pd.DataFrame([{"description": "text"}]), personality=" ")

    with pytest.raises(ValueError, match="greater than zero"):
        rewrite_news_with_personality(
            df=pd.DataFrame([{"description": "text"}]),
            personality="persona",
            max_requests_per_minute=0,
        )


def test_rewrite_news_uses_gemini_generator_and_records_success(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        rewrite,
        "create_llm_client",
        lambda **_kwargs: ("gemini", object()),
    )
    monkeypatch.setattr(rewrite.time, "sleep", lambda _seconds: None)

    def fake_generate_gemini_text_with_retry(_client, **kwargs) -> str:
        calls.append(kwargs)
        return "rewritten text"

    monkeypatch.setattr(
        rewrite,
        "generate_gemini_text_with_retry",
        fake_generate_gemini_text_with_retry,
    )
    df = pd.DataFrame(
        [
            {
                "title": "Title",
                "description": "Original text",
                "language": "Portuguese",
            }
        ]
    )

    result = rewrite_news_with_personality(
        df=df,
        personality="skeptical",
        provider="gemini",
        model="gemini-test",
        retry_attempts=2,
        sleep_seconds=0.1,
    )

    assert result.at[0, "rewritten_news"] == "rewritten text"
    assert result.at[0, "rewrite_status"] == "success"
    assert result.at[0, "source_text_column"] == "description"
    assert result.at[0, "target_language"] == "pt"
    assert result.at[0, "target_language_source"] == "row.language"
    assert calls[0]["model"] == "gemini-test"
    assert calls[0]["max_attempts"] == 2
    assert "Original text" in calls[0]["prompt"]


def test_rewrite_news_marks_rows_without_text_as_skipped(monkeypatch) -> None:
    monkeypatch.setattr(rewrite, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    df = pd.DataFrame([{"title": "Only title", "description": ""}])

    result = rewrite_news_with_personality(
        df=df,
        personality="persona",
        allow_title_fallback=False,
    )

    assert result.at[0, "rewrite_status"] == "skipped"
    assert "no usable text" in result.at[0, "rewrite_error"]


def test_rewrite_news_records_generator_errors_for_openai_provider(monkeypatch) -> None:
    monkeypatch.setattr(rewrite, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))

    def fake_generate_openai_text_with_retry(_client, **_kwargs) -> str:
        raise RuntimeError("provider failed")

    monkeypatch.setattr(
        rewrite,
        "generate_openai_text_with_retry",
        fake_generate_openai_text_with_retry,
    )
    df = pd.DataFrame([{"description": "Original text"}])

    result = rewrite_news_with_personality(
        df=df,
        personality="persona",
        provider="chatgpt",
    )

    assert result.at[0, "rewrite_status"] == "error"
    assert result.at[0, "rewrite_error"] == "provider failed"


def test_rewrite_news_respects_max_rows(monkeypatch) -> None:
    monkeypatch.setattr(rewrite, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    monkeypatch.setattr(
        rewrite,
        "generate_gemini_text_with_retry",
        lambda _client, **_kwargs: "rewritten",
    )
    df = pd.DataFrame([{"description": "first"}, {"description": "second"}])

    result = rewrite_news_with_personality(df=df, personality="persona", max_rows=1)

    assert result.at[0, "rewrite_status"] == "success"
    assert result.at[1, "rewrite_status"] == "not_requested"
