from __future__ import annotations

import pandas as pd
import pytest

from misinformation_simulation.llm import false_to_true
from misinformation_simulation.llm.false_to_true import (
    build_false_to_true_prompt,
    rewrite_false_news_as_true,
)


def test_build_false_to_true_prompt_includes_the_article_context() -> None:
    prompt = build_false_to_true_prompt(
        "Unsupported allegation.",
        topic="politics",
        title="False headline",
    )

    assert "truthful news article" in prompt
    assert "politics" in prompt
    assert "False headline" in prompt
    assert "Unsupported allegation." in prompt


def test_rewrite_false_news_as_true_records_success(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(false_to_true, "create_llm_client", lambda **_kwargs: ("gemini", object()))

    def fake_generate_gemini_text_with_retry(_client, **kwargs) -> str:
        calls.append(kwargs)
        return "Neutral rewrite."

    monkeypatch.setattr(
        false_to_true,
        "generate_gemini_text_with_retry",
        fake_generate_gemini_text_with_retry,
    )
    frame = pd.DataFrame(
        [{"subject": "politics", "title": "False claim", "original_article_text": "False text"}]
    )

    result = rewrite_false_news_as_true(
        frame,
        model="gemini-test",
        provider="gemini",
        retry_attempts=2,
    )

    assert result.at[0, "rewritten_article_text"] == "Neutral rewrite."
    assert result.at[0, "rewrite_status"] == "success"
    assert result.at[0, "rewrite_provider"] == "gemini"
    assert result.at[0, "rewrite_model"] == "gemini-test"
    assert "False text" in result.at[0, "rewrite_prompt"]
    assert calls[0]["max_attempts"] == 2


def test_rewrite_false_news_as_true_skips_successful_rows_and_marks_empty_text(monkeypatch) -> None:
    monkeypatch.setattr(false_to_true, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    calls: list[str] = []
    monkeypatch.setattr(
        false_to_true,
        "generate_gemini_text_with_retry",
        lambda _client, **_kwargs: calls.append("called") or "new rewrite",
    )
    frame = pd.DataFrame(
        [
            {
                "original_article_text": "Already done",
                "rewrite_status": "success",
                "rewritten_article_text": "Existing rewrite",
            },
            {"original_article_text": ""},
        ]
    )

    result = rewrite_false_news_as_true(frame)

    assert calls == []
    assert result.at[0, "rewritten_article_text"] == "Existing rewrite"
    assert result.at[1, "rewrite_status"] == "skipped"
    assert "No usable text" in result.at[1, "rewrite_error"]


def test_rewrite_false_news_as_true_saves_a_checkpoint_after_each_rewrite(
    monkeypatch, tmp_path
) -> None:
    checkpoint_path = tmp_path / "rewrites.csv"
    monkeypatch.setattr(false_to_true, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    generated_count = 0

    def fake_generate_gemini_text_with_retry(_client, **_kwargs) -> str:
        nonlocal generated_count
        generated_count += 1
        if generated_count == 2:
            saved = pd.read_csv(checkpoint_path)
            assert saved.at[0, "rewrite_status"] == "success"
            assert saved.at[0, "rewritten_article_text"] == "Rewrite 1"
        return f"Rewrite {generated_count}"

    monkeypatch.setattr(
        false_to_true,
        "generate_gemini_text_with_retry",
        fake_generate_gemini_text_with_retry,
    )
    frame = pd.DataFrame([{"original_article_text": "First"}, {"original_article_text": "Second"}])

    result = rewrite_false_news_as_true(
        frame,
        provider="gemini",
        checkpoint_path=checkpoint_path,
    )
    saved = pd.read_csv(checkpoint_path)

    assert result["rewrite_status"].tolist() == ["success", "success"]
    assert saved["rewritten_article_text"].tolist() == ["Rewrite 1", "Rewrite 2"]


def test_rewrite_false_news_as_true_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        rewrite_false_news_as_true(pd.DataFrame())

    with pytest.raises(ValueError, match="does not exist"):
        rewrite_false_news_as_true(pd.DataFrame([{"text": "article"}]))

    with pytest.raises(ValueError, match="greater than zero"):
        rewrite_false_news_as_true(
            pd.DataFrame([{"original_article_text": "article"}]),
            retry_attempts=0,
        )
