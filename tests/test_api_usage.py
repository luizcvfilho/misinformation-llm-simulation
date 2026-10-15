import json
from types import SimpleNamespace

import pytest
from google.genai import types
from openai.types.chat import ChatCompletion

from misinformation_simulation.llm import retry
from misinformation_simulation.llm.pricing import load_pricing
from misinformation_simulation.llm.usage import (
    UsageTracker,
    collect_usage,
    combine_usage,
    record_cache_hit,
    record_response,
    track_run_usage,
    usage_stage,
)


def response(*, usage=True, model="gpt-6-luna"):
    return SimpleNamespace(
        id="response-1",
        model=model,
        service_tier="default",
        choices=[SimpleNamespace(message=SimpleNamespace(content="response text"))],
        usage=SimpleNamespace(
            prompt_tokens=10000,
            completion_tokens=2000,
            prompt_tokens_details={"cached_tokens": 4000, "cache_write_tokens": 1000},
            completion_tokens_details={"reasoning_tokens": 1500},
        )
        if usage
        else None,
    )


def test_cost_accounts_for_cache_writes_and_includes_reasoning_only_once(tmp_path):
    tracker = UsageTracker(tmp_path / "usage.jsonl")
    tracker.response(response(), provider="chatgpt", model="gpt-6-luna")
    record = tracker.records[0]
    expected = (5000 * 0.1 + 4000 * 0.01 + 1000 * 0.125 + 2000 * 0.5) / 1_000_000
    assert record["estimated_cost_usd"] == pytest.approx(expected)
    assert record["reasoning_tokens"] == 1500
    assert record["raw_usage"]["completion_tokens"] == 2000
    assert json.loads(tracker.path.read_text())["run_id"] == tracker.run_id


def test_unavailable_usage_and_rates_do_not_report_free_requests():
    tracker = UsageTracker()
    tracker.response(response(usage=False), provider="chatgpt", model="gpt-6-luna")
    tracker.response(response(), provider="chatgpt", model="unpriced-model")
    summary = tracker.summary()
    assert summary["requests"] == summary["unpriced_requests"] == 2
    assert summary["estimated_cost_usd"] is None


def test_retry_records_failed_attempt_and_success_and_context_is_restored(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    replies = iter([TimeoutError("timeout"), response()])

    def create(**kwargs):
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item

    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    tracker = UsageTracker()
    with collect_usage(tracker):
        text = usage_stage("judge")(retry.generate_openai_text_with_retry)(
            client,
            model="gpt-6-luna",
            prompt="prompt",
            system_instruction="system",
            temperature=0.1,
            max_attempts=2,
        )
        record_cache_hit("extraction", source="shared")
    assert text == "response text"
    assert [record["status"] for record in tracker.records[:2]] == ["error", "response"]
    summary = tracker.summary()
    assert summary["stages"]["judge"]["requests"] == 2
    assert summary["stages"]["judge"]["failed_requests"] == 1
    assert summary["estimated_cost_usd"] is None
    assert summary["known_estimated_cost_usd"] > 0
    assert summary["stages"]["extraction"]["estimated_cost_usd"] == 0
    record_cache_hit("rewrite", source="shared")
    assert len(tracker.records) == 3


def test_validation_retry_counts_both_received_responses(monkeypatch):
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response())),
    )
    monkeypatch.setattr(retry.time, "sleep", lambda _: None)
    parses = []

    def parse(text):
        parses.append(text)
        if len(parses) == 1:
            raise ValueError("invalid response")
        return text

    tracker = UsageTracker()
    with collect_usage(tracker):
        retry.generate_and_parse_with_retry(
            lambda: retry.generate_openai_text_with_retry(
                client,
                model="gpt-6-luna",
                prompt="prompt",
                system_instruction="system",
                temperature=0.1,
            ),
            parse,
        )
    assert tracker.summary()["requests"] == 2
    assert tracker.summary()["estimated_cost_usd"] > tracker.records[0]["estimated_cost_usd"]


@pytest.mark.parametrize("tier,multiplier", [("flex", 0.5), ("fast", 2), ("unknown", None)])
def test_uses_returned_service_tier(tier, multiplier):
    tracker = UsageTracker()
    item = response()
    tracker.response(item, provider="chatgpt", model="gpt-6-luna")
    standard = tracker.records[0]["estimated_cost_usd"]
    item.service_tier = tier
    tracker.response(item, provider="chatgpt", model="gpt-6-luna")
    cost = tracker.records[-1]["estimated_cost_usd"]
    assert cost is None if multiplier is None else cost == pytest.approx(standard * multiplier)


def test_long_context_rates_are_applied():
    tracker = UsageTracker()
    item = response()
    item.usage.prompt_tokens = 300000
    tracker.response(item, provider="chatgpt", model="gpt-6-luna")
    expected = (
        295000 * 0.1 * 2 + 4000 * 0.01 * 2 + 1000 * 0.125 * 2 + 2000 * 0.5 * 1.5
    ) / 1_000_000
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(expected)


def test_batch_totals_keep_unpriced_requests_unknown():
    first, second = UsageTracker(), UsageTracker()
    first.response(response(), provider="chatgpt", model="gpt-6-luna")
    second.response(response(usage=False), provider="chatgpt", model="gpt-6-luna")
    combined = combine_usage([first.summary(), second.summary()])
    assert combined["requests"] == 2
    assert combined["unpriced_requests"] == 1
    assert combined["estimated_cost_usd"] is None
    assert combined["known_estimated_cost_usd"] == first.summary()["estimated_cost_usd"]


def test_failed_run_keeps_usage_and_invalid_input_does_not_create_reports(tmp_path):
    @track_run_usage
    def fail(*, record, **kwargs):
        if record:
            record_response(response(), provider="chatgpt", model="gpt-6-luna")
        raise ValueError("failed run")

    with pytest.raises(ValueError, match="failed run"):
        fail(record=False, output_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="failed run"):
        fail(record=True, output_dir=tmp_path / "failed")
    summary = json.loads((tmp_path / "failed" / "simulation_api_costs.json").read_text())
    assert summary["requests"] == 1
    assert (tmp_path / "failed" / "simulation_api_usage.jsonl").is_file()


def test_gemini_usage_keeps_thought_tokens_separate_without_assuming_pricing():
    tracker = UsageTracker()
    item = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=30,
            cached_content_token_count=10,
        )
    )
    tracker.response(item, provider="gemini", model="gemini-model")
    assert tracker.records[0]["output_tokens"] == 50
    assert tracker.records[0]["reasoning_tokens"] == 30
    assert tracker.summary()["estimated_cost_usd"] is None


def test_invalid_pricing_is_rejected_before_api_work(tmp_path):
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"chatgpt::model": {"input": -1}}))
    with pytest.raises(ValueError, match="nonnegative"):
        load_pricing(path)


def test_gemini_sdk_response_records_thinking_cache_and_paid_text_cost():
    item = types.GenerateContentResponse(
        model_version="gemini-3.5-flash-lite",
        response_id="gemini-response",
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="reply")]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=30,
            cached_content_token_count=10,
        ),
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: item))
    tracker = UsageTracker()
    with collect_usage(tracker):
        result = usage_stage("rewrite")(retry.generate_gemini_text_with_retry)(
            client,
            model="gemini-3.5-flash-lite",
            prompt="prompt",
            system_instruction="system",
            temperature=0.1,
        )
    record = tracker.records[0]
    assert result == "reply"
    assert record["response_id"] == "gemini-response"
    assert record["service_tier"] == "default"
    assert record["cost_source"] == "token_rates"
    assert record["estimated_cost_usd"] == pytest.approx(
        (90 * 0.30 + 10 * 0.03 + 50 * 2.50) / 1_000_000
    )
    assert tracker.summary()["stages"]["rewrite"]["requests"] == 1


@pytest.mark.parametrize("ticks", [0, 158500, 37756000])
def test_grok_openai_sdk_response_uses_provider_reported_cost(ticks):
    item = ChatCompletion.model_validate(
        {
            "id": "grok-response",
            "object": "chat.completion",
            "created": 0,
            "model": "grok-4.3",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "reply"},
                }
            ],
            "usage": {
                "prompt_tokens": 199,
                "completion_tokens": 1,
                "total_tokens": 200,
                "cost_in_usd_ticks": ticks,
            },
        }
    )
    client = SimpleNamespace(
        base_url="https://api.x.ai/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: item)),
    )
    tracker = UsageTracker()
    with collect_usage(tracker):
        result = usage_stage("judge")(retry.generate_openai_text_with_retry)(
            client,
            model="grok-4-1-fast-non-reasoning",
            prompt="prompt",
            system_instruction="system",
            temperature=0.1,
        )
    record = tracker.records[0]
    assert result == "reply"
    assert record["provider"] == "grok"
    assert record["requested_model"] == "grok-4-1-fast-non-reasoning"
    assert record["returned_model"] == "grok-4.3"
    assert record["provider_cost_in_usd_ticks"] == ticks
    assert record["estimated_cost_usd"] == pytest.approx(ticks / 10_000_000_000)
    assert record["cost_source"] == "provider_reported"
    assert tracker.summary()["provider_reported_requests"] == 1


def test_grok_alternate_token_names_and_missing_cost_are_supported():
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage={
                "input_tokens": 199,
                "output_tokens": 1,
                "cost_in_usd_ticks": 158500,
            }
        ),
        provider="grok",
        model="model",
    )
    assert tracker.records[0]["input_tokens"] == 199
    assert tracker.records[0]["output_tokens"] == 1
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(158500 / 10_000_000_000)
    tracker.response(
        SimpleNamespace(
            usage={
                "prompt_tokens": 199,
                "completion_tokens": 1,
            }
        ),
        provider="grok",
        model="model",
    )
    assert tracker.records[-1]["estimated_cost_usd"] is None
    assert tracker.summary()["estimated_cost_usd"] is None


@pytest.mark.parametrize(
    "traffic,tier",
    [
        ("ON_DEMAND", "default"),
        ("ON_DEMAND_FLEX", "flex"),
        ("ON_DEMAND_PRIORITY", "priority"),
        ("PROVISIONED_THROUGHPUT", "unknown"),
    ],
)
def test_gemini_explicit_traffic_type_is_not_confused_with_standard(traffic, tier):
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage_metadata={
                "prompt_token_count": 100,
                "candidates_token_count": 10,
                "traffic_type": traffic,
            }
        ),
        provider="gemini",
        model="gemini-3.5-flash-lite",
    )
    assert tracker.records[0]["service_tier"] == tier
    assert (tracker.records[0]["estimated_cost_usd"] is None) == (tier == "unknown")
