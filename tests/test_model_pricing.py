import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from misinformation_simulation.apps.interaction_graph_ui import AVAILABLE_MODELS
from misinformation_simulation.enums.models import Models
from misinformation_simulation.llm.pricing import (
    MODEL_PRICING,
    default_pricing,
    load_pricing,
    validate_pricing,
)
from misinformation_simulation.llm.usage import UsageTracker


def test_catalog_covers_every_preset_and_replaces_retired_choices():
    assert set(MODEL_PRICING) == set(Models)
    assert "gemini-3.5-flash-lite" in AVAILABLE_MODELS
    assert "grok-build-0.1" in AVAILABLE_MODELS
    assert "gemini-3.1-flash-lite-preview" not in AVAILABLE_MODELS
    assert "grok-4-1-fast-non-reasoning" not in AVAILABLE_MODELS
    validate_pricing(default_pricing())
    json.dumps(default_pricing(), allow_nan=False)


def test_catalog_is_immutable_and_each_tracker_gets_its_own_snapshot():
    entry = MODEL_PRICING[Models.GEMINI35FlashLite]
    with pytest.raises(FrozenInstanceError):
        entry.rates.input = 99
    with pytest.raises(TypeError):
        MODEL_PRICING[Models.GEMINI35FlashLite] = entry
    first, second = UsageTracker(), UsageTracker()
    first.pricing[entry.key]["tier_rates"]["flex"]["cached_input"] = 99
    assert second.pricing[entry.key]["tier_rates"]["flex"]["cached_input"] == 0.02
    assert default_pricing()[entry.key]["tier_rates"]["flex"]["cached_input"] == 0.02


@pytest.mark.parametrize(
    "model,input_rate,cached_rate,output_rate",
    [
        (Models.GPT6Luna, 0.10, 0.01, 0.50),
        (Models.GPT56Luna, 0.20, 0.02, 1.20),
        (Models.GPT5Mini, 0.25, 0.025, 2.00),
        (Models.GPT5Nano, 0.05, 0.005, 0.40),
        (Models.GPT41Mini, 0.40, 0.10, 1.60),
        (Models.GROKBuild01, 1.00, 0.20, 2.00),
        (Models.NEMOTRON3Super120B, 0, 0, 0),
        (Models.STEP35FLASH, 0, 0, 0),
        (Models.LLAMA318B, 0, 0, 0),
    ],
)
def test_standard_cost_uses_the_selected_models_rates(model, input_rate, cached_rate, output_rate):
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            service_tier="default",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        ),
        provider=MODEL_PRICING[model].provider,
        model=model,
    )
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(
        (90 * input_rate + 10 * cached_rate + 20 * output_rate) / 1_000_000
    )


@pytest.mark.parametrize(
    "traffic,input_rate,cached_rate,output_rate",
    [
        ("ON_DEMAND", 0.30, 0.03, 2.50),
        ("ON_DEMAND_FLEX", 0.15, 0.02, 1.25),
        ("ON_DEMAND_PRIORITY", 0.54, 0.05, 4.50),
        ("BATCH", 0.15, 0.02, 1.25),
    ],
)
def test_gemini_uses_exact_tier_rates_including_cached_tokens(
    traffic, input_rate, cached_rate, output_rate
):
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage_metadata={
                "prompt_token_count": 100,
                "cached_content_token_count": 10,
                "candidates_token_count": 20,
                "thoughts_token_count": 30,
                "traffic_type": traffic,
            }
        ),
        provider="gemini",
        model=Models.GEMINI35FlashLite,
    )
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(
        (90 * input_rate + 10 * cached_rate + 50 * output_rate) / 1_000_000
    )


@pytest.mark.parametrize("input_tokens,multiplier", [(199999, 1), (200000, 2), (200001, 2)])
def test_grok_long_context_starts_at_200k_tokens(input_tokens, multiplier):
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 10000},
            }
        ),
        provider="grok",
        model=Models.GROKBuild01,
    )
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(
        ((input_tokens - 10000) * 1 + 10000 * 0.2 + 20 * 2) * multiplier / 1_000_000
    )


def test_grok_provider_amount_overrides_catalog_estimate():
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage={"prompt_tokens": 100, "completion_tokens": 20, "cost_in_usd_ticks": 7}
        ),
        provider="grok",
        model=Models.GROKBuild01,
    )
    assert tracker.records[0]["estimated_cost_usd"] == pytest.approx(7 / 10_000_000_000)
    assert tracker.records[0]["cost_source"] == "provider_reported"


@pytest.mark.parametrize("model", [Models.NEMOTRON3Super120B, Models.STEP35FLASH, Models.LLAMA318B])
def test_free_and_local_presets_without_service_tier_have_zero_api_cost(model):
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(usage={"prompt_tokens": 100, "completion_tokens": 20}),
        provider=MODEL_PRICING[model].provider,
        model=model,
    )
    assert tracker.records[0]["service_tier"] == "default"
    assert tracker.summary()["estimated_cost_usd"] == 0


def test_overrides_replace_whole_model_record_without_mutating_catalog(tmp_path):
    entry = MODEL_PRICING[Models.GEMINI35FlashLite]
    rate = entry.as_record()
    rate["output"] = 99
    rate["source"] = "https://example.com/contract"
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps({entry.key: rate}), encoding="utf-8")
    pricing = load_pricing(path)
    assert pricing[entry.key]["output"] == 99
    assert len(pricing) == len(MODEL_PRICING)
    assert default_pricing()[entry.key]["output"] == 2.50
    tracker = UsageTracker(pricing=pricing)
    pricing[entry.key]["output"] = 0
    assert tracker.pricing[entry.key]["output"] == 99
    assert load_pricing(None) is None


def test_unsupported_cache_write_charge_stays_unknown():
    tracker = UsageTracker()
    tracker.response(
        SimpleNamespace(
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cache_write_tokens": 10},
            }
        ),
        provider="grok",
        model=Models.GROKBuild01,
    )
    assert tracker.summary()["estimated_cost_usd"] is None


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf"), True, "0.02"])
def test_invalid_tier_rates_are_rejected(bad_value):
    pricing = default_pricing()
    pricing["gemini::gemini-3.5-flash-lite"]["tier_rates"]["flex"]["cached_input"] = bad_value
    with pytest.raises(ValueError, match="finite nonnegative"):
        validate_pricing(pricing)
