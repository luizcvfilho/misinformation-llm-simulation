from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from misinformation_simulation.enums.models import Models
from misinformation_simulation.enums.providers import Provider

PRICING_CHECKED_ON = "2026-10-14"


@dataclass(frozen=True)
class TokenRates:
    """USD per million text tokens; None means an unsupported cache-write charge."""

    input: float
    cached_input: float
    output: float
    cache_write: float | None = None


@dataclass(frozen=True)
class ModelPricing:
    provider: Provider
    model: Models
    rates: TokenRates
    source: str
    basis: str = "Text token rates; excludes tool fees and storage."
    checked_on: str = PRICING_CHECKED_ON
    tier_multipliers: tuple[tuple[str, float], ...] = (("default", 1),)
    tier_rates: tuple[tuple[str, TokenRates], ...] = ()
    long_context_threshold: int | None = None
    long_context_threshold_inclusive: bool = False
    long_input_multiplier: float = 1
    long_output_multiplier: float = 1

    @property
    def key(self) -> str:
        return f"{self.provider}::{self.model}"

    def as_record(self) -> dict:
        record = {
            **asdict(self.rates),
            "source": self.source,
            "checked_on": self.checked_on,
            "basis": self.basis,
            "tier_multipliers": dict(self.tier_multipliers),
        }
        if self.tier_rates:
            record["tier_rates"] = {tier: asdict(rates) for tier, rates in self.tier_rates}
        if self.long_context_threshold is not None:
            record.update(
                long_context_threshold=self.long_context_threshold,
                long_context_threshold_inclusive=self.long_context_threshold_inclusive,
                long_input_multiplier=self.long_input_multiplier,
                long_output_multiplier=self.long_output_multiplier,
            )
        return record


_MODEL_PRICING = (
    ModelPricing(
        Provider.CHATGPT,
        Models.GPT6Luna,
        TokenRates(0.10, 0.01, 0.50, cache_write=0.125),
        "https://developers.openai.com/api/docs/models/gpt-6-luna",
        tier_multipliers=(("default", 1), ("flex", 0.5), ("fast", 2), ("priority", 2)),
        long_context_threshold=272000,
        long_input_multiplier=2,
        long_output_multiplier=1.5,
    ),
    ModelPricing(
        Provider.CHATGPT,
        Models.GPT56Luna,
        TokenRates(0.20, 0.02, 1.20),
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    ),
    ModelPricing(
        Provider.CHATGPT,
        Models.GPT5Mini,
        TokenRates(0.25, 0.025, 2.00),
        "https://developers.openai.com/api/docs/models/gpt-5-mini",
    ),
    ModelPricing(
        Provider.CHATGPT,
        Models.GPT5Nano,
        TokenRates(0.05, 0.005, 0.40),
        "https://developers.openai.com/api/docs/models/gpt-5-nano",
    ),
    ModelPricing(
        Provider.CHATGPT,
        Models.GPT41Mini,
        TokenRates(0.40, 0.10, 1.60),
        "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    ),
    ModelPricing(
        Provider.GEMINI,
        Models.GEMINI35FlashLite,
        TokenRates(0.30, 0.03, 2.50),
        "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite",
        basis="Paid-tier text rates; excludes explicit cache storage and tool fees.",
        tier_rates=(
            ("batch", TokenRates(0.15, 0.02, 1.25)),
            ("flex", TokenRates(0.15, 0.02, 1.25)),
            ("priority", TokenRates(0.54, 0.05, 4.50)),
        ),
    ),
    ModelPricing(
        Provider.GROK,
        Models.GROKBuild01,
        TokenRates(1.00, 0.20, 2.00),
        "https://docs.x.ai/developers/pricing",
        basis="Global endpoint text rates. Coding model; provider-reported cost takes precedence.",
        long_context_threshold=200000,
        long_context_threshold_inclusive=True,
        long_input_multiplier=2,
        long_output_multiplier=2,
    ),
    *(
        ModelPricing(
            Provider.OPENROUTER,
            model,
            TokenRates(0, 0, 0, cache_write=0),
            f"https://openrouter.ai/{model}",
            basis="Free endpoint token rates; availability and rate limits still apply.",
        )
        for model in (Models.NEMOTRON3Super120B, Models.STEP35FLASH)
    ),
    ModelPricing(
        Provider.LOCAL,
        Models.LLAMA318B,
        TokenRates(0, 0, 0, cache_write=0),
        "https://docs.ollama.com/api",
        basis="Local inference has no provider API token charge; excludes hardware and energy.",
    ),
)

MODEL_PRICING = MappingProxyType({entry.model: entry for entry in _MODEL_PRICING})


def default_pricing() -> dict:
    """Return fresh JSON-compatible records, isolated from the immutable catalog."""
    return {entry.key: entry.as_record() for entry in MODEL_PRICING.values()}


def load_pricing(path: str | Path | None) -> dict | None:
    if not path:
        return None
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_pricing(overrides)
    return {**default_pricing(), **overrides}


def _validate_rates(rate: dict) -> None:
    if not isinstance(rate, dict):
        raise ValueError("Token rates must be objects.")
    for key in ("input", "cached_input", "cache_write", "output"):
        value = rate.get(key)
        if key == "cache_write" and key in rate and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Pricing '{key}' must be a finite nonnegative USD-per-million rate.")


def validate_pricing(pricing: dict) -> None:
    if not isinstance(pricing, dict) or any(
        not isinstance(rate, dict) for rate in pricing.values()
    ):
        raise ValueError("Pricing must map provider::model identifiers to rate objects.")
    for rate in pricing.values():
        _validate_rates(rate)
        if not rate.get("source") or not rate.get("checked_on"):
            raise ValueError("Pricing requires a source and checked_on date.")
        for key in ("tier_multipliers", "tier_rates"):
            if not isinstance(rate.get(key, {}), dict):
                raise ValueError(f"Pricing '{key}' must map service tiers to values.")
        for tier_rate in rate.get("tier_rates", {}).values():
            _validate_rates(tier_rate)
        multipliers = [
            *rate.get("tier_multipliers", {}).values(),
            rate.get("long_input_multiplier", 1),
            rate.get("long_output_multiplier", 1),
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in multipliers
        ):
            raise ValueError("Pricing multipliers must be finite and nonnegative.")
        if "long_context_threshold" in rate:
            threshold = rate["long_context_threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
                raise ValueError("The long-context threshold must be a positive integer.")
        if not isinstance(rate.get("long_context_threshold_inclusive", False), bool):
            raise ValueError("The long-context threshold inclusive flag must be boolean.")


def estimate_cost(record: dict, pricing: dict) -> tuple[float | None, dict | None]:
    rate = pricing.get(f"{record['provider']}::{record['requested_model']}")
    if not rate or record["status"] != "response" or record["input_tokens"] is None:
        return None, rate
    service_tier = record["service_tier"]
    tier_rates = rate.get("tier_rates", {}).get(service_tier)
    tier = 1 if tier_rates is not None else rate.get("tier_multipliers", {}).get(service_tier)
    if tier is None or record["output_tokens"] is None:
        return None, rate
    effective = tier_rates if tier_rates is not None else rate
    incoming = record["input_tokens"]
    cached, written = record["cached_input_tokens"], record["cache_write_tokens"]
    if min(incoming, cached, written, record["output_tokens"]) < 0 or cached + written > incoming:
        return None, rate
    if written and effective["cache_write"] is None:
        return None, rate
    threshold = rate.get("long_context_threshold", float("inf"))
    long_context = incoming > threshold or (
        rate.get("long_context_threshold_inclusive", False) and incoming == threshold
    )
    input_multiplier = rate.get("long_input_multiplier", 1) if long_context else 1
    output_multiplier = rate.get("long_output_multiplier", 1) if long_context else 1
    cost = (
        (
            (
                (incoming - cached - written) * effective["input"]
                + cached * effective["cached_input"]
                + written * (effective["cache_write"] or 0)
            )
            * input_multiplier
            + record["output_tokens"] * effective["output"] * output_multiplier
        )
        * tier
        / 1_000_000
    )
    return cost, rate
