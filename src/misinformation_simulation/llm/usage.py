from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from misinformation_simulation.llm.pricing import default_pricing, estimate_cost, validate_pricing
from misinformation_simulation.llm.progress import emit_event
from misinformation_simulation.llm.service_tier import configured_openai_service_tier

_tracker: ContextVar[UsageTracker | None] = ContextVar("llm_usage_tracker", default=None)
_stage: ContextVar[str] = ContextVar("llm_usage_stage", default="other")


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return vars(value) if hasattr(value, "__dict__") else {}


def provider_for_client(client: Any) -> str:
    configuration = getattr(client, "_recovery_configuration", None)
    if isinstance(configuration, dict) and configuration.get("provider"):
        return configuration["provider"]
    host = urlparse(str(getattr(client, "base_url", ""))).hostname
    return {
        "api.openai.com": "chatgpt",
        "openrouter.ai": "openrouter",
        "api.deepseek.com": "deepseek",
        "api.x.ai": "grok",
        "127.0.0.1": "local",
        "localhost": "local",
    }.get(host, "unknown")


def response_tier(response: Any, usage: dict, provider: str) -> str:
    tier = getattr(response, "service_tier", None)
    if tier is not None:
        return str(tier).lower()
    if provider == "gemini":
        # This application's generate_content calls use standard traffic by default.
        traffic = usage.get("traffic_type")
        return {
            None: "default",
            "TRAFFIC_TYPE_UNSPECIFIED": "default",
            "ON_DEMAND": "default",
            "ON_DEMAND_FLEX": "flex",
            "ON_DEMAND_PRIORITY": "priority",
            "BATCH": "batch",
        }.get(traffic, "unknown")
    return "default" if provider in {"grok", "openrouter", "local"} else "unknown"


class UsageTracker:
    def __init__(self, path: Path | None = None, pricing: dict | None = None):
        self.path = path
        self.pricing = default_pricing() if pricing is None else deepcopy(pricing)
        validate_pricing(self.pricing)
        self.run_id = uuid4().hex
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": _stage.get(),
            **record,
        }
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def response(self, response: Any, *, provider: str, model: str) -> None:
        usage = _as_dict(
            getattr(response, "usage_metadata" if provider == "gemini" else "usage", None)
        )
        details = _as_dict(usage.get("prompt_tokens_details") or usage.get("input_tokens_details"))
        completion = _as_dict(
            usage.get("completion_tokens_details") or usage.get("output_tokens_details")
        )
        record = {
            "kind": "request",
            "status": "response",
            "provider": provider,
            "requested_model": str(model),
            "returned_model": getattr(response, "model", getattr(response, "model_version", None)),
            "response_id": getattr(response, "id", None) or getattr(response, "response_id", None),
            "service_tier": response_tier(response, usage, provider),
            "requested_service_tier": configured_openai_service_tier()
            if provider == "chatgpt"
            else None,
            "input_tokens": usage.get("prompt_token_count")
            if provider == "gemini"
            else (
                usage.get("prompt_tokens")
                if usage.get("prompt_tokens") is not None
                else usage.get("input_tokens")
            ),
            "output_tokens": usage.get("completion_tokens")
            if usage.get("completion_tokens") is not None
            else usage.get("output_tokens"),
            "cached_input_tokens": usage.get("cached_content_token_count", 0) or 0
            if provider == "gemini"
            else details.get("cached_tokens", 0) or 0,
            "cache_write_tokens": details.get("cache_write_tokens", 0) or 0,
            "reasoning_tokens": usage.get("thoughts_token_count", 0) or 0
            if provider == "gemini"
            else completion.get("reasoning_tokens", 0) or 0,
            "raw_usage": usage,
        }
        if provider == "gemini" and usage.get("candidates_token_count") is not None:
            record["output_tokens"] = usage["candidates_token_count"] + record["reasoning_tokens"]
        record["estimated_cost_usd"], record["pricing"] = estimate_cost(record, self.pricing)
        record["cost_source"] = (
            "token_rates" if record["estimated_cost_usd"] is not None else "unavailable"
        )
        ticks = usage.get("cost_in_usd_ticks")
        if (
            provider == "grok"
            and isinstance(ticks, int)
            and not isinstance(ticks, bool)
            and ticks >= 0
        ):
            record.update(
                provider_cost_in_usd_ticks=ticks,
                provider_reported_cost_usd=ticks / 10_000_000_000,
                estimated_cost_usd=ticks / 10_000_000_000,
                cost_source="provider_reported",
                cost_reference="https://docs.x.ai/developers/cost-tracking",
            )
        self.append(record)

    def summary(self) -> dict:
        stages = {}
        for stage in sorted({record["stage"] for record in self.records}):
            records = [record for record in self.records if record["stage"] == stage]
            requests = [record for record in records if record["kind"] == "request"]
            known = sum(record.get("estimated_cost_usd") or 0 for record in requests)
            unknown = sum(record.get("estimated_cost_usd") is None for record in requests)
            stages[stage] = {
                "requests": len(requests),
                "failed_requests": sum(record["status"] == "error" for record in requests),
                "cache_hits": sum(record["kind"] == "cache_hit" for record in records),
                "unpriced_requests": unknown,
                "provider_reported_requests": sum(
                    record.get("cost_source") == "provider_reported" for record in requests
                ),
                "known_estimated_cost_usd": known,
                "estimated_cost_usd": None if unknown else known,
                **{
                    key: sum(record.get(key) or 0 for record in requests)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cached_input_tokens",
                        "cache_write_tokens",
                        "reasoning_tokens",
                    )
                },
            }
        unknown = sum(stage["unpriced_requests"] for stage in stages.values())
        known = sum(stage["known_estimated_cost_usd"] for stage in stages.values())
        return {
            "version": "api_usage_v1",
            "currency": "USD",
            "stages": stages,
            "run_id": self.run_id,
            "requests": sum(stage["requests"] for stage in stages.values()),
            "cache_hits": sum(stage["cache_hits"] for stage in stages.values()),
            "unpriced_requests": unknown,
            "provider_reported_requests": sum(
                stage["provider_reported_requests"] for stage in stages.values()
            ),
            "known_estimated_cost_usd": known,
            "estimated_cost_usd": None if unknown else known,
            "note": "Costs use provider-reported amounts when available, otherwise recorded usage "
            "and dated rates. Token-rate estimates do not replace invoices. "
            "Unknown pricing and failed requests are not treated as free.",
        }


@contextmanager
def collect_usage(tracker: UsageTracker):
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


def combine_usage(summaries: list[dict]) -> dict:
    stages = {}
    for summary in summaries:
        for name, metrics in summary["stages"].items():
            combined = stages.setdefault(name, {})
            for key, value in metrics.items():
                if key != "estimated_cost_usd":
                    combined[key] = combined.get(key, 0) + value
    for metrics in stages.values():
        metrics["estimated_cost_usd"] = (
            None if metrics["unpriced_requests"] else metrics["known_estimated_cost_usd"]
        )
    unknown = sum(summary["unpriced_requests"] for summary in summaries)
    known = sum(summary["known_estimated_cost_usd"] for summary in summaries)
    return {
        "version": "api_usage_v1",
        "currency": "USD",
        "stages": stages,
        "graph_run_ids": [summary["run_id"] for summary in summaries],
        "requests": sum(summary["requests"] for summary in summaries),
        "cache_hits": sum(summary["cache_hits"] for summary in summaries),
        "unpriced_requests": unknown,
        "provider_reported_requests": sum(
            summary.get("provider_reported_requests", 0) for summary in summaries
        ),
        "known_estimated_cost_usd": known,
        "estimated_cost_usd": None if unknown else known,
    }


def usage_stage(stage: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            token = _stage.set(stage)
            try:
                return function(*args, **kwargs)
            finally:
                _stage.reset(token)

        return wrapped

    return decorate


def record_response(response: Any, *, provider: str, model: str) -> None:
    if tracker := _tracker.get():
        tracker.response(response, provider=provider, model=model)


@contextmanager
def api_request(*, provider: str, model: str):
    emit_event("request_started", requested_model=str(model), requested_provider=provider)
    try:
        yield
    except Exception as error:
        record_request_error(error, provider=provider, model=model)
        emit_event("request_finished", status="failed", reason=type(error).__name__)
        raise
    else:
        emit_event("request_finished", status="response")


def record_request_error(error: Exception, *, provider: str, model: str) -> None:
    if tracker := _tracker.get():
        tracker.append(
            {
                "kind": "request",
                "status": "error",
                "provider": provider,
                "requested_model": str(model),
                "exception_type": type(error).__name__,
                "estimated_cost_usd": None,
            }
        )


def record_cache_hit(namespace: str, *, source: str) -> None:
    if tracker := _tracker.get():
        tracker.append({"kind": "cache_hit", "stage": namespace, "source": source})


def track_run_usage(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        directory = None
        if kwargs.get("persist_results", True):
            from misinformation_simulation.simulation.graph import DEFAULT_SIMULATION_OUTPUT_DIR

            directory = Path(kwargs.get("output_dir") or DEFAULT_SIMULATION_OUTPUT_DIR)
        prefix = kwargs.get("output_prefix", "simulation")
        path = directory / f"{prefix}_api_usage.jsonl" if directory is not None else None
        tracker = UsageTracker(path, kwargs.get("api_pricing"))
        result = None
        try:
            with collect_usage(tracker):
                result = function(*args, **kwargs)
            return result
        finally:
            summary = tracker.summary()
            if directory is not None and (result is not None or tracker.records):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{prefix}_api_costs.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
                )
            if result is not None:
                result.summary["api_usage"] = summary
                if result.summary_path is not None:
                    result.summary_path.write_text(
                        json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
                    )

    return wrapped
