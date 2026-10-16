from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from misinformation_simulation.config.prompts import (
    REWRITE_SYSTEM_INSTRUCTION,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.progress import (
    track_run_progress,
)
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.llm.service_tier import (
    with_openai_service_tier,
)
from misinformation_simulation.llm.usage import track_run_usage, usage_stage
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.topology import (
    _normalize_edges as _normalize_edges,
)
from misinformation_simulation.simulation.topology import (
    _normalize_nodes as _normalize_nodes,
)
from misinformation_simulation.simulation.topology import (
    _resolve_start_node as _resolve_start_node,
)
from misinformation_simulation.simulation.topology import (
    _topological_path as _topological_path,
)
from misinformation_simulation.simulation.types import (
    SimulationEdge,
    SimulationNode,
    SimulationResult,
    SimulationStepResult,
)
from misinformation_simulation.text_metrics.vad import (
    VADModelBundle,
    VADScore,
)
from misinformation_simulation.topic_drift import (
    extract_topic_structure,
)
from misinformation_simulation.topic_drift.cluster_comparison import (
    TextEmbedder,
)
from misinformation_simulation.topic_drift.models import (
    TopicStructure,
)
from misinformation_simulation.topic_drift.structured_comparison import (
    DEFAULT_JUDGE_REPEATS,
    StructuredEmbeddingComparator,
    compare_dual_stdi,
)

DEFAULT_SIMULATION_OUTPUT_DIR = Path("output") / "interaction_graph"
DEFAULT_STDI_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GRAPH_REWRITE_TEMPERATURE = 0.8
ProgressCallback = Callable[[str], None]
WorkProgressCallback = Callable[[int, int, int, int], None]
CancelCheck = Callable[[], bool]
STDI_COMPONENTS = (
    "stdi",
    "theme_drift",
    "subtopic_drift",
    "entity_drift",
    "relation_drift",
    "contradiction_drift",
    "valence_drift",
    "arousal_drift",
    "dominance_drift",
    "vad_drift",
    "content_drift",
)


@usage_stage("rewrite")
def _generate_rewrite(
    *,
    provider_normalized: str,
    client: Any,
    model: str,
    prompt: str,
    retry_attempts: int,
    limiter: MinuteRateLimiter,
    system_instruction: str = REWRITE_SYSTEM_INSTRUCTION,
) -> str:
    if provider_normalized == "gemini":
        return generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=GRAPH_REWRITE_TEMPERATURE,
            max_attempts=retry_attempts,
            before_request_hook=limiter.acquire,
        )
    return generate_openai_text_with_retry(
        client,
        model=model,
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=GRAPH_REWRITE_TEMPERATURE,
        max_attempts=retry_attempts,
        before_request_hook=limiter.acquire,
    )


def _extract_compared_structure(
    *,
    compared_text: str,
    title: str | None,
    topic_drift_model: str,
    topic_drift_provider: Provider | str,
    topic_drift_api_key: str | None,
    topic_drift_base_url: str | None,
    max_requests_per_minute: int | None,
    retry_attempts: int,
) -> TopicStructure:
    return extract_topic_structure(
        text=compared_text,
        title=title,
        model=topic_drift_model,
        provider=topic_drift_provider,
        api_key=topic_drift_api_key,
        base_url=topic_drift_base_url,
        max_requests_per_minute=max_requests_per_minute,
        retry_attempts=retry_attempts,
    )


def _news_identifier(row_index: Any, row: pd.Series, news_id_column: str | None) -> str:
    if news_id_column and news_id_column in row.index and pd.notna(row[news_id_column]):
        candidate = str(row[news_id_column]).strip()
        if candidate:
            return candidate
    return f"row_{row_index}"


def _emit_progress(
    progress_callback: ProgressCallback | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _cancel_requested(cancel_check: CancelCheck | None) -> bool:
    return cancel_check is not None and cancel_check()


def _wait_between_steps(seconds: float, cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        time.sleep(seconds)
        return False
    deadline = time.monotonic() + seconds
    while not _cancel_requested(cancel_check):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(remaining, 0.1))
    return True


def _record_vad(step: SimulationStepResult, prefix: str, score: VADScore) -> None:
    for dimension in ("valence", "arousal", "dominance"):
        step.metadata[f"{prefix}_vad_{dimension}"] = getattr(score, dimension)


def _record_stdi_metrics(
    step: SimulationStepResult,
    *,
    suffix: str,
    metrics: dict[str, float],
) -> None:
    for component in STDI_COMPONENTS:
        setattr(step, f"{component}_{suffix}", metrics[component])


GRAPH_STEP_SCHEMA_VERSION = 6


def _record_dual_metrics(step: SimulationStepResult, suffix: str, result: dict[str, Any]) -> None:
    step.metadata[f"dual_stdi_{suffix}"] = result
    setattr(step, f"stdi_{suffix}", result["stdi"])
    setattr(step, f"stdi_status_{suffix}", result["status"])
    setattr(step, f"stdi_method_gap_{suffix}", result["method_gap"])
    for branch in ("embedding", "llm_judge"):
        metrics = result[branch].get("metrics")
        setattr(step, f"stdi_{branch}_{suffix}", metrics["stdi"] if metrics else None)
    # New records expose the selected evaluation; historical branch columns stay recoverable.
    embedding = result.get("metrics")
    if embedding:
        for component in STDI_COMPONENTS:
            if component != "stdi":
                setattr(step, f"{component}_{suffix}", embedding[component])


@with_openai_service_tier
@track_run_progress
@track_run_usage
def run_news_interaction_graph(
    df: pd.DataFrame,
    *,
    nodes: list[SimulationNode],
    edges: list[SimulationEdge] | None = None,
    start_node_id: str | None = None,
    text_column: str | None = "description",
    title_column: str = "title",
    news_id_column: str | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = 0.0,
    max_requests_per_minute: int | None = None,
    retry_attempts: int = 5,
    allow_title_fallback: bool = True,
    rewrite_mode: str = "faithful",
    topic_drift_model: str = DEFAULT_LLM_MODEL,
    topic_drift_provider: Provider | str = DEFAULT_LLM_PROVIDER,
    topic_drift_api_key: str | None = None,
    topic_drift_base_url: str | None = None,
    stdi_comparison_method: str = "dual",
    stdi_embedding_model: str = DEFAULT_STDI_EMBEDDING_MODEL,
    stdi_embedder: TextEmbedder | None = None,
    stdi_cache_dir: Path | str | None = None,
    shared_cache_dir: Path | str | None = None,
    reuse_rewrites: bool = False,
    api_pricing: dict | None = None,
    openai_service_tier: str | None = None,
    openai_flex_fallback: bool = False,
    openai_flex_retry_seconds: float = 300,
    stdi_judge_fn: Callable[..., Any] | None = None,
    stdi_judge_repeats: int = DEFAULT_JUDGE_REPEATS,
    vad_model_bundle: VADModelBundle | None = None,
    vad_scorer: Callable[[str], VADScore] | None = None,
    vad_method: str = "model",
    vad_llm_model: str | None = None,
    vad_llm_provider: Provider | str | None = None,
    vad_llm_api_key: str | None = None,
    vad_llm_base_url: str | None = None,
    vad_llm_scorer: Callable[[str], Any] | None = None,
    output_dir: Path | str | None = None,
    output_prefix: str = "simulation",
    persist_results: bool = True,
    progress_callback: ProgressCallback | None = None,
    work_progress_callback: WorkProgressCallback | None = None,
    operation_callback: Callable[[dict], None] | None = None,
    cancel_check: CancelCheck | None = None,
    defer_evaluation: bool = False,
) -> SimulationResult | PreparedSimulation:
    from misinformation_simulation.simulation.graph_execution import _GraphRun
    from misinformation_simulation.simulation.graph_preparation import _GraphRunOptions

    return _GraphRun(
        df,
        _GraphRunOptions(
            nodes=nodes,
            edges=edges,
            start_node_id=start_node_id,
            text_column=text_column,
            title_column=title_column,
            news_id_column=news_id_column,
            max_rows=max_rows,
            sleep_seconds=sleep_seconds,
            max_requests_per_minute=max_requests_per_minute,
            retry_attempts=retry_attempts,
            allow_title_fallback=allow_title_fallback,
            rewrite_mode=rewrite_mode,
            topic_drift_model=topic_drift_model,
            topic_drift_provider=topic_drift_provider,
            topic_drift_api_key=topic_drift_api_key,
            topic_drift_base_url=topic_drift_base_url,
            stdi_comparison_method=stdi_comparison_method,
            stdi_embedding_model=stdi_embedding_model,
            stdi_embedder=stdi_embedder,
            stdi_cache_dir=stdi_cache_dir,
            shared_cache_dir=shared_cache_dir,
            reuse_rewrites=reuse_rewrites,
            stdi_judge_fn=stdi_judge_fn,
            stdi_judge_repeats=stdi_judge_repeats,
            vad_model_bundle=vad_model_bundle,
            vad_scorer=vad_scorer,
            vad_method=vad_method,
            vad_llm_model=vad_llm_model,
            vad_llm_provider=vad_llm_provider,
            vad_llm_api_key=vad_llm_api_key,
            vad_llm_base_url=vad_llm_base_url,
            vad_llm_scorer=vad_llm_scorer,
            output_dir=output_dir,
            output_prefix=output_prefix,
            persist_results=persist_results,
            progress_callback=progress_callback,
            work_progress_callback=work_progress_callback,
            cancel_check=cancel_check,
            defer_evaluation=defer_evaluation,
        ),
        comparator_factory=StructuredEmbeddingComparator,
        _extract_compared_structure=_extract_compared_structure,
        _generate_rewrite=_generate_rewrite,
        compare_dual_stdi=compare_dual_stdi,
        create_llm_client=create_llm_client,
        extract_topic_structure=extract_topic_structure,
    ).run()
