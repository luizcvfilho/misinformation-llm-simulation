from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext as _null_operation
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from misinformation_simulation.config.prompts import (
    REWRITE_SYSTEM_INSTRUCTION,
    STRUCTURED_EXTRACTION_VERSION,
    resolve_graph_personality_prompt,
    resolve_graph_rewrite_prompt,
)
from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
    resolve_output_language,
    resolve_output_language_name,
    resolve_row_text,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import create_llm_client, normalize_provider
from misinformation_simulation.llm.progress import (
    emit_event,
    mark_cached,
    mark_failed,
    operation,
    progress_context,
    set_progress_context,
    timed_operation,
    track_run_progress,
)
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.llm.service_tier import (
    configured_openai_service_tier,
    with_openai_service_tier,
)
from misinformation_simulation.llm.usage import track_run_usage, usage_stage
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.paths import project_graph_paths
from misinformation_simulation.simulation.persistence import _persist_results
from misinformation_simulation.simulation.progress import simulation_plan, timing_key
from misinformation_simulation.simulation.topology import (
    _node_label,
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
    _root_to_leaf_paths,
    _topological_path,
)
from misinformation_simulation.simulation.types import (
    SimulationEdge,
    SimulationNode,
    SimulationResult,
    SimulationStepResult,
)
from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
    VADModelBundle,
    VADScore,
    predict_text_vad,
)
from misinformation_simulation.text_metrics.vad_evaluation import (
    VADTextEvaluator,
    compare_vad_evaluations,
    evaluation_score,
    vad_for_branch,
    validate_vad_method,
)
from misinformation_simulation.topic_drift import (
    calculate_stdi,
    extract_topic_structure,
    flatten_topic_structure,
)
from misinformation_simulation.topic_drift.cache_versions import runtime_cache_contracts
from misinformation_simulation.topic_drift.cluster_comparison import (
    CLUSTER_STDI_COMPARISON_VERSION,
    TextEmbedder,
    TopicStructurePair,
)
from misinformation_simulation.topic_drift.extraction import _build_topic_structure
from misinformation_simulation.topic_drift.models import (
    TopicStructure,
    empty_topic_structure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache
from misinformation_simulation.topic_drift.structured_comparison import (
    DEFAULT_JUDGE_REPEATS,
    DUAL_STDI_VERSION,
    STRUCTURED_CLUSTER_VERSION,
    StructuredEmbeddingComparator,
    compare_dual_stdi,
    complete_stdi,
    validate_judge_repeats,
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


def _score_vad(
    text: str,
    *,
    model_bundle: VADModelBundle | None,
    scorer: Callable[[str], VADScore] | None,
) -> VADScore:
    score = predict_text_vad(text, model_bundle=model_bundle, scorer=scorer)
    if any(getattr(score, dimension) is None for dimension in ("valence", "arousal", "dominance")):
        raise ValueError("VAD scoring must return valence, arousal, and dominance.")
    if not all(
        math.isfinite(getattr(score, dimension))
        for dimension in ("valence", "arousal", "dominance")
    ):
        raise ValueError("VAD scoring must return finite scores.")
    return score


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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")
    if stdi_comparison_method not in {"dual", "cluster", "llm", "lexical"}:
        raise ValueError("'stdi_comparison_method' must be 'dual', 'cluster', 'llm' or 'lexical'.")
    validate_vad_method(vad_method)
    validate_judge_repeats(stdi_judge_repeats)
    cache_directory = stdi_cache_dir
    if (
        cache_directory is None
        and persist_results
        and (stdi_comparison_method in {"dual", "llm"} or vad_method != "model")
    ):
        cache_directory = Path(output_dir or DEFAULT_SIMULATION_OUTPUT_DIR) / "evaluation_cache"
    cache_contracts = runtime_cache_contracts()
    if stdi_judge_fn is not None:
        cache_contracts["judge"]["custom_judge_session"] = uuid4().hex
    evaluation_cache = EvaluationCache(
        cache_directory,
        shared_directory=shared_cache_dir,
        compatibility=cache_contracts,
    )
    vad_cache: dict[str, VADScore] = {}
    vad_errors: dict[str, str] = {}
    judge_limiter = MinuteRateLimiter(max_requests_per_minute)
    vad_provider = vad_llm_provider or topic_drift_provider
    reuse_vad_endpoint = normalize_provider(vad_provider) == normalize_provider(
        topic_drift_provider
    )
    vad_evaluator = VADTextEvaluator(
        method=vad_method,
        model_bundle=vad_model_bundle,
        model_scorer=vad_scorer,
        llm_scorer=vad_llm_scorer,
        llm_model=vad_llm_model or topic_drift_model,
        llm_provider=vad_provider,
        api_key=vad_llm_api_key or (topic_drift_api_key if reuse_vad_endpoint else None),
        base_url=vad_llm_base_url or (topic_drift_base_url if reuse_vad_endpoint else None),
        before_request_hook=judge_limiter.acquire,
        cache=evaluation_cache,
    )

    @timed_operation(
        "extraction", model=topic_drift_model, provider=normalize_provider(topic_drift_provider)
    )
    def extract_structure(**kwargs: Any) -> TopicStructure:
        if stdi_comparison_method not in {"dual", "cluster", "llm"}:
            return extract_topic_structure(**kwargs)
        inputs = {
            key: str(value) if key in {"model", "provider"} else value
            for key, value in kwargs.items()
            if key in {"text", "title", "model", "provider", "base_url"}
        }
        inputs["version"] = STRUCTURED_EXTRACTION_VERSION
        saved = evaluation_cache.get("extraction", inputs)
        if saved is not None:
            mark_cached()
            return _build_topic_structure(saved)
        try:
            structure = extract_topic_structure(
                **kwargs,
                structured=True,
                before_request_hook=judge_limiter.acquire,
            )
        except Exception as exc:
            mark_failed(str(exc))
            structure = empty_topic_structure()
            structure.schema_version = 2
            structure.extraction_status = "failed"
            structure.extraction_issues = [str(exc)]
            structure.provenance = getattr(exc, "provenance", {})
            return structure
        if structure.extraction_status == "valid":
            evaluation_cache.put("extraction", inputs, topic_structure_to_dict(structure))
        if structure.extraction_status == "failed":
            mark_failed("; ".join(structure.extraction_issues))
        return structure

    vad_timing_model = (
        f"{vad_method}:{vad_llm_model or topic_drift_model}"
        if vad_method != "model"
        else DEFAULT_VAD_MODEL_NAME
    )

    @timed_operation(
        "vad",
        model=vad_timing_model,
        provider=normalize_provider(vad_provider) if vad_method != "model" else "local",
    )
    def score_vad(text: str) -> VADScore:
        memory_hit = text in vad_evaluator.records
        evaluation = vad_evaluator.evaluate(text)
        methods = ("model", "llm") if vad_method == "dual" else (vad_method,)
        if memory_hit or all(evaluation[method].get("cache_hit") for method in methods):
            mark_cached()
        source = "model" if vad_method == "dual" else vad_method
        vad_cache[text] = evaluation_score(evaluation, source)
        required = ("model", "llm") if vad_method == "dual" else (vad_method,)
        errors = [
            evaluation[method].get(
                "error", "VAD scoring must return valence, arousal, and dominance."
            )
            for method in required
            if evaluation[method]["status"] != "valid"
        ]
        if errors:
            mark_failed("; ".join(errors))
            vad_errors[text] = "; ".join(errors)
            if vad_method == "model" and stdi_comparison_method not in {"dual", "llm"}:
                raise ValueError(vad_errors[text])
        return vad_cache[text]

    rewrite_prompt = resolve_graph_rewrite_prompt(rewrite_mode)
    rewrite_metadata = {
        "rewrite_mode": rewrite_prompt.mode,
        "rewrite_prompt_version": rewrite_prompt.version,
        "rewrite_temperature_requested": GRAPH_REWRITE_TEMPERATURE,
        "rewrite_original_title_context": rewrite_prompt.original_title_context,
    }

    nodes_by_id = _normalize_nodes(nodes)
    effective_personalities = {
        node_id: resolve_graph_personality_prompt(node.personality, rewrite_mode=rewrite_mode)
        for node_id, node in nodes_by_id.items()
    }
    normalized_edges = _normalize_edges(nodes_by_id, edges)
    resolved_start_node = _resolve_start_node(nodes_by_id, normalized_edges, start_node_id)
    ordered_node_ids = _topological_path(nodes_by_id, normalized_edges, resolved_start_node)
    graph_paths = _root_to_leaf_paths(ordered_node_ids, normalized_edges)
    parents_by_node = {edge.target: edge.source for edge in normalized_edges}
    depths_by_node: dict[str, int] = {}
    for node_id in ordered_node_ids:
        depths_by_node[node_id] = depths_by_node.get(parents_by_node.get(node_id), 0) + 1
    branching = len(graph_paths) > 1
    shared_node_ids = {
        node_id for node_id in ordered_node_ids if sum(node_id in path for path in graph_paths) > 1
    }
    resolved_text_column = text_column or choose_news_text_column(df)
    rows_to_process = len(df) if max_rows is None else min(len(df), max_rows)
    _emit_progress(
        progress_callback,
        f"Starting interaction graph run with {rows_to_process} row(s) and "
        f"{len(ordered_node_ids)} node(s).",
    )

    clients_by_node_id: dict[str, tuple[str, Any]] = {}
    limiters_by_node_id: dict[str, MinuteRateLimiter] = {}
    with operation("preparation"):
        for node in nodes:
            provider_normalized = normalize_provider(node.provider)
            clients_by_node_id[node.node_id] = (
                provider_normalized,
                create_llm_client(
                    provider=provider_normalized,
                    api_key=node.api_key,
                    base_url=node.base_url,
                )[1],
            )
            limiters_by_node_id[node.node_id] = MinuteRateLimiter(max_requests_per_minute)

    evaluation_metadata = {
        "openai_service_tier": configured_openai_service_tier(),
        "vad_method": vad_method,
        "vad_evaluation_version": "vad_evaluation_v1",
        "vad_llm_model": str(vad_llm_model or topic_drift_model) if vad_method != "model" else None,
        "vad_llm_provider": normalize_provider(vad_llm_provider or topic_drift_provider)
        if vad_method != "model"
        else None,
        "stdi_judge_repeats": stdi_judge_repeats
        if stdi_comparison_method in {"dual", "llm"}
        else None,
        "stdi_judge_aggregation": "mean_complete_metrics"
        if stdi_comparison_method in {"dual", "llm"}
        else None,
        "topic_drift_model": str(topic_drift_model),
        "topic_drift_provider": normalize_provider(topic_drift_provider),
        "topic_extraction_original_title_context": rewrite_prompt.original_title_context,
        **rewrite_metadata,
    }
    step_results: list[SimulationStepResult] = []
    scoring_contexts: list[
        tuple[
            int,
            SimulationStepResult,
            TopicStructure,
            TopicStructure,
            TopicStructure,
            VADScore,
            VADScore,
            VADScore,
        ]
    ] = []
    target_indexes = list(df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]
    total_rows = len(target_indexes)
    emit_event(
        "plan",
        entries=simulation_plan(
            total_rows,
            list(nodes_by_id.values()),
            method=stdi_comparison_method,
            repeats=stdi_judge_repeats,
            model=topic_drift_model,
            provider=normalize_provider(topic_drift_provider),
            vad_model=vad_timing_model,
            vad_provider=normalize_provider(vad_provider) if vad_method != "model" else "local",
            embedding_model=stdi_embedding_model,
            sleep_seconds=sleep_seconds,
            persist=persist_results,
        ),
    )
    rows_started = 0
    cancelled = False
    rewrite_operations_started = 0

    for row_position, row_index in enumerate(target_indexes, start=1):
        if _cancel_requested(cancel_check):
            cancelled = True
            break
        rows_started = row_position
        row = df.loc[row_index]
        news_id = _news_identifier(row_index, row, news_id_column)
        title = ""
        if title_column in row.index and pd.notna(row[title_column]):
            title = str(row[title_column]).strip()
        category = ""
        if "category" in row.index and pd.notna(row["category"]):
            category = str(row["category"]).strip()
        _emit_progress(
            progress_callback,
            f"[{row_position}/{total_rows}] Preparing news '{news_id}' ({title or 'Untitled'}).",
        )

        set_progress_context(news_id=news_id, row=row_position)
        extraction_title = (title or "Untitled") if rewrite_prompt.original_title_context else None
        original_structure_ready = False
        try:
            source_column, original_text = resolve_row_text(
                row=row,
                preferred_column=resolved_text_column,
                allow_title_fallback=allow_title_fallback,
            )
            target_language_code, target_language_source = resolve_output_language(
                row=row,
                original_text=original_text,
            )
            original_structure = extract_structure(
                text=original_text,
                title=extraction_title,
                model=topic_drift_model,
                provider=topic_drift_provider,
                api_key=topic_drift_api_key,
                base_url=topic_drift_base_url,
                max_requests_per_minute=max_requests_per_minute,
                retry_attempts=retry_attempts,
            )
            original_structure_ready = True
            if _cancel_requested(cancel_check):
                cancelled = True
                break
            original_vad = score_vad(original_text)
            _emit_progress(
                progress_callback,
                f"[{row_position}/{total_rows}] Original text ready for '{news_id}'.",
            )
            if work_progress_callback is not None:
                work_progress_callback(row_position, total_rows, 0, len(ordered_node_ids))
        except Exception as exc:
            if _cancel_requested(cancel_check):
                cancelled = True
                break
            _emit_progress(
                progress_callback,
                f"[{row_position}/{total_rows}] Failed to prepare '{news_id}': {exc}",
            )
            for node_id in ordered_node_ids:
                node = nodes_by_id[node_id]
                provider_normalized, _ = clients_by_node_id[node.node_id]
                step_results.append(
                    SimulationStepResult(
                        news_id=news_id,
                        step_index=depths_by_node[node_id],
                        node_id=node.node_id,
                        node_label=_node_label(node),
                        source_node_id=(parents_by_node.get(node_id, "original")),
                        source_node_label="source_unavailable",
                        provider=provider_normalized,
                        model=node.model,
                        personality=node.personality,
                        source_text="",
                        rewritten_text=None,
                        target_language="unknown",
                        target_language_source="unresolved",
                        rewrite_status="blocked",
                        rewrite_error=f"Original text could not be prepared: {exc}",
                        original_topic_structure_status=(
                            "success" if original_structure_ready else "error"
                        ),
                        original_topic_structure_error=(
                            None if original_structure_ready else str(exc)
                        ),
                        original_vad_status=("error" if original_structure_ready else "blocked"),
                        original_vad_error=(
                            str(exc)
                            if original_structure_ready
                            else "Skipped because original preprocessing failed."
                        ),
                        rewritten_topic_structure_status="blocked",
                        rewritten_topic_structure_error=(
                            "Skipped because original preprocessing failed."
                        ),
                        metadata={
                            **evaluation_metadata,
                            "rewrite_effective_personality": effective_personalities[node_id],
                            "title": title,
                            "category": category,
                            "graph_step_id": f"{output_prefix}:{row_position}:{node_id}",
                            **(
                                flatten_topic_structure(original_structure, prefix="original")
                                if original_structure_ready
                                else {}
                            ),
                        },
                    )
                )
            if work_progress_callback is not None:
                work_progress_callback(
                    row_position, total_rows, len(ordered_node_ids), len(ordered_node_ids)
                )
            continue

        previous_node_id = "original"
        previous_node_label = source_column
        previous_text = original_text
        previous_structure = original_structure
        previous_vad = original_vad
        previous_rewritten_text: str | None = None
        cumulative_stdi = 0.0
        initial_state = (
            previous_node_id,
            previous_node_label,
            previous_text,
            previous_structure,
            previous_vad,
            previous_rewritten_text,
            cumulative_stdi,
        )
        states_by_node: dict[str, tuple] = {}
        failed_nodes: set[str] = set()

        for execution_index, node_id in enumerate(ordered_node_ids, start=1):
            if _cancel_requested(cancel_check):
                cancelled = True
                break
            node = nodes_by_id[node_id]
            step_index = depths_by_node[node_id]
            parent_id = parents_by_node.get(node_id)
            inherited_state = states_by_node.get(parent_id, initial_state)
            (
                previous_node_id,
                previous_node_label,
                previous_text,
                previous_structure,
                previous_vad,
                previous_rewritten_text,
                cumulative_stdi,
            ) = inherited_state
            states_by_node[node_id] = inherited_state
            provider_normalized, client = clients_by_node_id[node.node_id]
            node_label = _node_label(node)
            set_progress_context(news_id=news_id, row=row_position, node_id=node_id)
            _emit_progress(
                progress_callback,
                f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                f"node='{node_label}' provider='{provider_normalized}' "
                f"model='{node.model}': rewriting.",
            )
            prompt = rewrite_prompt.template.format(
                personality=effective_personalities[node_id],
                target_language_name=resolve_output_language_name(target_language_code),
                target_language_code=target_language_code,
                title=title or "Untitled",
                original_text=previous_text,
            )

            step_result = SimulationStepResult(
                news_id=news_id,
                step_index=step_index,
                node_id=node.node_id,
                node_label=node_label,
                source_node_id=previous_node_id,
                source_node_label=previous_node_label,
                provider=provider_normalized,
                model=node.model,
                personality=node.personality,
                source_text=previous_text,
                rewritten_text=None,
                target_language=target_language_code,
                target_language_source=target_language_source,
                rewrite_status="not_requested",
                rewrite_error=None,
                original_topic_structure_status=(
                    "success"
                    if stdi_comparison_method not in {"dual", "llm"}
                    or original_structure.extraction_status == "valid"
                    else "error"
                ),
                original_topic_structure_error=(
                    "; ".join(original_structure.extraction_issues) or None
                ),
                original_vad_status="success" if original_text not in vad_errors else "error",
                original_vad_error=(vad_errors.get(original_text)),
                metadata={
                    **evaluation_metadata,
                    "rewrite_effective_personality": effective_personalities[node_id],
                    "rewrite_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "title": title,
                    "category": category,
                    "source_text_column": source_column,
                    "graph_execution_index": execution_index,
                    "graph_step_id": f"{output_prefix}:{row_position}:{node_id}",
                    **(
                        {"original_text": original_text, "extraction_title": extraction_title}
                        if stdi_comparison_method in {"dual", "llm"}
                        else {}
                    ),
                    **flatten_topic_structure(original_structure, prefix="original"),
                },
            )
            _record_vad(step_result, "original", original_vad)
            _record_vad(step_result, "source", previous_vad)
            step_result.metadata["original_vad_evaluation"] = vad_evaluator.records[original_text]
            step_result.metadata["source_vad_evaluation"] = vad_evaluator.records[previous_text]

            rewritten_structure_ready = False
            rewritten_vad_ready = False
            if branching and parent_id in failed_nodes:
                step_result.source_node_id = parent_id
                step_result.source_node_label = _node_label(nodes_by_id[parent_id])
                step_result.source_text = ""
                step_result.rewrite_status = "blocked"
                step_result.rewrite_error = (
                    f"Parent node '{parent_id}' did not complete successfully."
                )
                failed_nodes.add(node_id)
                step_results.append(step_result)
                if work_progress_callback is not None:
                    work_progress_callback(
                        row_position, total_rows, execution_index, len(ordered_node_ids)
                    )
                continue
            try:
                lineage = [node_id]
                while lineage[-1] in parents_by_node:
                    lineage.append(parents_by_node[lineage[-1]])
                rewrite_inputs = {
                    "news_id": news_id,
                    "row_position": row_position,
                    "original_text": original_text,
                    "lineage": list(reversed(lineage)),
                    "prompt": prompt,
                    "system_instruction": rewrite_prompt.system_instruction,
                    "version": rewrite_prompt.version,
                    "temperature": GRAPH_REWRITE_TEMPERATURE,
                    "provider": provider_normalized,
                    "model": str(node.model),
                    "base_url": str(client.base_url)
                    if getattr(client, "base_url", None)
                    else node.base_url,
                }
                with operation("rewrite", model=node.model, provider=provider_normalized):
                    saved_rewrite = (
                        evaluation_cache.get("rewrite", rewrite_inputs) if reuse_rewrites else None
                    )
                    step_result.metadata["rewrite_cache_hit"] = saved_rewrite is not None
                    if saved_rewrite is not None:
                        mark_cached()
                        rewritten_text = saved_rewrite["text"]
                    else:
                        rewrite_operations_started += 1
                        step_result.metadata["rewrite_operation_started"] = True
                        rewritten_text = _generate_rewrite(
                            provider_normalized=provider_normalized,
                            client=client,
                            model=node.model,
                            prompt=prompt,
                            system_instruction=rewrite_prompt.system_instruction,
                            retry_attempts=retry_attempts,
                            limiter=limiters_by_node_id[node.node_id],
                        )
                        if reuse_rewrites:
                            evaluation_cache.put(
                                "rewrite", rewrite_inputs, {"text": rewritten_text}
                            )
                step_result.rewritten_text = rewritten_text
                step_result.rewrite_status = "success"
                if _cancel_requested(cancel_check):
                    cancelled = True
                    break
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': rewrite completed, extracting topic drift."
                    ),
                )

                with (
                    operation(
                        "extraction",
                        model=topic_drift_model,
                        provider=normalize_provider(topic_drift_provider),
                    )
                    if stdi_comparison_method not in {"dual", "llm"}
                    else _null_operation()
                ):
                    rewritten_structure = (
                        extract_structure(
                            text=rewritten_text,
                            title=extraction_title,
                            model=topic_drift_model,
                            provider=topic_drift_provider,
                            api_key=topic_drift_api_key,
                            base_url=topic_drift_base_url,
                            max_requests_per_minute=max_requests_per_minute,
                            retry_attempts=retry_attempts,
                        )
                        if stdi_comparison_method in {"dual", "llm"}
                        else _extract_compared_structure(
                            compared_text=rewritten_text,
                            title=extraction_title,
                            topic_drift_model=topic_drift_model,
                            topic_drift_provider=topic_drift_provider,
                            topic_drift_api_key=topic_drift_api_key,
                            topic_drift_base_url=topic_drift_base_url,
                            max_requests_per_minute=max_requests_per_minute,
                            retry_attempts=retry_attempts,
                        )
                    )
                rewritten_structure_ready = True
                if _cancel_requested(cancel_check):
                    cancelled = True
                    break
                step_result.rewritten_topic_structure_status = (
                    "success"
                    if stdi_comparison_method not in {"dual", "llm"}
                    or rewritten_structure.extraction_status == "valid"
                    else "error"
                )
                step_result.rewritten_topic_structure_error = (
                    "; ".join(rewritten_structure.extraction_issues) or None
                )
                step_result.metadata.update(
                    flatten_topic_structure(rewritten_structure, prefix="rewritten")
                )
                rewritten_vad = score_vad(rewritten_text)
                rewritten_vad_ready = True
                step_result.rewritten_vad_status = (
                    "success" if rewritten_text not in vad_errors else "error"
                )
                step_result.rewritten_vad_error = vad_errors.get(rewritten_text)
                _record_vad(step_result, "rewritten", rewritten_vad)
                step_result.metadata["rewritten_vad_evaluation"] = vad_evaluator.records[
                    rewritten_text
                ]
                for suffix, reference in (
                    ("vs_original", original_text),
                    ("incremental", previous_text),
                ):
                    step_result.metadata[f"vad_evaluation_{suffix}"] = compare_vad_evaluations(
                        vad_evaluator.records[reference],
                        vad_evaluator.records[rewritten_text],
                        method=vad_method,
                    )
                if stdi_comparison_method in {"cluster", "dual", "llm"}:
                    scoring_contexts.append(
                        (
                            row_position,
                            step_result,
                            original_structure,
                            previous_structure,
                            rewritten_structure,
                            original_vad,
                            previous_vad,
                            rewritten_vad,
                        )
                    )
                else:
                    vs_original_metrics = calculate_stdi(
                        original_structure,
                        rewritten_structure,
                        original_vad=original_vad,
                        compared_vad=rewritten_vad,
                    )
                    if previous_rewritten_text is None:
                        incremental_metrics = vs_original_metrics
                    else:
                        incremental_metrics = calculate_stdi(
                            previous_structure,
                            rewritten_structure,
                            original_vad=previous_vad,
                            compared_vad=rewritten_vad,
                        )
                    for suffix, metrics in (
                        ("vs_original", vs_original_metrics),
                        ("incremental", incremental_metrics),
                    ):
                        vad = vad_for_branch(
                            step_result.metadata[f"vad_evaluation_{suffix}"], "embedding"
                        )
                        if vad["status"] == "valid":
                            metrics.update(
                                {
                                    key: round(value, 6)
                                    for key, value in complete_stdi(metrics, vad).items()
                                }
                            )
                        else:
                            metrics.update(dict.fromkeys(STDI_COMPONENTS))
                        _record_stdi_metrics(step_result, suffix=suffix, metrics=metrics)
                    if incremental_metrics["stdi"] is not None:
                        cumulative_stdi += incremental_metrics["stdi"]
                    step_result.stdi_cumulative = round(cumulative_stdi, 6)

                previous_text = rewritten_text
                previous_rewritten_text = rewritten_text
                previous_structure = rewritten_structure
                previous_vad = rewritten_vad
                previous_node_id = node.node_id
                previous_node_label = node_label
                states_by_node[node_id] = (
                    previous_node_id,
                    previous_node_label,
                    previous_text,
                    previous_structure,
                    previous_vad,
                    previous_rewritten_text,
                    cumulative_stdi,
                )
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': rewrite processing completed; "
                        + (
                            "STDI evaluation pending."
                            if stdi_comparison_method in {"cluster", "dual", "llm"}
                            else "STDI evaluation complete; "
                            f"stdi_vs_original={step_result.stdi_vs_original}, "
                            f"stdi_incremental={step_result.stdi_incremental}."
                        )
                    ),
                )
            except Exception as exc:
                failed_nodes.add(node_id)
                step_result.rewrite_status = "error"
                step_result.rewrite_error = str(exc)
                if rewritten_structure_ready and not rewritten_vad_ready:
                    step_result.rewritten_vad_status = "error"
                    step_result.rewritten_vad_error = str(exc)
                elif step_result.rewritten_text is not None:
                    step_result.rewritten_topic_structure_status = "error"
                    step_result.rewritten_topic_structure_error = str(exc)
                    step_result.rewritten_vad_status = "blocked"
                    step_result.rewritten_vad_error = "Skipped because topic extraction failed."
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': error: {exc}"
                    ),
                )

            step_results.append(step_result)
            if work_progress_callback is not None:
                work_progress_callback(
                    row_position, total_rows, execution_index, len(ordered_node_ids)
                )

            with operation("pacing") if sleep_seconds > 0 else _null_operation():
                if sleep_seconds > 0 and _wait_between_steps(sleep_seconds, cancel_check):
                    cancelled = True
                    break

        if cancelled:
            break

    if _cancel_requested(cancel_check):
        cancelled = True
    if not cancelled:
        emit_event("settle", stages=["rewrite", "extraction", "vad", "pacing"])
    pairs = [
        TopicStructurePair(f"{index}:original", original, rewritten)
        for index, (_, _, original, _, rewritten, _, _, _) in enumerate(scoring_contexts)
    ] + [
        TopicStructurePair(f"{index}:incremental", previous, rewritten)
        for index, (_, _, _, previous, rewritten, _, _, _) in enumerate(scoring_contexts)
    ]

    def build_comparator(pairs):
        comparator = StructuredEmbeddingComparator(
            embedder=stdi_embedder,
            embedding_model=stdi_embedding_model,
        )
        try:
            if pairs and stdi_comparison_method != "llm":
                with operation("embeddings", model=stdi_embedding_model):
                    comparator.fit(pairs)
        except Exception:
            if stdi_comparison_method not in {"dual", "llm"}:
                raise
        return comparator

    def finish(comparator=None, *, evaluate=True):
        nonlocal cancelled
        set_progress_context()
        if evaluate and stdi_comparison_method in {"dual", "llm"}:
            emit_event(
                "plan",
                entries=[
                    dict(
                        key=timing_key(
                            "judge", topic_drift_model, normalize_provider(topic_drift_provider)
                        ),
                        total=len(scoring_contexts) * 2 * stdi_judge_repeats,
                    )
                ],
            )
        if evaluate and scoring_contexts:
            _emit_progress(
                progress_callback,
                f"Preparing STDI evaluations for {len(scoring_contexts)} processed step(s).",
            )
            comparator = comparator or build_comparator(pairs)
            last_row_position = 0
            cumulative_stdi = 0.0
            valid_steps = 0
            branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
            branch_valid_steps = {"cluster": 0, "llm_judge": 0}
            totals_by_node: dict[str, tuple] = {}
            for (
                row_position,
                step,
                original,
                previous,
                rewritten,
                original_vad,
                previous_vad,
                rewritten_vad,
            ) in scoring_contexts:
                set_progress_context(news_id=step.news_id, row=row_position, node_id=step.node_id)
                if stdi_comparison_method in {"dual", "llm"} and _cancel_requested(cancel_check):
                    cancelled = True
                    break
                if row_position != last_row_position:
                    cumulative_stdi = 0.0
                    valid_steps = 0
                    branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
                    branch_valid_steps = {"cluster": 0, "llm_judge": 0}
                    last_row_position = row_position
                    totals_by_node = {}
                if branching:
                    parent_totals = totals_by_node.get(parents_by_node.get(step.node_id))
                    if parent_totals is None:
                        cumulative_stdi, valid_steps = 0.0, 0
                        branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
                        branch_valid_steps = {"cluster": 0, "llm_judge": 0}
                    else:
                        cumulative_stdi, valid_steps, saved_cumulative, saved_valid = parent_totals
                        branch_cumulative = saved_cumulative.copy()
                        branch_valid_steps = saved_valid.copy()
                if stdi_comparison_method in {"dual", "llm"}:
                    _emit_progress(
                        progress_callback,
                        f"Evaluating {stdi_comparison_method} STDI for '{step.news_id}' "
                        f"node '{step.node_id}'.",
                    )
                    common = {
                        "modified_text": step.rewritten_text,
                        "modified_structure": rewritten,
                        "modified_vad": rewritten_vad,
                        "title": step.metadata.get("extraction_title"),
                        "comparator": comparator,
                        "model": topic_drift_model,
                        "provider": normalize_provider(topic_drift_provider),
                        "api_key": topic_drift_api_key,
                        "base_url": topic_drift_base_url,
                        "retry_attempts": retry_attempts,
                        "before_request_hook": judge_limiter.acquire,
                        "judge_repeats": stdi_judge_repeats,
                        "cancel_check": cancel_check,
                        "cache": evaluation_cache,
                        "comparison_method": stdi_comparison_method,
                        **({"judge_fn": stdi_judge_fn} if stdi_judge_fn is not None else {}),
                    }
                    set_progress_context(
                        news_id=step.news_id,
                        row=row_position,
                        node_id=step.node_id,
                        comparison="vs original",
                    )
                    vs_original_dual = compare_dual_stdi(
                        original_text=step.metadata["original_text"],
                        original_structure=original,
                        original_vad=original_vad,
                        vad_evaluation=step.metadata["vad_evaluation_vs_original"],
                        **common,
                    )
                    if _cancel_requested(cancel_check):
                        cancelled = True
                        _record_dual_metrics(step, "vs_original", vs_original_dual)
                        break
                    set_progress_context(
                        news_id=step.news_id,
                        row=row_position,
                        node_id=step.node_id,
                        comparison="vs previous",
                    )
                    incremental_dual = compare_dual_stdi(
                        original_text=step.source_text,
                        original_structure=previous,
                        original_vad=previous_vad,
                        vad_evaluation=step.metadata["vad_evaluation_incremental"],
                        **common,
                    )
                    _record_dual_metrics(step, "vs_original", vs_original_dual)
                    _record_dual_metrics(step, "incremental", incremental_dual)
                    _emit_progress(
                        progress_callback,
                        f"News '{step.news_id}' node '{step.node_id}': STDI evaluation complete; "
                        f"vs_original={vs_original_dual['status']}, "
                        f"incremental={incremental_dual['status']}.",
                    )
                    if incremental_dual["stdi"] is not None:
                        cumulative_stdi += incremental_dual["stdi"]
                        valid_steps += 1
                    step.stdi_cumulative = cumulative_stdi
                    step.stdi_cumulative_valid_steps = valid_steps
                    step.stdi_chain_complete = valid_steps == step.step_index
                    for branch, name in (("embedding", "cluster"), ("llm_judge", "llm_judge")):
                        metrics = incremental_dual[branch].get("metrics")
                        if metrics is not None:
                            branch_cumulative[name] += metrics["stdi"]
                            branch_valid_steps[name] += 1
                        setattr(step, f"stdi_{name}_cumulative", branch_cumulative[name])
                        setattr(
                            step, f"stdi_{name}_cumulative_valid_steps", branch_valid_steps[name]
                        )
                        setattr(
                            step,
                            f"stdi_{name}_chain_complete",
                            branch_valid_steps[name] == step.step_index,
                        )
                    totals_by_node[step.node_id] = (
                        cumulative_stdi,
                        valid_steps,
                        branch_cumulative.copy(),
                        branch_valid_steps.copy(),
                    )
                    continue
                vs_original = calculate_stdi(
                    original,
                    rewritten,
                    original_vad=original_vad,
                    compared_vad=rewritten_vad,
                    component_overrides=comparator.compare(original, rewritten).component_drifts,
                )
                incremental = calculate_stdi(
                    previous,
                    rewritten,
                    original_vad=previous_vad,
                    compared_vad=rewritten_vad,
                    component_overrides=comparator.compare(previous, rewritten).component_drifts,
                )
                for suffix, metrics in (("vs_original", vs_original), ("incremental", incremental)):
                    vad = vad_for_branch(step.metadata[f"vad_evaluation_{suffix}"], "embedding")
                    step.metadata[f"stdi_vad_source_{suffix}"] = vad["source"]
                    setattr(step, f"stdi_status_{suffix}", vad["status"])
                    if vad["status"] == "valid":
                        metrics.update(
                            {
                                key: round(value, 6)
                                for key, value in complete_stdi(metrics, vad).items()
                            }
                        )
                    else:
                        metrics.update(dict.fromkeys(STDI_COMPONENTS))
                _record_stdi_metrics(step, suffix="vs_original", metrics=vs_original)
                _record_stdi_metrics(step, suffix="incremental", metrics=incremental)
                if incremental["stdi"] is not None:
                    cumulative_stdi += incremental["stdi"]
                step.stdi_cumulative = round(cumulative_stdi, 6)
                totals_by_node[step.node_id] = (
                    cumulative_stdi,
                    valid_steps,
                    branch_cumulative.copy(),
                    branch_valid_steps.copy(),
                )

        if _cancel_requested(cancel_check):
            cancelled = True
        success_count = sum(1 for step in step_results if step.rewrite_status == "success")
        error_count = sum(1 for step in step_results if step.rewrite_status == "error")
        vad_model_name = DEFAULT_VAD_MODEL_NAME
        if vad_model_bundle is not None:
            vad_model_name = vad_model_bundle.model_name
        if vad_scorer is not None:
            vad_model_name = "custom_scorer"
        if vad_method == "llm":
            vad_model_name = None
        embedding_model_name = (
            (
                stdi_embedding_model
                if stdi_embedder is None
                else f"custom:{type(stdi_embedder).__name__}"
            )
            if stdi_comparison_method in {"cluster", "dual"}
            else None
        )
        comparison_version = (
            DUAL_STDI_VERSION
            if stdi_comparison_method in {"dual", "llm"}
            else (
                STRUCTURED_CLUSTER_VERSION
                if any(
                    structure.schema_version == 2
                    for context in scoring_contexts
                    for structure in (context[2], context[3], context[4])
                )
                else CLUSTER_STDI_COMPARISON_VERSION
            )
            if stdi_comparison_method == "cluster"
            else "lexical_v1"
        )
        for step in step_results:
            identity = progress_context()
            if run_id := identity.get("run_id"):
                step.metadata.update(
                    run_id=run_id,
                    dataset_row=identity["dataset_row"],
                    graph_step_id=f"{run_id}:{step.node_id}",
                )
            step.metadata.update(
                {
                    "rewrite_model": step.model,
                    "rewrite_provider": step.provider,
                    "vad_model": vad_model_name,
                    "stdi_embedding_model": embedding_model_name,
                    "stdi_comparison_version": comparison_version,
                    "graph_shared_node": step.node_id in shared_node_ids,
                    "graph_step_reused": False,
                    "rewrite_operation_attributed": step.metadata.get(
                        "rewrite_operation_started", False
                    ),
                }
            )
        summary = {
            "partial": not evaluate or cancelled,
            "schema_version": GRAPH_STEP_SCHEMA_VERSION,
            "cache_contracts": cache_contracts,
            "shared_cache_dir": str(shared_cache_dir) if shared_cache_dir is not None else None,
            "reuse_rewrites": reuse_rewrites,
            **evaluation_metadata,
            "rewrite_system_instruction": rewrite_prompt.system_instruction,
            "rewrite_prompt_template": rewrite_prompt.template,
            "rows_processed": rows_started,
            "cancelled": cancelled,
            "steps_total": len(step_results),
            "steps_success": success_count,
            "steps_error": error_count,
            "steps_blocked": sum(step.rewrite_status == "blocked" for step in step_results),
            "result_kind": "branching_graph" if branching else "chain",
            "rewrite_operations_started": rewrite_operations_started,
            "rewrite_reuse_scope": "current_graph_run",
            "planned_unique_rewrites": total_rows * len(ordered_node_ids),
            "planned_linear_rewrites": total_rows * sum(len(path) for path in graph_paths),
            "planned_rewrites_saved": total_rows
            * (sum(len(path) for path in graph_paths) - len(ordered_node_ids)),
            "vad_model": vad_model_name,
            "stdi_comparison_method": stdi_comparison_method,
            "stdi_comparison_version": comparison_version,
            "stdi_embedding_model": embedding_model_name,
            "stdi_complete_incremental_pairs": sum(
                step.stdi_status_incremental == "valid" for step in step_results
            )
            if stdi_comparison_method in {"dual", "llm"}
            else None,
            "stdi_components_branch": "selected_evaluation",
            "graph": {
                "start_node_id": resolved_start_node,
                "ordered_node_ids": ordered_node_ids,
                "edges": [asdict(edge) for edge in normalized_edges],
                "paths": graph_paths,
            },
            "nodes": [asdict(node) for node in nodes],
        }
        result = SimulationResult(summary=summary, step_results=step_results)
        if branching:
            result.path_results = project_graph_paths(result)
            summary["path_results"] = [
                {
                    "path_id": path.summary["path_id"],
                    "chain_code": path.summary["chain_code"],
                    "node_ids": path.summary["graph"]["ordered_node_ids"],
                }
                for path in result.path_results
            ]

        if persist_results:
            resolved_output_dir = (
                Path(output_dir) if output_dir is not None else DEFAULT_SIMULATION_OUTPUT_DIR
            )
            set_progress_context()
            with operation("saving") if evaluate else _null_operation():
                summary_path, steps_path = _persist_results(
                    result=result,
                    output_dir=resolved_output_dir,
                    output_prefix=output_prefix,
                )
            result.summary_path = summary_path
            result.steps_path = steps_path
            if evaluate:
                _emit_progress(
                    progress_callback,
                    f"Saved summary to {summary_path} and step records to {steps_path}.",
                )

        if evaluate:
            emit_event("run_finished", cancelled=cancelled)
            _emit_progress(
                progress_callback,
                (
                    "Run finished: "
                    f"rows_processed={summary['rows_processed']}, "
                    f"steps_total={summary['steps_total']}, "
                    f"steps_success={summary['steps_success']}, "
                    f"steps_error={summary['steps_error']}."
                ),
            )

        return result

    if defer_evaluation:
        return PreparedSimulation(
            pairs, finish, build_comparator, checkpoint=lambda: finish(evaluate=False)
        )
    return finish()
