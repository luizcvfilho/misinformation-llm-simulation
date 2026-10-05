from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

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
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.simulation.persistence import _persist_results
from misinformation_simulation.simulation.topology import (
    _node_label,
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
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
from misinformation_simulation.topic_drift import (
    calculate_stdi,
    extract_topic_structure,
    flatten_topic_structure,
)
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
    DUAL_STDI_VERSION,
    STRUCTURED_CLUSTER_VERSION,
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


GRAPH_STEP_SCHEMA_VERSION = 3


def _record_dual_metrics(step: SimulationStepResult, suffix: str, result: dict[str, Any]) -> None:
    step.metadata[f"dual_stdi_{suffix}"] = result
    setattr(step, f"stdi_{suffix}", result["stdi"])
    setattr(step, f"stdi_status_{suffix}", result["status"])
    setattr(step, f"stdi_method_gap_{suffix}", result["method_gap"])
    for branch in ("embedding", "llm_judge"):
        metrics = result[branch].get("metrics")
        setattr(step, f"stdi_{branch}_{suffix}", metrics["stdi"] if metrics else None)
    # Legacy component columns explicitly describe the embedding branch in dual mode.
    embedding = result["embedding"].get("metrics")
    if embedding:
        for component in STDI_COMPONENTS:
            if component != "stdi":
                setattr(step, f"{component}_{suffix}", embedding[component])


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
    stdi_judge_fn: Callable[..., Any] | None = None,
    vad_model_bundle: VADModelBundle | None = None,
    vad_scorer: Callable[[str], VADScore] | None = None,
    output_dir: Path | str | None = None,
    output_prefix: str = "simulation",
    persist_results: bool = True,
    progress_callback: ProgressCallback | None = None,
    work_progress_callback: WorkProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> SimulationResult:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")
    if stdi_comparison_method not in {"dual", "cluster", "lexical"}:
        raise ValueError("'stdi_comparison_method' must be 'dual', 'cluster' or 'lexical'.")
    cache_directory = stdi_cache_dir
    if cache_directory is None and persist_results and stdi_comparison_method == "dual":
        cache_directory = Path(output_dir or DEFAULT_SIMULATION_OUTPUT_DIR) / "evaluation_cache"
    evaluation_cache = EvaluationCache(cache_directory)
    vad_cache: dict[str, VADScore] = {}
    vad_errors: dict[str, str] = {}
    judge_limiter = MinuteRateLimiter(max_requests_per_minute)

    def extract_structure(**kwargs: Any) -> TopicStructure:
        if stdi_comparison_method not in {"dual", "cluster"}:
            return extract_topic_structure(**kwargs)
        inputs = {
            key: str(value) if key in {"model", "provider"} else value
            for key, value in kwargs.items()
            if key in {"text", "title", "model", "provider", "base_url"}
        }
        inputs["version"] = STRUCTURED_EXTRACTION_VERSION
        saved = evaluation_cache.get("extraction", inputs)
        if saved is not None:
            return _build_topic_structure(saved)
        try:
            structure = extract_topic_structure(
                **kwargs,
                structured=True,
                before_request_hook=judge_limiter.acquire,
            )
        except Exception as exc:
            structure = empty_topic_structure()
            structure.schema_version = 2
            structure.extraction_status = "failed"
            structure.extraction_issues = [str(exc)]
            structure.provenance = getattr(exc, "provenance", {})
            return structure
        evaluation_cache.put("extraction", inputs, topic_structure_to_dict(structure))
        return structure

    def score_vad(text: str) -> VADScore:
        if stdi_comparison_method != "dual" or text not in vad_cache:
            try:
                vad_cache[text] = _score_vad(text, model_bundle=vad_model_bundle, scorer=vad_scorer)
            except Exception as exc:
                if stdi_comparison_method != "dual":
                    raise
                vad_cache[text] = VADScore(None, None, None)
                vad_errors[text] = str(exc)
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
    resolved_text_column = text_column or choose_news_text_column(df)
    rows_to_process = len(df) if max_rows is None else min(len(df), max_rows)
    _emit_progress(
        progress_callback,
        f"Starting interaction graph run with {rows_to_process} row(s) and "
        f"{len(ordered_node_ids)} node(s).",
    )

    clients_by_node_id: dict[str, tuple[str, Any]] = {}
    limiters_by_node_id: dict[str, MinuteRateLimiter] = {}
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
    rows_started = 0
    cancelled = False

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
            for step_index, node_id in enumerate(ordered_node_ids, start=1):
                node = nodes_by_id[node_id]
                provider_normalized, _ = clients_by_node_id[node.node_id]
                step_results.append(
                    SimulationStepResult(
                        news_id=news_id,
                        step_index=step_index,
                        node_id=node.node_id,
                        node_label=_node_label(node),
                        source_node_id=(
                            "original" if step_index == 1 else ordered_node_ids[step_index - 2]
                        ),
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

        for step_index, node_id in enumerate(ordered_node_ids, start=1):
            if _cancel_requested(cancel_check):
                cancelled = True
                break
            node = nodes_by_id[node_id]
            provider_normalized, client = clients_by_node_id[node.node_id]
            node_label = _node_label(node)
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
                    if stdi_comparison_method != "dual"
                    or original_structure.extraction_status == "valid"
                    else "error"
                ),
                original_topic_structure_error=(
                    "; ".join(original_structure.extraction_issues) or None
                ),
                original_vad_status="success" if original_vad.valence is not None else "error",
                original_vad_error=(vad_errors.get(original_text)),
                metadata={
                    **evaluation_metadata,
                    "rewrite_effective_personality": effective_personalities[node_id],
                    "rewrite_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                    "title": title,
                    "category": category,
                    "source_text_column": source_column,
                    **(
                        {"original_text": original_text, "extraction_title": extraction_title}
                        if stdi_comparison_method == "dual"
                        else {}
                    ),
                    **flatten_topic_structure(original_structure, prefix="original"),
                },
            )
            _record_vad(step_result, "original", original_vad)
            _record_vad(step_result, "source", previous_vad)

            rewritten_structure_ready = False
            rewritten_vad_ready = False
            try:
                rewritten_text = _generate_rewrite(
                    provider_normalized=provider_normalized,
                    client=client,
                    model=node.model,
                    prompt=prompt,
                    system_instruction=rewrite_prompt.system_instruction,
                    retry_attempts=retry_attempts,
                    limiter=limiters_by_node_id[node.node_id],
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
                    if stdi_comparison_method == "dual"
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
                    if stdi_comparison_method != "dual"
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
                    "success" if rewritten_vad.valence is not None else "error"
                )
                step_result.rewritten_vad_error = vad_errors.get(rewritten_text)
                _record_vad(step_result, "rewritten", rewritten_vad)
                if stdi_comparison_method in {"cluster", "dual"}:
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
                    _record_stdi_metrics(
                        step_result, suffix="vs_original", metrics=vs_original_metrics
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
                    _record_stdi_metrics(
                        step_result, suffix="incremental", metrics=incremental_metrics
                    )
                    cumulative_stdi += incremental_metrics["stdi"]
                    step_result.stdi_cumulative = round(cumulative_stdi, 6)

                previous_text = rewritten_text
                previous_rewritten_text = rewritten_text
                previous_structure = rewritten_structure
                previous_vad = rewritten_vad
                previous_node_id = node.node_id
                previous_node_label = node_label
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': success; "
                        + (
                            "STDI pending shared embedding comparison."
                            if stdi_comparison_method in {"cluster", "dual"}
                            else f"stdi_vs_original={step_result.stdi_vs_original}, "
                            f"stdi_incremental={step_result.stdi_incremental}."
                        )
                    ),
                )
            except Exception as exc:
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
                work_progress_callback(row_position, total_rows, step_index, len(ordered_node_ids))

            if sleep_seconds > 0 and _wait_between_steps(sleep_seconds, cancel_check):
                cancelled = True
                break

        if cancelled:
            break

    if _cancel_requested(cancel_check):
        cancelled = True
    if scoring_contexts:
        _emit_progress(
            progress_callback,
            f"Fitting shared STDI embeddings from {len(scoring_contexts)} successful step(s).",
        )
        pairs = [
            TopicStructurePair(f"{index}:original", original, rewritten)
            for index, (_, _, original, _, rewritten, _, _, _) in enumerate(scoring_contexts)
        ] + [
            TopicStructurePair(f"{index}:incremental", previous, rewritten)
            for index, (_, _, _, previous, rewritten, _, _, _) in enumerate(scoring_contexts)
        ]
        comparator_class = StructuredEmbeddingComparator
        comparator = comparator_class(
            embedder=stdi_embedder,
            embedding_model=stdi_embedding_model,
        )
        try:
            comparator.fit(pairs)
        except Exception:
            if stdi_comparison_method != "dual":
                raise
        last_row_position = 0
        cumulative_stdi = 0.0
        valid_steps = 0
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
            if stdi_comparison_method == "dual" and _cancel_requested(cancel_check):
                cancelled = True
                break
            if row_position != last_row_position:
                cumulative_stdi = 0.0
                valid_steps = 0
                last_row_position = row_position
            if stdi_comparison_method == "dual":
                _emit_progress(
                    progress_callback,
                    f"Evaluating dual STDI for '{step.news_id}' step {step.step_index}.",
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
                    "cache": evaluation_cache,
                    **({"judge_fn": stdi_judge_fn} if stdi_judge_fn is not None else {}),
                }
                vs_original_dual = compare_dual_stdi(
                    original_text=step.metadata["original_text"],
                    original_structure=original,
                    original_vad=original_vad,
                    **common,
                )
                if _cancel_requested(cancel_check):
                    cancelled = True
                    _record_dual_metrics(step, "vs_original", vs_original_dual)
                    break
                incremental_dual = compare_dual_stdi(
                    original_text=step.source_text,
                    original_structure=previous,
                    original_vad=previous_vad,
                    **common,
                )
                _record_dual_metrics(step, "vs_original", vs_original_dual)
                _record_dual_metrics(step, "incremental", incremental_dual)
                if incremental_dual["stdi"] is not None:
                    cumulative_stdi += incremental_dual["stdi"]
                    valid_steps += 1
                step.stdi_cumulative = cumulative_stdi
                step.stdi_cumulative_valid_steps = valid_steps
                step.stdi_chain_complete = valid_steps == step.step_index
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
            _record_stdi_metrics(step, suffix="vs_original", metrics=vs_original)
            _record_stdi_metrics(step, suffix="incremental", metrics=incremental)
            cumulative_stdi += incremental["stdi"]
            step.stdi_cumulative = round(cumulative_stdi, 6)

    if _cancel_requested(cancel_check):
        cancelled = True
    success_count = sum(1 for step in step_results if step.rewrite_status == "success")
    error_count = sum(1 for step in step_results if step.rewrite_status == "error")
    vad_model_name = DEFAULT_VAD_MODEL_NAME
    if vad_model_bundle is not None:
        vad_model_name = vad_model_bundle.model_name
    if vad_scorer is not None:
        vad_model_name = "custom_scorer"
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
        if stdi_comparison_method == "dual"
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
        step.metadata.update(
            {
                "rewrite_model": step.model,
                "rewrite_provider": step.provider,
                "vad_model": vad_model_name,
                "stdi_embedding_model": embedding_model_name,
                "stdi_comparison_version": comparison_version,
            }
        )
    summary = {
        "schema_version": GRAPH_STEP_SCHEMA_VERSION,
        **evaluation_metadata,
        "rewrite_system_instruction": rewrite_prompt.system_instruction,
        "rewrite_prompt_template": rewrite_prompt.template,
        "rows_processed": rows_started,
        "cancelled": cancelled,
        "steps_total": len(step_results),
        "steps_success": success_count,
        "steps_error": error_count,
        "vad_model": vad_model_name,
        "stdi_comparison_method": stdi_comparison_method,
        "stdi_comparison_version": comparison_version,
        "stdi_embedding_model": embedding_model_name,
        "stdi_complete_incremental_pairs": sum(
            step.stdi_status_incremental == "valid" for step in step_results
        )
        if stdi_comparison_method == "dual"
        else None,
        "stdi_components_branch": "embedding" if stdi_comparison_method == "dual" else None,
        "graph": {
            "start_node_id": resolved_start_node,
            "ordered_node_ids": ordered_node_ids,
            "edges": [asdict(edge) for edge in normalized_edges],
        },
        "nodes": [asdict(node) for node in nodes],
    }
    result = SimulationResult(summary=summary, step_results=step_results)

    if persist_results:
        resolved_output_dir = (
            Path(output_dir) if output_dir is not None else DEFAULT_SIMULATION_OUTPUT_DIR
        )
        summary_path, steps_path = _persist_results(
            result=result,
            output_dir=resolved_output_dir,
            output_prefix=output_prefix,
        )
        result.summary_path = summary_path
        result.steps_path = steps_path
        _emit_progress(
            progress_callback,
            f"Saved summary to {summary_path} and step records to {steps_path}.",
        )

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
