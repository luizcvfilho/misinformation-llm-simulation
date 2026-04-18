from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from misinformation_simulation.config.prompts import PROMPT_TEMPLATE, REWRITE_SYSTEM_INSTRUCTION
from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
    resolve_output_language,
    resolve_output_language_name,
    resolve_row_text,
)
from misinformation_simulation.enums import Provider
from misinformation_simulation.llm.clients import create_llm_client, normalize_provider
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift import (
    calculate_stdi,
    extract_topic_structure,
    flatten_topic_structure,
)
from misinformation_simulation.topic_drift.models import TopicStructure

DEFAULT_SIMULATION_OUTPUT_DIR = Path("output") / "interaction_graph"
ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class SimulationNode:
    node_id: str
    model: str
    provider: Provider | str
    personality: str
    label: str | None = None
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True)
class SimulationEdge:
    source: str
    target: str


@dataclass(slots=True)
class SimulationStepResult:
    news_id: str
    step_index: int
    node_id: str
    node_label: str
    source_node_id: str
    source_node_label: str
    provider: str
    model: str
    personality: str
    source_text: str
    rewritten_text: str | None
    target_language: str
    target_language_source: str
    rewrite_status: str
    rewrite_error: str | None
    stdi_vs_original: float | None = None
    theme_drift_vs_original: float | None = None
    subtopic_drift_vs_original: float | None = None
    entity_drift_vs_original: float | None = None
    relation_drift_vs_original: float | None = None
    stdi_incremental: float | None = None
    theme_drift_incremental: float | None = None
    subtopic_drift_incremental: float | None = None
    entity_drift_incremental: float | None = None
    relation_drift_incremental: float | None = None
    original_topic_structure_status: str = "not_requested"
    original_topic_structure_error: str | None = None
    rewritten_topic_structure_status: str = "not_requested"
    rewritten_topic_structure_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        metadata = record.pop("metadata", {})
        for key, value in metadata.items():
            record[f"metadata_{key}"] = value
        return record


@dataclass(slots=True)
class SimulationResult:
    summary: dict[str, Any]
    step_results: list[SimulationStepResult]
    summary_path: Path | None = None
    steps_path: Path | None = None


def _node_label(node: SimulationNode) -> str:
    return node.label or node.node_id


def _normalize_nodes(nodes: list[SimulationNode]) -> dict[str, SimulationNode]:
    if not nodes:
        raise ValueError("Provide at least one simulation node.")

    nodes_by_id: dict[str, SimulationNode] = {}
    for node in nodes:
        if not node.node_id.strip():
            raise ValueError("Each node must have a non-empty 'node_id'.")
        if node.node_id in nodes_by_id:
            raise ValueError(f"Duplicate node_id detected: '{node.node_id}'.")
        if not node.personality.strip():
            raise ValueError(f"Node '{node.node_id}' must define a personality.")
        nodes_by_id[node.node_id] = node
    return nodes_by_id


def _normalize_edges(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge] | None,
) -> list[SimulationEdge]:
    if edges is None:
        ordered_nodes = list(nodes_by_id.values())
        return [
            SimulationEdge(
                source=ordered_nodes[index].node_id,
                target=ordered_nodes[index + 1].node_id,
            )
            for index in range(len(ordered_nodes) - 1)
        ]

    normalized_edges: list[SimulationEdge] = []
    seen_targets: set[str] = set()
    for edge in edges:
        if edge.source not in nodes_by_id:
            raise ValueError(f"Edge source '{edge.source}' is not a known node.")
        if edge.target not in nodes_by_id:
            raise ValueError(f"Edge target '{edge.target}' is not a known node.")
        if edge.target in seen_targets:
            raise ValueError(
                "Each node can receive text from only one previous node in this simulation."
            )
        seen_targets.add(edge.target)
        normalized_edges.append(edge)
    return normalized_edges


def _resolve_start_node(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge],
    start_node_id: str | None,
) -> str:
    if start_node_id is not None:
        if start_node_id not in nodes_by_id:
            raise ValueError(f"Unknown start node '{start_node_id}'.")
        return start_node_id

    inbound_counts = {node_id: 0 for node_id in nodes_by_id}
    for edge in edges:
        inbound_counts[edge.target] += 1

    candidates = [node_id for node_id, count in inbound_counts.items() if count == 0]
    if len(candidates) != 1:
        raise ValueError("Could not infer a unique start node. Provide 'start_node_id' explicitly.")
    return candidates[0]


def _topological_path(
    nodes_by_id: dict[str, SimulationNode],
    edges: list[SimulationEdge],
    start_node_id: str,
) -> list[str]:
    next_by_source: dict[str, str] = {}
    for edge in edges:
        if edge.source in next_by_source:
            raise ValueError(
                f"Node '{edge.source}' has multiple outgoing edges. "
                "This simulation currently supports a single chain path."
            )
        next_by_source[edge.source] = edge.target

    path = [start_node_id]
    seen = {start_node_id}
    current = start_node_id

    while current in next_by_source:
        current = next_by_source[current]
        if current in seen:
            raise ValueError("Cycle detected in the interaction graph.")
        seen.add(current)
        path.append(current)

    if len(path) != len(nodes_by_id):
        missing = [node_id for node_id in nodes_by_id if node_id not in seen]
        raise ValueError(
            "The interaction graph must form a single connected chain. "
            f"Unreachable nodes: {', '.join(missing)}."
        )
    return path


def _generate_rewrite(
    *,
    provider_normalized: str,
    client: Any,
    model: str,
    prompt: str,
    retry_attempts: int,
    limiter: MinuteRateLimiter,
) -> str:
    if provider_normalized == "gemini":
        return generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=REWRITE_SYSTEM_INSTRUCTION,
            temperature=0.8,
            max_attempts=retry_attempts,
            before_request_hook=limiter.acquire,
        )
    return generate_openai_text_with_retry(
        client,
        model=model,
        prompt=prompt,
        system_instruction=REWRITE_SYSTEM_INSTRUCTION,
        temperature=0.8,
        max_attempts=retry_attempts,
        before_request_hook=limiter.acquire,
    )


def _extract_structures_and_metrics(
    *,
    reference_text: str,
    compared_text: str,
    title: str,
    topic_drift_model: str,
    topic_drift_provider: Provider | str,
    topic_drift_api_key: str | None,
    topic_drift_base_url: str | None,
    max_requests_per_minute: int | None,
    retry_attempts: int,
) -> tuple[TopicStructure, TopicStructure]:
    reference_structure = extract_topic_structure(
        text=reference_text,
        title=title,
        model=topic_drift_model,
        provider=topic_drift_provider,
        api_key=topic_drift_api_key,
        base_url=topic_drift_base_url,
        max_requests_per_minute=max_requests_per_minute,
        retry_attempts=retry_attempts,
    )
    compared_structure = extract_topic_structure(
        text=compared_text,
        title=title,
        model=topic_drift_model,
        provider=topic_drift_provider,
        api_key=topic_drift_api_key,
        base_url=topic_drift_base_url,
        max_requests_per_minute=max_requests_per_minute,
        retry_attempts=retry_attempts,
    )
    return reference_structure, compared_structure


def _news_identifier(row_index: Any, row: pd.Series, news_id_column: str | None) -> str:
    if news_id_column and news_id_column in row.index and pd.notna(row[news_id_column]):
        candidate = str(row[news_id_column]).strip()
        if candidate:
            return candidate
    return f"row_{row_index}"


def _persist_results(
    *,
    result: SimulationResult,
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{output_prefix}_summary.json"
    steps_path = output_dir / f"{output_prefix}_steps.jsonl"

    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    steps_df = pd.DataFrame([step.to_record() for step in result.step_results])
    steps_df.to_json(steps_path, orient="records", lines=True, force_ascii=False)
    return summary_path, steps_path


def _emit_progress(
    progress_callback: ProgressCallback | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


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
    allow_title_fallback: bool = False,
    topic_drift_model: str = "gemini-3.1-flash-lite-preview",
    topic_drift_provider: Provider | str = Provider.GEMINI,
    topic_drift_api_key: str | None = None,
    topic_drift_base_url: str | None = None,
    output_dir: Path | str | None = None,
    output_prefix: str = "simulation",
    persist_results: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> SimulationResult:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas.DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if max_requests_per_minute is not None and max_requests_per_minute <= 0:
        raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")

    nodes_by_id = _normalize_nodes(nodes)
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

    step_results: list[SimulationStepResult] = []
    target_indexes = list(df.index)
    if max_rows is not None:
        target_indexes = target_indexes[:max_rows]
    total_rows = len(target_indexes)

    for row_position, row_index in enumerate(target_indexes, start=1):
        row = df.loc[row_index]
        news_id = _news_identifier(row_index, row, news_id_column)
        title = ""
        if title_column in row.index and pd.notna(row[title_column]):
            title = str(row[title_column]).strip()
        _emit_progress(
            progress_callback,
            f"[{row_position}/{total_rows}] Preparing news '{news_id}' ({title or 'Untitled'}).",
        )

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
            original_structure = extract_topic_structure(
                text=original_text,
                title=title or "Untitled",
                model=topic_drift_model,
                provider=topic_drift_provider,
                api_key=topic_drift_api_key,
                base_url=topic_drift_base_url,
                max_requests_per_minute=max_requests_per_minute,
                retry_attempts=retry_attempts,
            )
            _emit_progress(
                progress_callback,
                f"[{row_position}/{total_rows}] Original text ready for '{news_id}'.",
            )
        except Exception as exc:
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
                        original_topic_structure_status="error",
                        original_topic_structure_error=str(exc),
                        rewritten_topic_structure_status="blocked",
                        rewritten_topic_structure_error=(
                            "Skipped because original preprocessing failed."
                        ),
                        metadata={
                            "title": title,
                        },
                    )
                )
            continue

        previous_node_id = "original"
        previous_node_label = source_column
        previous_text = original_text
        previous_structure = original_structure
        previous_rewritten_text: str | None = None

        for step_index, node_id in enumerate(ordered_node_ids, start=1):
            node = nodes_by_id[node_id]
            provider_normalized, client = clients_by_node_id[node.node_id]
            node_label = _node_label(node)
            _emit_progress(
                progress_callback,
                f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                f"node='{node_label}' provider='{provider_normalized}' "
                f"model='{node.model}': rewriting.",
            )
            prompt = PROMPT_TEMPLATE.format(
                personality=node.personality,
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
                metadata={
                    "title": title,
                    "source_text_column": source_column,
                },
            )

            try:
                rewritten_text = _generate_rewrite(
                    provider_normalized=provider_normalized,
                    client=client,
                    model=node.model,
                    prompt=prompt,
                    retry_attempts=retry_attempts,
                    limiter=limiters_by_node_id[node.node_id],
                )
                step_result.rewritten_text = rewritten_text
                step_result.rewrite_status = "success"
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': rewrite completed, extracting topic drift."
                    ),
                )

                _, rewritten_structure = _extract_structures_and_metrics(
                    reference_text=original_text,
                    compared_text=rewritten_text,
                    title=title or "Untitled",
                    topic_drift_model=topic_drift_model,
                    topic_drift_provider=topic_drift_provider,
                    topic_drift_api_key=topic_drift_api_key,
                    topic_drift_base_url=topic_drift_base_url,
                    max_requests_per_minute=max_requests_per_minute,
                    retry_attempts=retry_attempts,
                )
                vs_original_metrics = calculate_stdi(original_structure, rewritten_structure)
                for key, value in {
                    **flatten_topic_structure(original_structure, prefix="original"),
                    **flatten_topic_structure(rewritten_structure, prefix="rewritten"),
                }.items():
                    step_result.metadata[key] = value
                step_result.original_topic_structure_status = "success"
                step_result.rewritten_topic_structure_status = "success"
                step_result.stdi_vs_original = vs_original_metrics["stdi"]
                step_result.theme_drift_vs_original = vs_original_metrics["theme_drift"]
                step_result.subtopic_drift_vs_original = vs_original_metrics["subtopic_drift"]
                step_result.entity_drift_vs_original = vs_original_metrics["entity_drift"]
                step_result.relation_drift_vs_original = vs_original_metrics["relation_drift"]

                if previous_rewritten_text is None:
                    step_result.stdi_incremental = step_result.stdi_vs_original
                    step_result.theme_drift_incremental = step_result.theme_drift_vs_original
                    step_result.subtopic_drift_incremental = step_result.subtopic_drift_vs_original
                    step_result.entity_drift_incremental = step_result.entity_drift_vs_original
                    step_result.relation_drift_incremental = step_result.relation_drift_vs_original
                else:
                    incremental_metrics = calculate_stdi(previous_structure, rewritten_structure)
                    step_result.stdi_incremental = incremental_metrics["stdi"]
                    step_result.theme_drift_incremental = incremental_metrics["theme_drift"]
                    step_result.subtopic_drift_incremental = incremental_metrics["subtopic_drift"]
                    step_result.entity_drift_incremental = incremental_metrics["entity_drift"]
                    step_result.relation_drift_incremental = incremental_metrics["relation_drift"]

                previous_text = rewritten_text
                previous_rewritten_text = rewritten_text
                previous_structure = rewritten_structure
                previous_node_id = node.node_id
                previous_node_label = node_label
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': success "
                        f"(stdi_vs_original={step_result.stdi_vs_original}, "
                        f"stdi_incremental={step_result.stdi_incremental})."
                    ),
                )
            except Exception as exc:
                step_result.rewrite_status = "error"
                step_result.rewrite_error = str(exc)
                _emit_progress(
                    progress_callback,
                    (
                        f"[{row_position}/{total_rows}] Step {step_index}/{len(ordered_node_ids)} "
                        f"node='{node_label}': error: {exc}"
                    ),
                )

            step_results.append(step_result)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    success_count = sum(1 for step in step_results if step.rewrite_status == "success")
    error_count = sum(1 for step in step_results if step.rewrite_status == "error")
    summary = {
        "rows_processed": len(target_indexes),
        "steps_total": len(step_results),
        "steps_success": success_count,
        "steps_error": error_count,
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
