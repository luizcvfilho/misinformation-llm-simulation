from __future__ import annotations

from contextlib import nullcontext as _null_operation
from hashlib import sha256

import pandas as pd

from misinformation_simulation.datasets.selection import (
    resolve_output_language,
    resolve_output_language_name,
    resolve_row_text,
)
from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.llm.progress import (
    emit_event,
    mark_cached,
    operation,
    set_progress_context,
)
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.graph import (
    GRAPH_REWRITE_TEMPERATURE,
    STDI_COMPONENTS,
    _cancel_requested,
    _emit_progress,
    _news_identifier,
    _record_stdi_metrics,
    _record_vad,
    _wait_between_steps,
)
from misinformation_simulation.simulation.graph_evaluation import _GraphEvaluation
from misinformation_simulation.simulation.graph_preparation import _GraphPreparation
from misinformation_simulation.simulation.topology import (
    _node_label,
)
from misinformation_simulation.simulation.types import (
    SimulationStepResult,
)
from misinformation_simulation.text_metrics.vad_evaluation import (
    compare_vad_evaluations,
    vad_for_branch,
)
from misinformation_simulation.topic_drift import calculate_stdi, flatten_topic_structure
from misinformation_simulation.topic_drift.cluster_comparison import (
    TopicStructurePair,
)
from misinformation_simulation.topic_drift.structured_comparison import (
    complete_stdi,
)


class _GraphRun(_GraphPreparation, _GraphEvaluation):
    """Keep rewrite, cancellation, cache and lineage state isolated to one run."""

    def __init__(
        self,
        df,
        options,
        *,
        comparator_factory,
        _extract_compared_structure,
        _generate_rewrite,
        compare_dual_stdi,
        create_llm_client,
        extract_topic_structure,
    ):
        self.df = df
        self.options = options
        self.comparator_factory = comparator_factory
        self._extract_compared_structure = _extract_compared_structure
        self._generate_rewrite = _generate_rewrite
        self.compare_dual_stdi = compare_dual_stdi
        self.create_llm_client = create_llm_client
        self.extract_topic_structure = extract_topic_structure

    def run(self):
        self._validate_input()

        self._prepare_evaluators()

        self._prepare_graph()

        self._prepare_clients()

        self._initialize_results()

        self.rows_started = 0
        self.cancelled = False
        self.rewrite_operations_started = 0
        self._process_rows()

        if _cancel_requested(self.options.cancel_check):
            self.cancelled = True
        if not self.cancelled:
            emit_event("settle", stages=["rewrite", "extraction", "vad", "pacing"])
        self.pairs = [
            TopicStructurePair(f"{index}:original", original, rewritten)
            for index, (_, _, original, _, rewritten, _, _, _) in enumerate(self.scoring_contexts)
        ] + [
            TopicStructurePair(f"{index}:incremental", previous, rewritten)
            for index, (_, _, _, previous, rewritten, _, _, _) in enumerate(self.scoring_contexts)
        ]
        if self.options.defer_evaluation:
            return PreparedSimulation(
                self.pairs,
                self.finish,
                self.build_comparator,
                checkpoint=lambda: self.finish(evaluate=False),
            )
        return self.finish()

    def _process_row(self):
        self.rows_started = self.row_position
        self._prepare_row_context()

        self.original_structure_ready = False
        try:
            self.source_column, self.original_text = resolve_row_text(
                row=self.row,
                preferred_column=self.resolved_text_column,
                allow_title_fallback=self.options.allow_title_fallback,
            )
            self.target_language_code, self.target_language_source = resolve_output_language(
                row=self.row, original_text=self.original_text
            )
            self.original_structure = self.extract_structure(
                text=self.original_text,
                title=self.extraction_title,
                model=self.options.topic_drift_model,
                provider=self.options.topic_drift_provider,
                api_key=self.options.topic_drift_api_key,
                base_url=self.options.topic_drift_base_url,
                max_requests_per_minute=self.options.max_requests_per_minute,
                retry_attempts=self.options.retry_attempts,
            )
            self.original_structure_ready = True
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                return
            self.original_vad = self.score_vad(self.original_text)
            _emit_progress(
                self.options.progress_callback,
                (
                    f"[{self.row_position}/{self.total_rows}] Original text ready for '"
                    f"{self.news_id}'."
                ),
            )
            if self.options.work_progress_callback is not None:
                self.options.work_progress_callback(
                    self.row_position, self.total_rows, 0, len(self.ordered_node_ids)
                )
        except Exception as exc:
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                return
            self._record_preparation_failure(exc)

            return
        self._initialize_row_lineage()

        self._process_nodes()

        if self.cancelled:
            return

    def _process_rows(self):
        for row_position, row_index in enumerate(self.target_indexes, start=1):
            self.row_position, self.row_index = (row_position, row_index)
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                break
            self._process_row()
            if self.cancelled:
                break

    def _process_node(self):
        self.node = self.nodes_by_id[self.node_id]
        self.step_index = self.depths_by_node[self.node_id]
        parent_id = self.parents_by_node.get(self.node_id)
        inherited_state = self.states_by_node.get(parent_id, self.initial_state)
        (
            self.previous_node_id,
            self.previous_node_label,
            self.previous_text,
            self.previous_structure,
            self.previous_vad,
            self.previous_rewritten_text,
            self.cumulative_stdi,
        ) = inherited_state
        self.states_by_node[self.node_id] = inherited_state
        self.provider_normalized, self.client = self.clients_by_node_id[self.node.node_id]
        self.node_label = _node_label(self.node)
        set_progress_context(news_id=self.news_id, row=self.row_position, node_id=self.node_id)
        _emit_progress(
            self.options.progress_callback,
            (
                f"[{self.row_position}/{self.total_rows}] Step {self.step_index}/"
                f"{len(self.ordered_node_ids)} node='{self.node_label}' provider='"
                f"{self.provider_normalized}' model='{self.node.model}': rewriting."
            ),
        )
        self.prompt = self.rewrite_prompt.template.format(
            personality=self.effective_personalities[self.node_id],
            target_language_name=resolve_output_language_name(self.target_language_code),
            target_language_code=self.target_language_code,
            title=self.title or "Untitled",
            original_text=self.previous_text,
        )
        self._create_step_result()

        self.rewritten_structure_ready = False
        self.rewritten_vad_ready = False
        if self.branching and parent_id in self.failed_nodes:
            self.step_result.source_node_id = parent_id
            self.step_result.source_node_label = _node_label(self.nodes_by_id[parent_id])
            self.step_result.source_text = ""
            self.step_result.rewrite_status = "blocked"
            self.step_result.rewrite_error = (
                f"Parent node '{parent_id}' did not complete successfully."
            )
            self.failed_nodes.add(self.node_id)
            self.step_results.append(self.step_result)
            if self.options.work_progress_callback is not None:
                self.options.work_progress_callback(
                    self.row_position,
                    self.total_rows,
                    self.execution_index,
                    len(self.ordered_node_ids),
                )
            return
        try:
            self._rewrite_node()

            self.step_result.rewritten_text = self.rewritten_text
            self.step_result.rewrite_status = "success"
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                return
            _emit_progress(
                self.options.progress_callback,
                (
                    f"[{self.row_position}/{self.total_rows}] Step {self.step_index}/"
                    f"{len(self.ordered_node_ids)} node='{self.node_label}"
                    f"': rewrite completed, extracting topic drift."
                ),
            )
            self._extract_rewritten_structure()

            self.rewritten_structure_ready = True
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                return
            self._record_rewritten_evaluation()

            if self.options.stdi_comparison_method in {"cluster", "dual", "llm"}:
                self.scoring_contexts.append(
                    (
                        self.row_position,
                        self.step_result,
                        self.original_structure,
                        self.previous_structure,
                        self.rewritten_structure,
                        self.original_vad,
                        self.previous_vad,
                        self.rewritten_vad,
                    )
                )
            else:
                self._score_lexical_step()

            self._advance_lineage()

        except Exception as exc:
            self._record_node_error(exc)

        self.step_results.append(self.step_result)
        if self.options.work_progress_callback is not None:
            self.options.work_progress_callback(
                self.row_position,
                self.total_rows,
                self.execution_index,
                len(self.ordered_node_ids),
            )
        with operation("pacing") if self.options.sleep_seconds > 0 else _null_operation():
            if self.options.sleep_seconds > 0 and _wait_between_steps(
                self.options.sleep_seconds, self.options.cancel_check
            ):
                self.cancelled = True
                return

    def _record_preparation_failure(self, exc):
        _emit_progress(
            self.options.progress_callback,
            f"[{self.row_position}/{self.total_rows}] Failed to prepare '{self.news_id}': {exc}",
        )
        for node_id in self.ordered_node_ids:
            self.node_id = node_id
            self.node = self.nodes_by_id[self.node_id]
            self.provider_normalized, self._ = self.clients_by_node_id[self.node.node_id]
            self.step_results.append(
                SimulationStepResult(
                    news_id=self.news_id,
                    step_index=self.depths_by_node[self.node_id],
                    node_id=self.node.node_id,
                    node_label=_node_label(self.node),
                    source_node_id=self.parents_by_node.get(self.node_id, "original"),
                    source_node_label="source_unavailable",
                    provider=self.provider_normalized,
                    model=self.node.model,
                    personality=self.node.personality,
                    source_text="",
                    rewritten_text=None,
                    target_language="unknown",
                    target_language_source="unresolved",
                    rewrite_status="blocked",
                    rewrite_error=f"Original text could not be prepared: {exc}",
                    original_topic_structure_status="success"
                    if self.original_structure_ready
                    else "error",
                    original_topic_structure_error=None
                    if self.original_structure_ready
                    else str(exc),
                    original_vad_status="error" if self.original_structure_ready else "blocked",
                    original_vad_error=str(exc)
                    if self.original_structure_ready
                    else "Skipped because original preprocessing failed.",
                    rewritten_topic_structure_status="blocked",
                    rewritten_topic_structure_error=(
                        "Skipped because original preprocessing failed."
                    ),
                    metadata={
                        **self.evaluation_metadata,
                        "rewrite_effective_personality": self.effective_personalities[self.node_id],
                        "title": self.title,
                        "category": self.category,
                        "graph_step_id": (
                            f"{self.options.output_prefix}:{self.row_position}:{self.node_id}"
                        ),
                        **(
                            flatten_topic_structure(self.original_structure, prefix="original")
                            if self.original_structure_ready
                            else {}
                        ),
                    },
                )
            )
        if self.options.work_progress_callback is not None:
            self.options.work_progress_callback(
                self.row_position,
                self.total_rows,
                len(self.ordered_node_ids),
                len(self.ordered_node_ids),
            )

    def _initialize_row_lineage(self):
        self.previous_node_id = "original"
        self.previous_node_label = self.source_column
        self.previous_text = self.original_text
        self.previous_structure = self.original_structure
        self.previous_vad = self.original_vad
        self.previous_rewritten_text: str | None = None
        self.cumulative_stdi = 0.0
        self.initial_state = (
            self.previous_node_id,
            self.previous_node_label,
            self.previous_text,
            self.previous_structure,
            self.previous_vad,
            self.previous_rewritten_text,
            self.cumulative_stdi,
        )
        self.states_by_node: dict[str, tuple] = {}
        self.failed_nodes: set[str] = set()

    def _process_nodes(self):
        for execution_index, node_id in enumerate(self.ordered_node_ids, start=1):
            self.execution_index, self.node_id = (execution_index, node_id)
            if _cancel_requested(self.options.cancel_check):
                self.cancelled = True
                break
            self._process_node()
            if self.cancelled:
                break

    def _prepare_row_context(self):
        self.row = self.df.loc[self.row_index]
        self.news_id = _news_identifier(self.row_index, self.row, self.options.news_id_column)
        self.title = ""
        if self.options.title_column in self.row.index and pd.notna(
            self.row[self.options.title_column]
        ):
            self.title = str(self.row[self.options.title_column]).strip()
        self.category = ""
        if "category" in self.row.index and pd.notna(self.row["category"]):
            self.category = str(self.row["category"]).strip()
        _emit_progress(
            self.options.progress_callback,
            (
                f"[{self.row_position}/{self.total_rows}] Preparing news '"
                f"{self.news_id}' ({self.title or 'Untitled'})."
            ),
        )
        set_progress_context(news_id=self.news_id, row=self.row_position)
        self.extraction_title = (
            self.title or "Untitled" if self.rewrite_prompt.original_title_context else None
        )

    def _create_step_result(self):
        self.step_result = SimulationStepResult(
            news_id=self.news_id,
            step_index=self.step_index,
            node_id=self.node.node_id,
            node_label=self.node_label,
            source_node_id=self.previous_node_id,
            source_node_label=self.previous_node_label,
            provider=self.provider_normalized,
            model=self.node.model,
            personality=self.node.personality,
            source_text=self.previous_text,
            rewritten_text=None,
            target_language=self.target_language_code,
            target_language_source=self.target_language_source,
            rewrite_status="not_requested",
            rewrite_error=None,
            original_topic_structure_status="success"
            if self.options.stdi_comparison_method not in {"dual", "llm"}
            or self.original_structure.extraction_status == "valid"
            else "error",
            original_topic_structure_error="; ".join(self.original_structure.extraction_issues)
            or None,
            original_vad_status="success" if self.original_text not in self.vad_errors else "error",
            original_vad_error=self.vad_errors.get(self.original_text),
            metadata={
                **self.evaluation_metadata,
                "rewrite_effective_personality": self.effective_personalities[self.node_id],
                "rewrite_prompt_sha256": sha256(self.prompt.encode("utf-8")).hexdigest(),
                "title": self.title,
                "category": self.category,
                "source_text_column": self.source_column,
                "graph_execution_index": self.execution_index,
                "graph_step_id": (
                    f"{self.options.output_prefix}:{self.row_position}:{self.node_id}"
                ),
                **(
                    {
                        "original_text": self.original_text,
                        "extraction_title": self.extraction_title,
                    }
                    if self.options.stdi_comparison_method in {"dual", "llm"}
                    else {}
                ),
                **flatten_topic_structure(self.original_structure, prefix="original"),
            },
        )
        _record_vad(self.step_result, "original", self.original_vad)
        _record_vad(self.step_result, "source", self.previous_vad)
        self.step_result.metadata["original_vad_evaluation"] = self.vad_evaluator.records[
            self.original_text
        ]
        self.step_result.metadata["source_vad_evaluation"] = self.vad_evaluator.records[
            self.previous_text
        ]

    def _rewrite_node(self):
        lineage = [self.node_id]
        while lineage[-1] in self.parents_by_node:
            lineage.append(self.parents_by_node[lineage[-1]])
        rewrite_inputs = {
            "news_id": self.news_id,
            "row_position": self.row_position,
            "original_text": self.original_text,
            "lineage": list(reversed(lineage)),
            "prompt": self.prompt,
            "system_instruction": self.rewrite_prompt.system_instruction,
            "version": self.rewrite_prompt.version,
            "temperature": GRAPH_REWRITE_TEMPERATURE,
            "provider": self.provider_normalized,
            "model": str(self.node.model),
            "base_url": str(self.client.base_url)
            if getattr(self.client, "base_url", None)
            else self.node.base_url,
        }
        with operation("rewrite", model=self.node.model, provider=self.provider_normalized):
            saved_rewrite = (
                self.evaluation_cache.get("rewrite", rewrite_inputs)
                if self.options.reuse_rewrites
                else None
            )
            self.step_result.metadata["rewrite_cache_hit"] = saved_rewrite is not None
            if saved_rewrite is not None:
                mark_cached()
                self.rewritten_text = saved_rewrite["text"]
            else:
                self.rewrite_operations_started += 1
                self.step_result.metadata["rewrite_operation_started"] = True
                self.rewritten_text = self._generate_rewrite(
                    provider_normalized=self.provider_normalized,
                    client=self.client,
                    model=self.node.model,
                    prompt=self.prompt,
                    system_instruction=self.rewrite_prompt.system_instruction,
                    retry_attempts=self.options.retry_attempts,
                    limiter=self.limiters_by_node_id[self.node.node_id],
                )
                if self.options.reuse_rewrites:
                    self.evaluation_cache.put(
                        "rewrite", rewrite_inputs, {"text": self.rewritten_text}
                    )

    def _extract_rewritten_structure(self):
        with (
            operation(
                "extraction",
                model=self.options.topic_drift_model,
                provider=normalize_provider(self.options.topic_drift_provider),
            )
            if self.options.stdi_comparison_method not in {"dual", "llm"}
            else _null_operation()
        ):
            self.rewritten_structure = (
                self.extract_structure(
                    text=self.rewritten_text,
                    title=self.extraction_title,
                    model=self.options.topic_drift_model,
                    provider=self.options.topic_drift_provider,
                    api_key=self.options.topic_drift_api_key,
                    base_url=self.options.topic_drift_base_url,
                    max_requests_per_minute=self.options.max_requests_per_minute,
                    retry_attempts=self.options.retry_attempts,
                )
                if self.options.stdi_comparison_method in {"dual", "llm"}
                else self._extract_compared_structure(
                    compared_text=self.rewritten_text,
                    title=self.extraction_title,
                    topic_drift_model=self.options.topic_drift_model,
                    topic_drift_provider=self.options.topic_drift_provider,
                    topic_drift_api_key=self.options.topic_drift_api_key,
                    topic_drift_base_url=self.options.topic_drift_base_url,
                    max_requests_per_minute=self.options.max_requests_per_minute,
                    retry_attempts=self.options.retry_attempts,
                )
            )

    def _record_rewritten_evaluation(self):
        self.step_result.rewritten_topic_structure_status = (
            "success"
            if self.options.stdi_comparison_method not in {"dual", "llm"}
            or self.rewritten_structure.extraction_status == "valid"
            else "error"
        )
        self.step_result.rewritten_topic_structure_error = (
            "; ".join(self.rewritten_structure.extraction_issues) or None
        )
        self.step_result.metadata.update(
            flatten_topic_structure(self.rewritten_structure, prefix="rewritten")
        )
        self.rewritten_vad = self.score_vad(self.rewritten_text)
        self.rewritten_vad_ready = True
        self.step_result.rewritten_vad_status = (
            "success" if self.rewritten_text not in self.vad_errors else "error"
        )
        self.step_result.rewritten_vad_error = self.vad_errors.get(self.rewritten_text)
        _record_vad(self.step_result, "rewritten", self.rewritten_vad)
        self.step_result.metadata["rewritten_vad_evaluation"] = self.vad_evaluator.records[
            self.rewritten_text
        ]
        for suffix, reference in (
            ("vs_original", self.original_text),
            ("incremental", self.previous_text),
        ):
            self.suffix, reference = (suffix, reference)
            self.step_result.metadata[f"vad_evaluation_{self.suffix}"] = compare_vad_evaluations(
                self.vad_evaluator.records[reference],
                self.vad_evaluator.records[self.rewritten_text],
                method=self.options.vad_method,
            )

    def _score_lexical_step(self):
        vs_original_metrics = calculate_stdi(
            self.original_structure,
            self.rewritten_structure,
            original_vad=self.original_vad,
            compared_vad=self.rewritten_vad,
        )
        if self.previous_rewritten_text is None:
            incremental_metrics = vs_original_metrics
        else:
            incremental_metrics = calculate_stdi(
                self.previous_structure,
                self.rewritten_structure,
                original_vad=self.previous_vad,
                compared_vad=self.rewritten_vad,
            )
        for suffix, metrics in (
            ("vs_original", vs_original_metrics),
            ("incremental", incremental_metrics),
        ):
            self.suffix, self.metrics = (suffix, metrics)
            self.vad = vad_for_branch(
                self.step_result.metadata[f"vad_evaluation_{self.suffix}"],
                "embedding",
            )
            if self.vad["status"] == "valid":
                self.metrics.update(
                    {
                        key: round(value, 6)
                        for key, value in complete_stdi(self.metrics, self.vad).items()
                    }
                )
            else:
                self.metrics.update(dict.fromkeys(STDI_COMPONENTS))
            _record_stdi_metrics(self.step_result, suffix=self.suffix, metrics=self.metrics)
        if incremental_metrics["stdi"] is not None:
            self.cumulative_stdi += incremental_metrics["stdi"]
        self.step_result.stdi_cumulative = round(self.cumulative_stdi, 6)

    def _advance_lineage(self):
        self.previous_text = self.rewritten_text
        self.previous_rewritten_text = self.rewritten_text
        self.previous_structure = self.rewritten_structure
        self.previous_vad = self.rewritten_vad
        self.previous_node_id = self.node.node_id
        self.previous_node_label = self.node_label
        self.states_by_node[self.node_id] = (
            self.previous_node_id,
            self.previous_node_label,
            self.previous_text,
            self.previous_structure,
            self.previous_vad,
            self.previous_rewritten_text,
            self.cumulative_stdi,
        )
        _emit_progress(
            self.options.progress_callback,
            (
                f"[{self.row_position}/{self.total_rows}] Step {self.step_index}/"
                f"{len(self.ordered_node_ids)} node='{self.node_label}"
                f"': rewrite processing completed; "
            )
            + (
                "STDI evaluation pending."
                if self.options.stdi_comparison_method in {"cluster", "dual", "llm"}
                else (
                    f"STDI evaluation complete; stdi_vs_original="
                    f"{self.step_result.stdi_vs_original}, stdi_incremental="
                    f"{self.step_result.stdi_incremental}."
                )
            ),
        )

    def _record_node_error(self, exc):
        self.failed_nodes.add(self.node_id)
        self.step_result.rewrite_status = "error"
        self.step_result.rewrite_error = str(exc)
        if self.rewritten_structure_ready and (not self.rewritten_vad_ready):
            self.step_result.rewritten_vad_status = "error"
            self.step_result.rewritten_vad_error = str(exc)
        elif self.step_result.rewritten_text is not None:
            self.step_result.rewritten_topic_structure_status = "error"
            self.step_result.rewritten_topic_structure_error = str(exc)
            self.step_result.rewritten_vad_status = "blocked"
            self.step_result.rewritten_vad_error = "Skipped because topic extraction failed."
        _emit_progress(
            self.options.progress_callback,
            (
                f"[{self.row_position}/{self.total_rows}] Step {self.step_index}/"
                f"{len(self.ordered_node_ids)} node='{self.node_label}': error: {exc}"
            ),
        )
