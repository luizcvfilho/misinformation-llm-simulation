from __future__ import annotations

from contextlib import nullcontext as _null_operation
from dataclasses import asdict
from pathlib import Path

from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.llm.progress import (
    emit_event,
    operation,
    progress_context,
    set_progress_context,
)
from misinformation_simulation.simulation.graph import (
    DEFAULT_SIMULATION_OUTPUT_DIR,
    GRAPH_STEP_SCHEMA_VERSION,
    STDI_COMPONENTS,
    _cancel_requested,
    _emit_progress,
    _record_dual_metrics,
    _record_stdi_metrics,
)
from misinformation_simulation.simulation.paths import project_graph_paths
from misinformation_simulation.simulation.persistence import _persist_results
from misinformation_simulation.simulation.progress import timing_key
from misinformation_simulation.simulation.types import (
    SimulationResult,
)
from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
)
from misinformation_simulation.text_metrics.vad_evaluation import (
    vad_for_branch,
)
from misinformation_simulation.topic_drift import calculate_stdi
from misinformation_simulation.topic_drift.cluster_comparison import (
    CLUSTER_STDI_COMPARISON_VERSION,
)
from misinformation_simulation.topic_drift.structured_comparison import (
    DUAL_STDI_VERSION,
    STRUCTURED_CLUSTER_VERSION,
    complete_stdi,
)


class _GraphEvaluation:
    """Evaluate prepared graph steps and persist complete or partial results."""

    def build_comparator(self, pairs):
        comparator = self.comparator_factory(
            embedder=self.options.stdi_embedder, embedding_model=self.options.stdi_embedding_model
        )
        try:
            if pairs and self.options.stdi_comparison_method != "llm":
                with operation("embeddings", model=self.options.stdi_embedding_model):
                    comparator.fit(pairs)
        except Exception:
            if self.options.stdi_comparison_method not in {"dual", "llm"}:
                raise
        return comparator

    def finish(self, comparator=None, *, evaluate=True):
        set_progress_context()
        if evaluate and self.options.stdi_comparison_method in {"dual", "llm"}:
            emit_event(
                "plan",
                entries=[
                    dict(
                        key=timing_key(
                            "judge",
                            self.options.topic_drift_model,
                            normalize_provider(self.options.topic_drift_provider),
                        ),
                        total=len(self.scoring_contexts) * 2 * self.options.stdi_judge_repeats,
                    )
                ],
            )
        if evaluate and self.scoring_contexts:
            _emit_progress(
                self.options.progress_callback,
                f"Preparing STDI evaluations for {len(self.scoring_contexts)} processed step(s).",
            )
            comparator = comparator or self.build_comparator(self.pairs)
            self._evaluate_steps(comparator)

        if _cancel_requested(self.options.cancel_check):
            self.cancelled = True
        self._annotate_results()

        self._build_result(evaluate)

        self._save_result(evaluate)

        if evaluate:
            emit_event("run_finished", cancelled=self.cancelled)
            _emit_progress(
                self.options.progress_callback,
                (
                    f"Run finished: rows_processed={self.summary['rows_processed']}"
                    f", steps_total={self.summary['steps_total']}, steps_success="
                    f"{self.summary['steps_success']}, steps_error="
                    f"{self.summary['steps_error']}."
                ),
            )
        return self.result

    def _annotate_results(self):
        self.success_count = sum(
            1 for step in self.step_results if step.rewrite_status == "success"
        )
        self.error_count = sum(1 for step in self.step_results if step.rewrite_status == "error")
        self.vad_model_name = DEFAULT_VAD_MODEL_NAME
        if self.options.vad_model_bundle is not None:
            self.vad_model_name = self.options.vad_model_bundle.model_name
        if self.options.vad_scorer is not None:
            self.vad_model_name = "custom_scorer"
        if self.options.vad_method == "llm":
            self.vad_model_name = None
        self.embedding_model_name = (
            (
                self.options.stdi_embedding_model
                if self.options.stdi_embedder is None
                else f"custom:{type(self.options.stdi_embedder).__name__}"
            )
            if self.options.stdi_comparison_method in {"cluster", "dual"}
            else None
        )
        self.comparison_version = (
            DUAL_STDI_VERSION
            if self.options.stdi_comparison_method in {"dual", "llm"}
            else (
                STRUCTURED_CLUSTER_VERSION
                if any(
                    structure.schema_version == 2
                    for context in self.scoring_contexts
                    for structure in (context[2], context[3], context[4])
                )
                else CLUSTER_STDI_COMPARISON_VERSION
            )
            if self.options.stdi_comparison_method == "cluster"
            else "lexical_v1"
        )
        for step in self.step_results:
            self.step = step
            identity = progress_context()
            if run_id := identity.get("run_id"):
                self.step.metadata.update(
                    run_id=run_id,
                    dataset_row=identity["dataset_row"],
                    graph_step_id=f"{run_id}:{self.step.node_id}",
                )
            self.step.metadata.update(
                {
                    "rewrite_model": self.step.model,
                    "rewrite_provider": self.step.provider,
                    "vad_model": self.vad_model_name,
                    "stdi_embedding_model": self.embedding_model_name,
                    "stdi_comparison_version": self.comparison_version,
                    "graph_shared_node": self.step.node_id in self.shared_node_ids,
                    "graph_step_reused": False,
                    "rewrite_operation_attributed": self.step.metadata.get(
                        "rewrite_operation_started", False
                    ),
                }
            )

    def _build_result(self, evaluate):
        self.summary = {
            "partial": not evaluate or self.cancelled,
            "schema_version": GRAPH_STEP_SCHEMA_VERSION,
            "cache_contracts": self.cache_contracts,
            "shared_cache_dir": str(self.options.shared_cache_dir)
            if self.options.shared_cache_dir is not None
            else None,
            "reuse_rewrites": self.options.reuse_rewrites,
            **self.evaluation_metadata,
            "rewrite_system_instruction": self.rewrite_prompt.system_instruction,
            "rewrite_prompt_template": self.rewrite_prompt.template,
            "rows_processed": self.rows_started,
            "cancelled": self.cancelled,
            "steps_total": len(self.step_results),
            "steps_success": self.success_count,
            "steps_error": self.error_count,
            "steps_blocked": sum(step.rewrite_status == "blocked" for step in self.step_results),
            "result_kind": "branching_graph" if self.branching else "chain",
            "rewrite_operations_started": self.rewrite_operations_started,
            "rewrite_reuse_scope": "current_graph_run",
            "planned_unique_rewrites": self.total_rows * len(self.ordered_node_ids),
            "planned_linear_rewrites": self.total_rows
            * sum(len(path) for path in self.graph_paths),
            "planned_rewrites_saved": self.total_rows
            * (sum(len(path) for path in self.graph_paths) - len(self.ordered_node_ids)),
            "vad_model": self.vad_model_name,
            "stdi_comparison_method": self.options.stdi_comparison_method,
            "stdi_comparison_version": self.comparison_version,
            "stdi_embedding_model": self.embedding_model_name,
            "stdi_complete_incremental_pairs": sum(
                step.stdi_status_incremental == "valid" for step in self.step_results
            )
            if self.options.stdi_comparison_method in {"dual", "llm"}
            else None,
            "stdi_components_branch": "selected_evaluation",
            "graph": {
                "start_node_id": self.resolved_start_node,
                "ordered_node_ids": self.ordered_node_ids,
                "edges": [asdict(edge) for edge in self.normalized_edges],
                "paths": self.graph_paths,
            },
            "nodes": [asdict(node) for node in self.options.nodes],
        }
        self.result = SimulationResult(summary=self.summary, step_results=self.step_results)
        if self.branching:
            self.result.path_results = project_graph_paths(self.result)
            self.summary["path_results"] = [
                {
                    "path_id": path.summary["path_id"],
                    "chain_code": path.summary["chain_code"],
                    "node_ids": path.summary["graph"]["ordered_node_ids"],
                }
                for path in self.result.path_results
            ]

    def _save_result(self, evaluate):
        if self.options.persist_results:
            resolved_output_dir = (
                Path(self.options.output_dir)
                if self.options.output_dir is not None
                else DEFAULT_SIMULATION_OUTPUT_DIR
            )
            set_progress_context()
            with operation("saving") if evaluate else _null_operation():
                summary_path, steps_path = _persist_results(
                    result=self.result,
                    output_dir=resolved_output_dir,
                    output_prefix=self.options.output_prefix,
                )
            self.result.summary_path = summary_path
            self.result.steps_path = steps_path
            if evaluate:
                _emit_progress(
                    self.options.progress_callback,
                    f"Saved summary to {summary_path} and step records to {steps_path}.",
                )

    def _restore_evaluation_totals(self):
        if self.row_position != self.last_row_position:
            self.cumulative_stdi = 0.0
            self.valid_steps = 0
            self.branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
            self.branch_valid_steps = {"cluster": 0, "llm_judge": 0}
            self.last_row_position = self.row_position
            self.totals_by_node = {}
        if self.branching:
            parent_totals = self.totals_by_node.get(self.parents_by_node.get(self.step.node_id))
            if parent_totals is None:
                self.cumulative_stdi, self.valid_steps = (0.0, 0)
                self.branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
                self.branch_valid_steps = {"cluster": 0, "llm_judge": 0}
            else:
                (
                    self.cumulative_stdi,
                    self.valid_steps,
                    saved_cumulative,
                    saved_valid,
                ) = parent_totals
                self.branch_cumulative = saved_cumulative.copy()
                self.branch_valid_steps = saved_valid.copy()

    def _score_dual_step(self, comparator):
        _emit_progress(
            self.options.progress_callback,
            (
                f"Evaluating {self.options.stdi_comparison_method} STDI for '"
                f"{self.step.news_id}' node '{self.step.node_id}'."
            ),
        )
        common = {
            "modified_text": self.step.rewritten_text,
            "modified_structure": self.rewritten,
            "modified_vad": self.rewritten_vad,
            "title": self.step.metadata.get("extraction_title"),
            "comparator": comparator,
            "model": self.options.topic_drift_model,
            "provider": normalize_provider(self.options.topic_drift_provider),
            "api_key": self.options.topic_drift_api_key,
            "base_url": self.options.topic_drift_base_url,
            "retry_attempts": self.options.retry_attempts,
            "before_request_hook": self.judge_limiter.acquire,
            "judge_repeats": self.options.stdi_judge_repeats,
            "cancel_check": self.options.cancel_check,
            "cache": self.evaluation_cache,
            "comparison_method": self.options.stdi_comparison_method,
            **(
                {"judge_fn": self.options.stdi_judge_fn}
                if self.options.stdi_judge_fn is not None
                else {}
            ),
        }
        set_progress_context(
            news_id=self.step.news_id,
            row=self.row_position,
            node_id=self.step.node_id,
            comparison="vs original",
        )
        vs_original_dual = self.compare_dual_stdi(
            original_text=self.step.metadata["original_text"],
            original_structure=self.original,
            original_vad=self.original_vad,
            vad_evaluation=self.step.metadata["vad_evaluation_vs_original"],
            **common,
        )
        if _cancel_requested(self.options.cancel_check):
            self.cancelled = True
            _record_dual_metrics(self.step, "vs_original", vs_original_dual)
            return
        set_progress_context(
            news_id=self.step.news_id,
            row=self.row_position,
            node_id=self.step.node_id,
            comparison="vs previous",
        )
        incremental_dual = self.compare_dual_stdi(
            original_text=self.step.source_text,
            original_structure=self.previous,
            original_vad=self.previous_vad,
            vad_evaluation=self.step.metadata["vad_evaluation_incremental"],
            **common,
        )
        _record_dual_metrics(self.step, "vs_original", vs_original_dual)
        _record_dual_metrics(self.step, "incremental", incremental_dual)
        _emit_progress(
            self.options.progress_callback,
            (
                f"News '{self.step.news_id}' node '{self.step.node_id}"
                f"': STDI evaluation complete; vs_original="
                f"{vs_original_dual['status']}, incremental="
                f"{incremental_dual['status']}."
            ),
        )
        if incremental_dual["stdi"] is not None:
            self.cumulative_stdi += incremental_dual["stdi"]
            self.valid_steps += 1
        self.step.stdi_cumulative = self.cumulative_stdi
        self.step.stdi_cumulative_valid_steps = self.valid_steps
        self.step.stdi_chain_complete = self.valid_steps == self.step.step_index
        for branch, name in (("embedding", "cluster"), ("llm_judge", "llm_judge")):
            self.metrics = incremental_dual[branch].get("metrics")
            if self.metrics is not None:
                self.branch_cumulative[name] += self.metrics["stdi"]
                self.branch_valid_steps[name] += 1
            setattr(
                self.step,
                f"stdi_{name}_cumulative",
                self.branch_cumulative[name],
            )
            setattr(
                self.step,
                f"stdi_{name}_cumulative_valid_steps",
                self.branch_valid_steps[name],
            )
            setattr(
                self.step,
                f"stdi_{name}_chain_complete",
                self.branch_valid_steps[name] == self.step.step_index,
            )
        self.totals_by_node[self.step.node_id] = (
            self.cumulative_stdi,
            self.valid_steps,
            self.branch_cumulative.copy(),
            self.branch_valid_steps.copy(),
        )

    def _score_cluster_step(self, comparator):
        vs_original = calculate_stdi(
            self.original,
            self.rewritten,
            original_vad=self.original_vad,
            compared_vad=self.rewritten_vad,
            component_overrides=comparator.compare(self.original, self.rewritten).component_drifts,
        )
        incremental = calculate_stdi(
            self.previous,
            self.rewritten,
            original_vad=self.previous_vad,
            compared_vad=self.rewritten_vad,
            component_overrides=comparator.compare(self.previous, self.rewritten).component_drifts,
        )
        for suffix, metrics in (
            ("vs_original", vs_original),
            ("incremental", incremental),
        ):
            self.suffix, self.metrics = (suffix, metrics)
            self.vad = vad_for_branch(
                self.step.metadata[f"vad_evaluation_{self.suffix}"], "embedding"
            )
            self.step.metadata[f"stdi_vad_source_{self.suffix}"] = self.vad["source"]
            setattr(self.step, f"stdi_status_{self.suffix}", self.vad["status"])
            if self.vad["status"] == "valid":
                self.metrics.update(
                    {
                        key: round(value, 6)
                        for key, value in complete_stdi(self.metrics, self.vad).items()
                    }
                )
            else:
                self.metrics.update(dict.fromkeys(STDI_COMPONENTS))
        _record_stdi_metrics(self.step, suffix="vs_original", metrics=vs_original)
        _record_stdi_metrics(self.step, suffix="incremental", metrics=incremental)
        if incremental["stdi"] is not None:
            self.cumulative_stdi += incremental["stdi"]
        self.step.stdi_cumulative = round(self.cumulative_stdi, 6)
        self.totals_by_node[self.step.node_id] = (
            self.cumulative_stdi,
            self.valid_steps,
            self.branch_cumulative.copy(),
            self.branch_valid_steps.copy(),
        )

    def _evaluate_steps(self, comparator):
        self.last_row_position = 0
        self.cumulative_stdi = 0.0
        self.valid_steps = 0
        self.branch_cumulative = {"cluster": 0.0, "llm_judge": 0.0}
        self.branch_valid_steps = {"cluster": 0, "llm_judge": 0}
        self.totals_by_node: dict[str, tuple] = {}
        for (
            row_position,
            step,
            original,
            previous,
            rewritten,
            original_vad,
            previous_vad,
            rewritten_vad,
        ) in self.scoring_contexts:
            (
                self.row_position,
                self.step,
                self.original,
                self.previous,
                self.rewritten,
                self.original_vad,
                self.previous_vad,
                self.rewritten_vad,
            ) = (
                row_position,
                step,
                original,
                previous,
                rewritten,
                original_vad,
                previous_vad,
                rewritten_vad,
            )
            set_progress_context(
                news_id=self.step.news_id, row=self.row_position, node_id=self.step.node_id
            )
            if self.options.stdi_comparison_method in {"dual", "llm"} and _cancel_requested(
                self.options.cancel_check
            ):
                self.cancelled = True
                break
            self._restore_evaluation_totals()

            if self.options.stdi_comparison_method in {"dual", "llm"}:
                self._score_dual_step(comparator)
                if self.cancelled:
                    break

                continue
            self._score_cluster_step(comparator)
