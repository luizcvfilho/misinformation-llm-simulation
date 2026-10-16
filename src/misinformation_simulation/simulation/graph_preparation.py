from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from misinformation_simulation.config.prompts import (
    STRUCTURED_EXTRACTION_VERSION,
    resolve_graph_personality_prompt,
    resolve_graph_rewrite_prompt,
)
from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, Provider
from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.llm.flex_fallback import flex_fallback_settings
from misinformation_simulation.llm.progress import (
    emit_event,
    mark_cached,
    mark_failed,
    operation,
    timed_operation,
)
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.service_tier import configured_openai_service_tier
from misinformation_simulation.simulation.graph import (
    DEFAULT_SIMULATION_OUTPUT_DIR,
    DEFAULT_STDI_EMBEDDING_MODEL,
    GRAPH_REWRITE_TEMPERATURE,
    CancelCheck,
    ProgressCallback,
    WorkProgressCallback,
    _emit_progress,
)
from misinformation_simulation.simulation.progress import simulation_plan
from misinformation_simulation.simulation.topology import (
    _normalize_edges,
    _normalize_nodes,
    _resolve_start_node,
    _root_to_leaf_paths,
    _topological_path,
)
from misinformation_simulation.simulation.types import (
    SimulationEdge,
    SimulationNode,
    SimulationStepResult,
)
from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
    VADModelBundle,
    VADScore,
)
from misinformation_simulation.text_metrics.vad_evaluation import (
    VADTextEvaluator,
    evaluation_score,
    validate_vad_method,
)
from misinformation_simulation.topic_drift.cache_versions import runtime_cache_contracts
from misinformation_simulation.topic_drift.cluster_comparison import (
    TextEmbedder,
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
    validate_judge_repeats,
)


@dataclass
class _GraphRunOptions:
    nodes: list[SimulationNode]
    edges: list[SimulationEdge] | None = None
    start_node_id: str | None = None
    text_column: str | None = "description"
    title_column: str = "title"
    news_id_column: str | None = None
    max_rows: int | None = None
    sleep_seconds: float = 0.0
    max_requests_per_minute: int | None = None
    retry_attempts: int = 5
    allow_title_fallback: bool = True
    rewrite_mode: str = "faithful"
    topic_drift_model: str = DEFAULT_LLM_MODEL
    topic_drift_provider: Provider | str = DEFAULT_LLM_PROVIDER
    topic_drift_api_key: str | None = None
    topic_drift_base_url: str | None = None
    stdi_comparison_method: str = "dual"
    stdi_embedding_model: str = DEFAULT_STDI_EMBEDDING_MODEL
    stdi_embedder: TextEmbedder | None = None
    stdi_cache_dir: Path | str | None = None
    shared_cache_dir: Path | str | None = None
    reuse_rewrites: bool = False
    stdi_judge_fn: Callable[..., Any] | None = None
    stdi_judge_repeats: int = DEFAULT_JUDGE_REPEATS
    vad_model_bundle: VADModelBundle | None = None
    vad_scorer: Callable[[str], VADScore] | None = None
    vad_method: str = "model"
    vad_llm_model: str | None = None
    vad_llm_provider: Provider | str | None = None
    vad_llm_api_key: str | None = None
    vad_llm_base_url: str | None = None
    vad_llm_scorer: Callable[[str], Any] | None = None
    output_dir: Path | str | None = None
    output_prefix: str = "simulation"
    persist_results: bool = True
    progress_callback: ProgressCallback | None = None
    work_progress_callback: WorkProgressCallback | None = None
    cancel_check: CancelCheck | None = None
    defer_evaluation: bool = False


class _GraphPreparation:
    """Validate topology and prepare the shared evaluators and run metadata."""

    def _extract_structure(self, **kwargs: Any) -> TopicStructure:
        if self.options.stdi_comparison_method not in {"dual", "cluster", "llm"}:
            return self.extract_topic_structure(**kwargs)
        inputs = {
            key: str(value) if key in {"model", "provider"} else value
            for key, value in kwargs.items()
            if key in {"text", "title", "model", "provider", "base_url"}
        }
        inputs["version"] = STRUCTURED_EXTRACTION_VERSION
        saved = self.evaluation_cache.get("extraction", inputs)
        if saved is not None:
            mark_cached()
            return _build_topic_structure(saved)
        try:
            structure = self.extract_topic_structure(
                **kwargs, structured=True, before_request_hook=self.judge_limiter.acquire
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
            self.evaluation_cache.put("extraction", inputs, topic_structure_to_dict(structure))
        if structure.extraction_status == "failed":
            mark_failed("; ".join(structure.extraction_issues))
        return structure

    def _score_vad(self, text: str) -> VADScore:
        memory_hit = text in self.vad_evaluator.records
        evaluation = self.vad_evaluator.evaluate(text)
        methods = (
            ("model", "llm") if self.options.vad_method == "dual" else (self.options.vad_method,)
        )
        if memory_hit or all(evaluation[method].get("cache_hit") for method in methods):
            mark_cached()
        source = "model" if self.options.vad_method == "dual" else self.options.vad_method
        self.vad_cache[text] = evaluation_score(evaluation, source)
        required = (
            ("model", "llm") if self.options.vad_method == "dual" else (self.options.vad_method,)
        )
        errors = [
            evaluation[method].get(
                "error", "VAD scoring must return valence, arousal, and dominance."
            )
            for method in required
            if evaluation[method]["status"] != "valid"
        ]
        if errors:
            mark_failed("; ".join(errors))
            self.vad_errors[text] = "; ".join(errors)
            if self.options.vad_method == "model" and self.options.stdi_comparison_method not in {
                "dual",
                "llm",
            }:
                raise ValueError(self.vad_errors[text])
        return self.vad_cache[text]

    def _validate_input(self):
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("'df' must be a pandas.DataFrame.")
        if self.df.empty:
            raise ValueError("The DataFrame is empty.")
        if (
            self.options.max_requests_per_minute is not None
            and self.options.max_requests_per_minute <= 0
        ):
            raise ValueError("'max_requests_per_minute' must be greater than zero when provided.")
        if self.options.stdi_comparison_method not in {"dual", "cluster", "llm", "lexical"}:
            raise ValueError(
                "'stdi_comparison_method' must be 'dual', 'cluster', 'llm' or 'lexical'."
            )
        validate_vad_method(self.options.vad_method)
        validate_judge_repeats(self.options.stdi_judge_repeats)

    def _prepare_evaluators(self):
        cache_directory = self.options.stdi_cache_dir
        if (
            cache_directory is None
            and self.options.persist_results
            and (
                self.options.stdi_comparison_method in {"dual", "llm"}
                or self.options.vad_method != "model"
            )
        ):
            cache_directory = (
                Path(self.options.output_dir or DEFAULT_SIMULATION_OUTPUT_DIR) / "evaluation_cache"
            )
        self.cache_contracts = runtime_cache_contracts()
        if self.options.stdi_judge_fn is not None:
            self.cache_contracts["judge"]["custom_judge_session"] = uuid4().hex
        self.evaluation_cache = EvaluationCache(
            cache_directory,
            shared_directory=self.options.shared_cache_dir,
            compatibility=self.cache_contracts,
        )
        self.vad_cache: dict[str, VADScore] = {}
        self.vad_errors: dict[str, str] = {}
        self.judge_limiter = MinuteRateLimiter(self.options.max_requests_per_minute)
        self.vad_provider = self.options.vad_llm_provider or self.options.topic_drift_provider
        reuse_vad_endpoint = normalize_provider(self.vad_provider) == normalize_provider(
            self.options.topic_drift_provider
        )
        self.vad_evaluator = VADTextEvaluator(
            method=self.options.vad_method,
            model_bundle=self.options.vad_model_bundle,
            model_scorer=self.options.vad_scorer,
            llm_scorer=self.options.vad_llm_scorer,
            llm_model=self.options.vad_llm_model or self.options.topic_drift_model,
            llm_provider=self.vad_provider,
            api_key=self.options.vad_llm_api_key
            or (self.options.topic_drift_api_key if reuse_vad_endpoint else None),
            base_url=self.options.vad_llm_base_url
            or (self.options.topic_drift_base_url if reuse_vad_endpoint else None),
            before_request_hook=self.judge_limiter.acquire,
            cache=self.evaluation_cache,
        )
        self.extract_structure = timed_operation(
            "extraction",
            model=self.options.topic_drift_model,
            provider=normalize_provider(self.options.topic_drift_provider),
        )(self._extract_structure)
        self.vad_timing_model = (
            (
                f"{self.options.vad_method}:"
                f"{self.options.vad_llm_model or self.options.topic_drift_model}"
            )
            if self.options.vad_method != "model"
            else DEFAULT_VAD_MODEL_NAME
        )
        self.score_vad = timed_operation(
            "vad",
            model=self.vad_timing_model,
            provider=normalize_provider(self.vad_provider)
            if self.options.vad_method != "model"
            else "local",
        )(self._score_vad)

    def _prepare_graph(self):
        self.rewrite_prompt = resolve_graph_rewrite_prompt(self.options.rewrite_mode)
        self.rewrite_metadata = {
            "rewrite_mode": self.rewrite_prompt.mode,
            "rewrite_prompt_version": self.rewrite_prompt.version,
            "rewrite_temperature_requested": GRAPH_REWRITE_TEMPERATURE,
            "rewrite_original_title_context": self.rewrite_prompt.original_title_context,
        }
        self.nodes_by_id = _normalize_nodes(self.options.nodes)
        self.effective_personalities = {
            node_id: resolve_graph_personality_prompt(
                node.personality, rewrite_mode=self.options.rewrite_mode
            )
            for node_id, node in self.nodes_by_id.items()
        }
        self.normalized_edges = _normalize_edges(self.nodes_by_id, self.options.edges)
        self.resolved_start_node = _resolve_start_node(
            self.nodes_by_id, self.normalized_edges, self.options.start_node_id
        )
        self.ordered_node_ids = _topological_path(
            self.nodes_by_id, self.normalized_edges, self.resolved_start_node
        )
        self.graph_paths = _root_to_leaf_paths(self.ordered_node_ids, self.normalized_edges)
        self.parents_by_node = {edge.target: edge.source for edge in self.normalized_edges}
        self.depths_by_node: dict[str, int] = {}
        for node_id in self.ordered_node_ids:
            self.node_id = node_id
            self.depths_by_node[self.node_id] = (
                self.depths_by_node.get(self.parents_by_node.get(self.node_id), 0) + 1
            )
        self.branching = len(self.graph_paths) > 1
        self.shared_node_ids = {
            node_id
            for node_id in self.ordered_node_ids
            if sum(node_id in path for path in self.graph_paths) > 1
        }
        self.resolved_text_column = self.options.text_column or choose_news_text_column(self.df)
        rows_to_process = (
            len(self.df)
            if self.options.max_rows is None
            else min(len(self.df), self.options.max_rows)
        )
        _emit_progress(
            self.options.progress_callback,
            (
                f"Starting interaction graph run with {rows_to_process} row(s) and "
                f"{len(self.ordered_node_ids)} node(s)."
            ),
        )

    def _prepare_clients(self):
        self.clients_by_node_id: dict[str, tuple[str, Any]] = {}
        self.limiters_by_node_id: dict[str, MinuteRateLimiter] = {}
        with operation("preparation"):
            for node in self.options.nodes:
                self.node = node
                self.provider_normalized = normalize_provider(self.node.provider)
                self.clients_by_node_id[self.node.node_id] = (
                    self.provider_normalized,
                    self.create_llm_client(
                        provider=self.provider_normalized,
                        api_key=self.node.api_key,
                        base_url=self.node.base_url,
                    )[1],
                )
                self.limiters_by_node_id[self.node.node_id] = MinuteRateLimiter(
                    self.options.max_requests_per_minute
                )

    def _initialize_results(self):
        self.evaluation_metadata = {
            "openai_service_tier": configured_openai_service_tier(),
            **flex_fallback_settings(),
            "vad_method": self.options.vad_method,
            "vad_evaluation_version": "vad_evaluation_v1",
            "vad_llm_model": str(self.options.vad_llm_model or self.options.topic_drift_model)
            if self.options.vad_method != "model"
            else None,
            "vad_llm_provider": normalize_provider(
                self.options.vad_llm_provider or self.options.topic_drift_provider
            )
            if self.options.vad_method != "model"
            else None,
            "stdi_judge_repeats": self.options.stdi_judge_repeats
            if self.options.stdi_comparison_method in {"dual", "llm"}
            else None,
            "stdi_judge_aggregation": "mean_complete_metrics"
            if self.options.stdi_comparison_method in {"dual", "llm"}
            else None,
            "topic_drift_model": str(self.options.topic_drift_model),
            "topic_drift_provider": normalize_provider(self.options.topic_drift_provider),
            "topic_extraction_original_title_context": self.rewrite_prompt.original_title_context,
            **self.rewrite_metadata,
        }
        self.step_results: list[SimulationStepResult] = []
        self.scoring_contexts: list[
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
        self.target_indexes = list(self.df.index)
        if self.options.max_rows is not None:
            self.target_indexes = self.target_indexes[: self.options.max_rows]
        self.total_rows = len(self.target_indexes)
        emit_event(
            "plan",
            entries=simulation_plan(
                self.total_rows,
                list(self.nodes_by_id.values()),
                method=self.options.stdi_comparison_method,
                repeats=self.options.stdi_judge_repeats,
                model=self.options.topic_drift_model,
                provider=normalize_provider(self.options.topic_drift_provider),
                vad_model=self.vad_timing_model,
                vad_provider=normalize_provider(self.vad_provider)
                if self.options.vad_method != "model"
                else "local",
                embedding_model=self.options.stdi_embedding_model,
                sleep_seconds=self.options.sleep_seconds,
                persist=self.options.persist_results,
            ),
        )
