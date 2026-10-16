from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from misinformation_simulation.config.prompts import STRUCTURED_EXTRACTION_VERSION
from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.text_metrics.vad import (
    VADScore,
)
from misinformation_simulation.text_metrics.vad_evaluation import (
    VADTextEvaluator,
    compare_vad_evaluations,
    evaluation_score,
    vad_pair_columns,
)
from misinformation_simulation.topic_drift.cluster_comparison import (
    ClusterSTDIComparator,
    TextEmbedder,
    TopicStructurePair,
    TransformerTextEmbedder,
)
from misinformation_simulation.topic_drift.extraction import _build_topic_structure
from misinformation_simulation.topic_drift.metrics import calculate_stdi
from misinformation_simulation.topic_drift.models import (
    TopicStructure,
    empty_topic_structure,
    flatten_topic_structure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache, input_hash
from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison
from misinformation_simulation.topic_drift.structured_comparison import (
    CONTENT_COMPONENTS,
    DUAL_STDI_VERSION,
    FORMULA_VERSION,
    NUMERICAL_TOLERANCE,
    StructuredEmbeddingComparator,
    compare_dual_stdi,
    structure_issues,
)

if TYPE_CHECKING:
    from misinformation_simulation.topic_drift.comparison_workflow import ComparisonWorkflowResult


def run_dual_workflow(
    df: pd.DataFrame,
    *,
    original_text_column: str,
    modified_text_column: str,
    title_column: str,
    pair_id_column: str | None,
    extraction_model: str,
    extraction_provider: str,
    api_key: str | None,
    base_url: str | None,
    extraction_fn: Callable[..., TopicStructure],
    judge_model: str,
    judge_provider: str,
    judge_fn: Callable[..., SemanticSTDIComparison],
    embedder: TextEmbedder | None,
    embedding_model: str,
    n_clusters: int | None,
    random_state: int,
    reuse_structures: bool,
    progress_callback: Callable[[str], None] | None,
    cache_dir: Path | str | None,
    vad_scorer: Callable[[str], VADScore] | None,
    uncached_judge: bool,
    judge_repeats: int,
    vad_method: str = "model",
    vad_llm_model: str | None = None,
    vad_llm_provider: str | None = None,
    vad_llm_scorer=None,
    vad_llm_api_key: str | None = None,
    vad_llm_base_url: str | None = None,
) -> ComparisonWorkflowResult:
    return _DualWorkflow(
        df=df,
        original_text_column=original_text_column,
        modified_text_column=modified_text_column,
        title_column=title_column,
        pair_id_column=pair_id_column,
        extraction_model=extraction_model,
        extraction_provider=extraction_provider,
        api_key=api_key,
        base_url=base_url,
        extraction_fn=extraction_fn,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_fn=judge_fn,
        embedder=embedder,
        embedding_model=embedding_model,
        n_clusters=n_clusters,
        random_state=random_state,
        reuse_structures=reuse_structures,
        progress_callback=progress_callback,
        cache_dir=cache_dir,
        vad_scorer=vad_scorer,
        uncached_judge=uncached_judge,
        judge_repeats=judge_repeats,
        vad_method=vad_method,
        vad_llm_model=vad_llm_model,
        vad_llm_provider=vad_llm_provider,
        vad_llm_scorer=vad_llm_scorer,
        vad_llm_api_key=vad_llm_api_key,
        vad_llm_base_url=vad_llm_base_url,
    ).run()


class _DualWorkflow:
    """Prepare shared structures and collect independent comparison branches."""

    def __init__(
        self,
        *,
        df,
        original_text_column,
        modified_text_column,
        title_column,
        pair_id_column,
        extraction_model,
        extraction_provider,
        api_key,
        base_url,
        extraction_fn,
        judge_model,
        judge_provider,
        judge_fn,
        embedder,
        embedding_model,
        n_clusters,
        random_state,
        reuse_structures,
        progress_callback,
        cache_dir,
        vad_scorer,
        uncached_judge,
        judge_repeats,
        vad_method,
        vad_llm_model,
        vad_llm_provider,
        vad_llm_scorer,
        vad_llm_api_key,
        vad_llm_base_url,
    ):
        self.df = df
        self.original_text_column = original_text_column
        self.modified_text_column = modified_text_column
        self.title_column = title_column
        self.pair_id_column = pair_id_column
        self.extraction_model = extraction_model
        self.extraction_provider = extraction_provider
        self.api_key = api_key
        self.base_url = base_url
        self.extraction_fn = extraction_fn
        self.judge_model = judge_model
        self.judge_provider = judge_provider
        self.judge_fn = judge_fn
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.reuse_structures = reuse_structures
        self.progress_callback = progress_callback
        self.cache_dir = cache_dir
        self.vad_scorer = vad_scorer
        self.uncached_judge = uncached_judge
        self.judge_repeats = judge_repeats
        self.vad_method = vad_method
        self.vad_llm_model = vad_llm_model
        self.vad_llm_provider = vad_llm_provider
        self.vad_llm_scorer = vad_llm_scorer
        self.vad_llm_api_key = vad_llm_api_key
        self.vad_llm_base_url = vad_llm_base_url

    def run(self):
        from misinformation_simulation.topic_drift.comparison_workflow import (
            ComparisonWorkflowResult,
            _as_non_empty_text,
            _resolved_pair_ids,
            topic_structure_from_row,
        )

        self._as_non_empty_text = _as_non_empty_text
        self._resolved_pair_ids = _resolved_pair_ids
        self.topic_structure_from_row = topic_structure_from_row
        self.original_column = self.original_text_column
        self.modified_column = self.modified_text_column
        missing = {self.original_column, self.modified_column} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
        self._prepare_result()

        self._prepare_evaluator()

        for row_index, row in self.result.iterrows():
            self.row_index, self.row = (row_index, row)
            self._prepare_pair()

        self._fit_comparators()

        for row_index, texts, structures, scores, title in self.contexts:
            self.row_index, self.texts, self.structures, self.scores, self.title = (
                row_index,
                texts,
                structures,
                scores,
                title,
            )
            self._evaluate_pair()

        self._add_cluster_aliases()

        manifest = {
            "method": "dual",
            "version": DUAL_STDI_VERSION,
            "formula_version": FORMULA_VERSION,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "rows": len(self.result),
            "valid_pairs": int((self.result.get("comparison_status") == "valid").sum()),
            "embedding_model": self.embedding_model,
            "judge_model": str(self.judge_model),
            "judge_provider": str(self.judge_provider),
            "random_state": self.random_state,
            "n_clusters": self.n_clusters,
            "extraction_version": STRUCTURED_EXTRACTION_VERSION,
            "extraction_model": str(self.extraction_model),
            "extraction_provider": str(self.extraction_provider),
            "branch_weights": [0.5, 0.5],
            "polarity_weight": 0.2,
            "duration_weight": 0.2,
            "numeric_weight": 0.2,
            "numeric_aggregation": "maximum",
            "legacy_component_columns": "embedding",
            "uncached_judge": self.uncached_judge,
            "judge_repeats": self.judge_repeats,
            "judge_aggregation": "mean_complete_metrics",
            "vad_model": ("custom_scorer" if self.vad_scorer else "RobroKools/vad-bert")
            if self.vad_method != "llm"
            else None,
            "vad_method": self.vad_method,
            "vad_llm_model": self.evaluator.llm_model if self.vad_method != "model" else None,
            "vad_llm_provider": self.evaluator.llm_provider if self.vad_method != "model" else None,
            "dual_aggregation": "mean_complete_branches_with_selected_vad",
        }
        return ComparisonWorkflowResult(self.result, manifest)

    def _prepare_result(self):
        self.result = self.df.copy()
        for key in (
            "stdi",
            "method_gap",
            "dual_evaluation_json",
            "comparison_status",
            "embedding_status",
            "llm_judge_status",
            "embedding_error",
            "llm_judge_error",
            "cluster_evaluation_json",
            "llm_judge_evaluation_json",
            "comparison_error",
            "historical_embedding_stdi",
            "historical_lexical_stdi",
        ):
            self.key = key
            self.result[self.key] = pd.Series(index=self.result.index, dtype=object)
        for branch in ("embedding", "llm_judge", "dual"):
            self.branch = branch
            for component in (
                *CONTENT_COMPONENTS,
                "contradiction_drift",
                "valence_drift",
                "arousal_drift",
                "dominance_drift",
                "vad_drift",
                "content_drift",
                "stdi",
            ):
                self.result[f"{self.branch}_{component}"] = pd.Series(
                    index=self.result.index, dtype=object
                )
        for key in (
            *CONTENT_COMPONENTS,
            "contradiction_drift",
            "vad_drift",
            "content_drift",
            "original_vad_error",
            "modified_vad_error",
        ):
            self.key = key
            self.result[self.key] = pd.Series(index=self.result.index, dtype=object)
        self.result["comparison_method"] = "dual"
        self.result["pair_id"] = self._resolved_pair_ids(self.df, self.pair_id_column)

    def _fit_comparators(self):
        resolved_embedder = self.embedder or TransformerTextEmbedder(self.embedding_model)
        self.comparator = StructuredEmbeddingComparator(
            embedder=resolved_embedder,
            embedding_model=self.embedding_model,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        try:
            self.comparator.fit(self.prepared)
        except Exception:
            pass
        self.historical_comparator = ClusterSTDIComparator(
            embedder=resolved_embedder,
            embedding_model=self.embedding_model,
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        try:
            self.historical_comparator.fit(self.prepared)
        except Exception:
            self.historical_comparator = None

    def _prepare_pair(self):
        self.title = self._as_non_empty_text(self.row.get(self.title_column)) or None
        self.structures = []
        self.scores = []
        self.texts = []
        try:
            for prefix, column in (
                ("original", self.original_column),
                ("modified", self.modified_column),
            ):
                self.prefix, self.column = (prefix, column)
                self._prepare_text()

            self.prepared.append(TopicStructurePair(str(self.row["pair_id"]), *self.structures))
            self.contexts.append(
                (self.row_index, self.texts, self.structures, self.scores, self.title)
            )
        except Exception as exc:
            self.result.at[self.row_index, "comparison_status"] = "failed"
            self.result.at[self.row_index, "comparison_error"] = str(exc)

    def _prepare_text(self):
        self.text = self._as_non_empty_text(self.row[self.column])
        if not self.text:
            raise ValueError("Both texts must be non-empty")
        self.texts.append(self.text)
        self._resolve_structure()

        self.structures.append(self.structure)
        self._write_structure()

        self._score_text()

    def _resolve_structure(self):
        inputs = {
            "text": self.text,
            "title": self.title,
            "model": str(self.extraction_model),
            "provider": str(self.extraction_provider),
            "base_url": self.base_url,
            "version": STRUCTURED_EXTRACTION_VERSION,
        }
        try:
            self.structure = (
                self.topic_structure_from_row(self.row, prefix=self.prefix)
                if self.reuse_structures
                else None
            )
        except (ValueError, TypeError):
            self.structure = None
        if self.structure is not None and (
            structure_issues(self.structure)
            or self.structure.provenance.get("input_sha256")
            != input_hash({"text": self.text, "title": self.title})
            or self.structure.provenance.get("model") != str(self.extraction_model)
            or (self.structure.provenance.get("provider") != str(self.extraction_provider))
            or (self.structure.provenance.get("base_url") != self.base_url)
            or (self.structure.provenance.get("prompt_version") != STRUCTURED_EXTRACTION_VERSION)
        ):
            self.structure = None
        saved = self.cache.get("extraction", inputs) if self.reuse_structures else None
        if self.structure is None and saved is not None:
            self.structure = _build_topic_structure(saved)
        if self.structure is None:
            try:
                self.structure = self.extraction_fn(
                    text=self.text,
                    title=self.title,
                    model=self.extraction_model,
                    provider=self.extraction_provider,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    structured=True,
                )
                self.cache.put("extraction", inputs, topic_structure_to_dict(self.structure))
            except Exception as exc:
                self.structure = empty_topic_structure()
                self.structure.schema_version = 2
                self.structure.extraction_status = "failed"
                self.structure.extraction_issues = [str(exc)]
                self.structure.provenance = getattr(exc, "provenance", {})

    def _write_structure(self):
        for key, value in flatten_topic_structure(self.structure, prefix=self.prefix).items():
            self.key, self.value = (key, value)
            if self.key not in self.result:
                self.result[self.key] = pd.Series(index=self.result.index, dtype=object)
            else:
                self.result[self.key] = self.result[self.key].astype(object)
            self.result.at[self.row_index, self.key] = self.value

    def _score_text(self):
        if self.text not in self.vad_scores:
            self.evaluation = self.evaluator.evaluate(self.text)
            self.source = "model" if self.vad_method == "dual" else self.vad_method
            self.vad_scores[self.text] = evaluation_score(self.evaluation, self.source)
            required = ("model", "llm") if self.vad_method == "dual" else (self.vad_method,)
            errors = [
                self.evaluation[method].get("error", "Incomplete VAD scores.")
                for method in required
                if self.evaluation[method]["status"] != "valid"
            ]
            if errors:
                self.vad_errors[self.text] = "; ".join(errors)
        self.column = f"{self.prefix}_vad_evaluation_json"
        if self.column not in self.result:
            self.result[self.column] = pd.Series(index=self.result.index, dtype=object)
        self.result.at[self.row_index, self.column] = json.dumps(
            self.evaluator.records[self.text], ensure_ascii=False
        )
        self.scores.append(self.vad_scores[self.text])
        self.result.at[self.row_index, f"{self.prefix}_vad_error"] = self.vad_errors.get(self.text)
        if self.scores[-1] is not None:
            for dimension, value in asdict(self.scores[-1]).items():
                dimension, self.value = (dimension, value)
                self.result.at[self.row_index, f"{self.prefix}_vad_{dimension}"] = self.value

    def _evaluate_pair(self):
        callback = self.progress_callback
        if callback:
            callback(f"Evaluating dual STDI pair '{self.result.at[self.row_index, 'pair_id']}'.")
        self.evaluation = compare_dual_stdi(
            original_text=self.texts[0],
            modified_text=self.texts[1],
            title=self.title,
            original_structure=self.structures[0],
            modified_structure=self.structures[1],
            original_vad=self.scores[0],
            modified_vad=self.scores[1],
            comparator=self.comparator,
            model=self.judge_model,
            provider=self.judge_provider,
            api_key=self.api_key,
            base_url=self.base_url,
            judge_fn=self.judge_fn,
            cache=self.cache,
            uncached_judge=self.uncached_judge,
            judge_repeats=self.judge_repeats,
            vad_evaluation=compare_vad_evaluations(
                self.evaluator.records[self.texts[0]],
                self.evaluator.records[self.texts[1]],
                method=self.vad_method,
            ),
        )
        self._write_evaluation()

        self.result.at[self.row_index, "historical_lexical_stdi"] = calculate_stdi(
            *self.structures, original_vad=self.scores[0], compared_vad=self.scores[1]
        )["stdi"]
        if self.historical_comparator is not None and (
            not (structure_issues(self.structures[0]) or structure_issues(self.structures[1]))
        ):
            self.result.at[self.row_index, "historical_embedding_stdi"] = calculate_stdi(
                *self.structures,
                original_vad=self.scores[0],
                compared_vad=self.scores[1],
                component_overrides=self.historical_comparator.compare(
                    *self.structures
                ).component_drifts,
            )["stdi"]

    def _write_evaluation(self):
        for key, value in {
            "comparison_method": "dual",
            "comparison_status": self.evaluation["status"],
            "stdi": self.evaluation["stdi"],
            "method_gap": self.evaluation["method_gap"],
            "dual_evaluation_json": json.dumps(
                self.evaluation, ensure_ascii=False, allow_nan=False
            ),
        }.items():
            self.key, self.value = (key, value)
            if self.key not in self.result:
                self.result[self.key] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, self.key] = self.value
        for branch in ("embedding", "llm_judge"):
            self.branch = branch
            self._write_branch_evaluation()

        for key, value in vad_pair_columns(self.evaluation["vad_evaluation"]).items():
            self.key, self.value = (key, value)
            if self.key not in self.result:
                self.result[self.key] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, self.key] = self.value
        for branch, source in self.evaluation["vad_sources"].items():
            self.branch, self.source = (branch, source)
            self.column = f"{self.branch}_vad_source"
            if self.column not in self.result:
                self.result[self.column] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, self.column] = self.source
        for key, value in (self.evaluation.get("metrics") or {}).items():
            self.key, self.value = (key, value)
            self.column = f"dual_{self.key}"
            if self.column not in self.result:
                self.result[self.column] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, self.column] = self.value

    def _prepare_evaluator(self):
        self.cache = EvaluationCache(self.cache_dir)
        vad_provider = self.vad_llm_provider or self.judge_provider
        reuse_vad_endpoint = normalize_provider(vad_provider) == normalize_provider(
            self.extraction_provider
        )
        self.evaluator = VADTextEvaluator(
            method=self.vad_method,
            model_scorer=self.vad_scorer,
            llm_scorer=self.vad_llm_scorer,
            llm_model=self.vad_llm_model or self.judge_model,
            llm_provider=vad_provider,
            cache=self.cache,
            api_key=self.vad_llm_api_key or (self.api_key if reuse_vad_endpoint else None),
            base_url=self.vad_llm_base_url or (self.base_url if reuse_vad_endpoint else None),
        )
        self.prepared = []
        self.vad_scores = {}
        self.vad_errors = {}
        self.contexts = []

    def _add_cluster_aliases(self):
        cluster_aliases = {
            column.replace("embedding_", "cluster_", 1): self.result[column]
            for column in self.result.columns
            if column.startswith("embedding_")
        }
        self.result = pd.concat(
            [
                self.result.drop(columns=list(cluster_aliases), errors="ignore"),
                pd.DataFrame(cluster_aliases),
            ],
            axis=1,
        )

    def _write_branch_evaluation(self):
        branch_result = self.evaluation[self.branch]
        self.result.at[self.row_index, f"{self.branch}_status"] = branch_result["status"]
        self.result.at[self.row_index, f"{self.branch}_error"] = branch_result.get("error")
        name = "cluster" if self.branch == "embedding" else self.branch
        self.result.at[self.row_index, f"{name}_evaluation_json"] = json.dumps(
            branch_result, ensure_ascii=False, allow_nan=False
        )
        metrics = branch_result.get("metrics") or {}
        for key in (
            *CONTENT_COMPONENTS,
            "contradiction_drift",
            "valence_drift",
            "arousal_drift",
            "dominance_drift",
            "vad_drift",
            "content_drift",
            "stdi",
        ):
            self.key = key
            self.result.at[self.row_index, f"{self.branch}_{self.key}"] = metrics.get(self.key)
        if self.branch == "embedding":
            for key in (
                *CONTENT_COMPONENTS,
                "contradiction_drift",
                "vad_drift",
                "content_drift",
            ):
                self.key = key
                self.result.at[self.row_index, self.key] = metrics.get(self.key)
