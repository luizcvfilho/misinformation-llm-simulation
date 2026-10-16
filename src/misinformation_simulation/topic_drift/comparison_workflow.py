from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.text_metrics.vad_evaluation import (
    VADTextEvaluator,
    compare_vad_evaluations,
    vad_for_branch,
    vad_pair_columns,
    validate_vad_method,
)
from misinformation_simulation.topic_drift.cluster_comparison import (
    CLUSTER_STDI_COMPARISON_VERSION,
    ClusterSTDIComparator,
    TextEmbedder,
    TopicStructurePair,
)
from misinformation_simulation.topic_drift.extraction import extract_topic_structure
from misinformation_simulation.topic_drift.metrics import calculate_stdi
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    flatten_topic_structure,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache
from misinformation_simulation.topic_drift.semantic_comparison import (
    SemanticSTDIComparison,
    compare_stdi_components_semantically,
)
from misinformation_simulation.topic_drift.structured_comparison import (
    DEFAULT_JUDGE_REPEATS,
    STRUCTURED_CLUSTER_VERSION,
    StructuredEmbeddingComparator,
    complete_stdi,
    validate_judge_repeats,
)

ComparisonMethod = Literal["llm_semantic", "cluster", "dual"]
SUPPORTED_COMPARISON_METHODS: tuple[ComparisonMethod, ...] = ("llm_semantic", "cluster", "dual")


@dataclass(frozen=True, slots=True)
class ComparisonWorkflowResult:
    results: pd.DataFrame
    manifest: dict[str, Any]
    cluster_artifacts: pd.DataFrame | None = None


def _decode_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        parsed = [value]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _decode_relations(value: Any) -> list[TopicRelation]:
    if value is None:
        return []
    if not isinstance(value, list) and pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value)) if not isinstance(value, list) else value
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    relations: list[TopicRelation] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        action = str(item.get("action", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and action and obj:
            relations.append(TopicRelation(subject, action, obj))
    return relations


def _as_non_empty_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _write_structure(
    result: pd.DataFrame, *, row_index: Any, structure: TopicStructure, prefix: str
) -> None:
    for column, value in flatten_topic_structure(structure, prefix=prefix).items():
        if column in result.columns:
            result[column] = result[column].astype("object")
        result.at[row_index, column] = value


def topic_structure_from_row(row: pd.Series, *, prefix: str) -> TopicStructure | None:
    from misinformation_simulation.topic_drift.extraction import _build_topic_structure

    saved = row.get(f"{prefix}_json")
    if isinstance(saved, str) and saved.strip():
        payload = json.loads(saved)
        return _build_topic_structure(payload)
    main_topic_value = row.get(f"{prefix}_main_topic")
    subtopics_value = row.get(f"{prefix}_subtopics")
    entities_value = row.get(f"{prefix}_central_entities")
    relations_value = row.get(f"{prefix}_central_relations")
    values = (main_topic_value, subtopics_value, entities_value, relations_value)
    if all(value is None or pd.isna(value) for value in values):
        return None
    main_topic = _as_non_empty_text(main_topic_value) or None
    return TopicStructure(
        main_topic=main_topic,
        subtopics=_decode_list(subtopics_value),
        central_entities=_decode_list(entities_value),
        central_relations=_decode_relations(relations_value),
    )


def _resolved_pair_ids(df: pd.DataFrame, pair_id_column: str | None) -> list[str]:
    if pair_id_column and pair_id_column in df.columns:
        raw_ids = df[pair_id_column].astype(str).tolist()
    elif "pair_id" in df.columns:
        raw_ids = df["pair_id"].astype(str).tolist()
    elif "evaluation_id" in df.columns:
        raw_ids = df["evaluation_id"].astype(str).tolist()
    else:
        raw_ids = [f"pair_{index:06d}" for index in range(len(df))]
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("Pair identifiers must be unique.")
    return raw_ids


def _validate_method(method: str) -> ComparisonMethod:
    if method not in SUPPORTED_COMPARISON_METHODS:
        raise ValueError(f"Unsupported comparison method '{method}'.")
    return method  # type: ignore[return-value]


def run_comparison_workflow(
    df: pd.DataFrame,
    *,
    method: ComparisonMethod | str,
    original_text_column: str = "original_text",
    modified_text_column: str = "modified_text",
    title_column: str = "title",
    pair_id_column: str | None = None,
    extraction_model: str = DEFAULT_LLM_MODEL,
    extraction_provider: str = DEFAULT_LLM_PROVIDER,
    extraction_api_key: str | None = None,
    extraction_base_url: str | None = None,
    extraction_fn: Callable[..., TopicStructure] = extract_topic_structure,
    llm_comparison_model: str | None = None,
    llm_comparison_provider: str | None = None,
    llm_comparison_fn: Callable[..., SemanticSTDIComparison] = compare_stdi_components_semantically,
    embedder: TextEmbedder | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_clusters: int | None = None,
    random_state: int = 42,
    reuse_structures: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    cache_dir: Path | str | None = None,
    vad_scorer: Callable[..., Any] | None = None,
    uncached_judge: bool = False,
    judge_repeats: int = DEFAULT_JUDGE_REPEATS,
    vad_method: str = "model",
    vad_llm_model: str | None = None,
    vad_llm_provider: str | None = None,
    vad_llm_scorer: Callable[..., Any] | None = None,
    vad_llm_api_key: str | None = None,
    vad_llm_base_url: str | None = None,
) -> ComparisonWorkflowResult:
    """Run one comparison method over shared LLM-extracted topic structures."""
    return _ComparisonWorkflow(
        df=df,
        method=method,
        original_text_column=original_text_column,
        modified_text_column=modified_text_column,
        title_column=title_column,
        pair_id_column=pair_id_column,
        extraction_model=extraction_model,
        extraction_provider=extraction_provider,
        extraction_api_key=extraction_api_key,
        extraction_base_url=extraction_base_url,
        extraction_fn=extraction_fn,
        llm_comparison_model=llm_comparison_model,
        llm_comparison_provider=llm_comparison_provider,
        llm_comparison_fn=llm_comparison_fn,
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


class _ComparisonWorkflow:
    """Run extraction and one comparison method over paired texts."""

    def __init__(
        self,
        *,
        df,
        method,
        original_text_column,
        modified_text_column,
        title_column,
        pair_id_column,
        extraction_model,
        extraction_provider,
        extraction_api_key,
        extraction_base_url,
        extraction_fn,
        llm_comparison_model,
        llm_comparison_provider,
        llm_comparison_fn,
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
        self.method = method
        self.original_text_column = original_text_column
        self.modified_text_column = modified_text_column
        self.title_column = title_column
        self.pair_id_column = pair_id_column
        self.extraction_model = extraction_model
        self.extraction_provider = extraction_provider
        self.extraction_api_key = extraction_api_key
        self.extraction_base_url = extraction_base_url
        self.extraction_fn = extraction_fn
        self.llm_comparison_model = llm_comparison_model
        self.llm_comparison_provider = llm_comparison_provider
        self.llm_comparison_fn = llm_comparison_fn
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
        """Run one comparison method over shared LLM-extracted topic structures."""
        self.resolved_method = _validate_method(str(self.method))
        validate_judge_repeats(self.judge_repeats)
        validate_vad_method(self.vad_method)
        if self.resolved_method == "dual":
            from misinformation_simulation.topic_drift.dual_workflow import run_dual_workflow

            return run_dual_workflow(
                self.df,
                original_text_column=self.original_text_column,
                modified_text_column=self.modified_text_column,
                title_column=self.title_column,
                pair_id_column=self.pair_id_column,
                extraction_model=self.extraction_model,
                extraction_provider=self.extraction_provider,
                api_key=self.extraction_api_key,
                base_url=self.extraction_base_url,
                extraction_fn=self.extraction_fn,
                judge_model=self.llm_comparison_model or self.extraction_model,
                judge_provider=self.llm_comparison_provider or self.extraction_provider,
                judge_fn=self.llm_comparison_fn,
                embedder=self.embedder,
                embedding_model=self.embedding_model,
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                reuse_structures=self.reuse_structures,
                progress_callback=self.progress_callback,
                cache_dir=self.cache_dir,
                vad_scorer=self.vad_scorer,
                uncached_judge=self.uncached_judge,
                judge_repeats=self.judge_repeats,
                vad_method=self.vad_method,
                vad_llm_model=self.vad_llm_model,
                vad_llm_provider=self.vad_llm_provider,
                vad_llm_scorer=self.vad_llm_scorer,
                vad_llm_api_key=self.vad_llm_api_key,
                vad_llm_base_url=self.vad_llm_base_url,
            )
        self._prepare_result()

        self.total_rows = len(self.result)
        for row_position, (row_index, row) in enumerate(self.result.iterrows(), start=1):
            self.row_position, (self.row_index, self.row) = (row_position, (row_index, row))
            self._prepare_pair()

        self._fit_clusters()

        for row_position, (row_index, (original_structure, modified_structure)) in enumerate(
            self.row_structures.items(), start=1
        ):
            (
                self.row_position,
                (self.row_index, (self.original_structure, self.modified_structure)),
            ) = (row_position, (row_index, (original_structure, modified_structure)))
            self._compare_pair()

        manifest = {
            "schema_version": 1,
            "vad_method": self.vad_method,
            "vad_llm_model": self.vad_evaluator.llm_model if self.vad_method != "model" else None,
            "vad_llm_provider": self.vad_evaluator.llm_provider
            if self.vad_method != "model"
            else None,
            "comparison_method": self.resolved_method,
            "pair_count": len(self.result),
            "successful_pair_count": int(self.result["comparison_status"].eq("success").sum()),
            "original_text_column": self.original_text_column,
            "modified_text_column": self.modified_text_column,
            "title_column": self.title_column,
            "shared_extraction": {
                "model": self.extraction_model,
                "provider": self.extraction_provider,
                "reused_existing_structures": self.reuse_structures,
            },
            "llm_comparison": {
                "model": self.llm_comparison_model or self.extraction_model,
                "provider": self.llm_comparison_provider or self.extraction_provider,
            }
            if self.resolved_method == "llm_semantic"
            else None,
            "cluster": {
                "comparison_version": STRUCTURED_CLUSTER_VERSION
                if any(
                    pair.original.schema_version == 2 or pair.modified.schema_version == 2
                    for pair in self.prepared_pairs
                )
                else CLUSTER_STDI_COMPARISON_VERSION,
                "embedding_model": self.embedding_model,
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
            }
            if self.resolved_method == "cluster"
            else None,
        }
        return ComparisonWorkflowResult(
            results=self.result, manifest=manifest, cluster_artifacts=self.cluster_artifacts
        )

    def _prepare_result(self):
        required_columns = {self.original_text_column, self.modified_text_column}
        missing_columns = sorted(required_columns - set(self.df.columns))
        if missing_columns:
            raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")
        self.result = self.df.drop(
            columns=[
                "original_topic_domain",
                "modified_topic_domain",
                "cluster_theme_domain_match",
                "cluster_theme_domain_gate_applied",
            ],
            errors="ignore",
        ).copy()
        for column in (
            "original_extraction_model",
            "original_extraction_provider",
            "modified_extraction_model",
            "modified_extraction_provider",
            "comparison_model",
            "comparison_provider",
        ):
            self.column = column
            if self.column in self.result:
                self.result[self.column] = self.result[self.column].astype("object")
        pair_ids = _resolved_pair_ids(self.result, self.pair_id_column)
        self.result["pair_id"] = pair_ids
        self.prepared_pairs: list[TopicStructurePair] = []
        self.row_structures: dict[Any, tuple[TopicStructure, TopicStructure]] = {}
        vad_provider = (
            self.vad_llm_provider or self.llm_comparison_provider or self.extraction_provider
        )
        reuse_vad_endpoint = normalize_provider(vad_provider) == normalize_provider(
            self.extraction_provider
        )
        self.vad_evaluator = VADTextEvaluator(
            method=self.vad_method,
            model_scorer=self.vad_scorer,
            llm_scorer=self.vad_llm_scorer,
            llm_model=self.vad_llm_model or self.llm_comparison_model or self.extraction_model,
            llm_provider=vad_provider,
            cache=EvaluationCache(self.cache_dir),
            api_key=self.vad_llm_api_key
            or (self.extraction_api_key if reuse_vad_endpoint else None),
            base_url=self.vad_llm_base_url
            or (self.extraction_base_url if reuse_vad_endpoint else None),
        )

    def _prepare_pair(self):
        pair_id = self.result.at[self.row_index, "pair_id"]
        if self.progress_callback is not None:
            self.progress_callback(
                f"[{self.row_position}/{self.total_rows}] Preparing pair '{pair_id}'."
            )
        original_text = _as_non_empty_text(self.row[self.original_text_column])
        modified_text = _as_non_empty_text(self.row[self.modified_text_column])
        if not original_text or not modified_text:
            self.result.at[self.row_index, "comparison_status"] = "skipped"
            self.result.at[self.row_index, "comparison_error"] = "Both texts must be non-empty."
            return
        title = _as_non_empty_text(self.row.get(self.title_column, ""))
        self.original_structure = (
            topic_structure_from_row(self.row, prefix="original") if self.reuse_structures else None
        )
        self.modified_structure = (
            topic_structure_from_row(self.row, prefix="modified") if self.reuse_structures else None
        )
        try:
            if self.original_structure is None:
                if self.progress_callback is not None:
                    self.progress_callback(
                        f"[{self.row_position}/{self.total_rows}] Extracting original structure."
                    )
                self.result.at[self.row_index, "original_extraction_model"] = str(
                    self.extraction_model
                )
                self.result.at[self.row_index, "original_extraction_provider"] = str(
                    self.extraction_provider
                )
                self.original_structure = self.extraction_fn(
                    text=original_text,
                    title=title,
                    model=self.extraction_model,
                    provider=self.extraction_provider,
                    api_key=self.extraction_api_key,
                    base_url=self.extraction_base_url,
                    **{"structured": True} if self.resolved_method == "cluster" else {},
                )
            if self.modified_structure is None:
                if self.progress_callback is not None:
                    self.progress_callback(
                        f"[{self.row_position}/{self.total_rows}] Extracting modified structure."
                    )
                self.result.at[self.row_index, "modified_extraction_model"] = str(
                    self.extraction_model
                )
                self.result.at[self.row_index, "modified_extraction_provider"] = str(
                    self.extraction_provider
                )
                self.modified_structure = self.extraction_fn(
                    text=modified_text,
                    title=title,
                    model=self.extraction_model,
                    provider=self.extraction_provider,
                    api_key=self.extraction_api_key,
                    base_url=self.extraction_base_url,
                    **{"structured": True} if self.resolved_method == "cluster" else {},
                )
        except Exception as exc:
            self.result.at[self.row_index, "comparison_status"] = "error"
            self.result.at[self.row_index, "comparison_error"] = str(exc)
            return
        if self.progress_callback is not None:
            self.progress_callback(
                f"[{self.row_position}/{self.total_rows}] Shared structures ready."
            )
        self.row_structures[self.row_index] = (self.original_structure, self.modified_structure)
        self.prepared_pairs.append(
            TopicStructurePair(
                pair_id=self.result.at[self.row_index, "pair_id"],
                original=self.original_structure,
                modified=self.modified_structure,
            )
        )
        _write_structure(
            self.result,
            row_index=self.row_index,
            structure=self.original_structure,
            prefix="original",
        )
        _write_structure(
            self.result,
            row_index=self.row_index,
            structure=self.modified_structure,
            prefix="modified",
        )

    def _fit_clusters(self):
        self.cluster_comparator: ClusterSTDIComparator | None = None
        self.cluster_artifacts: pd.DataFrame | None = None
        if self.resolved_method == "cluster" and self.prepared_pairs:
            if self.progress_callback is not None:
                self.progress_callback(
                    f"Fitting shared clusters from {len(self.prepared_pairs)} prepared pair(s)."
                )
            self.cluster_comparator = StructuredEmbeddingComparator(
                embedder=self.embedder,
                embedding_model=self.embedding_model,
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            ).fit(self.prepared_pairs)
            self.cluster_artifacts = pd.DataFrame(self.cluster_comparator.artifact_rows())

    def _compare_pair(self):
        self.row = self.result.loc[self.row_index]
        try:
            if self.progress_callback is not None:
                self.progress_callback(
                    f"[{self.row_position}/{len(self.row_structures)}] Comparing pair '"
                    f"{self.result.at[self.row_index, 'pair_id']}' with "
                    f"{self.resolved_method}."
                )
            self._compare_content()

            self.metrics = calculate_stdi(
                self.original_structure,
                self.modified_structure,
                component_overrides=self.component_overrides,
            )
            self._evaluate_pair_vad()

            self.metrics = complete_stdi(self.metrics, self.vad)
            self._write_comparison()

        except Exception as exc:
            self.result.at[self.row_index, "comparison_method"] = self.resolved_method
            self.result.at[self.row_index, "comparison_status"] = "error"
            self.result.at[self.row_index, "comparison_error"] = str(exc)
            for metric in (
                "theme_drift",
                "subtopic_drift",
                "entity_drift",
                "relation_drift",
                "content_drift",
                "contradiction_drift",
                "valence_drift",
                "arousal_drift",
                "dominance_drift",
                "vad_drift",
                "stdi",
            ):
                self.result.at[self.row_index, metric] = np.nan

    def _compare_content(self):
        self.details: dict[str, dict[str, float | int]] = {}
        self.rationales: dict[str, str] = {}
        if self.resolved_method == "llm_semantic":
            self.result.at[self.row_index, "comparison_model"] = str(
                self.llm_comparison_model or self.extraction_model
            )
            self.result.at[self.row_index, "comparison_provider"] = str(
                self.llm_comparison_provider or self.extraction_provider
            )
            semantic_result = self.llm_comparison_fn(
                original_text=str(self.row[self.original_text_column]),
                modified_text=str(self.row[self.modified_text_column]),
                title=_as_non_empty_text(self.row.get(self.title_column, "")),
                original_structure=self.original_structure,
                modified_structure=self.modified_structure,
                model=self.llm_comparison_model or self.extraction_model,
                provider=self.llm_comparison_provider or self.extraction_provider,
                api_key=self.extraction_api_key,
                base_url=self.extraction_base_url,
            )
            self.component_overrides = {
                component: semantic_result.component_drifts[component]
                for component in (
                    "theme_drift",
                    "subtopic_drift",
                    "entity_drift",
                    "relation_drift",
                )
            }
            self.rationales = semantic_result.rationales
        else:
            if self.cluster_comparator is None:
                raise RuntimeError("The cluster comparator was not initialized.")
            cluster_result = self.cluster_comparator.compare(
                self.original_structure, self.modified_structure
            )
            self.component_overrides = cluster_result.component_drifts
            self.details = cluster_result.details

    def _evaluate_pair_vad(self):
        text_evaluations = {
            prefix: self.vad_evaluator.evaluate(_as_non_empty_text(self.row[column]))
            for prefix, column in (
                ("original", self.original_text_column),
                ("modified", self.modified_text_column),
            )
        }
        vad_evaluation = compare_vad_evaluations(
            text_evaluations["original"],
            text_evaluations["modified"],
            method=self.vad_method,
        )
        for key, value in vad_pair_columns(vad_evaluation).items():
            key, self.value = (key, value)
            if key not in self.result:
                self.result[key] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, key] = self.value
        for prefix, evaluation in text_evaluations.items():
            key = f"{prefix}_vad_evaluation_json"
            if key not in self.result:
                self.result[key] = pd.Series(index=self.result.index, dtype=object)
            self.result.at[self.row_index, key] = json.dumps(evaluation, ensure_ascii=False)
        self.vad = vad_for_branch(
            vad_evaluation,
            "embedding" if self.resolved_method == "cluster" else "llm_judge",
        )
        if self.vad["status"] != "valid":
            raise ValueError("Selected VAD evaluation is incomplete.")

    def _write_comparison(self):
        self.result.at[self.row_index, "comparison_method"] = self.resolved_method
        self.result.at[self.row_index, "comparison_status"] = "success"
        self.result.at[self.row_index, "comparison_error"] = pd.NA
        for column, value in self.metrics.items():
            self.column, self.value = (column, value)
            self.result.at[self.row_index, self.column] = self.value
        for component, rationale in self.rationales.items():
            self.result.at[self.row_index, f"comparison_{component}_rationale"] = rationale
        for component, component_details in self.details.items():
            for detail_name, value in component_details.items():
                detail_name, self.value = (detail_name, value)
                self.result.at[self.row_index, f"cluster_{component}_{detail_name}"] = self.value


def load_comparison_input(input_directory: Path) -> pd.DataFrame:
    """Load a reusable output folder from either comparison method."""
    candidates = (
        input_directory / "comparison_results.csv",
        input_directory / "scored_stdi_pairs.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        "No reusable comparison CSV was found. Expected 'comparison_results.csv' or "
        "'scored_stdi_pairs.csv'."
    )


def write_comparison_output(output_directory: Path, workflow: ComparisonWorkflowResult) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    workflow.results.to_csv(output_directory / "comparison_results.csv", index=False)
    (output_directory / "manifest.json").write_text(
        json.dumps(workflow.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if workflow.cluster_artifacts is not None:
        artifacts_directory = output_directory / "cluster_artifacts"
        artifacts_directory.mkdir(exist_ok=True)
        workflow.cluster_artifacts.to_csv(
            artifacts_directory / "cluster_assignments.csv", index=False
        )


def compare_method_outputs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    """Join two outputs by pair_id to compare equivalent component scores."""
    columns = [
        "pair_id",
        "theme_drift",
        "subtopic_drift",
        "entity_drift",
        "relation_drift",
        "content_drift",
        "stdi",
    ]
    for frame, label in ((left, left_label), (right, right_label)):
        missing_columns = sorted(set(columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"'{label}' is missing: {', '.join(missing_columns)}")
    left_frame = left[columns].rename(
        columns={column: f"{left_label}_{column}" for column in columns if column != "pair_id"}
    )
    right_frame = right[columns].rename(
        columns={column: f"{right_label}_{column}" for column in columns if column != "pair_id"}
    )
    merged = left_frame.merge(right_frame, on="pair_id", how="inner", validate="one_to_one")
    for component in columns[1:]:
        merged[f"delta_{component}"] = (
            merged[f"{right_label}_{component}"] - merged[f"{left_label}_{component}"]
        )
    return merged
