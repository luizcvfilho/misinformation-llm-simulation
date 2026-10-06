from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from misinformation_simulation.config.prompts import STRUCTURED_EXTRACTION_VERSION
from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
    VADScore,
    predict_text_vad,
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
) -> ComparisonWorkflowResult:
    from misinformation_simulation.topic_drift.comparison_workflow import (
        ComparisonWorkflowResult,
        _as_non_empty_text,
        _resolved_pair_ids,
        topic_structure_from_row,
    )

    original_column = original_text_column
    modified_column = modified_text_column
    missing = {original_column, modified_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    result = df.copy()
    for key in (
        "stdi",
        "method_gap",
        "dual_evaluation_json",
        "comparison_status",
        "embedding_status",
        "llm_judge_status",
        "comparison_error",
        "historical_embedding_stdi",
        "historical_lexical_stdi",
    ):
        result[key] = pd.Series(index=result.index, dtype=object)
    for branch in ("embedding", "llm_judge"):
        for component in (
            *CONTENT_COMPONENTS,
            "contradiction_drift",
            "vad_drift",
            "content_drift",
            "stdi",
        ):
            result[f"{branch}_{component}"] = pd.Series(index=result.index, dtype=object)
    for key in (
        *CONTENT_COMPONENTS,
        "contradiction_drift",
        "vad_drift",
        "content_drift",
        "original_vad_error",
        "modified_vad_error",
    ):
        result[key] = pd.Series(index=result.index, dtype=object)
    result["comparison_method"] = "dual"
    result["pair_id"] = _resolved_pair_ids(df, pair_id_column)
    cache = EvaluationCache(cache_dir)
    prepared = []
    vad_scores = {}
    vad_errors = {}
    contexts = []
    for row_index, row in result.iterrows():
        title = _as_non_empty_text(row.get(title_column)) or None
        structures = []
        scores = []
        texts = []
        try:
            for prefix, column in (("original", original_column), ("modified", modified_column)):
                text = _as_non_empty_text(row[column])
                if not text:
                    raise ValueError("Both texts must be non-empty")
                texts.append(text)
                inputs = {
                    "text": text,
                    "title": title,
                    "model": str(extraction_model),
                    "provider": str(extraction_provider),
                    "base_url": base_url,
                    "version": STRUCTURED_EXTRACTION_VERSION,
                }
                try:
                    structure = (
                        topic_structure_from_row(row, prefix=prefix) if reuse_structures else None
                    )
                except (ValueError, TypeError):
                    structure = None
                if structure is not None and (
                    structure_issues(structure)
                    or structure.provenance.get("input_sha256")
                    != input_hash({"text": text, "title": title})
                    or structure.provenance.get("model") != str(extraction_model)
                    or structure.provenance.get("provider") != str(extraction_provider)
                    or structure.provenance.get("base_url") != base_url
                    or structure.provenance.get("prompt_version") != STRUCTURED_EXTRACTION_VERSION
                ):
                    structure = None
                saved = cache.get("extraction", inputs) if reuse_structures else None
                if structure is None and saved is not None:
                    structure = _build_topic_structure(saved)
                if structure is None:
                    try:
                        structure = extraction_fn(
                            text=text,
                            title=title,
                            model=extraction_model,
                            provider=extraction_provider,
                            api_key=api_key,
                            base_url=base_url,
                            structured=True,
                        )
                        cache.put("extraction", inputs, topic_structure_to_dict(structure))
                    except Exception as exc:
                        structure = empty_topic_structure()
                        structure.schema_version = 2
                        structure.extraction_status = "failed"
                        structure.extraction_issues = [str(exc)]
                        structure.provenance = getattr(exc, "provenance", {})
                structures.append(structure)
                for key, value in flatten_topic_structure(structure, prefix=prefix).items():
                    if key not in result:
                        result[key] = pd.Series(index=result.index, dtype=object)
                    else:
                        result[key] = result[key].astype(object)
                    result.at[row_index, key] = value
                if text not in vad_scores:
                    try:
                        vad_inputs = {"text": text, "model": DEFAULT_VAD_MODEL_NAME}
                        saved_vad = cache.get("vad", vad_inputs) if vad_scorer is None else None
                        vad_scores[text] = (
                            VADScore(**saved_vad)
                            if saved_vad
                            else predict_text_vad(text, scorer=vad_scorer)
                        )
                        if vad_scorer is None and all(
                            value is not None for value in asdict(vad_scores[text]).values()
                        ):
                            cache.put("vad", vad_inputs, asdict(vad_scores[text]))
                    except Exception as exc:
                        vad_scores[text] = None
                        vad_errors[text] = str(exc)
                scores.append(vad_scores[text])
                result.at[row_index, f"{prefix}_vad_error"] = vad_errors.get(text)
                if scores[-1] is not None:
                    for dimension, value in asdict(scores[-1]).items():
                        result.at[row_index, f"{prefix}_vad_{dimension}"] = value
            prepared.append(TopicStructurePair(str(row["pair_id"]), *structures))
            contexts.append((row_index, texts, structures, scores, title))
        except Exception as exc:
            result.at[row_index, "comparison_status"] = "failed"
            result.at[row_index, "comparison_error"] = str(exc)
    resolved_embedder = embedder or TransformerTextEmbedder(embedding_model)
    comparator = StructuredEmbeddingComparator(
        embedder=resolved_embedder,
        embedding_model=embedding_model,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    try:
        comparator.fit(prepared)
    except Exception:
        # The judge can still produce an independently available branch.
        pass
    historical_comparator = ClusterSTDIComparator(
        embedder=resolved_embedder,
        embedding_model=embedding_model,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    try:
        historical_comparator.fit(prepared)
    except Exception:
        historical_comparator = None
    for row_index, texts, structures, scores, title in contexts:
        callback = progress_callback
        if callback:
            callback(f"Evaluating dual STDI pair '{result.at[row_index, 'pair_id']}'.")
        evaluation = compare_dual_stdi(
            original_text=texts[0],
            modified_text=texts[1],
            title=title,
            original_structure=structures[0],
            modified_structure=structures[1],
            original_vad=scores[0],
            modified_vad=scores[1],
            comparator=comparator,
            model=judge_model,
            provider=judge_provider,
            api_key=api_key,
            base_url=base_url,
            judge_fn=judge_fn,
            cache=cache,
            uncached_judge=uncached_judge,
        )
        for key, value in {
            "comparison_method": "dual",
            "comparison_status": evaluation["status"],
            "stdi": evaluation["stdi"],
            "method_gap": evaluation["method_gap"],
            "dual_evaluation_json": json.dumps(evaluation, ensure_ascii=False, allow_nan=False),
        }.items():
            if key not in result:
                result[key] = pd.Series(index=result.index, dtype=object)
            result.at[row_index, key] = value
        for branch in ("embedding", "llm_judge"):
            branch_result = evaluation[branch]
            result.at[row_index, f"{branch}_status"] = branch_result["status"]
            metrics = branch_result.get("metrics") or {}
            for key in (
                *CONTENT_COMPONENTS,
                "contradiction_drift",
                "vad_drift",
                "content_drift",
                "stdi",
            ):
                result.at[row_index, f"{branch}_{key}"] = metrics.get(key)
            if branch == "embedding":
                for key in (
                    *CONTENT_COMPONENTS,
                    "contradiction_drift",
                    "vad_drift",
                    "content_drift",
                ):
                    result.at[row_index, key] = metrics.get(key)
        result.at[row_index, "historical_lexical_stdi"] = calculate_stdi(
            *structures,
            original_vad=scores[0],
            compared_vad=scores[1],
        )["stdi"]
        if historical_comparator is not None and not (
            structure_issues(structures[0]) or structure_issues(structures[1])
        ):
            result.at[row_index, "historical_embedding_stdi"] = calculate_stdi(
                *structures,
                original_vad=scores[0],
                compared_vad=scores[1],
                component_overrides=historical_comparator.compare(*structures).component_drifts,
            )["stdi"]
    manifest = {
        "method": "dual",
        "version": DUAL_STDI_VERSION,
        "formula_version": FORMULA_VERSION,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "rows": len(result),
        "valid_pairs": int((result.get("comparison_status") == "valid").sum()),
        "embedding_model": embedding_model,
        "judge_model": str(judge_model),
        "judge_provider": str(judge_provider),
        "random_state": random_state,
        "n_clusters": n_clusters,
        "extraction_version": STRUCTURED_EXTRACTION_VERSION,
        "extraction_model": str(extraction_model),
        "extraction_provider": str(extraction_provider),
        "branch_weights": [0.5, 0.5],
        "polarity_weight": 0.2,
        "duration_weight": 0.2,
        "numeric_weight": 0.2,
        "numeric_aggregation": "maximum",
        "legacy_component_columns": "embedding",
        "uncached_judge": uncached_judge,
        "vad_model": "custom_scorer" if vad_scorer else "RobroKools/vad-bert",
    }
    return ComparisonWorkflowResult(result, manifest)
