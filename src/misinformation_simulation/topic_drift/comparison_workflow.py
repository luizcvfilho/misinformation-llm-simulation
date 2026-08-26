from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from misinformation_simulation.topic_drift.cluster_comparison import (
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
from misinformation_simulation.topic_drift.semantic_comparison import (
    SemanticSTDIComparison,
    compare_stdi_components_semantically,
)

ComparisonMethod = Literal["llm_semantic", "cluster"]
SUPPORTED_COMPARISON_METHODS: tuple[ComparisonMethod, ...] = ("llm_semantic", "cluster")


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


def topic_structure_from_row(row: pd.Series, *, prefix: str) -> TopicStructure | None:
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
    extraction_model: str = "gpt-5.6-luna",
    extraction_provider: str = "chatgpt",
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
) -> ComparisonWorkflowResult:
    """Run one comparison method over shared LLM-extracted topic structures."""
    resolved_method = _validate_method(str(method))
    required_columns = {original_text_column, modified_text_column}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    result = df.copy()
    pair_ids = _resolved_pair_ids(result, pair_id_column)
    result["pair_id"] = pair_ids
    prepared_pairs: list[TopicStructurePair] = []
    row_structures: dict[Any, tuple[TopicStructure, TopicStructure]] = {}

    for row_index, row in result.iterrows():
        original_text = _as_non_empty_text(row[original_text_column])
        modified_text = _as_non_empty_text(row[modified_text_column])
        if not original_text or not modified_text:
            result.at[row_index, "comparison_status"] = "skipped"
            result.at[row_index, "comparison_error"] = "Both texts must be non-empty."
            continue
        title = _as_non_empty_text(row.get(title_column, ""))
        original_structure = topic_structure_from_row(row, prefix="original")
        modified_structure = topic_structure_from_row(row, prefix="modified")
        try:
            if original_structure is None:
                original_structure = extraction_fn(
                    text=original_text,
                    title=title,
                    model=extraction_model,
                    provider=extraction_provider,
                    api_key=extraction_api_key,
                    base_url=extraction_base_url,
                )
            if modified_structure is None:
                modified_structure = extraction_fn(
                    text=modified_text,
                    title=title,
                    model=extraction_model,
                    provider=extraction_provider,
                    api_key=extraction_api_key,
                    base_url=extraction_base_url,
                )
        except Exception as exc:
            result.at[row_index, "comparison_status"] = "error"
            result.at[row_index, "comparison_error"] = str(exc)
            continue
        row_structures[row_index] = (original_structure, modified_structure)
        prepared_pairs.append(
            TopicStructurePair(
                pair_id=result.at[row_index, "pair_id"],
                original=original_structure,
                modified=modified_structure,
            )
        )
        for column, value in flatten_topic_structure(original_structure, prefix="original").items():
            result.at[row_index, column] = value
        for column, value in flatten_topic_structure(modified_structure, prefix="modified").items():
            result.at[row_index, column] = value

    cluster_comparator: ClusterSTDIComparator | None = None
    cluster_artifacts: pd.DataFrame | None = None
    if resolved_method == "cluster" and prepared_pairs:
        cluster_comparator = ClusterSTDIComparator(
            embedder=embedder,
            embedding_model=embedding_model,
            n_clusters=n_clusters,
            random_state=random_state,
        ).fit(prepared_pairs)
        cluster_artifacts = pd.DataFrame(cluster_comparator.artifact_rows())

    for row_index, (original_structure, modified_structure) in row_structures.items():
        row = result.loc[row_index]
        try:
            details: dict[str, dict[str, float | int]] = {}
            rationales: dict[str, str] = {}
            if resolved_method == "llm_semantic":
                semantic_result = llm_comparison_fn(
                    original_text=str(row[original_text_column]),
                    modified_text=str(row[modified_text_column]),
                    title=_as_non_empty_text(row.get(title_column, "")),
                    original_structure=original_structure,
                    modified_structure=modified_structure,
                    model=llm_comparison_model or extraction_model,
                    provider=llm_comparison_provider or extraction_provider,
                    api_key=extraction_api_key,
                    base_url=extraction_base_url,
                )
                component_overrides = {
                    component: semantic_result.component_drifts[component]
                    for component in (
                        "theme_drift",
                        "subtopic_drift",
                        "entity_drift",
                        "relation_drift",
                    )
                }
                rationales = semantic_result.rationales
            else:
                if cluster_comparator is None:
                    raise RuntimeError("The cluster comparator was not initialized.")
                cluster_result = cluster_comparator.compare(original_structure, modified_structure)
                component_overrides = cluster_result.component_drifts
                details = cluster_result.details
            metrics = calculate_stdi(
                original_structure,
                modified_structure,
                component_overrides=component_overrides,
            )
            result.at[row_index, "comparison_method"] = resolved_method
            result.at[row_index, "comparison_status"] = "success"
            result.at[row_index, "comparison_error"] = pd.NA
            for column, value in metrics.items():
                result.at[row_index, column] = value
            for component, rationale in rationales.items():
                result.at[row_index, f"comparison_{component}_rationale"] = rationale
            for component, component_details in details.items():
                for detail_name, value in component_details.items():
                    result.at[row_index, f"cluster_{component}_{detail_name}"] = value
        except Exception as exc:
            result.at[row_index, "comparison_method"] = resolved_method
            result.at[row_index, "comparison_status"] = "error"
            result.at[row_index, "comparison_error"] = str(exc)
            for metric in (
                "theme_drift",
                "subtopic_drift",
                "entity_drift",
                "relation_drift",
                "content_drift",
                "stdi",
            ):
                result.at[row_index, metric] = np.nan

    manifest = {
        "schema_version": 1,
        "comparison_method": resolved_method,
        "pair_count": len(result),
        "successful_pair_count": int(result["comparison_status"].eq("success").sum()),
        "original_text_column": original_text_column,
        "modified_text_column": modified_text_column,
        "title_column": title_column,
        "shared_extraction": {
            "model": extraction_model,
            "provider": extraction_provider,
        },
        "cluster": (
            {
                "embedding_model": embedding_model,
                "n_clusters": n_clusters,
                "random_state": random_state,
            }
            if resolved_method == "cluster"
            else None
        ),
    }
    return ComparisonWorkflowResult(
        results=result,
        manifest=manifest,
        cluster_artifacts=cluster_artifacts,
    )


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
