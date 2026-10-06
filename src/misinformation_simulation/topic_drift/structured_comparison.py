from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from statistics import fmean, stdev
from typing import Any

from misinformation_simulation.config.prompts import STRUCTURED_JUDGE_VERSION
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.cluster_comparison import (
    ClusterSTDIComparator,
    ClusterSTDIComparison,
    TopicStructurePair,
)
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.provenance import EvaluationCache, input_hash
from misinformation_simulation.topic_drift.qualifiers import (
    adjust_relation_distance,
)
from misinformation_simulation.topic_drift.semantic_comparison import (
    SemanticSTDIComparison,
    compare_stdi_components_semantically,
)

DUAL_STDI_VERSION = "dual_stdi_v5"
DEFAULT_JUDGE_REPEATS = 3
STRUCTURED_CLUSTER_VERSION = "cluster_v4"
FORMULA_VERSION = "stdi_remaining_distance_v1_mean_50_50"
NUMERICAL_TOLERANCE = 1e-12
CONTENT_COMPONENTS = ("theme_drift", "subtopic_drift", "entity_drift", "relation_drift")


def validate_judge_repeats(judge_repeats: int) -> None:
    if isinstance(judge_repeats, bool) or not isinstance(judge_repeats, int) or judge_repeats < 1:
        raise ValueError("'judge_repeats' must be a positive integer.")


def _core(structure: TopicStructure) -> TopicStructure:
    return replace(
        structure,
        central_relations=[
            replace(relation, action=relation.base_action or relation.action)
            for relation in structure.central_relations
        ],
    )


def structure_issues(structure: TopicStructure) -> list[str]:
    issues = [
        issue
        for issue in structure.extraction_issues
        if issue == "Invalid contradiction score"
        or issue == "Invalid relation record"
        or issue.startswith("Invalid list:")
        and not issue.endswith("opinions")
        or issue == "Missing required field: central_relations"
    ]
    if structure.schema_version != 2:
        issues.append("A version-2 extraction is required")
    if structure.extraction_status == "failed":
        issues.append("Extraction failed")
    if not structure.main_topic:
        issues.append("Main topic is unavailable")
    for index, relation in enumerate(structure.central_relations):
        if not relation.subject or not relation.action:
            issues.append(f"Relation {index} has incomplete required fields")
    return issues


class StructuredEmbeddingComparator(ClusterSTDIComparator):
    def _relation_similarity(self, left: TopicRelation, right: TopicRelation) -> float:
        similarity = super()._relation_similarity(left, right)
        if not left.object and not right.object:
            similarity = min(similarity + 0.15, 1.0)
        return similarity

    def compare(
        self,
        original: TopicStructure,
        modified: TopicStructure,
        *,
        round_scores: bool = True,
    ) -> ClusterSTDIComparison:
        if original.schema_version == modified.schema_version == 1:
            return super().compare(original, modified, round_scores=round_scores)
        result = self.compare_structured(original, modified)
        if result["status"] != "valid":
            raise ValueError("; ".join(result["issues"]))
        components = result["components"]
        return ClusterSTDIComparison(
            component_drifts={
                key: round(value, 6) if round_scores else value for key, value in components.items()
            },
            details={
                **result["cluster_details"],
                "qualifiers": {
                    "comparison_version": STRUCTURED_CLUSTER_VERSION,
                    "warnings_json": json.dumps(result["warnings"], ensure_ascii=False),
                    "relations_json": json.dumps(result["relations"], ensure_ascii=False),
                },
            },
        )

    def fit(self, pairs: Sequence[TopicStructurePair]) -> StructuredEmbeddingComparator:
        self.fit_error = None
        try:
            super().fit(
                [
                    TopicStructurePair(pair.pair_id, _core(pair.original), _core(pair.modified))
                    for pair in pairs
                ]
            )
        except Exception as exc:
            self.fit_error = str(exc)
            raise
        return self

    def compare_structured(
        self, original: TopicStructure, modified: TopicStructure
    ) -> dict[str, Any]:
        if getattr(self, "fit_error", None):
            raise RuntimeError(self.fit_error)
        left, right = _core(original), _core(modified)
        baseline = super().compare(left, right, round_scores=False)
        candidates = sorted(
            (
                (self._relation_similarity(a, b), i, j)
                for i, a in enumerate(left.central_relations)
                for j, b in enumerate(right.central_relations)
            ),
            reverse=True,
        )
        matched_left, matched_right = set(), set()
        details = []
        for similarity, i, j in candidates:
            if i in matched_left or j in matched_right:
                continue
            matched_left.add(i)
            matched_right.add(j)
            details.append(
                {
                    "reference_index": i,
                    "rewrite_index": j,
                    **adjust_relation_distance(
                        1 - similarity, original.central_relations[i], modified.central_relations[j]
                    ),
                }
            )
        unmatched = len(left.central_relations) + len(right.central_relations) - 2 * len(details)
        issues = structure_issues(original) + structure_issues(modified)
        warnings = list(original.extraction_issues) + list(modified.extraction_issues)
        warnings.extend(warning for item in details for warning in item["warnings"])
        count = max(len(left.central_relations), len(right.central_relations))
        components = dict(baseline.component_drifts)
        if not issues:
            components["relation_drift"] = (
                (sum(item["distance"] for item in details) + unmatched) / count if count else 0.0
            )
        components["contradiction_drift"] = modified.internal_contradiction_score
        return {
            "status": "partial" if issues else "valid",
            "issues": issues,
            "warnings": list(dict.fromkeys(warnings)),
            "components": components,
            "semantic_only_components": baseline.component_drifts,
            "cluster_details": baseline.details,
            "relations": details,
            "unmatched_relations": unmatched,
            "unmatched_reference_indices": [
                i for i in range(len(left.central_relations)) if i not in matched_left
            ],
            "unmatched_rewrite_indices": [
                i for i in range(len(right.central_relations)) if i not in matched_right
            ],
            "relation_denominator": count,
        }


def shared_vad_drift(original: VADScore | None, modified: VADScore | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for dimension in ("valence", "arousal", "dominance"):
        a = getattr(original, dimension, None)
        b = getattr(modified, dimension, None)
        if a is None or b is None or not all(math.isfinite(x) for x in (a, b)):
            return {"status": "partial", "vad_drift": None}
        metrics[f"{dimension}_drift"] = min(abs(b - a) / 4.0, 1.0)
    return {"status": "valid", **metrics, "vad_drift": sum(metrics.values()) / 3.0}


def complete_stdi(components: dict[str, float], vad: dict[str, Any]) -> dict[str, float]:
    scores = [components[key] for key in (*CONTENT_COMPONENTS, "contradiction_drift")]
    if vad["status"] != "valid" or not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
        raise ValueError("Complete STDI requires valid components and complete VAD")
    content = sum(components[key] for key in CONTENT_COMPONENTS) / 4
    contradiction = content + (1 - content) * 0.2 * components["contradiction_drift"]
    return {
        **components,
        **{k: v for k, v in vad.items() if k != "status"},
        "content_drift": content,
        "stdi": contradiction + (1 - contradiction) * 0.2 * vad["vad_drift"],
    }


def compare_dual_stdi(
    *,
    original_text: str,
    modified_text: str,
    title: str | None,
    original_structure: TopicStructure,
    modified_structure: TopicStructure,
    original_vad: VADScore | None,
    modified_vad: VADScore | None,
    comparator: StructuredEmbeddingComparator,
    model: str,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    retry_attempts: int = 5,
    before_request_hook: Callable[[], None] | None = None,
    judge_fn: Callable[..., SemanticSTDIComparison] = compare_stdi_components_semantically,
    cache: EvaluationCache | None = None,
    uncached_judge: bool = False,
    judge_repeats: int = DEFAULT_JUDGE_REPEATS,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    validate_judge_repeats(judge_repeats)
    cache = cache or EvaluationCache()
    result: dict[str, Any] = {
        "version": DUAL_STDI_VERSION,
        "formula_version": FORMULA_VERSION,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "polarity_weight": 0.2,
        "duration_weight": 0.2,
        "numeric_weight": 0.2,
        "numeric_aggregation": "maximum",
        "branch_weights": [0.5, 0.5],
        "judge_repeats": judge_repeats,
        "judge_aggregation": "mean_complete_metrics",
        "stdi": None,
        "method_gap": None,
        "input_sha256": input_hash(
            {"original": original_text, "modified": modified_text, "title": title}
        ),
    }
    vad = shared_vad_drift(original_vad, modified_vad)
    result["shared_vad"] = vad
    if (
        original_text == modified_text
        and not structure_issues(original_structure)
        and vad["status"] == "valid"
    ):
        zero = dict.fromkeys(
            (
                *CONTENT_COMPONENTS,
                "contradiction_drift",
                "valence_drift",
                "arousal_drift",
                "dominance_drift",
                "vad_drift",
                "content_drift",
                "stdi",
            ),
            0.0,
        )
        result.update(
            status="valid",
            identity_shortcut=True,
            stdi=0.0,
            method_gap=0.0,
            embedding={"status": "valid", "metrics": zero},
            llm_judge={
                "status": "valid",
                "metrics": zero,
                "requested_repeats": judge_repeats,
                "valid_repeats": 0,
                "executed_repeats": 0,
                "identity_shortcut": True,
                "samples": [],
            },
            shared_vad={"status": "valid", **{k: 0.0 for k in vad if k != "status"}},
        )
        return result
    try:
        embedding = comparator.compare_structured(original_structure, modified_structure)
        embedding["metrics"] = (
            complete_stdi(embedding["components"], vad)
            if embedding["status"] == "valid" and vad["status"] == "valid"
            else None
        )
        if embedding["metrics"] is None:
            embedding["status"] = "partial"
        result["embedding"] = embedding
    except Exception as exc:
        result["embedding"] = {"status": "failed", "error": str(exc), "metrics": None}
    inputs = {
        "original_text": original_text,
        "modified_text": modified_text,
        "title": title,
        "original_structure": topic_structure_to_dict(original_structure),
        "modified_structure": topic_structure_to_dict(modified_structure),
        "model": str(model),
        "provider": str(provider),
        "base_url": base_url,
        "prompt_version": STRUCTURED_JUDGE_VERSION,
    }
    samples = []
    cancelled = False
    for repeat_index in range(judge_repeats):
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        # Keep the historical first draw; subsequent draws have distinct durable keys.
        sample_inputs = inputs if repeat_index == 0 else {**inputs, "repeat_index": repeat_index}
        sample: dict[str, Any] = {"repeat": repeat_index + 1, "metrics": None}
        try:
            saved = None if uncached_judge else cache.get("judge", sample_inputs)
            sample["cache_hit"] = saved is not None
            if saved is None:
                saved = asdict(
                    judge_fn(
                        original_text=original_text,
                        modified_text=modified_text,
                        title=title,
                        original_structure=original_structure,
                        modified_structure=modified_structure,
                        model=model,
                        provider=provider,
                        api_key=api_key,
                        base_url=base_url,
                        retry_attempts=retry_attempts,
                        before_request_hook=before_request_hook,
                        structured=True,
                    )
                )
            scores = [
                saved["component_drifts"][key]
                for key in (*CONTENT_COMPONENTS, "contradiction_drift")
            ]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores):
                raise ValueError("Judge components must be finite scores between zero and one")
            sample["judgment"] = saved
            if not uncached_judge and not sample["cache_hit"]:
                cache.put("judge", sample_inputs, saved)
            sample["status"] = "valid"
            if vad["status"] == "valid":
                sample["metrics"] = complete_stdi(saved["component_drifts"], vad)
        except Exception as exc:
            sample.update(
                status="failed", error=str(exc), provenance=getattr(exc, "provenance", {})
            )
        samples.append(sample)
    valid_samples = [sample for sample in samples if sample["status"] == "valid"]
    all_valid = len(valid_samples) == judge_repeats
    complete_metrics = [sample["metrics"] for sample in valid_samples if sample["metrics"]]
    statistics = {}
    for key in (*CONTENT_COMPONENTS, "contradiction_drift", "stdi"):
        values = (
            [metrics[key] for metrics in complete_metrics]
            if key == "stdi"
            else [sample["judgment"]["component_drifts"][key] for sample in valid_samples]
        )
        statistics[key] = {
            "count": len(values),
            "mean": fmean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    judge_metrics = (
        {key: fmean(metrics[key] for metrics in complete_metrics) for key in complete_metrics[0]}
        if all_valid and len(complete_metrics) == judge_repeats
        else None
    )
    branch: dict[str, Any] = {
        "status": "valid"
        if judge_metrics is not None
        else "partial"
        if valid_samples
        else "failed",
        "metrics": judge_metrics,
        "aggregation": "mean_complete_metrics",
        "requested_repeats": judge_repeats,
        "executed_repeats": len(samples),
        "valid_repeats": len(valid_samples),
        "cache_hit": bool(samples) and all(sample.get("cache_hit", False) for sample in samples),
        "cancelled": cancelled,
        "samples": samples,
        "statistics": statistics,
    }
    if judge_repeats == 1 and valid_samples:
        branch["judgment"] = valid_samples[0]["judgment"]
    elif judge_repeats == 1 and samples:
        branch["provenance"] = samples[0].get("provenance", {})
    if not all_valid:
        errors = [
            f"Repeat {sample['repeat']}: {sample['error']}"
            for sample in samples
            if sample["status"] == "failed"
        ]
        if cancelled:
            errors.append("Judge evaluation cancelled before all repeats completed")
        branch["error"] = "; ".join(errors)
        if judge_repeats == 1 and samples and not cancelled:
            branch["error"] = samples[0]["error"]
    result["llm_judge"] = branch
    a, b = result["embedding"]["metrics"], result["llm_judge"]["metrics"]
    if a is not None and b is not None:
        result.update(
            status="valid", stdi=(a["stdi"] + b["stdi"]) / 2, method_gap=abs(a["stdi"] - b["stdi"])
        )
    else:
        result["status"] = (
            "partial"
            if any(
                result[branch]["status"] in {"valid", "partial"}
                for branch in ("embedding", "llm_judge")
            )
            else "failed"
        )
    return result
