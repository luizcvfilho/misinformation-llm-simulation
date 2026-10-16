from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from statistics import fmean, stdev
from typing import Any

from misinformation_simulation.config.prompts import STRUCTURED_JUDGE_VERSION
from misinformation_simulation.llm.progress import operation
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.text_metrics.vad_evaluation import vad_for_branch
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

DUAL_STDI_VERSION = "dual_stdi_v6"
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
        **{
            key: vad[key]
            for key in ("valence_drift", "arousal_drift", "dominance_drift", "vad_drift")
            if key in vad
        },
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
    vad_evaluation: dict[str, Any] | None = None,
    comparison_method: str = "dual",
) -> dict[str, Any]:
    return _DualSTDIComparison(
        original_text=original_text,
        modified_text=modified_text,
        title=title,
        original_structure=original_structure,
        modified_structure=modified_structure,
        original_vad=original_vad,
        modified_vad=modified_vad,
        comparator=comparator,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        retry_attempts=retry_attempts,
        before_request_hook=before_request_hook,
        judge_fn=judge_fn,
        cache=cache,
        uncached_judge=uncached_judge,
        judge_repeats=judge_repeats,
        cancel_check=cancel_check,
        vad_evaluation=vad_evaluation,
        comparison_method=comparison_method,
    ).run()


class _DualSTDIComparison:
    """Combine structural comparisons with cached, repeated judge evaluations."""

    def __init__(
        self,
        *,
        original_text,
        modified_text,
        title,
        original_structure,
        modified_structure,
        original_vad,
        modified_vad,
        comparator,
        model,
        provider,
        api_key,
        base_url,
        retry_attempts,
        before_request_hook,
        judge_fn,
        cache,
        uncached_judge,
        judge_repeats,
        cancel_check,
        vad_evaluation,
        comparison_method,
    ):
        self.original_text = original_text
        self.modified_text = modified_text
        self.title = title
        self.original_structure = original_structure
        self.modified_structure = modified_structure
        self.original_vad = original_vad
        self.modified_vad = modified_vad
        self.comparator = comparator
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.retry_attempts = retry_attempts
        self.before_request_hook = before_request_hook
        self.judge_fn = judge_fn
        self.cache = cache
        self.uncached_judge = uncached_judge
        self.judge_repeats = judge_repeats
        self.cancel_check = cancel_check
        self.vad_evaluation = vad_evaluation
        self.comparison_method = comparison_method

    def run(self):
        if self.comparison_method not in {"dual", "cluster", "llm"}:
            raise ValueError("STDI comparison method must be 'dual', 'cluster' or 'llm'.")
        validate_judge_repeats(self.judge_repeats)
        self.cache = self.cache or EvaluationCache()
        self.result: dict[str, Any] = {
            "version": DUAL_STDI_VERSION,
            "formula_version": FORMULA_VERSION,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "polarity_weight": 0.2,
            "duration_weight": 0.2,
            "numeric_weight": 0.2,
            "numeric_aggregation": "maximum",
            "branch_weights": [0.5, 0.5],
            "judge_repeats": self.judge_repeats,
            "judge_aggregation": "mean_complete_metrics",
            "stdi": None,
            "method_gap": None,
            "metrics": None,
            "comparison_method": self.comparison_method,
            "input_sha256": input_hash(
                {
                    "original": self.original_text,
                    "modified": self.modified_text,
                    "title": self.title,
                }
            ),
        }
        legacy_vad = shared_vad_drift(self.original_vad, self.modified_vad)
        evaluation = self.vad_evaluation or {"method": "model", "model": legacy_vad}
        paired = self.comparison_method == "dual"
        self.embedding_vad = vad_for_branch(evaluation, "embedding", paired=paired)
        self.judge_vad = vad_for_branch(evaluation, "llm_judge", paired=paired)
        self.vad = vad_for_branch(evaluation, "dual", paired=paired)
        self.result["shared_vad"] = self.vad
        self.result["vad_evaluation"] = evaluation
        self.result["vad_method"] = evaluation["method"]
        self.result["vad_sources"] = dict(
            embedding=self.embedding_vad["source"],
            llm_judge=self.judge_vad["source"],
            dual=self.vad["source"],
        )
        self.result["dual_aggregation"] = "mean_complete_branches_with_selected_vad"
        if (
            self.original_text == self.modified_text
            and (not structure_issues(self.original_structure))
            and (self.vad["status"] == "valid")
        ):
            return self._identity_result()

        self._evaluate_embedding()

        self._evaluate_judge()

        self._aggregate_judge()

        if self.comparison_method == "cluster":
            self.result["llm_judge"].update(status="not_requested")
        if self.comparison_method != "dual":
            chosen = self.result[
                "embedding" if self.comparison_method == "cluster" else "llm_judge"
            ]
            metrics = chosen.get("metrics")
            self.result.update(
                status=chosen["status"],
                metrics=metrics,
                stdi=metrics["stdi"] if metrics else None,
            )
            return self.result
        a, b = (
            self.result["embedding"].get("dual_metrics"),
            self.result["llm_judge"].get("dual_metrics"),
        )
        if a is not None and b is not None:
            metrics = {key: (a[key] + b[key]) / 2 for key in a}
            own_a, own_b = (
                self.result["embedding"]["metrics"],
                self.result["llm_judge"]["metrics"],
            )
            self.result.update(
                status="valid",
                stdi=metrics["stdi"],
                metrics=metrics,
                method_gap=abs(own_a["stdi"] - own_b["stdi"])
                if own_a is not None and own_b is not None
                else None,
            )
        else:
            self.result["status"] = (
                "partial"
                if any(
                    self.result[branch]["status"] in {"valid", "partial"}
                    for branch in ("embedding", "llm_judge")
                )
                else "failed"
            )
        return self.result

    def _identity_result(self):
        if self.comparison_method != "cluster":
            for repeat_index in range(self.judge_repeats):
                self.repeat_index = repeat_index
                with operation(
                    "judge",
                    model=self.model,
                    provider=self.provider,
                    repeat=self.repeat_index + 1,
                    repeats=self.judge_repeats,
                ) as self.progress:
                    self.progress.update(status="skipped", reason="identical texts")
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
        self.result.update(
            status="valid",
            identity_shortcut=True,
            stdi=0.0,
            method_gap=0.0,
            embedding={
                "status": "valid",
                "metrics": zero,
                "vad_source": self.embedding_vad["source"],
            },
            llm_judge={
                "status": "valid",
                "metrics": zero,
                "requested_repeats": self.judge_repeats,
                "valid_repeats": 0,
                "executed_repeats": 0,
                "identity_shortcut": True,
                "samples": [],
                "vad_source": self.judge_vad["source"],
            },
            shared_vad={
                "status": "valid",
                "source": self.vad["source"],
                **{
                    key: 0.0
                    for key in (
                        "valence_drift",
                        "arousal_drift",
                        "dominance_drift",
                        "vad_drift",
                    )
                },
            },
            metrics=zero,
        )
        if self.comparison_method != "dual":
            excluded = "embedding" if self.comparison_method == "llm" else "llm_judge"
            self.result[excluded] = {"status": "not_requested", "metrics": None}
            self.result["method_gap"] = None
        return self.result

    def _evaluate_embedding(self):
        try:
            embedding = (
                self.comparator.compare_structured(self.original_structure, self.modified_structure)
                if self.comparison_method != "llm"
                else {"status": "not_requested", "components": None}
            )
            embedding["metrics"] = (
                complete_stdi(embedding["components"], self.embedding_vad)
                if embedding["status"] == "valid" and self.embedding_vad["status"] == "valid"
                else None
            )
            embedding["dual_metrics"] = (
                complete_stdi(embedding["components"], self.vad)
                if embedding["status"] == "valid" and self.vad["status"] == "valid"
                else None
            )
            embedding["vad_source"] = self.embedding_vad["source"]
            if embedding["metrics"] is None and embedding["status"] != "not_requested":
                embedding["status"] = "partial"
            self.result["embedding"] = embedding
        except Exception as exc:
            self.result["embedding"] = {"status": "failed", "error": str(exc), "metrics": None}

    def _evaluate_judge_sample(self):
        sample_inputs = (
            self.inputs
            if self.repeat_index == 0
            else {**self.inputs, "repeat_index": self.repeat_index}
        )
        sample: dict[str, Any] = {"repeat": self.repeat_index + 1, "metrics": None}
        with operation(
            "judge",
            model=self.model,
            provider=self.provider,
            repeat=self.repeat_index + 1,
            repeats=self.judge_repeats,
        ) as self.progress:
            try:
                saved = None if self.uncached_judge else self.cache.get("judge", sample_inputs)
                sample["cache_hit"] = saved is not None
                self.progress["cache_hit"] = saved is not None
                if saved is None:
                    saved = asdict(
                        self.judge_fn(
                            original_text=self.original_text,
                            modified_text=self.modified_text,
                            title=self.title,
                            original_structure=self.original_structure,
                            modified_structure=self.modified_structure,
                            model=self.model,
                            provider=self.provider,
                            api_key=self.api_key,
                            base_url=self.base_url,
                            retry_attempts=self.retry_attempts,
                            before_request_hook=self.before_request_hook,
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
                if not self.uncached_judge and (not sample["cache_hit"]):
                    self.cache.put("judge", sample_inputs, saved)
                sample["status"] = "valid"
                if self.judge_vad["status"] == "valid":
                    sample["metrics"] = complete_stdi(saved["component_drifts"], self.judge_vad)
                if self.vad["status"] == "valid":
                    sample["dual_metrics"] = complete_stdi(saved["component_drifts"], self.vad)
            except Exception as exc:
                sample.update(
                    status="failed", error=str(exc), provenance=getattr(exc, "provenance", {})
                )
                self.progress.update(status="failed", error=str(exc))
        self.samples.append(sample)

    def _evaluate_judge(self):
        self.inputs = {
            "original_text": self.original_text,
            "modified_text": self.modified_text,
            "title": self.title,
            "original_structure": topic_structure_to_dict(self.original_structure),
            "modified_structure": topic_structure_to_dict(self.modified_structure),
            "model": str(self.model),
            "provider": str(self.provider),
            "base_url": self.base_url,
            "prompt_version": STRUCTURED_JUDGE_VERSION,
        }
        self.samples = []
        self.cancelled = False
        for repeat_index in range(self.judge_repeats if self.comparison_method != "cluster" else 0):
            self.repeat_index = repeat_index
            if self.cancel_check is not None and self.cancel_check():
                self.cancelled = True
                break
            self._evaluate_judge_sample()

    def _aggregate_judge(self):
        self.valid_samples = [sample for sample in self.samples if sample["status"] == "valid"]
        all_valid = len(self.valid_samples) == self.judge_repeats
        self.complete_metrics = [
            sample["metrics"] for sample in self.valid_samples if sample["metrics"]
        ]
        self._record_judge_statistics()

        judge_metrics = (
            {
                key: fmean(metrics[key] for metrics in self.complete_metrics)
                for key in self.complete_metrics[0]
            }
            if all_valid and len(self.complete_metrics) == self.judge_repeats
            else None
        )
        dual_samples = [
            sample["dual_metrics"]
            for sample in self.valid_samples
            if sample.get("dual_metrics") is not None
        ]
        dual_judge_metrics = (
            {key: fmean(metrics[key] for metrics in dual_samples) for key in dual_samples[0]}
            if all_valid and len(dual_samples) == self.judge_repeats
            else None
        )
        branch: dict[str, Any] = {
            "status": "valid"
            if judge_metrics is not None
            else "partial"
            if self.valid_samples
            else "failed",
            "metrics": judge_metrics,
            "dual_metrics": dual_judge_metrics,
            "vad_source": self.judge_vad["source"],
            "aggregation": "mean_complete_metrics",
            "requested_repeats": self.judge_repeats,
            "executed_repeats": len(self.samples),
            "valid_repeats": len(self.valid_samples),
            "cache_hit": bool(self.samples)
            and all(sample.get("cache_hit", False) for sample in self.samples),
            "cancelled": self.cancelled,
            "samples": self.samples,
            "statistics": self.statistics,
        }
        if self.judge_repeats == 1 and self.valid_samples:
            branch["judgment"] = self.valid_samples[0]["judgment"]
        elif self.judge_repeats == 1 and self.samples:
            branch["provenance"] = self.samples[0].get("provenance", {})
        if not all_valid:
            errors = [
                f"Repeat {sample['repeat']}: {sample['error']}"
                for sample in self.samples
                if sample["status"] == "failed"
            ]
            if self.cancelled:
                errors.append("Judge evaluation cancelled before all repeats completed")
            branch["error"] = "; ".join(errors)
            if self.judge_repeats == 1 and self.samples and (not self.cancelled):
                branch["error"] = self.samples[0]["error"]
        self.result["llm_judge"] = branch

    def _record_judge_statistics(self):
        self.statistics = {}
        for key in (*CONTENT_COMPONENTS, "contradiction_drift", "stdi"):
            values = (
                [metrics[key] for metrics in self.complete_metrics]
                if key == "stdi"
                else [sample["judgment"]["component_drifts"][key] for sample in self.valid_samples]
            )
            self.statistics[key] = {
                "count": len(values),
                "mean": fmean(values) if values else None,
                "std": stdev(values) if len(values) > 1 else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
