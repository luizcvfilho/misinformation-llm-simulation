from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from misinformation_simulation.enums import Provider

EVALUATION_METRICS = (
    "stdi",
    "theme_drift",
    "subtopic_drift",
    "entity_drift",
    "relation_drift",
    "contradiction_drift",
    "valence_drift",
    "arousal_drift",
    "dominance_drift",
    "vad_drift",
    "content_drift",
)


def expand_dual_evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Expose each branch independently, including historical dual records."""
    for role in ("original", "source", "rewritten"):
        evaluation = record.get(f"metadata_{role}_vad_evaluation")
        if not isinstance(evaluation, dict):
            continue
        for source in ("model", "llm"):
            scores = evaluation.get(source, {}).get("score") or {}
            for dimension in ("valence", "arousal", "dominance"):
                record[f"vad_{source}_{role}_{dimension}"] = scores.get(dimension)
    for suffix in ("vs_original", "incremental"):
        from misinformation_simulation.text_metrics.vad_evaluation import vad_pair_columns

        vad = record.get(f"metadata_vad_evaluation_{suffix}")
        if isinstance(vad, dict):
            record.update(vad_pair_columns(vad, suffix=suffix))
        dual = record.get(f"metadata_dual_stdi_{suffix}")
        if not isinstance(dual, dict):
            continue
        for branch, name in (("embedding", "cluster"), ("llm_judge", "llm_judge")):
            evaluation = dual.get(branch, {})
            record[f"{name}_evaluation_{suffix}"] = evaluation
            record[f"stdi_status_{name}_{suffix}"] = evaluation.get("status", "not_requested")
            record[f"stdi_error_{name}_{suffix}"] = evaluation.get("error")
            metrics = evaluation.get("metrics") or {}
            record[f"stdi_vad_source_{name}_{suffix}"] = evaluation.get("vad_source")
            for metric in EVALUATION_METRICS:
                record[f"{metric}_{name}_{suffix}"] = metrics.get(metric)
        metrics = dual.get("metrics") or {}
        if dual.get("comparison_method", "dual") == "dual":
            for metric in EVALUATION_METRICS:
                record[f"{metric}_dual_{suffix}"] = metrics.get(metric)
            record[f"stdi_vad_source_dual_{suffix}"] = dual.get("vad_sources", {}).get("dual")
    return record


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
    contradiction_drift_vs_original: float | None = None
    valence_drift_vs_original: float | None = None
    arousal_drift_vs_original: float | None = None
    dominance_drift_vs_original: float | None = None
    vad_drift_vs_original: float | None = None
    content_drift_vs_original: float | None = None
    stdi_incremental: float | None = None
    theme_drift_incremental: float | None = None
    subtopic_drift_incremental: float | None = None
    entity_drift_incremental: float | None = None
    relation_drift_incremental: float | None = None
    contradiction_drift_incremental: float | None = None
    valence_drift_incremental: float | None = None
    arousal_drift_incremental: float | None = None
    dominance_drift_incremental: float | None = None
    vad_drift_incremental: float | None = None
    content_drift_incremental: float | None = None
    stdi_cumulative: float | None = None
    stdi_embedding_vs_original: float | None = None
    stdi_llm_judge_vs_original: float | None = None
    stdi_method_gap_vs_original: float | None = None
    stdi_status_vs_original: str = "not_requested"
    stdi_embedding_incremental: float | None = None
    stdi_llm_judge_incremental: float | None = None
    stdi_method_gap_incremental: float | None = None
    stdi_status_incremental: str = "not_requested"
    stdi_cumulative_valid_steps: int = 0
    stdi_chain_complete: bool | None = None
    stdi_cluster_cumulative: float | None = None
    stdi_cluster_cumulative_valid_steps: int = 0
    stdi_cluster_chain_complete: bool | None = None
    stdi_llm_judge_cumulative: float | None = None
    stdi_llm_judge_cumulative_valid_steps: int = 0
    stdi_llm_judge_chain_complete: bool | None = None
    original_topic_structure_status: str = "not_requested"
    original_topic_structure_error: str | None = None
    original_vad_status: str = "not_requested"
    original_vad_error: str | None = None
    rewritten_topic_structure_status: str = "not_requested"
    rewritten_topic_structure_error: str | None = None
    rewritten_vad_status: str = "not_requested"
    rewritten_vad_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        metadata = record.pop("metadata", {})
        for key, value in metadata.items():
            record[f"metadata_{key}"] = value
        return expand_dual_evaluation_record(record)


@dataclass(slots=True)
class SimulationResult:
    summary: dict[str, Any]
    step_results: list[SimulationStepResult]
    summary_path: Path | None = None
    steps_path: Path | None = None
    path_results: list[SimulationResult] = field(default_factory=list)
