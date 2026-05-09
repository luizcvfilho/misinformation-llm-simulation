from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from misinformation_simulation.enums import Provider


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
    stdi_incremental: float | None = None
    theme_drift_incremental: float | None = None
    subtopic_drift_incremental: float | None = None
    entity_drift_incremental: float | None = None
    relation_drift_incremental: float | None = None
    original_topic_structure_status: str = "not_requested"
    original_topic_structure_error: str | None = None
    rewritten_topic_structure_status: str = "not_requested"
    rewritten_topic_structure_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        metadata = record.pop("metadata", {})
        for key, value in metadata.items():
            record[f"metadata_{key}"] = value
        return record


@dataclass(slots=True)
class SimulationResult:
    summary: dict[str, Any]
    step_results: list[SimulationStepResult]
    summary_path: Path | None = None
    steps_path: Path | None = None
