from __future__ import annotations

import json

import numpy as np
import pandas as pd

from misinformation_simulation.topic_drift.comparison_workflow import (
    compare_method_outputs,
    load_comparison_input,
    run_comparison_workflow,
    write_comparison_output,
)
from misinformation_simulation.topic_drift.models import TopicStructure
from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison


class StableEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float("econom" in normalized),
                    float("health" in normalized),
                    float("bank" in normalized),
                    float("minister" in normalized),
                    1.0,
                ]
            )
        return np.asarray(vectors, dtype=float)


def _structure_columns(prefix: str, topic: str, entity: str, action: str) -> dict[str, str]:
    return {
        f"{prefix}_main_topic": topic,
        f"{prefix}_subtopics": json.dumps([topic]),
        f"{prefix}_central_entities": json.dumps([entity]),
        f"{prefix}_central_relations": json.dumps(
            [{"subject": entity, "action": action, "object": "policy"}]
        ),
    }


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluation_id": "pair_a",
                "title": "Policy update",
                "original_text": "The central bank announces policy.",
                "modified_text": "The central bank announces policy.",
                **_structure_columns("original", "economy policy", "Central Bank", "announces"),
                **_structure_columns("modified", "economy policy", "Central Bank", "announces"),
            },
            {
                "evaluation_id": "pair_b",
                "title": "Policy update",
                "original_text": "The central bank announces policy.",
                "modified_text": "The health minister cancels policy.",
                **_structure_columns("original", "economy policy", "Central Bank", "announces"),
                **_structure_columns("modified", "health policy", "Health Minister", "cancels"),
            },
        ]
    )


def test_cluster_workflow_reuses_persisted_structures_without_extraction() -> None:
    calls: list[str] = []
    progress_messages: list[str] = []

    def extractor(**_kwargs: object) -> TopicStructure:
        calls.append("called")
        raise AssertionError("The extractor should not run for persisted structures.")

    workflow = run_comparison_workflow(
        _pairs(),
        method="cluster",
        extraction_fn=extractor,
        embedder=StableEmbedder(),
        random_state=1,
        progress_callback=progress_messages.append,
    )

    assert not calls
    assert workflow.results["comparison_method"].eq("cluster").all()
    assert workflow.results["comparison_status"].eq("success").all()
    assert workflow.results.loc[0, "theme_drift"] == 0.0
    assert workflow.results.loc[1, "theme_drift"] > 0.0
    assert workflow.cluster_artifacts is not None
    assert any("Reusing shared structures" in message for message in progress_messages)
    assert any("Fitting shared clusters" in message for message in progress_messages)
    assert any("Comparing pair" in message for message in progress_messages)


def test_llm_workflow_and_output_reuse(tmp_path) -> None:
    def semantic_comparator(**_kwargs: object) -> SemanticSTDIComparison:
        return SemanticSTDIComparison(
            component_drifts={
                "theme_drift": 0.25,
                "subtopic_drift": 0.5,
                "entity_drift": 0.75,
                "relation_drift": 1.0,
                "contradiction_drift": 0.0,
            },
            rationales={"theme_drift": "Test rationale."},
        )

    llm_workflow = run_comparison_workflow(
        _pairs(), method="llm_semantic", llm_comparison_fn=semantic_comparator
    )
    output_directory = tmp_path / "llm"
    write_comparison_output(output_directory, llm_workflow)
    loaded = load_comparison_input(output_directory)

    assert loaded["pair_id"].tolist() == ["pair_a", "pair_b"]
    assert (output_directory / "manifest.json").exists()
    assert loaded["comparison_theme_drift_rationale"].eq("Test rationale.").all()

    cluster_workflow = run_comparison_workflow(
        loaded,
        method="cluster",
        embedder=StableEmbedder(),
        random_state=1,
    )
    comparison = compare_method_outputs(
        llm_workflow.results,
        cluster_workflow.results,
        left_label="llm",
        right_label="cluster",
    )

    assert len(comparison) == 2
    assert "delta_relation_drift" in comparison
