from __future__ import annotations

import json

import pandas as pd

from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.graph import (
    SimulationNode,
    SimulationStepResult,
    run_news_interaction_graph,
)
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure


def make_structure(topic: str) -> TopicStructure:
    return TopicStructure(
        main_topic=topic,
        subtopics=["subtopic"],
        central_entities=["entity"],
        central_relations=[TopicRelation("entity", "does", "thing")],
        narrative_frame="frame",
    )


def test_simulation_step_result_flattens_metadata() -> None:
    result = SimulationStepResult(
        news_id="news-1",
        step_index=1,
        node_id="node",
        node_label="Node",
        source_node_id="original",
        source_node_label="description",
        provider="gemini",
        model="model",
        personality="persona",
        source_text="source",
        rewritten_text="rewrite",
        target_language="en",
        target_language_source="default",
        rewrite_status="success",
        rewrite_error=None,
        metadata={"title": "Title"},
    )

    record = result.to_record()

    assert "metadata" not in record
    assert record["metadata_title"] == "Title"


def test_run_news_interaction_graph_records_success_and_persists_outputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: "rewritten text")
    monkeypatch.setattr(
        graph, "extract_topic_structure", lambda **_kwargs: make_structure("original")
    )
    monkeypatch.setattr(
        graph,
        "_extract_compared_structure",
        lambda **_kwargs: make_structure("rewritten"),
    )
    monkeypatch.setattr(
        graph,
        "calculate_stdi",
        lambda _original, _rewritten: {
            "stdi": 0.5,
            "theme_drift": 0.1,
            "subtopic_drift": 0.2,
            "entity_drift": 0.3,
            "relation_drift": 0.4,
        },
    )
    progress: list[str] = []
    df = pd.DataFrame(
        [
            {
                "article_id": "news-1",
                "title": "Title",
                "description": "Original text",
                "language": "en",
            }
        ]
    )

    result = run_news_interaction_graph(
        df,
        nodes=[SimulationNode("node-1", "model", "gemini", "persona", label="Node 1")],
        news_id_column="article_id",
        output_dir=tmp_path,
        output_prefix="run",
        progress_callback=progress.append,
    )

    assert result.summary["rows_processed"] == 1
    assert result.summary["steps_success"] == 1
    assert result.step_results[0].rewritten_text == "rewritten text"
    assert result.step_results[0].stdi_vs_original == 0.5
    assert result.summary_path == tmp_path / "run_summary.json"
    assert result.steps_path == tmp_path / "run_steps.jsonl"
    assert json.loads(result.summary_path.read_text(encoding="utf-8"))["steps_success"] == 1
    assert "metadata_title" in result.steps_path.read_text(encoding="utf-8")
    assert any("Run finished" in message for message in progress)


def test_run_news_interaction_graph_blocks_steps_when_original_text_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    df = pd.DataFrame([{"title": "Title", "description": ""}])

    result = run_news_interaction_graph(
        df,
        nodes=[SimulationNode("node-1", "model", "gemini", "persona")],
        allow_title_fallback=False,
        persist_results=False,
    )

    assert result.summary["steps_success"] == 0
    assert result.step_results[0].rewrite_status == "blocked"
    assert result.step_results[0].original_topic_structure_status == "error"


def test_run_news_interaction_graph_records_rewrite_error(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        graph, "extract_topic_structure", lambda **_kwargs: make_structure("original")
    )

    def fail_rewrite(**_kwargs) -> str:
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(graph, "_generate_rewrite", fail_rewrite)
    df = pd.DataFrame([{"title": "Title", "description": "Original text"}])

    result = run_news_interaction_graph(
        df,
        nodes=[SimulationNode("node-1", "model", "chatgpt", "persona")],
        persist_results=False,
    )

    assert result.summary["steps_error"] == 1
    assert result.step_results[0].rewrite_status == "error"
    assert result.step_results[0].rewrite_error == "rewrite failed"
