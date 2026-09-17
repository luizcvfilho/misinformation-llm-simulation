from __future__ import annotations

import json

import pandas as pd
import pytest

from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.graph import (
    SimulationNode,
    SimulationStepResult,
    run_news_interaction_graph,
)
from misinformation_simulation.text_metrics.vad import VADScore
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
        vad_scorer=lambda text: (
            VADScore(3.0, 3.0, 3.0) if text == "Original text" else VADScore(2.0, 4.0, 3.0)
        ),
    )

    assert result.summary["rows_processed"] == 1
    assert result.summary["steps_success"] == 1
    assert result.step_results[0].rewritten_text == "rewritten text"
    assert result.step_results[0].stdi_vs_original == pytest.approx(0.275)
    assert result.step_results[0].stdi_cumulative == pytest.approx(0.275)
    assert result.step_results[0].vad_drift_vs_original == pytest.approx(0.166667)
    assert result.step_results[0].content_drift_vs_original == 0.25
    assert result.step_results[0].contradiction_drift_vs_original == 0.0
    assert result.summary_path == tmp_path / "run_summary.json"
    assert result.steps_path == tmp_path / "run_steps.jsonl"
    assert json.loads(result.summary_path.read_text(encoding="utf-8"))["steps_success"] == 1
    persisted_step = json.loads(result.steps_path.read_text(encoding="utf-8").strip())
    assert persisted_step["metadata_title"] == "Title"
    assert persisted_step["metadata_original_vad_valence"] == 3.0
    assert persisted_step["metadata_rewritten_vad_arousal"] == 4.0
    assert persisted_step["vad_drift_vs_original"] == pytest.approx(0.166667)
    assert persisted_step["stdi_cumulative"] == pytest.approx(0.275)
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
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
    )

    assert result.summary["steps_error"] == 1
    assert result.step_results[0].rewrite_status == "error"
    assert result.step_results[0].rewrite_error == "rewrite failed"


def test_graph_stdi_uses_vad_and_contradiction_against_both_references(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    rewritten_structures = [make_structure("topic"), make_structure("topic")]
    rewritten_structures[1].internal_contradiction_score = 0.5
    monkeypatch.setattr(
        graph,
        "_extract_compared_structure",
        lambda **_kwargs: rewritten_structures.pop(0),
    )
    rewritten_texts = iter(["first version", "second version"])
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: next(rewritten_texts))
    vad_scores = {
        "Original text": VADScore(3.0, 2.0, 2.0),
        "first version": VADScore(2.0, 3.0, 2.0),
        "second version": VADScore(3.0, 2.0, 2.0),
    }

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[
            SimulationNode("first", "model", "chatgpt", "persona"),
            SimulationNode("second", "model", "chatgpt", "persona"),
        ],
        vad_scorer=vad_scores.__getitem__,
        persist_results=False,
    )

    first, second = result.step_results
    assert first.stdi_incremental == first.stdi_vs_original
    assert first.stdi_cumulative == pytest.approx(0.033333)
    assert first.vad_drift_vs_original == pytest.approx(0.166667)
    assert first.stdi_vs_original == pytest.approx(0.033333)
    assert second.vad_drift_vs_original == 0.0
    assert second.vad_drift_incremental == pytest.approx(0.166667)
    assert second.contradiction_drift_vs_original == 0.5
    assert second.contradiction_drift_incremental == 0.5
    assert second.stdi_vs_original == pytest.approx(0.1)
    assert second.stdi_incremental == pytest.approx(0.13)
    assert second.stdi_cumulative == pytest.approx(0.163333)
    assert second.metadata["source_vad_valence"] == 2.0


def test_graph_blocks_original_when_vad_is_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[SimulationNode("node", "model", "chatgpt", "persona")],
        vad_scorer=lambda _text: VADScore(None, 3.0, 3.0),
        persist_results=False,
    )

    step = result.step_results[0]
    assert step.rewrite_status == "blocked"
    assert step.original_topic_structure_status == "success"
    assert step.original_vad_status == "error"
    assert "VAD scoring must return" in step.original_vad_error


def test_graph_does_not_report_stdi_when_rewritten_vad_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: "rewritten text")

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[SimulationNode("node", "model", "chatgpt", "persona")],
        vad_scorer=lambda text: (
            VADScore(3.0, 3.0, 3.0) if text == "Original text" else VADScore(None, 3.0, 3.0)
        ),
        persist_results=False,
    )

    step = result.step_results[0]
    assert step.rewrite_status == "error"
    assert step.rewritten_text == "rewritten text"
    assert step.rewritten_topic_structure_status == "success"
    assert step.rewritten_vad_status == "error"
    assert step.stdi_vs_original is None
    assert step.stdi_cumulative is None


def test_graph_cumulative_stdi_skips_failed_steps(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    rewrite_calls = iter(["first", RuntimeError("rewrite failed"), "third"])

    def rewrite(**_kwargs) -> str:
        value = next(rewrite_calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    scores = {
        "Original text": VADScore(3.0, 2.0, 2.0),
        "first": VADScore(2.0, 3.0, 2.0),
        "third": VADScore(3.0, 2.0, 2.0),
    }

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "First", "description": "Original text"}]),
        nodes=[
            SimulationNode("one", "model", "chatgpt", "persona"),
            SimulationNode("two", "model", "chatgpt", "persona"),
            SimulationNode("three", "model", "chatgpt", "persona"),
        ],
        vad_scorer=scores.__getitem__,
        persist_results=False,
    )

    first, failed, third = result.step_results
    assert first.stdi_cumulative == pytest.approx(0.033333)
    assert failed.stdi_cumulative is None
    assert third.source_node_id == "one"
    assert third.stdi_incremental == pytest.approx(0.033333)
    assert third.stdi_cumulative == pytest.approx(0.066666)


def test_graph_cumulative_stdi_resets_for_each_news_item(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: "rewritten")

    result = run_news_interaction_graph(
        pd.DataFrame(
            [
                {"title": "First", "description": "Original text"},
                {"title": "Second", "description": "Original text"},
            ]
        ),
        nodes=[SimulationNode("node", "model", "chatgpt", "persona")],
        vad_scorer=lambda text: (
            VADScore(3.0, 2.0, 2.0) if text == "Original text" else VADScore(2.0, 3.0, 2.0)
        ),
        persist_results=False,
    )

    assert [step.stdi_cumulative for step in result.step_results] == [
        pytest.approx(0.033333),
        pytest.approx(0.033333),
    ]
