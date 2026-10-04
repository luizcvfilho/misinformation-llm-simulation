from __future__ import annotations

import json
from hashlib import sha256
from threading import Event

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.config.prompts import (
    INTERPRETIVE_PERSONALITY_EXTENSIONS,
    PROMPT_TEMPLATE,
    REWRITE_SYSTEM_INSTRUCTION,
    resolve_graph_personality_prompt,
    resolve_graph_rewrite_prompt,
)
from misinformation_simulation.enums import DefaultPersonality
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
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


class KeywordEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(
            [
                [
                    float("iran" in text.casefold()),
                    float("proposal" in text.casefold()),
                    float("health" in text.casefold()),
                    1.0,
                ]
                for text in texts
            ]
        )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.mark.parametrize("mode", ["faithful", "interpretive"])
def test_graph_transmission_modes_pass_previous_message_and_persist_prompt_provenance(
    mode, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    extraction_titles = []

    def extract(**kwargs):
        extraction_titles.append(kwargs["title"])
        return make_structure("political policy")

    monkeypatch.setattr(graph, "extract_topic_structure", extract)
    monkeypatch.setattr(graph, "_extract_compared_structure", extract)
    calls = []
    outputs = iter(["First person's interpretation", "Second person's interpretation"])

    def rewrite(**kwargs):
        calls.append(kwargs)
        return next(outputs)

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    nodes = [
        SimulationNode("first", "model", "chatgpt", DefaultPersonality.ConspiracyDenialist),
        SimulationNode("second", "model", "chatgpt", DefaultPersonality.InvestigativeSkeptic),
    ]
    title = "Distinct original headline"
    original = "The government proposed a policy after an open debate."
    result = run_news_interaction_graph(
        pd.DataFrame([{"title": title, "description": original, "language": "en"}]),
        nodes=nodes,
        **({"rewrite_mode": mode} if mode != "faithful" else {}),
        stdi_embedder=KeywordEmbedder(),
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        output_dir=tmp_path,
    )
    prompt_config = resolve_graph_rewrite_prompt(mode)
    assert original in calls[0]["prompt"]
    assert "First person's interpretation" in calls[1]["prompt"]
    assert original not in calls[1]["prompt"]
    assert all(call["system_instruction"] == prompt_config.system_instruction for call in calls)
    assert extraction_titles == [title, title, title]
    assert result.step_results[1].source_text == "First person's interpretation"
    assert result.summary["rewrite_mode"] == mode
    assert result.summary["rewrite_prompt_version"] == prompt_config.version
    assert result.summary["rewrite_original_title_context"] == (mode == "faithful")
    assert result.summary["rewrite_temperature_requested"] == 0.8
    assert result.summary["rewrite_prompt_template"] == prompt_config.template
    assert result.summary["rewrite_system_instruction"] == prompt_config.system_instruction
    persisted = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert persisted["rewrite_mode"] == mode
    saved_steps = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    assert all(step["metadata_rewrite_mode"] == mode for step in saved_steps)
    assert all(len(step["metadata_rewrite_prompt_sha256"]) == 64 for step in saved_steps)
    for index, step in enumerate(saved_steps):
        reconstructed_prompt = prompt_config.template.format(
            personality=step["metadata_rewrite_effective_personality"],
            target_language_name="English",
            target_language_code=step["target_language"],
            title=title,
            original_text=step["source_text"],
        )
        assert reconstructed_prompt == calls[index]["prompt"]
        assert (
            sha256(reconstructed_prompt.encode("utf-8")).hexdigest()
            == (step["metadata_rewrite_prompt_sha256"])
        )
    if mode == "faithful":
        assert calls[0]["prompt"] == PROMPT_TEMPLATE.format(
            personality=nodes[0].personality,
            target_language_name="English",
            target_language_code="en",
            title=title,
            original_text=original,
        )
        assert calls[0]["system_instruction"] == REWRITE_SYSTEM_INSTRUCTION
    else:
        assert all(title not in call["prompt"] for call in calls)
        assert "Keep factual content unchanged" not in calls[0]["prompt"]
        conspiracy_extension = INTERPRETIVE_PERSONALITY_EXTENSIONS["ConspiracyDenialist"]
        skeptic_extension = INTERPRETIVE_PERSONALITY_EXTENSIONS["InvestigativeSkeptic"]
        assert conspiracy_extension in calls[0]["prompt"]
        assert skeptic_extension not in calls[0]["prompt"]
        assert skeptic_extension in calls[1]["prompt"]
        assert conspiracy_extension not in calls[1]["prompt"]
        assert "conspir" not in prompt_config.template.casefold()
        assert "skept" not in prompt_config.template.casefold()
        assert result.summary["rewrite_prompt_version"] == "interpretive_v2"


@pytest.mark.parametrize(
    "preset", [DefaultPersonality.ConspiracyDenialist, DefaultPersonality.InvestigativeSkeptic]
)
@pytest.mark.parametrize("opening_sentence_only", [False, True])
def test_personality_extensions_support_full_and_legacy_presets(preset, opening_sentence_only):
    personality = preset.value.split(".", 1)[0] + "." if opening_sentence_only else preset.value
    extended = resolve_graph_personality_prompt(personality, rewrite_mode="interpretive")
    assert extended.startswith(personality)
    assert INTERPRETIVE_PERSONALITY_EXTENSIONS[preset.name] in extended
    assert resolve_graph_personality_prompt(personality, rewrite_mode="faithful") == personality


@pytest.mark.parametrize(
    "personality",
    [
        DefaultPersonality.ConservativeRight.value,
        "A skeptical reader with a custom perspective.",
        DefaultPersonality.InvestigativeSkeptic.value + " Use only my custom transmission rules.",
    ],
)
def test_interpretive_extensions_leave_other_and_custom_personalities_unchanged(personality):
    assert resolve_graph_personality_prompt(personality, rewrite_mode="interpretive") == personality


def test_invalid_transmission_mode_fails_before_creating_clients(monkeypatch) -> None:
    monkeypatch.setattr(
        graph, "create_llm_client", lambda **_kwargs: pytest.fail("Unexpected model client.")
    )
    with pytest.raises(ValueError, match="rewrite_mode"):
        run_news_interaction_graph(
            pd.DataFrame([{"description": "Original"}]),
            nodes=[SimulationNode("node", "model", "chatgpt", "persona")],
            rewrite_mode="unknown",
        )


@pytest.mark.parametrize("provider", ["chatgpt", "gemini"])
def test_rewrite_provider_receives_selected_system_instruction(provider, monkeypatch) -> None:
    calls = []

    def generate(_client, **kwargs):
        calls.append(kwargs)
        return "retold message"

    monkeypatch.setattr(graph, "generate_gemini_text_with_retry", generate)
    monkeypatch.setattr(graph, "generate_openai_text_with_retry", generate)
    instruction = resolve_graph_rewrite_prompt("interpretive").system_instruction
    result = graph._generate_rewrite(
        provider_normalized=provider,
        client=object(),
        model="model",
        prompt="received message",
        system_instruction=instruction,
        retry_attempts=2,
        limiter=MinuteRateLimiter(None),
    )
    assert result == "retold message"
    assert calls[0]["system_instruction"] == instruction
    assert calls[0]["temperature"] == 0.8


def test_graph_uses_shared_embedding_comparison_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(
        graph,
        "extract_topic_structure",
        lambda **_kwargs: make_structure("Iran rejects a proposal"),
    )
    rewritten_structures = iter(
        [make_structure("Iran's rejection of a proposal"), make_structure("Health policy")]
    )
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: next(rewritten_structures)
    )
    rewritten_texts = iter(["first version", "second version"])
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: next(rewritten_texts))

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[
            SimulationNode("first", "model", "chatgpt", "persona"),
            SimulationNode("second", "model", "chatgpt", "persona"),
        ],
        stdi_embedder=KeywordEmbedder(),
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        output_dir=tmp_path,
    )

    first, second = result.step_results
    assert result.summary["stdi_comparison_method"] == "cluster"
    assert result.summary["stdi_comparison_version"] == "cluster_v2"
    assert result.summary["stdi_embedding_model"] == "custom:KeywordEmbedder"
    assert first.theme_drift_vs_original == 0.0
    assert first.stdi_vs_original == 0.0
    assert first.stdi_incremental == 0.0
    assert second.theme_drift_vs_original > 0.0
    assert second.stdi_incremental == second.stdi_vs_original
    assert second.stdi_cumulative == second.stdi_incremental
    saved_steps = [json.loads(line) for line in result.steps_path.read_text().splitlines()]
    assert saved_steps[0]["stdi_vs_original"] == 0.0
    assert saved_steps[0]["metadata_stdi_comparison_version"] == "cluster_v2"
    assert "metadata_original_topic_domain" not in saved_steps[0]
    assert "topic_domain" not in json.loads(saved_steps[0]["metadata_original_json"])
    assert saved_steps[1]["stdi_cumulative"] == second.stdi_incremental


def test_graph_cancellation_saves_completed_steps(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    rewrites = []

    def rewrite(**_kwargs):
        rewrites.append("rewrite")
        return "rewritten text"

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    cancel_event = Event()

    def on_progress(message: str) -> None:
        if "STDI pending" in message:
            cancel_event.set()

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[
            SimulationNode("first", "model", "chatgpt", "persona"),
            SimulationNode("second", "model", "chatgpt", "persona"),
        ],
        stdi_embedder=KeywordEmbedder(),
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        output_dir=tmp_path,
        cancel_check=cancel_event.is_set,
        progress_callback=on_progress,
    )

    assert result.summary["cancelled"] is True
    assert result.summary["steps_total"] == 1
    assert result.summary["steps_success"] == 1
    assert len(rewrites) == 1
    assert result.step_results[0].stdi_incremental == 0.0
    assert json.loads(result.summary_path.read_text())["cancelled"] is True
    assert len(result.steps_path.read_text().splitlines()) == 1


def test_graph_cancellation_after_rewrite_skips_next_model_call(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    cancel_event = Event()

    def rewrite(**_kwargs):
        cancel_event.set()
        return "rewritten text"

    monkeypatch.setattr(graph, "_generate_rewrite", rewrite)
    monkeypatch.setattr(
        graph,
        "_extract_compared_structure",
        lambda **_kwargs: pytest.fail("Topic extraction must not start after cancellation."),
    )

    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[SimulationNode("first", "model", "chatgpt", "persona")],
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        output_dir=tmp_path,
        cancel_check=cancel_event.is_set,
    )

    assert result.summary["cancelled"] is True
    assert result.summary["steps_total"] == 0
    assert result.steps_path.read_text().strip() == ""


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
    work_progress: list[tuple[int, int, int, int]] = []
    df = pd.DataFrame(
        [
            {
                "article_id": "news-1",
                "title": "Title",
                "description": "Original text",
                "language": "en",
                "category": "politics; top",
            }
        ]
    )

    result = run_news_interaction_graph(
        stdi_comparison_method="lexical",
        df=df,
        nodes=[SimulationNode("node-1", "model", "gemini", "persona", label="Node 1")],
        news_id_column="article_id",
        output_dir=tmp_path,
        output_prefix="run",
        progress_callback=progress.append,
        work_progress_callback=lambda *counts: work_progress.append(counts),
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
    assert persisted_step["metadata_category"] == "politics; top"
    assert persisted_step["metadata_original_vad_valence"] == 3.0
    assert persisted_step["metadata_rewritten_vad_arousal"] == 4.0
    assert json.loads(persisted_step["metadata_original_json"])["main_topic"] == "original"
    assert json.loads(persisted_step["metadata_rewritten_json"])["main_topic"] == "rewritten"
    assert persisted_step["vad_drift_vs_original"] == pytest.approx(0.166667)
    assert persisted_step["stdi_cumulative"] == pytest.approx(0.275)
    assert any("Run finished" in message for message in progress)
    assert work_progress == [(1, 1, 0, 1), (1, 1, 1, 1)]


def test_run_news_interaction_graph_blocks_steps_when_original_text_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("gemini", object()))
    df = pd.DataFrame([{"title": "Title", "description": "", "category": "science"}])

    work_progress = []
    result = run_news_interaction_graph(
        stdi_comparison_method="lexical",
        df=df,
        nodes=[SimulationNode("node-1", "model", "gemini", "persona")],
        work_progress_callback=lambda *counts: work_progress.append(counts),
        allow_title_fallback=False,
        persist_results=False,
    )

    assert result.summary["steps_success"] == 0
    assert result.step_results[0].rewrite_status == "blocked"
    assert result.step_results[0].to_record()["metadata_category"] == "science"
    assert result.step_results[0].original_topic_structure_status == "error"
    assert work_progress == [(1, 1, 1, 1)]


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
        stdi_comparison_method="lexical",
        df=df,
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
        stdi_comparison_method="lexical",
        df=pd.DataFrame([{"title": "Title", "description": "Original text"}]),
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
        stdi_comparison_method="lexical",
        df=pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[SimulationNode("node", "model", "chatgpt", "persona")],
        vad_scorer=lambda _text: VADScore(None, 3.0, 3.0),
        persist_results=False,
    )

    step = result.step_results[0]
    assert step.rewrite_status == "blocked"
    assert step.original_topic_structure_status == "success"
    assert step.original_vad_status == "error"
    assert json.loads(step.metadata["original_json"])["main_topic"] == "topic"
    assert "VAD scoring must return" in step.original_vad_error


def test_graph_does_not_report_stdi_when_rewritten_vad_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: "rewritten text")

    result = run_news_interaction_graph(
        stdi_comparison_method="lexical",
        df=pd.DataFrame([{"title": "Title", "description": "Original text"}]),
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
    assert json.loads(step.metadata["rewritten_json"])["main_topic"] == "topic"
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
        stdi_comparison_method="lexical",
        df=pd.DataFrame([{"title": "First", "description": "Original text"}]),
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
        stdi_comparison_method="lexical",
        df=pd.DataFrame(
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


@pytest.mark.parametrize("rewrite_model", ["gpt-5.6-luna", "gpt-6-luna"])
def test_graph_persists_separate_rewrite_and_evaluation_models(
    monkeypatch, tmp_path, rewrite_model
):
    monkeypatch.setattr(graph, "create_llm_client", lambda **_kwargs: ("chatgpt", object()))
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_kwargs: "rewritten text")
    monkeypatch.setattr(graph, "extract_topic_structure", lambda **_kwargs: make_structure("topic"))
    monkeypatch.setattr(
        graph, "_extract_compared_structure", lambda **_kwargs: make_structure("topic")
    )
    result = run_news_interaction_graph(
        pd.DataFrame([{"title": "Title", "description": "Original text"}]),
        nodes=[SimulationNode("node", rewrite_model, "chatgpt", "persona")],
        topic_drift_model="evaluation-model",
        topic_drift_provider="gemini",
        stdi_comparison_method="lexical",
        vad_scorer=lambda _text: VADScore(3.0, 3.0, 3.0),
        output_dir=tmp_path,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    step = json.loads(result.steps_path.read_text(encoding="utf-8"))
    assert summary["topic_drift_model"] == "evaluation-model"
    assert summary["topic_drift_provider"] == "gemini"
    assert summary["nodes"][0]["model"] == rewrite_model
    assert step["model"] == step["metadata_rewrite_model"] == rewrite_model
    assert step["metadata_rewrite_provider"] == "chatgpt"
    assert step["metadata_topic_drift_model"] == "evaluation-model"
    assert step["metadata_topic_drift_provider"] == "gemini"
    assert step["metadata_vad_model"] == "custom_scorer"
    assert step["metadata_stdi_embedding_model"] is None
