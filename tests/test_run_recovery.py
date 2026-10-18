from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps.interaction_graph_attempts import (
    RunAttempt,
    find_attempts,
    load_attempt,
    load_corpus,
    recovery_inputs,
    restore_evaluation_cache,
    result_state,
    save_corpus,
)
from misinformation_simulation.apps.interaction_graph_parallel import run_parallel_queue
from misinformation_simulation.apps.interaction_graph_run_job import (
    GraphRunJob,
    _run_graph_queue,
    start_graph_run_job,
)
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form
from misinformation_simulation.llm.checkpoint import checkpoint_response, use_call_checkpoint
from misinformation_simulation.llm.progress import operation
from misinformation_simulation.llm.recovery import (
    PermissionStore,
    RecoveryCancelled,
    RecoveryDecision,
    RecoverySession,
    get_recovery,
    use_recovery,
)
from misinformation_simulation.llm.retry import generate_and_parse_with_retry
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.types import SimulationResult, SimulationStepResult
from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
from misinformation_simulation.topic_drift.models import TopicStructure


def settings(tmp_path, rows=1, limit=1):
    return dict(
        output_dir=str(tmp_path),
        output_prefix="batch",
        max_rows=rows,
        max_simultaneous_runs=limit,
        text_column="description",
        title_column="title",
        news_id_column="",
        sleep_seconds=0,
        max_requests_per_minute=0,
        retry_attempts=1,
        allow_title_fallback=True,
        topic_drift_model="test",
        topic_drift_provider="gemini",
        stdi_comparison_method="cluster",
        reuse_previous_results=True,
    )


def graphs():
    return [dict(name="Graph", nodes=[create_default_node_form(1)])]


def frame():
    return pd.DataFrame([dict(title="001", description="001", article_id="0002")], index=["002"])


def make_attempt(tmp_path):
    return RunAttempt(
        job=GraphRunJob(),
        directory=tmp_path / "run",
        run_id="run",
        graph=graphs()[0],
        df=frame(),
        settings=settings(tmp_path),
    )


@pytest.mark.parametrize("had_error", [False, True])
def test_cancellation_preserves_cause_without_becoming_a_failure(tmp_path, had_error):
    attempt = make_attempt(tmp_path)
    attempt.observe(dict(kind="started", stage="rewrite", node_id="n2", news_id="news"))
    if had_error:
        attempt.observe(dict(kind="recovery_wait", error="Provider unavailable"))
    attempt.observe(
        dict(
            kind="finished",
            status="failed",
            exception_type="RecoveryCancelled",
            error="Execution cancelled while waiting for API quota.",
        )
    )
    attempt.finish("cancelled", error=RecoveryCancelled("Execution cancelled"))
    saved = load_attempt(attempt.path)
    assert saved["state"] == "cancelled"
    assert saved["had_error"] is had_error
    assert saved["interruption"] == ("cancelled_after_error" if had_error else "cancelled_by_user")
    assert len(saved["errors"]) == int(had_error)
    assert saved["location"]["node_id"] == "n2"


def test_recovery_restores_inputs_and_starts_a_fresh_linked_attempt(tmp_path):
    attempt = make_attempt(tmp_path)
    attempt.record["settings"]["provider_rpm_limits"] = {"gemini": 12, "chatgpt": 60}
    attempt.finish("failed", error=RuntimeError("Provider unavailable"))
    (attempt.path.parent / "call_checkpoint").mkdir()
    before = attempt.path.read_bytes()
    for mode in ("retry", "restart"):
        data, selected, configuration = recovery_inputs(load_attempt(attempt.path), mode)
        pd.testing.assert_frame_equal(data, frame())
        assert selected[0]["name"] == "Graph"
        assert configuration["reuse_previous_results"] is False
        assert configuration["provider_rpm_limits"] == {"gemini": 12, "chatgpt": 60}
        assert configuration["_parent_attempt"] == str(attempt.path.resolve())
        assert ("_recovery_checkpoint" in configuration) == (mode == "retry")
    assert attempt.path.read_bytes() == before


def test_snapshots_never_include_credentials_and_redact_failure_details(tmp_path):
    configuration = settings(tmp_path)
    configuration["topic_drift_api_key"] = "sk-test-secret-key"
    graph = graphs()[0]
    graph["nodes"][0]["api_key"] = "sk-test-secret-key"
    attempt = RunAttempt(
        job=GraphRunJob(),
        directory=tmp_path / "run",
        run_id="run",
        graph=graph,
        df=frame(),
        settings=configuration,
    )
    attempt.finish("failed", error="Failed using sk-test-secret-key")
    content = attempt.path.read_text("utf-8")
    assert "sk-test-secret-key" not in content
    assert "topic_drift_api_key" not in content
    records, errors = find_attempts(tmp_path)
    assert not errors and len(records) == 1


def test_completed_runs_and_missing_checkpoints_cannot_be_retried(tmp_path):
    attempt = make_attempt(tmp_path)
    attempt.finish("completed")
    with pytest.raises(ValueError, match="Only failed or cancelled"):
        recovery_inputs(load_attempt(attempt.path), "retry")
    attempt.finish("failed", error="error")
    with pytest.raises(ValueError, match="No call checkpoint"):
        recovery_inputs(load_attempt(attempt.path), "retry")


def test_history_reports_invalid_manifests_without_losing_valid_attempts(tmp_path):
    make_attempt(tmp_path)
    invalid = tmp_path / "invalid" / "run_attempt.json"
    invalid.parent.mkdir()
    invalid.write_text("[]", encoding="utf-8")
    records, errors = find_attempts(tmp_path)
    assert len(records) == len(errors) == 1
    assert "valid run attempt" in errors[0]


def test_only_retry_copies_prior_evaluations_into_its_own_directory(tmp_path):
    source = tmp_path / "original_cache"
    source.mkdir()
    (source / "result.json").write_text('{"output": "saved"}', encoding="utf-8")
    for mode in ("retry", "restart"):
        destination = tmp_path / mode
        destination.mkdir()
        restore_evaluation_cache(
            destination,
            dict(
                _recovery_evaluation_cache=str(source),
                _recovery_mode=mode,
            ),
        )
        assert (destination / "evaluation_cache" / "result.json").is_file() == (mode == "retry")
    assert (source / "result.json").read_text("utf-8") == '{"output": "saved"}'


def test_recovery_does_not_ignore_an_empty_fallback_override(monkeypatch):
    from misinformation_simulation.apps import interaction_graph_credentials as credentials

    monkeypatch.setattr(
        credentials,
        "st",
        SimpleNamespace(
            session_state={
                "_session_api_key_overrides": {"chatgpt_fallback": True},
            }
        ),
    )
    monkeypatch.setattr(credentials, "environment_api_key", lambda _: "configured-primary")
    monkeypatch.setattr(credentials, "environment_openai_fallback", lambda: "configured-secondary")
    supplied, missing = credentials.session_api_credentials(
        [dict(nodes=[dict(provider="chatgpt")])],
        dict(topic_drift_provider="local"),
    )
    assert not supplied
    assert missing == ["OpenAI / GPT fallback"]


def test_failed_evaluation_branch_is_not_reported_as_completed():
    step = SimulationStepResult(
        news_id="news",
        step_index=1,
        node_id="node",
        node_label="Node",
        source_node_id="original",
        source_node_label="Original",
        provider="gemini",
        model="test",
        personality="neutral",
        source_text="Original",
        rewritten_text="Rewritten",
        target_language="en",
        target_language_source="dataset",
        rewrite_status="success",
        rewrite_error=None,
        metadata={
            "dual_stdi_vs_original": {
                "status": "partial",
                "llm_judge": {"status": "failed", "error": "Unavailable"},
            }
        },
    )
    result = SimulationResult(dict(cancelled=False, steps_error=0), [step])
    assert result_state(result) == "failed"
    result.summary["cancelled"] = True
    assert result_state(result) == "cancelled"


def checkpoint_generator(responses, calls):
    @checkpoint_response
    def generate(client, *, model, prompt, system_instruction, temperature):
        calls.append(prompt)
        value = responses[prompt].pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    return lambda prompt: generate(
        object(),
        model="test",
        prompt=prompt,
        system_instruction="instruction",
        temperature=0,
    )


def session(tmp_path, decisions=None):
    choices = iter(decisions or ["cancel"])
    return RecoverySession(
        lambda _: RecoveryDecision(next(choices)),
        store=PermissionStore(tmp_path / "permissions.json"),
    )


def test_retry_replays_validated_responses_but_reissues_invalid_call(tmp_path):
    calls = []
    generate = checkpoint_generator({"first": ['{"ok": 1}'], "second": ["invalid"]}, calls)
    source = tmp_path / "source"
    with use_call_checkpoint(source), use_recovery(session(tmp_path)):
        assert generate_and_parse_with_retry(lambda: generate("first"), json.loads) == {"ok": 1}
        with pytest.raises(RecoveryCancelled):
            generate_and_parse_with_retry(lambda: generate("second"), json.loads)
    assert len(list(source.glob("*.json"))) == 1
    calls.clear()
    generate = checkpoint_generator({"second": ['{"ok": 2}']}, calls)
    with use_call_checkpoint(tmp_path / "retry", source), use_recovery(session(tmp_path)):
        assert generate_and_parse_with_retry(lambda: generate("first"), json.loads) == {"ok": 1}
        assert generate_and_parse_with_retry(lambda: generate("second"), json.loads) == {"ok": 2}
    assert calls == ["second"]
    assert len(list((tmp_path / "retry").glob("*.json"))) == 2
    calls.clear()
    generate = checkpoint_generator({"first": ['{"ok": 3}']}, calls)
    with use_call_checkpoint(tmp_path / "restart"), use_recovery(session(tmp_path)):
        assert generate_and_parse_with_retry(lambda: generate("first"), json.loads) == {"ok": 3}
    assert calls == ["first"]


def test_checkpoint_keeps_judge_repetitions_independent(tmp_path):
    calls = []
    generate = checkpoint_generator({"judge": ["first", "second"]}, calls)
    with use_call_checkpoint(tmp_path / "source"):
        for repeat in (1, 2):
            with operation("judge", repeat=repeat):
                assert generate("judge") == ("first" if repeat == 1 else "second")
    calls.clear()
    with use_call_checkpoint(tmp_path / "retry", tmp_path / "source"):
        for repeat in (1, 2):
            with operation("judge", repeat=repeat):
                assert generate("judge") == ("first" if repeat == 1 else "second")
    assert not calls


def test_rejected_cached_response_is_not_replayed_on_manual_retry(tmp_path):
    calls = []
    generate = checkpoint_generator({"value": ['"old"', '"corrected"']}, calls)
    with use_call_checkpoint(tmp_path / "source"):
        generate("value")

    def parse(raw):
        value = json.loads(raw)
        if value == "old":
            raise ValueError("New validation requirement")
        return value

    with use_call_checkpoint(tmp_path / "retry", tmp_path / "source"):
        with use_recovery(session(tmp_path, ["retry"])):
            assert generate_and_parse_with_retry(lambda: generate("value"), parse) == "corrected"
    assert calls == ["value", "value"]


def test_parallel_failure_and_pure_cancellation_are_separate(tmp_path):
    job = GraphRunJob()

    def runner(**kwargs):
        row = kwargs["df"].index[0]
        if row == 0:
            raise RuntimeError("Provider failed")
        if row == 1:
            return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=True), [])
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    run_parallel_queue(
        job=job,
        df=pd.concat([frame()] * 3, ignore_index=True),
        graphs=graphs(),
        settings=settings(tmp_path, rows=3, limit=2),
        runner=runner,
        queue_mode=False,
    )
    result = next(value for kind, value in job.events.queue if kind == "bundle")
    assert result["status"] == "failed"
    assert len(result["summary"]["run_failures"]) == 1
    assert len(result["summary"]["run_cancellations"]) == 1
    records, _ = find_attempts(tmp_path)
    assert sorted(item["state"] for item in records) == ["cancelled", "completed", "failed"]
    assert next(item for item in records if item["state"] == "cancelled")["had_error"] is False


def test_sequential_rerun_preserves_original_files_and_reuses_valid_calls(tmp_path):
    calls = []
    generate = checkpoint_generator({"first": ["first"], "second": ["second"]}, calls)
    job = GraphRunJob()

    def runner(**kwargs):
        assert generate("first") == "first"
        job.cancel_event.set()
        return SimulationResult(dict(rows_processed=0, steps_total=0, cancelled=True), [])

    _run_graph_queue(
        job=job,
        df=frame(),
        graphs=graphs(),
        settings=settings(tmp_path),
        runner=runner,
        queue_mode=False,
    )
    original = next((tmp_path / "batch").rglob("run_attempt.json"))
    original_bytes = original.read_bytes()
    data, selected, configuration = recovery_inputs(load_attempt(original), "retry")
    calls.clear()

    def recovered(**kwargs):
        assert generate("first") == "first"
        assert generate("second") == "second"
        assert kwargs["reuse_rewrites"] is True
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    retry_job = GraphRunJob()
    _run_graph_queue(
        job=retry_job,
        df=data,
        graphs=selected,
        settings=configuration,
        runner=recovered,
        queue_mode=False,
    )
    assert calls == ["second"]
    assert original.read_bytes() == original_bytes
    records, _ = find_attempts(tmp_path)
    assert len(records) == 2
    assert records[0]["parent_attempt"] == str(original.resolve())
    assert records[0]["state"] == "completed"


def test_recovered_news_fits_with_saved_sibling_corpus(tmp_path):
    structure = TopicStructure("news", ["health"], ["agency"], [])
    corpus = tmp_path / "corpus.json"
    save_corpus(
        corpus,
        [
            (1, [TopicStructurePair("old:original", structure, structure)]),
            (2, [TopicStructurePair("sibling:original", structure, structure)]),
        ],
    )
    assert len(load_corpus(corpus, exclude_row=1)) == 1
    fits = []

    def runner(**kwargs):
        return PreparedSimulation(
            [TopicStructurePair("new:original", structure, structure)],
            lambda _: SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), []),
            lambda pairs: fits.append([pair.pair_id for pair in pairs]) or object(),
        )

    configuration = settings(tmp_path, limit=2)
    configuration.update(_recovery_corpus=str(corpus), _source_dataset_row=1)
    run_parallel_queue(
        job=GraphRunJob(),
        df=frame(),
        graphs=graphs(),
        settings=configuration,
        runner=runner,
        queue_mode=False,
    )
    assert fits == [["new:original", "sibling:original"]]


@pytest.mark.parametrize("mode", ["retry", "restart"])
def test_recovery_ui_separates_causes_and_preserves_results(tmp_path, monkeypatch, mode):
    attempt = make_attempt(tmp_path)
    attempt.finish("cancelled")
    (attempt.path.parent / "call_checkpoint").mkdir()
    captured = []
    from misinformation_simulation.apps import interaction_graph_attempt_ui as ui_module

    monkeypatch.setattr(ui_module, "session_api_credentials", lambda *_: ({}, []))
    monkeypatch.setattr(
        ui_module,
        "start_graph_run_job",
        lambda **kwargs: captured.append(kwargs) or SimpleNamespace(),
    )
    app = AppTest.from_string(
        "from misinformation_simulation.apps import interaction_graph_attempt_ui as ui\n"
        "ui.render_run_history(lambda **_: None)"
    )
    app.session_state["run_attempts"] = {str(attempt.path): load_attempt(attempt.path)}
    app.session_state["run_bundles"] = [{"name": "Original result"}]
    app.session_state["run_job"] = None
    app.run(timeout=25)
    assert not app.exception
    assert [(item.label, item.value) for item in app.metric] == [
        ("Failed", "0"),
        ("Cancelled without error", "1"),
        ("Cancelled after error", "0"),
    ]
    assert not app.button(key="recover_retry").disabled
    app.button(key=f"recover_{mode}").click().run(timeout=25)
    assert not app.exception
    assert captured[0]["settings"]["_recovery_mode"] == mode
    assert app.session_state["run_bundles"] == [{"name": "Original result"}]
    assert app.button(key="recover_restart").disabled


def test_one_worker_still_tracks_each_news_run_independently(tmp_path):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["df"].index.tolist())
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    data = pd.concat([frame()] * 3, ignore_index=True)
    job = start_graph_run_job(
        df=data,
        graphs=graphs(),
        settings=settings(tmp_path, rows=3, limit=1),
        runner=runner,
        queue_mode=False,
    )
    job.thread.join(10)
    assert not job.thread.is_alive()
    assert calls == [[0], [1], [2]]
    records, errors = find_attempts(tmp_path)
    assert not errors
    assert len(records) == 3 and all(item["scope"] == "news" for item in records)
    assert all(item["state"] == "completed" for item in records)


def test_cancelling_one_sequential_graph_does_not_cancel_the_next(tmp_path):
    job = GraphRunJob()
    job.recovery.ask = lambda *_, **__: RecoveryDecision("cancel_run")
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["output_prefix"])
        if len(calls) == 1:

            def fail():
                raise RuntimeError("Provider unavailable")

            return get_recovery().run(fail)
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    _run_graph_queue(
        job=job,
        df=frame(),
        graphs=graphs() * 2,
        settings=settings(tmp_path),
        runner=runner,
        queue_mode=True,
    )
    assert len(calls) == 2 and not job.cancel_event.is_set()
    records, _ = find_attempts(tmp_path)
    assert records[0]["state"] == "completed"
    assert records[1]["state"] == "cancelled"
    assert records[1]["interruption"] == "cancelled_after_error"
    assert records[1]["errors"][0]["message"] == "Provider unavailable"
