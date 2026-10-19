from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from threading import Barrier, Event, Lock, Thread
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from misinformation_simulation.apps.interaction_graph_parallel import run_parallel_queue
from misinformation_simulation.apps.interaction_graph_recovery import RecoveryBroker
from misinformation_simulation.apps.interaction_graph_run_job import GraphRunJob
from misinformation_simulation.apps.interaction_graph_ui import create_default_node_form
from misinformation_simulation.llm.flex_fallback import get_flex_fallback
from misinformation_simulation.llm.progress import progress_context
from misinformation_simulation.llm.rate_limit import (
    SharedRequestLimiter,
    get_shared_limiter,
    use_shared_limiter,
)
from misinformation_simulation.llm.recovery import (
    PermissionStore,
    RecoveryCancelled,
    RecoveryDecision,
    RecoveryRequest,
    RecoverySession,
    RequestContext,
)
from misinformation_simulation.llm.usage import api_request
from misinformation_simulation.simulation import graph
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.types import SimulationResult
from misinformation_simulation.text_metrics.vad import VADScore
from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure
from misinformation_simulation.topic_drift.provenance import EvaluationCache


def settings(tmp_path, *, rows=2, limit=3):
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
        reuse_previous_results=False,
    )


def graphs(count=2):
    return [
        dict(name=f"graph {index}", nodes=[create_default_node_form(1)]) for index in range(count)
    ]


def frame(rows=2):
    return pd.DataFrame([dict(title=f"Headline {i}", description=f"News {i}") for i in range(rows)])


def test_background_job_shares_one_fallback_controller_across_news_runs(tmp_path):
    from misinformation_simulation.apps.interaction_graph_run_job import start_graph_run_job

    controllers = []
    barrier = Barrier(3)

    def runner(**kwargs):
        controllers.append(get_flex_fallback())
        assert kwargs["openai_flex_fallback"] is True
        assert kwargs["openai_flex_retry_seconds"] == 120
        barrier.wait(timeout=5)
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    configuration = settings(tmp_path, rows=3)
    configuration.update(
        openai_service_tier="flex", openai_flex_fallback=True, openai_flex_retry_seconds=120
    )
    job = start_graph_run_job(
        df=frame(3), graphs=graphs(1), settings=configuration, runner=runner, queue_mode=False
    )
    job.thread.join(timeout=10)
    assert not job.thread.is_alive()
    assert len(controllers) == 3
    assert all(controller is job.flex_fallback for controller in controllers)
    assert job.flex_fallback.retry_seconds == 120
    assert get_flex_fallback() is None


def test_global_cap_prefers_same_graph_and_uses_spare_slots(tmp_path):
    job = GraphRunJob()
    release = Event()
    full = Event()
    lock = Lock()
    calls = []
    active = peak = 0

    def runner(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append((progress_context()["graph_index"], kwargs["df"].index[0]))
            if len(calls) == 3:
                full.set()
        assert release.wait(5)
        with lock:
            active -= 1
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    thread = Thread(
        target=run_parallel_queue,
        kwargs=dict(
            job=job,
            df=frame(),
            graphs=graphs(),
            settings=settings(tmp_path),
            runner=runner,
            queue_mode=True,
        ),
    )
    thread.start()
    try:
        assert full.wait(5)
        assert set(calls) == {(1, 0), (1, 1), (2, 0)}
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert peak == 3
    bundles = [value for kind, value in job.events.queue if kind == "bundle"]
    assert len(bundles) == 2
    assert all(bundle["summary"]["rows_processed"] == 2 for bundle in bundles)
    assert job.events.queue[-1][1]["completed"] == 2
    assert "[g001_news000001]" in (tmp_path / "batch" / "execution.log").read_text("utf-8")


def test_deferred_evaluations_fit_the_whole_graph_once_without_deadlock(tmp_path):
    fits, evaluations = [], []
    lock = Lock()

    def runner(**kwargs):
        row = int(kwargs["df"].index[0])
        pairs = [
            SimpleNamespace(pair_id=f"{row}:original"),
            SimpleNamespace(pair_id=f"{row}:incremental"),
        ]

        def fit(pairs):
            fits.append([pair.pair_id for pair in pairs])
            return object()

        def finish(comparator):
            assert comparator is not None
            with lock:
                evaluations.append(row)
            return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

        return PreparedSimulation(pairs, finish, fit)

    job = GraphRunJob()
    run_parallel_queue(
        job=job,
        df=frame(7),
        graphs=graphs(1),
        settings=settings(tmp_path, rows=7, limit=2),
        runner=runner,
        queue_mode=False,
    )
    assert fits == [[f"{i}:original" for i in range(7)] + [f"{i}:incremental" for i in range(7)]]
    assert sorted(evaluations) == list(range(7))
    assert job.events.queue[-1][1]["completed"] == 1


@pytest.mark.parametrize("limits", [6000, {"gemini": 6000, "chatgpt": 3000}])
def test_rpm_is_shared_by_actual_attempts_and_context_propagates(limits):
    limiter = SharedRequestLimiter(limits)
    barrier = Barrier(3)
    admitted = []
    lock = Lock()

    def worker():
        barrier.wait(timeout=3)
        for _ in range(2):
            with api_request(provider="gemini", model="test"):
                with lock:
                    admitted.append(monotonic())

    with use_shared_limiter(limiter), ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(copy_context().run, worker) for _ in range(3)]
        for future in futures:
            future.result(timeout=5)
    gaps = [right - left for left, right in zip(admitted, admitted[1:], strict=False)]
    assert len(admitted) == 6
    assert all(gap >= 0.008 for gap in gaps)


@pytest.mark.parametrize("limits", [{"gemini": 1}, {"gemini": 1, "chatgpt": 60}])
def test_other_provider_bypasses_pending_rpm_wait_without_releasing_limited_provider(
    monkeypatch, limits
):
    from misinformation_simulation.llm import rate_limit

    waiting = Event()
    stop = Event()
    limiter = SharedRequestLimiter(limits, stop.is_set)
    monkeypatch.setattr(rate_limit, "emit_event", lambda *_args, **_kwargs: waiting.set())
    limiter.acquire("gemini")
    with ThreadPoolExecutor(max_workers=2) as pool:
        gemini = pool.submit(limiter.acquire, "gemini")
        try:
            assert waiting.wait(2)
            gpt = pool.submit(limiter.acquire, "chatgpt")
            assert gpt.result(timeout=2) is None
            assert not gemini.done()
        finally:
            stop.set()
        with pytest.raises(RecoveryCancelled):
            gemini.result(timeout=2)
    assert not limiter.queue


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
def test_execution_retains_provider_rpm_limits(tmp_path, background, workers):
    from misinformation_simulation.apps.interaction_graph_run_job import start_graph_run_job

    limiters = []

    def runner(**kwargs):
        limiter = get_shared_limiter()
        assert limiter.intervals == {"gemini": 0.01, "chatgpt": 0.02}
        with api_request(provider="gemini", model="rewrite"):
            pass
        next_request = limiter.next_requests["gemini"]
        with api_request(provider="chatgpt", model="evaluate"):
            pass
        assert limiter.next_requests["gemini"] == next_request
        assert "chatgpt" in limiter.next_requests
        limiters.append(limiter)
        return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=False), [])

    configuration = settings(tmp_path, rows=1, limit=workers)
    configuration.pop("max_requests_per_minute")
    configuration["provider_rpm_limits"] = {"gemini": 6000, "chatgpt": 3000}
    kwargs = dict(
        df=frame(1), graphs=graphs(1), settings=configuration, runner=runner, queue_mode=False
    )
    if background:
        job = start_graph_run_job(**kwargs)
        job.thread.join(timeout=10)
        assert not job.thread.is_alive()
    else:
        job = GraphRunJob()
        run_parallel_queue(job=job, **kwargs)
    assert len(limiters) == 1
    assert job.events.queue[-1][1]["completed"] == 1
    assert get_shared_limiter() is None


def test_failed_graph_fit_finalizes_every_prepared_run(tmp_path):
    finalized = []

    def runner(**kwargs):
        row = int(kwargs["df"].index[0])

        def broken_fit(pairs):
            raise RuntimeError("Embedding model unavailable")

        return PreparedSimulation(
            [], lambda _: None, broken_fit, finalize=lambda _: finalized.append(row)
        )

    job = GraphRunJob()
    run_parallel_queue(
        job=job,
        df=frame(3),
        graphs=graphs(1),
        settings=settings(tmp_path, rows=3, limit=2),
        runner=runner,
        queue_mode=False,
    )
    assert sorted(finalized) == [0, 1, 2]
    assert job.events.queue[-1][1]["failed"] == 1


def test_quota_wait_is_cancellable_and_cleans_fifo_queue():
    stop = Event()
    limiter = SharedRequestLimiter(1, stop.is_set)
    limiter.acquire("gemini")
    errors = []

    def worker():
        try:
            limiter.acquire("gemini")
        except RecoveryCancelled as error:
            errors.append(error)

    thread = Thread(target=worker)
    thread.start()
    sleep(0.05)
    stop.set()
    thread.join(2)
    assert not thread.is_alive() and len(errors) == 1
    assert not limiter.queue


def test_simultaneous_errors_stay_separate_and_equivalent_allow_wakes_all():
    broker = RecoveryBroker(Event())
    requests = [
        RecoveryRequest(
            str(i), RequestContext(), f"error {i}", ("same",) if i < 2 else ("different",)
        )
        for i in range(3)
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(broker.ask, request) for request in requests]
        deadline = monotonic() + 3
        while len(broker.pending_requests) < 3 and monotonic() < deadline:
            sleep(0.01)
        try:
            assert len(broker.pending_requests) == 3
            assert broker.submit("0", RecoveryDecision("allow"))
            assert futures[0].result(timeout=2).action == "allow"
            assert futures[1].result(timeout=2).action == "allow"
            assert [request.id for request in broker.pending_requests] == ["2"]
            assert broker.submit("2", RecoveryDecision("cancel_run"))
            assert futures[2].result(timeout=2).action == "cancel_run"
            assert not broker.cancel_event.is_set()
        finally:
            broker.cancel_event.set()


def test_skip_occurrence_does_not_release_other_matching_failures():
    broker = RecoveryBroker(Event())
    requests = [
        RecoveryRequest(str(i), RequestContext(), "same failure", ("same",)) for i in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(broker.ask, request) for request in requests]
        deadline = monotonic() + 3
        while len(broker.pending_requests) < 2 and monotonic() < deadline:
            sleep(0.01)
        try:
            assert len(broker.pending_requests) == 2
            assert broker.submit("0", RecoveryDecision("skip"))
            assert futures[0].result(timeout=2).action == "skip"
            assert [request.id for request in broker.pending_requests] == ["1"]
            assert not broker._allowed
            assert not broker.cancel_event.is_set()
            assert broker.submit("1", RecoveryDecision("skip"))
            assert futures[1].result(timeout=2).action == "skip"
        finally:
            broker.cancel_event.set()


def test_concurrent_cache_misses_keep_both_outputs_and_first_publication(tmp_path):
    first, second = EvaluationCache(tmp_path), EvaluationCache(tmp_path)
    inputs = {"text": "same news"}
    assert first.get("rewrite", inputs) is second.get("rewrite", inputs) is None
    # Compatibility contracts cause reusable entries to carry their output hash.
    first.compatibility = second.compatibility = {"rewrite": {"version": 1}}
    first.put("rewrite", inputs, {"text": "first output"})
    second.put("rewrite", inputs, {"text": "second output"})
    assert first.get("rewrite", inputs)["text"] == "first output"
    assert second.get("rewrite", inputs)["text"] == "second output"
    later = EvaluationCache(tmp_path, compatibility=first.compatibility)
    assert later.get("rewrite", inputs)["text"] == "first output"


def test_temporary_api_errors_use_bounded_shared_cooldown_before_prompt(tmp_path):
    limiter = SharedRequestLimiter(None)
    waits, prompts = [], []
    limiter.cooldown = waits.append

    def decide(request):
        prompts.append(request)
        return RecoveryDecision("allow")

    session = RecoverySession(decide, store=PermissionStore(tmp_path / "permissions.json"))
    calls = []

    class RateError(Exception):
        status_code = 429

    def request():
        calls.append(1)
        raise RateError("Temporary rate limit")

    from misinformation_simulation.llm.recovery import ErrorPermitted

    with use_shared_limiter(limiter), pytest.raises(ErrorPermitted):
        session.run(request, context=RequestContext("gemini", "test"))
    assert len(calls) == 5
    assert waits == [2, 4, 8, 16]
    assert len(prompts) == 1


def test_cancelled_batch_still_finishes_prepared_runs_and_saves_costs(tmp_path):
    job = GraphRunJob()
    resumed = []
    preparing = Barrier(2)

    def runner(**kwargs):
        row = kwargs["df"].index[0]
        if row == 1:
            job.cancel_event.set()
        preparing.wait(timeout=3)

        def finish(comparator):
            resumed.append(row)
            return SimulationResult(dict(rows_processed=1, steps_total=0, cancelled=True), [])

        return PreparedSimulation([], finish, lambda _: object())

    run_parallel_queue(
        job=job,
        df=frame(5),
        graphs=graphs(1),
        settings=settings(tmp_path, rows=5, limit=2),
        runner=runner,
        queue_mode=False,
    )
    assert resumed
    assert set(resumed) <= {0, 1}
    assert job.events.queue[-1][1]["cancelled"] is True


def test_ui_exposes_and_retains_simultaneous_run_setting(tmp_path, monkeypatch):
    from streamlit.elements.lib import policies
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    warnings = []
    monkeypatch.setattr(policies._LOGGER, "warning", lambda *args, **_: warnings.append(args))
    path = tmp_path / "settings_app.py"
    path.write_text(
        """
import streamlit as st
from misinformation_simulation.apps.interaction_graph_sections import _render_advanced_settings
from misinformation_simulation.apps.interaction_graph_workspace import retain_workspace_widgets
retain_workspace_widgets()
st.session_state.settings = _render_advanced_settings()
""",
        encoding="utf-8",
    )
    ui = AppTest.from_file(str(path)).run(timeout=25)
    assert not ui.exception
    control = ui.number_input(key="simulation_advanced_settings_max_simultaneous_runs")
    assert control.value == 1
    control.set_value(5).run()
    assert not ui.exception
    assert ui.session_state.settings["max_simultaneous_runs"] == 5
    providers = ui.multiselect(key="simulation_advanced_settings_rpm_providers")
    assert "local" not in providers.options
    assert ui.session_state.settings["provider_rpm_limits"] == {}
    providers.set_value(["gemini", "chatgpt"]).run()
    ui.number_input(key="simulation_advanced_settings_provider_rpm_gemini").set_value(12)
    ui.number_input(key="simulation_advanced_settings_provider_rpm_chatgpt").set_value(60).run()
    assert not ui.exception
    assert ui.session_state.settings["provider_rpm_limits"] == {"gemini": 12, "chatgpt": 60}
    ui.multiselect(key="simulation_advanced_settings_rpm_providers").set_value(["chatgpt"]).run()
    assert not ui.exception
    assert ui.session_state.settings["provider_rpm_limits"] == {"chatgpt": 60}
    ui.multiselect(key="simulation_advanced_settings_rpm_providers").set_value(
        ["gemini", "chatgpt", "grok", "deepseek", "openrouter"]
    ).run()
    assert not ui.exception
    assert ui.session_state.settings["provider_rpm_limits"] == {
        "gemini": 12,
        "chatgpt": 60,
        "grok": 0,
        "deepseek": 0,
        "openrouter": 0,
    }
    ui.multiselect(key="simulation_advanced_settings_rpm_providers").set_value([]).run()
    assert not ui.exception
    assert ui.session_state.settings["provider_rpm_limits"] == {}
    assert warnings == []


def test_ui_restores_previous_single_provider_rpm_configuration():
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_string(
        "import streamlit as st\n"
        "from misinformation_simulation.apps.interaction_graph_sections import "
        "_render_provider_rpm_limits\n"
        "st.session_state.limits = _render_provider_rpm_limits()\n"
    )
    ui.session_state["simulation_advanced_settings_rpm_provider"] = "grok"
    ui.session_state["simulation_advanced_settings_max_requests_per_minute"] = 25
    ui.run(timeout=25)
    assert not ui.exception
    assert ui.session_state.limits == {"grok": 25}


class ConstantEmbedder:
    def encode(self, texts):
        return np.tile([1.0, 0.0], (len(texts), 1))


@pytest.mark.parametrize("method", ["cluster", "llm", "dual"])
def test_real_parallel_engine_preserves_graph_metrics_and_duplicate_occurrences(
    tmp_path, monkeypatch, method
):
    monkeypatch.setattr(graph, "create_llm_client", lambda **_: ("gemini", object()))
    monkeypatch.setattr(graph, "_generate_rewrite", lambda **_: "Rewritten news")
    monkeypatch.setattr(
        graph,
        "extract_topic_structure",
        lambda **_: TopicStructure(
            main_topic="news",
            subtopics=["health"],
            central_entities=["agency"],
            central_relations=[TopicRelation("agency", "reports", "news")],
            narrative_frame="neutral",
            schema_version=2,
            extraction_status="valid",
        ),
    )
    fits = []
    base_fit = graph.StructuredEmbeddingComparator.fit

    def fit(self, pairs):
        fits.append(len(pairs))
        return base_fit(self, pairs)

    monkeypatch.setattr(graph.StructuredEmbeddingComparator, "fit", fit)
    from misinformation_simulation.topic_drift.semantic_comparison import SemanticSTDIComparison

    def judge(**kwargs):
        components = (
            "theme_drift",
            "subtopic_drift",
            "entity_drift",
            "relation_drift",
            "contradiction_drift",
        )
        return SemanticSTDIComparison(
            dict.fromkeys(components, 0.0), dict.fromkeys(components, "Unchanged")
        )

    runner = partial(
        graph.run_news_interaction_graph,
        stdi_embedder=ConstantEmbedder(),
        vad_scorer=lambda _: VADScore(3, 3, 3),
        stdi_judge_fn=judge,
    )
    job = GraphRunJob()
    data = frame()
    data.index = ["duplicate", "duplicate"]
    options = settings(tmp_path, limit=2)
    options["stdi_comparison_method"] = method
    run_parallel_queue(
        job=job,
        df=data,
        graphs=graphs(1),
        settings=options,
        runner=runner,
        queue_mode=False,
    )
    bundles = [value for kind, value in job.events.queue if kind == "bundle"]
    assert len(bundles) == 1 and bundles[0]["status"] == "completed"
    bundle = bundles[0]
    assert bundle["summary"]["rows_processed"] == 2
    assert fits == ([] if method == "llm" else [4])
    assert len(bundle["steps_df"]) == 2
    assert bundle["steps_df"]["metadata_run_id"].nunique() == 2
    assert bundle["steps_df"]["metadata_graph_step_id"].nunique() == 2
    assert bundle["steps_df"]["stdi_vs_original"].tolist() == pytest.approx([0, 0])
    assert len(list((tmp_path / "batch").glob("*/news_runs/*/*_steps.jsonl"))) == 2
    from misinformation_simulation.analysis.interaction_graph_visualization import (
        discover_step_paths,
    )
    from misinformation_simulation.apps.interaction_graph_results import find_saved_results

    assert len(discover_step_paths(tmp_path / "batch")) == 1
    assert len(find_saved_results(tmp_path / "batch")) == 1
    if method == "dual":
        from streamlit.testing.v1 import AppTest

        ui = AppTest.from_string(
            "from misinformation_simulation.apps.interaction_graph_sections import "
            "render_results_tab\nrender_results_tab()"
        )
        ui.session_state.run_bundles = [bundle]
        ui.run(timeout=30)
        assert not ui.exception


def test_cancelled_graph_without_results_can_be_opened_in_ui():
    from streamlit.testing.v1 import AppTest

    ui = AppTest.from_string(
        "from misinformation_simulation.apps.interaction_graph_sections import "
        "render_results_tab\nrender_results_tab()"
    )
    ui.session_state.run_bundles = [
        dict(name="Graph", execution_name="batch", output_prefix="01_graph", status="cancelled")
    ]
    ui.run(timeout=30)
    assert not ui.exception


def test_unexpected_worker_setup_error_finishes_monitor(monkeypatch):
    from misinformation_simulation.apps import interaction_graph_run_job as run_job

    def broken(**kwargs):
        raise OSError("Cannot reserve output directory")

    monkeypatch.setattr(run_job, "_run_graph_queue", broken)
    job = run_job.start_graph_run_job(
        df=frame(), graphs=graphs(1), settings={}, runner=lambda **_: None, queue_mode=False
    )
    job.thread.join(3)
    assert not job.thread.is_alive()
    assert job.finished
    assert [payload for kind, payload in job.events.queue if kind == "done"] == [
        dict(completed=0, failed=1, cancelled=False)
    ]
