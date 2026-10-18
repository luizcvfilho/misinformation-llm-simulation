"""Bounded news scheduling with graph-wide embedding fits and isolated run state."""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, RLock
from time import monotonic

from misinformation_simulation.apps.interaction_graph_attempts import (
    RunAttempt,
    load_corpus,
    restore_evaluation_cache,
    result_state,
    save_corpus,
)
from misinformation_simulation.apps.interaction_graph_queue import output_prefix_for_graph
from misinformation_simulation.apps.interaction_graph_run_job import _reserve_output_directory
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    steps_to_dataframe,
)
from misinformation_simulation.llm.checkpoint import use_call_checkpoint
from misinformation_simulation.llm.pricing import load_pricing
from misinformation_simulation.llm.progress import use_run_identity
from misinformation_simulation.llm.rate_limit import (
    SharedRequestLimiter,
    get_shared_limiter,
    request_limits_from_settings,
    use_shared_limiter,
)
from misinformation_simulation.llm.recovery import RecoverySession, safe_message, use_recovery
from misinformation_simulation.llm.usage import combine_usage
from misinformation_simulation.simulation.deferred import PreparedSimulation
from misinformation_simulation.simulation.io import graph_config_from_payload, resolve_project_path
from misinformation_simulation.simulation.paths import project_graph_paths
from misinformation_simulation.simulation.persistence import _persist_results
from misinformation_simulation.simulation.progress import (
    ProgressTracker,
    event_message,
    simulation_plan,
)
from misinformation_simulation.simulation.types import SimulationResult
from misinformation_simulation.text_metrics.vad import DEFAULT_VAD_MODEL_NAME


@dataclass
class NewsRun:
    graph_index: int
    row: int
    run_id: str
    directory: Path
    cancel_event: Event = field(default_factory=Event)
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    prepared: PreparedSimulation | None = None
    result: SimulationResult | None = None
    state: str = "queued"
    error: str | None = None
    checkpoint: SimulationResult | None = None
    attempt: RunAttempt | None = None


class LockedComparator:
    def __init__(self, comparator):
        self.comparator = comparator
        self.lock = RLock()

    def __getattr__(self, name):
        return getattr(self.comparator, name)

    def compare(self, *args, **kwargs):
        with self.lock:
            return self.comparator.compare(*args, **kwargs)

    def compare_structured(self, *args, **kwargs):
        with self.lock:
            return self.comparator.compare_structured(*args, **kwargs)


def run_parallel_queue(*, job, df, graphs, settings, runner, queue_mode):
    return _ParallelQueue(
        job=job, df=df, graphs=graphs, settings=settings, runner=runner, queue_mode=queue_mode
    ).run()


class _ParallelQueue:
    """Schedule news runs and coordinate each graph's shared embedding fit."""

    def __init__(self, *, job, df, graphs, settings, runner, queue_mode):
        self.job = job
        self.df = df
        self.graphs = graphs
        self.settings = settings
        self.runner = runner
        self.queue_mode = queue_mode

    def run(self):
        self.limit = max(1, int(self.settings["max_simultaneous_runs"]))
        self.rows = min(len(self.df), max(0, int(self.settings["max_rows"])))
        self.runs_dir = resolve_project_path(Path(self.settings["output_dir"]).expanduser())
        self.prefix = Path(self.settings["output_prefix"].strip()).name
        if not self.prefix or self.prefix in {".", ".."}:
            self.job.events.put(("progress", "Enter an output prefix before running."))
            self.job.finish(dict(completed=0, failed=len(self.graphs), cancelled=False))
            return
        self._prepare_batch()

        self.completed_graphs = self.failed_graphs = 0
        self.futures = {}
        try:
            with ThreadPoolExecutor(
                max_workers=self.limit, thread_name_prefix="news-run"
            ) as self.pool:
                while self.pending or self.futures:
                    self._cancel_pending()

                    self._dispatch_pending()

                    if not self.futures:
                        self._schedule_idle_fits()

                        if self.pending:
                            continue
                        break
                    self.done, _ = wait(self.futures, timeout=0.1, return_when=FIRST_COMPLETED)
                    self._collect_completed()

                    self._advance_graphs()

                    with self.job.progress_lock:
                        self.snapshot()
        finally:
            self._finish_batch()

    def log(self, task, message):
        label = (
            (
                f"[{task.run_id}] Graph {task.graph_index}/{len(self.graphs)} ("
                f"{self.graphs[task.graph_index - 1]['name']}) · news {task.row + 1}/"
                f"{self.rows}"
            )
            if task
            else "[batch]"
        )
        with self.log_lock:
            message = f"{label}: {message}"
            self.job.events.put(("progress", message))
            with (self.batch_dir / "execution.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def snapshot(self):
        snapshots = [task.tracker.snapshot() for task in self.tasks]
        resolved = sum(task.state in {"completed", "failed", "cancelled"} for task in self.tasks)
        operation_total = sum(item["total"] for item in snapshots)
        operation_done = sum(item["completed"] for item in snapshots)
        progress = dict(
            completed=resolved,
            total=len(self.tasks),
            graph_index=0,
            graph_total=len(self.graphs),
            row=0,
            row_total=self.rows,
            step=0,
            step_total=0,
            operation_total=operation_total,
            operation_completed=operation_done,
            parallel=True,
            max_simultaneous_runs=self.limit,
            runs=[
                dict(
                    run_id=task.run_id,
                    graph=self.graphs[task.graph_index - 1]["name"],
                    news_row=task.attempt.record["dataset_row"],
                    state=task.state,
                    current=item["current"],
                    had_error=task.attempt.record["had_error"],
                    location=task.attempt.record["location"],
                )
                for task, item in zip(self.tasks, snapshots, strict=True)
                if task.state != "queued"
            ],
        )
        waiting = bool(self.job.recovery.pending_requests)
        elapsed = max(0.001, monotonic() - self.job.started_at)
        throughput = operation_done / elapsed
        remaining = max(0, operation_total - operation_done)
        progress.update(
            awaiting_decisions=waiting,
            estimate_low=remaining / throughput * 0.65
            if operation_done >= 3 and (not waiting)
            else None,
            estimate_high=remaining / throughput * 1.6
            if operation_done >= 3 and (not waiting)
            else None,
        )
        self.job.latest_progress = progress
        return progress

    def observe(self, task, event):
        task.attempt.observe(event)
        with self.job.progress_lock:
            task.tracker.observe(1, event)
            if event["kind"] == "recovery_wait":
                task.state = "awaiting_decision"
            elif event["kind"] in {"recovery_retry", "error_permitted", "request_started"}:
                task.state = "evaluating" if task.prepared else "preparing"
            elif event["kind"] == "rate_limit_wait":
                task.state = "awaiting_quota"
            self.snapshot()
        if message := event_message(event):
            self.log(task, message)

    def prepare(self, task):
        graph = self.graphs[task.graph_index - 1]
        payload = build_editor_graph_payload(graph["nodes"])
        nodes, edges, start_node_id = graph_config_from_payload(payload)

        def cancelled():
            return self.job.cancel_event.is_set() or task.cancel_event.is_set()

        def decide(request):
            task.attempt.awaiting_recovery(request)
            decision = self.job.recovery.ask(request, cancel_check=cancelled)
            if decision.action == "cancel_run":
                task.cancel_event.set()
            return decision

        session = RecoverySession(
            decide, store=self.job.recovery_session.store, cancel_check=cancelled
        )
        reuse = bool(self.settings.get("reuse_previous_results", False))
        with (
            use_recovery(session),
            use_call_checkpoint(
                task.directory / "call_checkpoint",
                Path(self.settings["_recovery_checkpoint"])
                if self.settings.get("_recovery_checkpoint")
                else None,
            ),
            use_shared_limiter(self.limiter),
            use_run_identity(
                run_id=task.run_id,
                graph_index=task.graph_index,
                graph_name=graph["name"],
                dataset_row=task.row + 1,
            ),
        ):
            result = self.runner(
                df=self.df.iloc[[task.row]].copy(),
                nodes=nodes,
                edges=edges,
                start_node_id=start_node_id,
                text_column=self.settings["text_column"],
                title_column=self.settings["title_column"],
                news_id_column=self.settings["news_id_column"] or None,
                max_rows=1,
                sleep_seconds=float(self.settings["sleep_seconds"]),
                max_requests_per_minute=None,
                retry_attempts=int(self.settings["retry_attempts"]),
                allow_title_fallback=self.settings["allow_title_fallback"],
                rewrite_mode=self.settings.get("rewrite_mode", "faithful"),
                topic_drift_model=self.settings["topic_drift_model"],
                topic_drift_provider=self.settings["topic_drift_provider"],
                stdi_comparison_method=self.settings.get("stdi_comparison_method", "dual"),
                stdi_judge_repeats=int(self.settings.get("stdi_judge_repeats", 3)),
                vad_method=self.settings.get("vad_method", "model"),
                vad_llm_model=self.settings.get("vad_llm_model"),
                vad_llm_provider=self.settings.get("vad_llm_provider"),
                stdi_cache_dir=self.batch_dir / "evaluation_cache",
                shared_cache_dir=self.runs_dir / "shared_cache" if reuse else None,
                reuse_rewrites=reuse or self.settings.get("_recovery_mode") == "retry",
                api_pricing=self.pricing,
                openai_service_tier=self.settings.get("openai_service_tier"),
                openai_flex_fallback=self.settings.get("openai_flex_fallback", False),
                openai_flex_retry_seconds=self.settings.get("openai_flex_retry_seconds", 300),
                output_dir=task.directory,
                output_prefix=task.run_id,
                progress_callback=lambda message: self.log(task, message),
                operation_callback=lambda event: self.observe(task, event),
                cancel_check=cancelled,
                defer_evaluation=True,
            )
            if isinstance(result, PreparedSimulation) and result.checkpoint:
                task.checkpoint = result.context.copy().run(result.checkpoint)
            return result

    def fit_graph(self, index):
        prepared = [
            task.prepared for task in self.tasks if task.graph_index == index and task.prepared
        ]
        groups = [
            (task.attempt.record["dataset_row"], task.prepared.pairs)
            for task in self.tasks
            if task.graph_index == index and task.prepared
        ]
        if self.settings.get("_recovery_corpus"):
            groups += load_corpus(
                self.settings["_recovery_corpus"],
                exclude_row=self.settings["_source_dataset_row"],
            )
        groups.sort(key=lambda item: item[0])
        save_corpus(self.graph_dirs[index] / "recovery_corpus.json", groups)
        pairs = [pair for _, items in groups for pair in items]
        pairs = [pair for pair in pairs if pair.pair_id.endswith(":original")] + [
            pair for pair in pairs if pair.pair_id.endswith(":incremental")
        ]
        first = prepared[0]
        comparator = first.context.copy().run(first.build_comparator, pairs)
        return LockedComparator(comparator)

    def save_graph(self, index, final=False):
        group = [task for task in self.tasks if task.graph_index == index]
        results = [
            task.result or task.checkpoint
            for task in group
            if task.result is not None or task.checkpoint is not None
        ]
        failures = [
            dict(run_id=task.run_id, row=task.attempt.record["dataset_row"], error=task.error)
            for task in group
            if task.state == "failed"
        ]
        cancelled_runs = [
            dict(
                run_id=task.run_id,
                row=task.attempt.record["dataset_row"],
                interruption=task.attempt.record.get("interruption", "cancelled_by_user"),
            )
            for task in group
            if task.state == "cancelled"
        ]
        graph = self.graphs[index - 1]
        costs = [
            json.loads(path.read_text(encoding="utf-8"))
            for task in group
            if (path := (task.directory / f"{task.run_id}_api_costs.json")).is_file()
        ]
        if costs:
            (self.graph_dirs[index] / f"{self.graph_prefixes[index]}_api_costs.json").write_text(
                json.dumps(combine_usage(costs), indent=2, allow_nan=False), encoding="utf-8"
            )
        if not results:
            if final:
                self.job.events.put(
                    (
                        "bundle",
                        dict(
                            name=graph["name"],
                            execution_name=self.batch_dir.name,
                            status="failed" if failures else "cancelled",
                            error="; ".join(item["error"] for item in failures),
                            output_prefix=self.graph_prefixes[index],
                            attempts=[str(task.attempt.path) for task in group],
                        ),
                    )
                )
            return
        summary = _combine_run_summaries(results, self.rows)

        summary.update(
            graph_name=graph["name"],
            execution_name=self.batch_dir.name,
            max_simultaneous_runs=self.limit,
            runs_total=len(group),
            runs_completed=sum(task.state == "completed" for task in group),
            run_failures=failures,
            run_cancellations=cancelled_runs,
            attempts=[str(task.attempt.path) for task in group],
            partial=not final or bool(failures) or any(task.state == "cancelled" for task in group),
            cancelled=self.job.cancel_event.is_set()
            or any(task.state == "cancelled" for task in group),
        )
        summary["execution_status"] = (
            "failed" if failures else "cancelled" if summary["cancelled"] else "completed"
        )
        if costs:
            summary["api_usage"] = combine_usage(costs)
        result = SimulationResult(summary, [step for item in results for step in item.step_results])
        if summary.get("result_kind") == "branching_graph":
            result.path_results = project_graph_paths(result)
        result.summary_path, result.steps_path = _persist_results(
            result=result,
            output_dir=self.graph_dirs[index],
            output_prefix=self.graph_prefixes[index],
        )
        if final:
            steps_df = steps_to_dataframe(result.step_results)
            self.job.events.put(
                (
                    "bundle",
                    dict(
                        name=graph["name"],
                        execution_name=self.batch_dir.name,
                        status=summary["execution_status"],
                        error="; ".join(item["error"] for item in failures),
                        attempts=summary["attempts"],
                        summary=summary,
                        summary_path=str(result.summary_path),
                        steps_path=str(result.steps_path),
                        steps_df=steps_df,
                        node_summary_df=build_node_summary_dataframe(steps_df),
                        news_summary_df=build_news_summary_dataframe(steps_df),
                        output_prefix=self.graph_prefixes[index],
                        graph_payload=build_editor_graph_payload(graph["nodes"]),
                    ),
                )
            )

    def _prepare_batch(self):
        self.batch_dir, _ = _reserve_output_directory(self.runs_dir, self.prefix)
        restore_evaluation_cache(self.batch_dir, self.settings)
        self.log_lock = Lock()
        self.graph_dirs = {}
        self.graph_prefixes = {}
        self.tasks = []
        self.pending = deque()
        self.comparators = {}
        self.fitting = set()
        self.published = set()
        histories = sorted(
            self.runs_dir.glob("*/timings.json"), key=lambda path: path.stat().st_mtime
        )
        self.history = histories[-1] if histories else None
        self.pricing = load_pricing(
            resolve_project_path(self.settings["api_pricing_path"])
            if self.settings.get("api_pricing_path")
            else None
        )
        self.limiter = get_shared_limiter() or SharedRequestLimiter(
            request_limits_from_settings(self.settings),
            self.job.cancel_event.is_set,
        )
        self._create_tasks()

    def _create_tasks(self):
        for index, graph in enumerate(self.graphs, 1):
            self.index, graph = (index, graph)
            graph_prefix = output_prefix_for_graph(
                self.prefix, self.index, graph["name"], include_base=False
            )
            self.graph_prefixes[self.index] = graph_prefix
            self.graph_dirs[self.index] = self.batch_dir / graph_prefix
            self.graph_dirs[self.index].mkdir()
            for row in range(self.rows):
                run_id = f"g{self.index:03d}_news{row + 1:06d}"
                self.task = NewsRun(
                    self.index,
                    row,
                    run_id,
                    self.graph_dirs[self.index] / "news_runs" / run_id,
                )
                self.task.attempt = RunAttempt(
                    job=self.job,
                    directory=self.task.directory,
                    run_id=run_id,
                    graph=graph,
                    df=self.df.iloc[[row]],
                    settings=self.settings,
                    row=row + 1,
                )
                self.task.attempt.record["corpus_path"] = str(
                    self.graph_dirs[self.index] / "recovery_corpus.json"
                )
                self.task.attempt.record["evaluation_cache_path"] = str(
                    self.batch_dir / "evaluation_cache"
                )
                self.task.attempt.save()
                self.task.tracker = ProgressTracker(self.history)
                vad_method = self.settings.get("vad_method", "model")
                vad_llm_model = (
                    self.settings.get("vad_llm_model") or self.settings["topic_drift_model"]
                )
                self.task.tracker.plan(
                    1,
                    simulation_plan(
                        1,
                        build_simulation_nodes(graph["nodes"]),
                        method=self.settings.get("stdi_comparison_method", "dual"),
                        repeats=int(self.settings.get("stdi_judge_repeats", 3)),
                        model=self.settings["topic_drift_model"],
                        provider=self.settings["topic_drift_provider"],
                        vad_model=DEFAULT_VAD_MODEL_NAME
                        if vad_method == "model"
                        else (f"{vad_method}:{vad_llm_model}"),
                        vad_provider="local"
                        if vad_method == "model"
                        else self.settings.get("vad_llm_provider")
                        or self.settings["topic_drift_provider"],
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                        sleep_seconds=float(self.settings["sleep_seconds"]),
                    ),
                )
                self.tasks.append(self.task)
                self.pending.append((self.task, "prepare"))
                self.job.run_cancellations[run_id] = self.task.cancel_event

    def _cancel_pending(self):
        if self.job.cancel_event.is_set():
            for task, phase in list(self.pending):
                self.task, self.phase = (task, phase)
                if self.phase == "prepare":
                    self.task.state = "cancelled"
                    self.task.attempt.finish("cancelled")
                    self.pending.remove((self.task, self.phase))

    def _dispatch_pending(self):
        while self.pending and len(self.futures) < self.limit:
            self.task, self.phase = self.pending.popleft()
            if self.phase == "prepare" and (
                self.job.cancel_event.is_set() or self.task.cancel_event.is_set()
            ):
                self.task.state = "cancelled"
                self.task.attempt.finish("cancelled")
                self.task.tracker.finish_graph(1)
                continue
            self.task.state = {
                "prepare": "preparing",
                "fit": "fitting_graph",
                "evaluate": "evaluating",
            }[self.phase]
            self.task.attempt.record["state"] = self.task.state
            self.task.attempt.save()
            function = (
                (lambda task=self.task: self.prepare(task))
                if self.phase == "prepare"
                else (lambda task=self.task: self.fit_graph(task.graph_index))
                if self.phase == "fit"
                else lambda task=self.task: task.prepared.resume(self.comparators[task.graph_index])
            )
            self.futures[self.pool.submit(copy_context().run, function)] = (
                self.task,
                self.phase,
            )
            self.log(self.task, self.task.state)

    def _schedule_idle_fits(self):
        for index in range(1, len(self.graphs) + 1):
            self.index = index
            self.prepared = next(
                (
                    task
                    for task in self.tasks
                    if task.graph_index == self.index and task.state == "prepared"
                ),
                None,
            )
            if self.prepared and self.index not in self.fitting:
                self.fitting.add(self.index)
                self.pending.appendleft((self.prepared, "fit"))

    def _collect_completed(self):
        for future in self.done:
            self.task, self.phase = self.futures.pop(future)
            try:
                value = future.result()
                if self.phase == "fit":
                    self.comparators[self.task.graph_index] = value
                    self.task.state = "prepared"
                    ready = [
                        item
                        for item in self.tasks
                        if item.graph_index == self.task.graph_index and item.state == "prepared"
                    ]
                    self.pending.extendleft((item, "evaluate") for item in reversed(ready))
                elif isinstance(value, PreparedSimulation):
                    self.task.prepared, self.task.state = (value, "prepared")
                else:
                    self.task.result = value
                    self.task.state = result_state(value)
                    self._finish_task(self.task, value)
                    self.task.tracker.finish_graph(1)
                    for step in value.step_results:
                        step.metadata.update(
                            run_id=self.task.run_id,
                            dataset_row=self.task.attempt.record["dataset_row"],
                        )
                        step.metadata["graph_step_id"] = f"{self.task.run_id}:{step.node_id}"
                    self.task.prepared = None
                    self.task.checkpoint = None
            except Exception as error:
                if self.phase == "fit":
                    self._fail_prepared_graph(error)
                elif self.phase == "evaluate":
                    self.task.prepared = None
                self.task.error = safe_message(error)
                self.task.state = (
                    "cancelled"
                    if self.task.cancel_event.is_set() or self.job.cancel_event.is_set()
                    else "failed"
                )
                self.task.attempt.finish(self.task.state, error=error)
                self.log(self.task, f"{self.task.state}: {error}")
                self.task.tracker.finish_graph(1)

    def _finish_task(self, task, result):
        if task.state == "failed":
            task.error = (
                task.attempt.record["errors"][-1]["message"]
                if task.attempt.record["errors"]
                else "One or more simulation operations failed."
            )
        result.summary["execution_status"] = task.state
        result.summary["attempt_path"] = str(task.attempt.path)
        if result.summary_path is not None:
            result.summary_path.write_text(
                json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        task.attempt.finish(task.state, result=result, error=task.error)

    def _fail_prepared_graph(self, error):
        for item in self.tasks:
            if item.graph_index == self.task.graph_index and item.prepared is not None:
                item.error = safe_message(error)
                item.state = (
                    "cancelled"
                    if item.cancel_event.is_set() or self.job.cancel_event.is_set()
                    else "failed"
                )
                item.attempt.finish(item.state, error=error)
                item.prepared.finalize(None)
                item.prepared = None
                item.tracker.finish_graph(1)

    def _advance_graphs(self):
        for index in range(1, len(self.graphs) + 1):
            self.index = index
            group = [task for task in self.tasks if task.graph_index == self.index]
            if (
                self.index not in self.fitting
                and group
                and all(
                    task.state
                    not in {
                        "queued",
                        "preparing",
                        "awaiting_quota",
                        "awaiting_decision",
                    }
                    for task in group
                )
            ):
                self.prepared = next((task for task in group if task.state == "prepared"), None)
                if self.prepared:
                    self.fitting.add(self.index)
                    self.pending.appendleft((self.prepared, "fit"))
            if self.index not in self.published and all(
                task.state in {"completed", "failed", "cancelled"} for task in group
            ):
                self.save_graph(self.index, final=True)
                self.published.add(self.index)
                self.comparators.pop(self.index, None)
                if any(task.state == "failed" for task in group):
                    self.failed_graphs += 1
                elif not any(task.state == "cancelled" for task in group):
                    self.completed_graphs += 1

    def _finish_batch(self):
        for index in range(1, len(self.graphs) + 1):
            self.index = index
            if self.index not in self.published:
                self.save_graph(self.index, final=True)
            corpus = self.graph_dirs[index] / "recovery_corpus.json"
            if not corpus.is_file():
                save_corpus(corpus, [])
        samples: dict[str, list] = {}
        for task in self.tasks:
            self.task = task
            for key, values in self.task.tracker.samples.items():
                samples.setdefault(key, []).extend(values)
        (self.batch_dir / "timings.json").write_text(
            json.dumps(
                dict(
                    version=1,
                    samples={key: values[-60:] for key, values in samples.items()},
                )
            ),
            encoding="utf-8",
        )
        costs = [
            json.loads(path.read_text(encoding="utf-8"))
            for task in self.tasks
            if (path := (task.directory / f"{task.run_id}_api_costs.json")).is_file()
        ]
        if costs:
            (self.batch_dir / "api_costs.json").write_text(
                json.dumps(combine_usage(costs), indent=2, allow_nan=False),
                encoding="utf-8",
            )
        self.job.finish(
            dict(
                completed=self.completed_graphs,
                failed=self.failed_graphs,
                cancelled=self.job.cancel_event.is_set(),
                completed_runs=sum(task.state == "completed" for task in self.tasks),
                failed_runs=sum(task.state == "failed" for task in self.tasks),
                cancelled_runs=sum(task.state == "cancelled" for task in self.tasks),
            )
        )


def _combine_run_summaries(results, rows):
    summary = dict(results[0].summary)
    for key in (
        "rows_processed",
        "steps_total",
        "steps_success",
        "steps_error",
        "steps_blocked",
        "rewrite_operations_started",
        "planned_unique_rewrites",
        "planned_linear_rewrites",
        "planned_rewrites_saved",
        "stdi_complete_incremental_pairs",
    ):
        if isinstance(summary.get(key), (int, float)):
            summary[key] = sum(result.summary.get(key) or 0 for result in results)
    for key in ("planned_unique_rewrites", "planned_linear_rewrites", "planned_rewrites_saved"):
        if isinstance(results[0].summary.get(key), (int, float)):
            summary[key] = results[0].summary[key] * rows
    return summary
