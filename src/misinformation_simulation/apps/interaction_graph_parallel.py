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

from misinformation_simulation.apps.interaction_graph_queue import output_prefix_for_graph
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    steps_to_dataframe,
)
from misinformation_simulation.llm.pricing import load_pricing
from misinformation_simulation.llm.progress import use_run_identity
from misinformation_simulation.llm.rate_limit import (
    SharedRequestLimiter,
    get_shared_limiter,
    use_shared_limiter,
)
from misinformation_simulation.llm.recovery import RecoverySession, use_recovery
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
    from misinformation_simulation.apps.interaction_graph_run_job import _reserve_output_directory

    limit = max(1, int(settings["max_simultaneous_runs"]))
    rows = min(len(df), max(0, int(settings["max_rows"])))
    runs_dir = resolve_project_path(Path(settings["output_dir"]).expanduser())
    prefix = Path(settings["output_prefix"].strip()).name
    if not prefix or prefix in {".", ".."}:
        job.events.put(("progress", "Enter an output prefix before running."))
        job.finish(dict(completed=0, failed=len(graphs), cancelled=False))
        return
    batch_dir, _ = _reserve_output_directory(runs_dir, prefix)
    log_lock = Lock()
    graph_dirs = {}
    graph_prefixes = {}
    tasks = []
    pending = deque()
    comparators = {}
    fitting = set()
    published = set()
    histories = sorted(runs_dir.glob("*/timings.json"), key=lambda path: path.stat().st_mtime)
    history = histories[-1] if histories else None
    pricing = load_pricing(
        resolve_project_path(settings["api_pricing_path"])
        if settings.get("api_pricing_path")
        else None
    )
    limiter = get_shared_limiter() or SharedRequestLimiter(
        int(settings.get("max_requests_per_minute", 0)) or None, job.cancel_event.is_set
    )
    for index, graph in enumerate(graphs, 1):
        graph_prefix = output_prefix_for_graph(prefix, index, graph["name"], include_base=False)
        graph_prefixes[index] = graph_prefix
        graph_dirs[index] = batch_dir / graph_prefix
        graph_dirs[index].mkdir()
        for row in range(rows):
            run_id = f"g{index:03d}_news{row + 1:06d}"
            task = NewsRun(index, row, run_id, graph_dirs[index] / "news_runs" / run_id)
            task.tracker = ProgressTracker(history)
            vad_method = settings.get("vad_method", "model")
            task.tracker.plan(
                1,
                simulation_plan(
                    1,
                    build_simulation_nodes(graph["nodes"]),
                    method=settings.get("stdi_comparison_method", "dual"),
                    repeats=int(settings.get("stdi_judge_repeats", 3)),
                    model=settings["topic_drift_model"],
                    provider=settings["topic_drift_provider"],
                    vad_model=(
                        DEFAULT_VAD_MODEL_NAME
                        if vad_method == "model"
                        else f"{vad_method}:"
                        f"{settings.get('vad_llm_model') or settings['topic_drift_model']}"
                    ),
                    vad_provider="local"
                    if vad_method == "model"
                    else (settings.get("vad_llm_provider") or settings["topic_drift_provider"]),
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    sleep_seconds=float(settings["sleep_seconds"]),
                ),
            )
            tasks.append(task)
            pending.append((task, "prepare"))
            job.run_cancellations[run_id] = task.cancel_event

    def log(task, message):
        label = (
            f"[{task.run_id}] Graph {task.graph_index}/{len(graphs)} "
            f"({graphs[task.graph_index - 1]['name']}) · news {task.row + 1}/{rows}"
            if task
            else "[batch]"
        )
        with log_lock:
            message = f"{label}: {message}"
            job.events.put(("progress", message))
            with (batch_dir / "execution.log").open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def snapshot():
        snapshots = [task.tracker.snapshot() for task in tasks]
        resolved = sum(task.state in {"completed", "failed", "cancelled"} for task in tasks)
        operation_total = sum(item["total"] for item in snapshots)
        operation_done = sum(item["completed"] for item in snapshots)
        progress = dict(
            completed=resolved,
            total=len(tasks),
            graph_index=0,
            graph_total=len(graphs),
            row=0,
            row_total=rows,
            step=0,
            step_total=0,
            operation_total=operation_total,
            operation_completed=operation_done,
            parallel=True,
            max_simultaneous_runs=limit,
            runs=[
                dict(
                    run_id=task.run_id,
                    graph=graphs[task.graph_index - 1]["name"],
                    news_row=task.row + 1,
                    state=task.state,
                    current=item["current"],
                )
                for task, item in zip(tasks, snapshots, strict=True)
                if task.state not in {"completed", "failed", "cancelled", "queued"}
            ],
        )
        waiting = bool(job.recovery.pending_requests)
        elapsed = max(0.001, monotonic() - job.started_at)
        throughput = operation_done / elapsed
        remaining = max(0, operation_total - operation_done)
        progress.update(
            awaiting_decisions=waiting,
            estimate_low=remaining / throughput * 0.65
            if operation_done >= 3 and not waiting
            else None,
            estimate_high=remaining / throughput * 1.6
            if operation_done >= 3 and not waiting
            else None,
        )
        job.latest_progress = progress
        return progress

    def observe(task, event):
        with job.progress_lock:
            task.tracker.observe(1, event)
            if event["kind"] == "recovery_wait":
                task.state = "awaiting_decision"
            elif event["kind"] in {"recovery_retry", "error_permitted", "request_started"}:
                task.state = "evaluating" if task.prepared else "preparing"
            elif event["kind"] == "rate_limit_wait":
                task.state = "awaiting_quota"
            snapshot()
        if message := event_message(event):
            log(task, message)

    def prepare(task):
        graph = graphs[task.graph_index - 1]
        payload = build_editor_graph_payload(graph["nodes"])
        nodes, edges, start_node_id = graph_config_from_payload(payload)

        def cancelled():
            return job.cancel_event.is_set() or task.cancel_event.is_set()

        def decide(request):
            decision = job.recovery.ask(request, cancel_check=cancelled)
            if decision.action == "cancel_run":
                task.cancel_event.set()
            return decision

        session = RecoverySession(decide, store=job.recovery_session.store, cancel_check=cancelled)
        reuse = bool(settings.get("reuse_previous_results", False))
        with (
            use_recovery(session),
            use_shared_limiter(limiter),
            use_run_identity(
                run_id=task.run_id,
                graph_index=task.graph_index,
                graph_name=graph["name"],
                dataset_row=task.row + 1,
            ),
        ):
            result = runner(
                df=df.iloc[[task.row]].copy(),
                nodes=nodes,
                edges=edges,
                start_node_id=start_node_id,
                text_column=settings["text_column"],
                title_column=settings["title_column"],
                news_id_column=settings["news_id_column"] or None,
                max_rows=1,
                sleep_seconds=float(settings["sleep_seconds"]),
                max_requests_per_minute=None,
                retry_attempts=int(settings["retry_attempts"]),
                allow_title_fallback=settings["allow_title_fallback"],
                rewrite_mode=settings.get("rewrite_mode", "faithful"),
                topic_drift_model=settings["topic_drift_model"],
                topic_drift_provider=settings["topic_drift_provider"],
                stdi_comparison_method=settings.get("stdi_comparison_method", "dual"),
                stdi_judge_repeats=int(settings.get("stdi_judge_repeats", 3)),
                vad_method=settings.get("vad_method", "model"),
                vad_llm_model=settings.get("vad_llm_model"),
                vad_llm_provider=settings.get("vad_llm_provider"),
                stdi_cache_dir=batch_dir / "evaluation_cache",
                shared_cache_dir=runs_dir / "shared_cache" if reuse else None,
                reuse_rewrites=reuse,
                api_pricing=pricing,
                openai_service_tier=settings.get("openai_service_tier"),
                output_dir=task.directory,
                output_prefix=task.run_id,
                progress_callback=lambda message: log(task, message),
                operation_callback=lambda event: observe(task, event),
                cancel_check=cancelled,
                defer_evaluation=True,
            )
            if isinstance(result, PreparedSimulation) and result.checkpoint:
                task.checkpoint = result.context.copy().run(result.checkpoint)
            return result

    def fit_graph(index):
        prepared = [task.prepared for task in tasks if task.graph_index == index and task.prepared]
        pairs = [pair for item in prepared for pair in item.pairs]
        # Match sequential corpus ordering: all original pairs, then incremental pairs.
        pairs = [pair for pair in pairs if pair.pair_id.endswith(":original")] + [
            pair for pair in pairs if pair.pair_id.endswith(":incremental")
        ]
        first = prepared[0]
        comparator = first.context.copy().run(first.build_comparator, pairs)
        return LockedComparator(comparator)

    def save_graph(index, final=False):
        group = [task for task in tasks if task.graph_index == index]
        results = [
            task.result or task.checkpoint
            for task in group
            if task.result is not None or task.checkpoint is not None
        ]
        failures = [
            dict(run_id=task.run_id, row=task.row + 1, error=task.error)
            for task in group
            if task.error
        ]
        graph = graphs[index - 1]
        costs = [
            json.loads(path.read_text(encoding="utf-8"))
            for task in group
            if (path := task.directory / f"{task.run_id}_api_costs.json").is_file()
        ]
        if costs:
            (graph_dirs[index] / f"{graph_prefixes[index]}_api_costs.json").write_text(
                json.dumps(combine_usage(costs), indent=2, allow_nan=False), encoding="utf-8"
            )
        if not results:
            if final:
                job.events.put(
                    (
                        "bundle",
                        dict(
                            name=graph["name"],
                            execution_name=batch_dir.name,
                            status="cancelled" if job.cancel_event.is_set() else "failed",
                            error="; ".join(item["error"] for item in failures),
                            output_prefix=graph_prefixes[index],
                        ),
                    )
                )
            return
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
        summary.update(
            graph_name=graph["name"],
            execution_name=batch_dir.name,
            max_simultaneous_runs=limit,
            runs_total=len(group),
            runs_completed=sum(task.state == "completed" for task in group),
            run_failures=failures,
            partial=not final or bool(failures) or any(task.state == "cancelled" for task in group),
            cancelled=job.cancel_event.is_set() or any(task.state == "cancelled" for task in group),
        )
        if costs:
            summary["api_usage"] = combine_usage(costs)
        result = SimulationResult(summary, [step for item in results for step in item.step_results])
        if summary.get("result_kind") == "branching_graph":
            result.path_results = project_graph_paths(result)
        result.summary_path, result.steps_path = _persist_results(
            result=result, output_dir=graph_dirs[index], output_prefix=graph_prefixes[index]
        )
        if final:
            steps_df = steps_to_dataframe(result.step_results)
            job.events.put(
                (
                    "bundle",
                    dict(
                        name=graph["name"],
                        execution_name=batch_dir.name,
                        status="cancelled" if summary["cancelled"] else "completed",
                        summary=summary,
                        summary_path=str(result.summary_path),
                        steps_path=str(result.steps_path),
                        steps_df=steps_df,
                        node_summary_df=build_node_summary_dataframe(steps_df),
                        news_summary_df=build_news_summary_dataframe(steps_df),
                        output_prefix=graph_prefixes[index],
                        graph_payload=build_editor_graph_payload(graph["nodes"]),
                    ),
                )
            )

    completed_graphs = failed_graphs = 0
    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="news-run") as pool:
            while pending or futures:
                if job.cancel_event.is_set():
                    for task, phase in list(pending):
                        if phase == "prepare":
                            task.state = "cancelled"
                            pending.remove((task, phase))
                while pending and len(futures) < limit:
                    task, phase = pending.popleft()
                    if phase == "prepare" and (
                        job.cancel_event.is_set() or task.cancel_event.is_set()
                    ):
                        task.state = "cancelled"
                        task.tracker.finish_graph(1)
                        continue
                    task.state = {
                        "prepare": "preparing",
                        "fit": "fitting_graph",
                        "evaluate": "evaluating",
                    }[phase]
                    function = (
                        (lambda task=task: prepare(task))
                        if phase == "prepare"
                        else (lambda task=task: fit_graph(task.graph_index))
                        if phase == "fit"
                        else (lambda task=task: task.prepared.resume(comparators[task.graph_index]))
                    )
                    futures[pool.submit(copy_context().run, function)] = (task, phase)
                    log(task, task.state)
                if not futures:
                    for index in range(1, len(graphs) + 1):
                        prepared = next(
                            (
                                task
                                for task in tasks
                                if task.graph_index == index and task.state == "prepared"
                            ),
                            None,
                        )
                        if prepared and index not in fitting:
                            fitting.add(index)
                            pending.appendleft((prepared, "fit"))
                    if pending:
                        continue
                    break
                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    task, phase = futures.pop(future)
                    try:
                        value = future.result()
                        if phase == "fit":
                            comparators[task.graph_index] = value
                            task.state = "prepared"
                            ready = [
                                item
                                for item in tasks
                                if item.graph_index == task.graph_index and item.state == "prepared"
                            ]
                            pending.extendleft((item, "evaluate") for item in reversed(ready))
                        elif isinstance(value, PreparedSimulation):
                            task.prepared, task.state = value, "prepared"
                        else:
                            task.result = value
                            task.state = (
                                "cancelled" if value.summary.get("cancelled") else "completed"
                            )
                            task.tracker.finish_graph(1)
                            for step in value.step_results:
                                step.metadata.update(run_id=task.run_id, dataset_row=task.row + 1)
                                step.metadata["graph_step_id"] = f"{task.run_id}:{step.node_id}"
                            task.prepared = None
                            task.checkpoint = None
                    except Exception as error:
                        if phase == "fit":
                            for item in tasks:
                                if (
                                    item.graph_index == task.graph_index
                                    and item.prepared is not None
                                ):
                                    item.error, item.state = str(error), "failed"
                                    item.prepared.finalize(None)
                                    item.prepared = None
                                    item.tracker.finish_graph(1)
                        elif phase == "evaluate":
                            task.prepared = None
                        task.error = str(error)
                        task.state = (
                            "cancelled"
                            if (task.cancel_event.is_set() or job.cancel_event.is_set())
                            else "failed"
                        )
                        log(task, f"{task.state}: {error}")
                        task.tracker.finish_graph(1)
                for index in range(1, len(graphs) + 1):
                    group = [task for task in tasks if task.graph_index == index]
                    if (
                        index not in fitting
                        and group
                        and all(
                            task.state
                            not in {"queued", "preparing", "awaiting_quota", "awaiting_decision"}
                            for task in group
                        )
                    ):
                        prepared = next((task for task in group if task.state == "prepared"), None)
                        if prepared:
                            fitting.add(index)
                            pending.appendleft((prepared, "fit"))
                    if index not in published and all(
                        task.state in {"completed", "failed", "cancelled"} for task in group
                    ):
                        save_graph(index, final=True)
                        published.add(index)
                        comparators.pop(index, None)
                        if any(task.state == "failed" for task in group):
                            failed_graphs += 1
                        elif not any(task.state == "cancelled" for task in group):
                            completed_graphs += 1
                with job.progress_lock:
                    snapshot()
    finally:
        for index in range(1, len(graphs) + 1):
            if index not in published:
                save_graph(index, final=True)
        samples: dict[str, list] = {}
        for task in tasks:
            for key, values in task.tracker.samples.items():
                samples.setdefault(key, []).extend(values)
        (batch_dir / "timings.json").write_text(
            json.dumps(
                dict(version=1, samples={key: values[-60:] for key, values in samples.items()})
            ),
            encoding="utf-8",
        )
        costs = [
            json.loads(path.read_text(encoding="utf-8"))
            for task in tasks
            if (path := task.directory / f"{task.run_id}_api_costs.json").is_file()
        ]
        if costs:
            (batch_dir / "api_costs.json").write_text(
                json.dumps(combine_usage(costs), indent=2, allow_nan=False), encoding="utf-8"
            )
        job.finish(
            dict(
                completed=completed_graphs,
                failed=failed_graphs,
                cancelled=job.cancel_event.is_set(),
                completed_runs=sum(task.state == "completed" for task in tasks),
                failed_runs=sum(task.state == "failed" for task in tasks),
                cancelled_runs=sum(task.state == "cancelled" for task in tasks),
            )
        )
