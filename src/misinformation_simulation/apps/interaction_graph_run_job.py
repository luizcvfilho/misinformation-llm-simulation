from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import pandas as pd

from misinformation_simulation.apps.interaction_graph_attempts import (
    RunAttempt,
    restore_evaluation_cache,
    result_state,
)
from misinformation_simulation.apps.interaction_graph_queue import output_prefix_for_graph
from misinformation_simulation.apps.interaction_graph_recovery import RecoveryBroker
from misinformation_simulation.apps.interaction_graph_ui import (
    build_editor_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    steps_to_dataframe,
)
from misinformation_simulation.llm.checkpoint import use_call_checkpoint
from misinformation_simulation.llm.clients import normalize_provider
from misinformation_simulation.llm.credentials import ProviderAPIKeys, use_api_credentials
from misinformation_simulation.llm.flex_fallback import FlexFallback, use_flex_fallback
from misinformation_simulation.llm.pricing import load_pricing
from misinformation_simulation.llm.progress import use_run_identity
from misinformation_simulation.llm.rate_limit import (
    SharedRequestLimiter,
    request_limits_from_settings,
    use_shared_limiter,
)
from misinformation_simulation.llm.recovery import (
    RecoverySession,
    RequestContext,
    get_recovery,
    safe_message,
    use_recovery,
)
from misinformation_simulation.llm.usage import combine_usage
from misinformation_simulation.simulation.graph import DEFAULT_STDI_EMBEDDING_MODEL
from misinformation_simulation.simulation.io import graph_config_from_payload, resolve_project_path
from misinformation_simulation.simulation.progress import (
    ProgressTracker,
    event_message,
    simulation_plan,
)
from misinformation_simulation.simulation.types import SimulationResult
from misinformation_simulation.text_metrics.vad import DEFAULT_VAD_MODEL_NAME

GraphRunner = Callable[..., SimulationResult]


@dataclass(slots=True)
class GraphRunJob:
    cancel_event: Event = field(default_factory=Event)
    events: Queue[tuple[str, Any]] = field(default_factory=Queue)
    thread: Thread | None = None
    started_at: float = field(default_factory=monotonic)
    progress_lock: Lock = field(default_factory=Lock)
    latest_progress: dict[str, Any] | None = None
    tracker: ProgressTracker | None = None
    observing: bool = False
    recovery: RecoveryBroker = field(init=False)
    recovery_session: RecoverySession = field(init=False)
    run_cancellations: dict[str, Event] = field(default_factory=dict)
    finished: bool = False
    flex_fallback: FlexFallback | None = None

    def finish(self, payload):
        if not self.finished:
            self.finished = True
            self.events.put(("done", payload))

    def __post_init__(self):
        self.recovery = RecoveryBroker(self.cancel_event)
        self.recovery_session = RecoverySession(
            self.recovery.ask,
            cancel_check=self.cancel_event.is_set,
        )


def start_graph_run_job(
    *,
    df: pd.DataFrame,
    graphs: list[dict[str, Any]],
    settings: dict[str, Any],
    runner: GraphRunner,
    queue_mode: bool,
    api_credentials: Mapping[str, ProviderAPIKeys] | None = None,
) -> GraphRunJob:
    job = GraphRunJob()
    settings = dict(settings)
    settings.setdefault("_track_news_runs", True)
    settings.setdefault("max_simultaneous_runs", 1)
    thread = Thread(
        target=_run_with_recovery,
        kwargs={
            "job": job,
            "df": df.copy(),
            "graphs": deepcopy(graphs),
            "settings": dict(settings),
            "runner": runner,
            "queue_mode": queue_mode,
            "api_credentials": dict(api_credentials or {}),
        },
        daemon=True,
        name="interaction-graph-simulation",
    )
    job.thread = thread
    thread.start()
    return job


def _run_with_recovery(
    *,
    job: GraphRunJob,
    api_credentials: Mapping[str, ProviderAPIKeys] | None = None,
    **kwargs,
) -> None:
    limiter = SharedRequestLimiter(
        request_limits_from_settings(kwargs["settings"]), job.cancel_event.is_set
    )
    try:
        with (
            use_api_credentials(api_credentials),
            use_recovery(job.recovery_session),
            use_shared_limiter(limiter),
            use_flex_fallback(
                kwargs["settings"].get("openai_flex_fallback", False),
                kwargs["settings"].get("openai_flex_retry_seconds", 300),
            ) as fallback,
        ):
            job.flex_fallback = fallback
            _run_graph_queue(job=job, **kwargs)
    except Exception as error:
        job.events.put(("progress", f"Execution failed: {safe_message(error)}"))
        job.finish(
            dict(completed=0, failed=len(kwargs["graphs"]), cancelled=job.cancel_event.is_set())
        )


def _execute_graph(job: GraphRunJob, function: Callable):
    if get_recovery() is job.recovery_session:
        return job.recovery_session.run(function, context=RequestContext(), isolated=True)
    return function()


def _reserve_output_directory(base_dir: Path, prefix: str) -> tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        name = prefix if attempt == 1 else f"{prefix}_{attempt:02d}"
        directory = base_dir / name
        try:
            directory.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        return directory, name


def _run_graph_queue(
    *,
    job: GraphRunJob,
    df: pd.DataFrame,
    graphs: list[dict[str, Any]],
    settings: dict[str, Any],
    runner: GraphRunner,
    queue_mode: bool,
) -> None:
    return _SequentialQueue(
        job=job, df=df, graphs=graphs, settings=settings, runner=runner, queue_mode=queue_mode
    ).run()


class _SequentialQueue:
    """Execute queued graphs and report persisted results and progress."""

    def __init__(self, *, job, df, graphs, settings, runner, queue_mode):
        self.job = job
        self.df = df
        self.graphs = graphs
        self.settings = settings
        self.runner = runner
        self.queue_mode = queue_mode

    def run(self):
        if (
            self.settings.get("_track_news_runs")
            or int(self.settings.get("max_simultaneous_runs", 1)) > 1
        ):
            from misinformation_simulation.apps.interaction_graph_parallel import run_parallel_queue

            run_parallel_queue(
                job=self.job,
                df=self.df,
                graphs=self.graphs,
                settings=self.settings,
                runner=self.runner,
                queue_mode=self.queue_mode,
            )
            return
        self._initialize_queue()

        try:
            self._prepare_batch()

            for index, graph in enumerate(self.graphs, start=1):
                self.index, self.graph = (index, graph)
                if self.job.cancel_event.is_set():
                    self.cancelled = True
                    break
                self._execute_queue_item()
                if self.cancelled:
                    break

        finally:
            if self.job.tracker is not None:
                try:
                    self.job.tracker.save_history()
                except OSError as exc:
                    self.job.events.put(("progress", f"Could not save timing history: {exc}"))
            self.job.finish(
                {
                    "completed": self.completed,
                    "failed": self.failed,
                    "cancelled": self.cancelled or self.cancelled_count > 0,
                }
            )

    def report_work(self, index: int, row: int, steps: int) -> None:
        units_per_row = len(self.graphs[index - 1]["nodes"]) + 1
        completed_units = self.finished_units + max(0, row - 1) * units_per_row
        if row:
            completed_units += 1 + steps
        progress = {
            "completed": completed_units,
            "total": self.total_units,
            "graph_index": index,
            "graph_total": len(self.graphs),
            "row": row,
            "row_total": self.rows_per_graph,
            "step": steps,
            "step_total": units_per_row - 1,
        }
        with self.job.progress_lock:
            if self.job.observing:
                progress.update(self.job.tracker.snapshot())
            self.job.latest_progress = progress

    def log(self, message):
        self.job.events.put(("progress", message))
        with (self.batch_dir / "execution.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    def observe(self, index, event):
        self.attempt.observe(event)
        with self.job.progress_lock:
            self.job.observing = True
            self.job.tracker.observe(index, event)
            if self.job.latest_progress is not None:
                self.job.latest_progress = {
                    **self.job.latest_progress,
                    **self.job.tracker.snapshot(),
                }
                if "stage" in event:
                    self.job.latest_progress["row"] = event.get("row", 0)
                    self.job.latest_progress["phase"] = event["stage"]
                    self.job.latest_progress["node_id"] = event.get("node_id")
        message = event_message(event)
        if message:
            self.log(f"Graph {index}/{len(self.graphs)}: {message}")

    def _initialize_queue(self):
        self.completed = 0
        self.failed = 0
        self.cancelled = False
        self.cancelled_count = 0
        self.rows_per_graph = min(len(self.df), max(0, int(self.settings["max_rows"])))
        self.graph_units = [
            self.rows_per_graph * (len(graph["nodes"]) + 1) + 1 for graph in self.graphs
        ]
        self.total_units = sum(self.graph_units)
        self.finished_units = 0
        self.runs_dir = resolve_project_path(Path(self.settings["output_dir"]).expanduser())
        self.batch_dir = self.runs_dir
        self.batch_usage = []

    def _prepare_batch(self):
        batch_prefix = Path(self.settings["output_prefix"].strip()).name
        if not batch_prefix or batch_prefix in {".", ".."}:
            raise ValueError("Enter an output prefix before running.")
        self.batch_dir, _ = _reserve_output_directory(self.runs_dir, batch_prefix)
        restore_evaluation_cache(self.batch_dir, self.settings)
        self.reuse = bool(self.settings.get("reuse_previous_results", False))
        histories = sorted(
            self.runs_dir.glob("*/timings.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        self.job.tracker = ProgressTracker(histories[0] if histories else None)
        self.job.tracker.history_path = self.batch_dir / "timings.json"
        self._plan_graphs()

    def _execute_queue_item(self):
        self.name = self.graph["name"]
        self.prefix = output_prefix_for_graph(
            self.settings["output_prefix"],
            self.index if self.queue_mode else 1,
            self.name,
            include_base=False,
        )
        self.label = f"Graph {self.index}/{len(self.graphs)}: {self.name}"
        self.log(f"{self.label} — starting")
        self.report_work(self.index, 0, 0)
        self.run_dir = None
        self.attempt = None
        try:
            self._run_graph()

            self._publish_result()

            if self.status == "cancelled":
                self.cancelled_count += 1
                self.cancelled = self.job.cancel_event.is_set()
                self.log(f"{self.label} — cancelled; partial results saved")
                return
            if self.status == "failed":
                self.failed += 1
                self.job.tracker.finish_graph(self.index)
                self.finished_units += self.graph_units[self.index - 1]
                self.log(f"{self.label} — failed; partial results saved")
                return
            self.completed += 1
            with self.job.progress_lock:
                self.job.tracker.finish_graph(self.index)
            self.finished_units += self.graph_units[self.index - 1]
            self.report_work(self.index, 0, 0)
            self.log(f"{self.label} — finished")
            if self.job.cancel_event.is_set():
                self.cancelled = True
                return
        except Exception as exc:
            is_cancelled = self.job.cancel_event.is_set() or (
                hasattr(self, "run_cancel") and self.run_cancel.is_set()
            )
            self.cancelled = self.job.cancel_event.is_set()
            status = "cancelled" if is_cancelled else "failed"
            self.cancelled_count += is_cancelled
            if self.attempt is not None:
                self.attempt.finish(status, error=exc)
            self.failed += not is_cancelled
            with self.job.progress_lock:
                self.job.tracker.finish_graph(self.index)
            self.finished_units += self.graph_units[self.index - 1]
            self.report_work(self.index, 0, 0)
            self.job.events.put(
                (
                    "bundle",
                    {
                        "name": self.name,
                        "execution_name": self.batch_dir.name,
                        "status": status,
                        "error": safe_message(exc),
                        "attempt_path": str(self.attempt.path) if self.attempt else None,
                        "output_prefix": self.prefix,
                    },
                )
            )
            self.log(f"{self.label} — {status}: {safe_message(exc)}")
        finally:
            cost_path = self.run_dir / f"{self.prefix}_api_costs.json" if self.run_dir else None
            if cost_path is not None and cost_path.is_file():
                self.batch_usage.append(json.loads(cost_path.read_text(encoding="utf-8")))
                (self.batch_dir / "api_costs.json").write_text(
                    json.dumps(combine_usage(self.batch_usage), indent=2, allow_nan=False),
                    encoding="utf-8",
                )

    def _run_graph(self):
        self.run_dir, _ = _reserve_output_directory(self.batch_dir, self.prefix)
        self.attempt = RunAttempt(
            job=self.job,
            directory=self.run_dir,
            run_id=self.prefix,
            graph=self.graph,
            df=self.df.iloc[: self.rows_per_graph],
            settings=self.settings,
        )
        self.attempt.record["evaluation_cache_path"] = str(self.batch_dir / "evaluation_cache")
        self.attempt.record["state"] = "running"
        self.attempt.save()
        self.run_cancel = Event()
        self.job.run_cancellations[self.prefix] = self.run_cancel

        def cancelled():
            return self.job.cancel_event.is_set() or self.run_cancel.is_set()

        def decide(request):
            self.attempt.awaiting_recovery(request)
            decision = self.job.recovery.ask(request, cancel_check=cancelled)
            if decision.action == "cancel_run":
                self.run_cancel.set()
            return decision

        session = RecoverySession(
            decide, store=self.job.recovery_session.store, cancel_check=cancelled
        )
        self.graph_payload = build_editor_graph_payload(self.graph["nodes"])
        nodes, edges, start_node_id = graph_config_from_payload(self.graph_payload)
        with (
            use_recovery(session),
            use_run_identity(run_id=self.prefix, graph_name=self.name, graph_index=self.index),
            use_call_checkpoint(
                self.run_dir / "call_checkpoint",
                Path(self.settings["_recovery_checkpoint"])
                if self.settings.get("_recovery_checkpoint")
                else None,
            ),
        ):
            self.result = self._invoke_graph(nodes, edges, start_node_id, cancelled)

    def _invoke_graph(self, nodes, edges, start_node_id, cancelled):
        return _execute_graph(
            self.job,
            partial(
                self.runner,
                df=self.df,
                nodes=nodes,
                edges=edges,
                start_node_id=start_node_id,
                text_column=self.settings["text_column"],
                title_column=self.settings["title_column"],
                news_id_column=self.settings["news_id_column"] or None,
                max_rows=int(self.settings["max_rows"]),
                sleep_seconds=float(self.settings["sleep_seconds"]),
                max_requests_per_minute=int(self.settings.get("max_requests_per_minute", 0))
                or None,
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
                shared_cache_dir=self.runs_dir / "shared_cache" if self.reuse else None,
                reuse_rewrites=self.reuse or self.settings.get("_recovery_mode") == "retry",
                api_pricing=load_pricing(
                    resolve_project_path(self.settings["api_pricing_path"])
                    if self.settings.get("api_pricing_path")
                    else None
                ),
                openai_service_tier=self.settings.get("openai_service_tier"),
                openai_flex_fallback=self.settings.get("openai_flex_fallback", False),
                openai_flex_retry_seconds=self.settings.get("openai_flex_retry_seconds", 300),
                output_dir=self.run_dir,
                output_prefix=self.prefix,
                progress_callback=lambda message, label=self.label: self.log(f"{label}: {message}"),
                operation_callback=lambda event, index=self.index: self.observe(index, event),
                work_progress_callback=lambda row, _total, steps, _nodes, index=self.index: (
                    self.report_work(index, row, steps)
                ),
                cancel_check=cancelled,
            ),
        )

    def _publish_result(self):
        self.status = result_state(self.result)
        if self.status == "failed" and not self.attempt.record["errors"]:
            self.attempt.add_error("One or more simulation operations failed.")
        self.attempt.finish(self.status, result=self.result)
        self.result.summary["graph_name"] = self.name
        self.result.summary["execution_name"] = self.batch_dir.name
        self.result.summary["execution_status"] = self.status
        self.result.summary["attempt_path"] = str(self.attempt.path)
        for path_result in getattr(self.result, "path_results", []):
            path_result.summary["execution_name"] = self.batch_dir.name
            if path_result.summary_path is not None:
                path_result.summary_path.write_text(
                    json.dumps(path_result.summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        if self.result.summary_path is not None:
            self.result.summary_path.write_text(
                json.dumps(self.result.summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        steps_df = steps_to_dataframe(self.result.step_results)
        bundle = {
            "name": self.name,
            "execution_name": self.batch_dir.name,
            "status": self.status,
            "attempt_path": str(self.attempt.path),
            "error": "; ".join(item["message"] for item in self.attempt.record["errors"]),
            "summary": self.result.summary,
            "summary_path": str(self.result.summary_path)
            if self.result.summary_path is not None
            else None,
            "steps_path": str(self.result.steps_path)
            if self.result.steps_path is not None
            else None,
            "steps_df": steps_df,
            "node_summary_df": build_node_summary_dataframe(steps_df),
            "news_summary_df": build_news_summary_dataframe(steps_df),
            "output_prefix": self.prefix,
            "graph_payload": self.graph_payload,
        }
        self.job.events.put(("bundle", bundle))

    def _plan_graphs(self):
        for graph_index, planned_graph in enumerate(self.graphs, 1):
            planned_nodes = build_simulation_nodes(planned_graph["nodes"])
            vad_method = self.settings.get("vad_method", "model")
            vad_model = (
                (
                    f"{vad_method}:"
                    f"{self.settings.get('vad_llm_model') or self.settings['topic_drift_model']}"
                )
                if vad_method != "model"
                else DEFAULT_VAD_MODEL_NAME
            )
            self.job.tracker.plan(
                graph_index,
                simulation_plan(
                    self.rows_per_graph,
                    planned_nodes,
                    method=self.settings.get("stdi_comparison_method", "dual"),
                    repeats=int(self.settings.get("stdi_judge_repeats", 3)),
                    model=self.settings["topic_drift_model"],
                    provider=normalize_provider(self.settings["topic_drift_provider"]),
                    vad_model=vad_model,
                    vad_provider=normalize_provider(
                        self.settings.get("vad_llm_provider")
                        or self.settings["topic_drift_provider"]
                    )
                    if vad_method != "model"
                    else "local",
                    embedding_model=DEFAULT_STDI_EMBEDDING_MODEL,
                    sleep_seconds=float(self.settings["sleep_seconds"]),
                ),
            )
