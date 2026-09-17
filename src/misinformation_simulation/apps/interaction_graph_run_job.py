from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

import pandas as pd

from misinformation_simulation.apps.interaction_graph_queue import output_prefix_for_graph
from misinformation_simulation.apps.interaction_graph_ui import (
    build_linear_graph_payload,
    build_news_summary_dataframe,
    build_node_summary_dataframe,
    build_simulation_nodes,
    steps_to_dataframe,
)
from misinformation_simulation.simulation.types import SimulationResult

GraphRunner = Callable[..., SimulationResult]


@dataclass(slots=True)
class GraphRunJob:
    cancel_event: Event = field(default_factory=Event)
    events: Queue[tuple[str, Any]] = field(default_factory=Queue)
    thread: Thread | None = None


def start_graph_run_job(
    *,
    df: pd.DataFrame,
    graphs: list[dict[str, Any]],
    settings: dict[str, Any],
    runner: GraphRunner,
    queue_mode: bool,
) -> GraphRunJob:
    job = GraphRunJob()
    thread = Thread(
        target=_run_graph_queue,
        kwargs={
            "job": job,
            "df": df.copy(),
            "graphs": deepcopy(graphs),
            "settings": dict(settings),
            "runner": runner,
            "queue_mode": queue_mode,
        },
        daemon=True,
        name="interaction-graph-simulation",
    )
    job.thread = thread
    thread.start()
    return job


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
    completed = 0
    failed = 0
    cancelled = False
    try:
        for index, graph in enumerate(graphs, start=1):
            if job.cancel_event.is_set():
                cancelled = True
                break
            name = graph["name"]
            prefix = output_prefix_for_graph(
                settings["output_prefix"], index if queue_mode else 1, name
            )
            label = f"Graph {index}/{len(graphs)}: {name}"
            job.events.put(("progress", f"{label} — starting"))
            try:
                run_dir, prefix = _reserve_output_directory(Path(settings["output_dir"]), prefix)
                nodes = build_simulation_nodes(graph["nodes"])
                result = runner(
                    df=df,
                    nodes=nodes,
                    start_node_id=nodes[0].node_id,
                    text_column=settings["text_column"],
                    title_column=settings["title_column"],
                    news_id_column=settings["news_id_column"] or None,
                    max_rows=int(settings["max_rows"]),
                    sleep_seconds=float(settings["sleep_seconds"]),
                    max_requests_per_minute=(int(settings["max_requests_per_minute"]) or None),
                    retry_attempts=int(settings["retry_attempts"]),
                    allow_title_fallback=settings["allow_title_fallback"],
                    topic_drift_model=settings["topic_drift_model"],
                    topic_drift_provider=settings["topic_drift_provider"],
                    stdi_comparison_method="cluster",
                    output_dir=run_dir,
                    output_prefix=prefix,
                    progress_callback=lambda message, label=label: job.events.put(
                        ("progress", f"{label}: {message}")
                    ),
                    cancel_check=job.cancel_event.is_set,
                )
                status = "cancelled" if result.summary.get("cancelled") else "completed"
                steps_df = steps_to_dataframe(result.step_results)
                bundle = {
                    "name": name,
                    "status": status,
                    "summary": result.summary,
                    "summary_path": str(result.summary_path)
                    if result.summary_path is not None
                    else None,
                    "steps_path": str(result.steps_path) if result.steps_path is not None else None,
                    "steps_df": steps_df,
                    "node_summary_df": build_node_summary_dataframe(steps_df),
                    "news_summary_df": build_news_summary_dataframe(steps_df),
                    "output_prefix": prefix,
                    "graph_payload": build_linear_graph_payload(graph["nodes"]),
                }
                job.events.put(("bundle", bundle))
                if status == "cancelled":
                    cancelled = True
                    job.events.put(("progress", f"{label} — cancelled; partial results saved"))
                    break
                completed += 1
                job.events.put(("progress", f"{label} — finished"))
                if job.cancel_event.is_set():
                    cancelled = True
                    break
            except Exception as exc:
                if job.cancel_event.is_set():
                    cancelled = True
                    job.events.put(("progress", f"{label} — cancelled: {exc}"))
                    break
                failed += 1
                job.events.put(
                    (
                        "bundle",
                        {
                            "name": name,
                            "status": "failed",
                            "error": str(exc),
                            "output_prefix": prefix,
                        },
                    )
                )
                job.events.put(("progress", f"{label} — failed: {exc}"))
    finally:
        job.events.put(
            (
                "done",
                {"completed": completed, "failed": failed, "cancelled": cancelled},
            )
        )
