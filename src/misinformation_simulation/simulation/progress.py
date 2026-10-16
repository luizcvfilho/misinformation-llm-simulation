"""Operation counts and model-specific timing estimates for queued graph runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from time import monotonic


def timing_key(stage, model="", provider="") -> str:
    return json.dumps([stage, str(model), str(provider).strip().lower()])


def simulation_plan(
    rows,
    nodes,
    *,
    method,
    repeats,
    model,
    provider,
    vad_model,
    vad_provider,
    embedding_model,
    sleep_seconds=0,
    persist=True,
) -> list[dict]:
    counts = Counter(timing_key("rewrite", node.model, node.provider) for node in nodes)
    entries = [dict(key=timing_key("preparation"), total=1, expected=1)]
    entries.extend(dict(key=key, total=count * rows) for key, count in counts.items())
    texts = rows * (len(nodes) + 1)
    entries.extend(
        [
            dict(key=timing_key("extraction", model, provider), total=texts),
            dict(key=timing_key("vad", vad_model, vad_provider), total=texts),
        ]
    )
    if method in {"dual", "llm"}:
        entries.append(
            dict(key=timing_key("judge", model, provider), total=rows * len(nodes) * 2 * repeats)
        )
    if method in {"dual", "cluster"}:
        entries.append(dict(key=timing_key("embeddings", embedding_model), total=1))
    if sleep_seconds > 0:
        entries.append(
            dict(key=timing_key("pacing"), total=rows * len(nodes), expected=sleep_seconds)
        )
    if persist:
        entries.append(dict(key=timing_key("saving"), total=1, expected=1))
    return entries


class ProgressTracker:
    def __init__(self, history_path: Path | None = None):
        self.history_path = history_path
        self.samples = defaultdict(list)
        if history_path is not None:
            try:
                raw = json.loads(history_path.read_text(encoding="utf-8"))
                if raw.get("version") == 1:
                    for key, values in raw["samples"].items():
                        self.samples[key] = [
                            float(value)
                            for value in values[-60:]
                            if isinstance(value, (int, float))
                            and math.isfinite(value)
                            and value > 0
                        ]
            except (OSError, ValueError, TypeError, AttributeError, KeyError):
                pass
        self.planned = {}
        self.completed = Counter()
        self.cached = Counter()
        self.skipped = Counter()
        self.failed = Counter()
        self.expected = {}
        self.active = None
        self.state = None

    def plan(self, graph: int, entries: list[dict]) -> None:
        for entry in entries:
            key = (graph, entry["key"])
            self.planned[key] = max(entry["total"], self.completed[key])
            if "expected" in entry:
                self.expected[entry["key"]] = entry["expected"]

    def observe(self, graph: int, event: dict) -> None:
        kind = event["kind"]
        if kind == "plan":
            self.plan(graph, event["entries"])
            return
        if kind == "settle":
            for key in self.planned:
                if key[0] == graph and json.loads(key[1])[0] in event["stages"]:
                    missing = self.planned[key] - self.completed[key]
                    self.skipped[key] += missing
                    self.completed[key] += missing
            return
        key = (
            graph,
            timing_key(event.get("stage", ""), event.get("model", ""), event.get("provider", "")),
        )
        if kind == "started":
            self.active = (key, event)
            self.state = event
        elif kind == "finished":
            self._finish_operation(key, event)

        elif kind in {"request_started", "request_finished", "retry_wait", "rate_limit_wait"}:
            self.state = event

    def finish_graph(self, graph: int) -> None:
        self.observe(
            graph,
            {
                "kind": "settle",
                "stages": [json.loads(key)[0] for index, key in self.planned if index == graph],
            },
        )
        self.active = None
        self.state = None

    def snapshot(self, now: float | None = None) -> dict:
        now = monotonic() if now is None else now
        low = high = 0.0
        missing = set()
        stages = {}
        for key, total in self.planned.items():
            stage = json.loads(key[1])[0]
            counts = stages.setdefault(
                stage, dict(completed=0, total=0, cache_hits=0, skipped=0, failed=0)
            )
            counts["total"] += total
            for name, source in (
                ("completed", self.completed),
                ("cache_hits", self.cached),
                ("skipped", self.skipped),
                ("failed", self.failed),
            ):
                counts[name] += source[key]
            pending = max(0, total - self.completed[key])
            if not pending:
                continue
            samples = self.samples[key[1]][-20:]
            if samples:
                recent = samples[-5:]
                typical = fmean(recent)
                ordered = sorted(samples)
                lower = min(ordered[len(ordered) // 4], typical * 0.75)
                upper = max(ordered[min(len(ordered) - 1, 3 * len(ordered) // 4)], typical * 1.5)
            elif key[1] in self.expected:
                lower, upper = self.expected[key[1]] * 0.5, self.expected[key[1]] * 2
            else:
                missing.add(stage)
                continue
            low += lower * pending
            high += upper * pending
            if self.active is not None and self.active[0] == key:
                age = max(0, now - self.active[1]["monotonic"])
                low -= min(lower, age)
                high += max(upper * 0.5, upper - age) - upper
        state = dict(self.state) if self.state else None
        if state:
            state["age"] = max(0, now - state["monotonic"])
            if state["kind"] in {"retry_wait", "rate_limit_wait"}:
                waiting = max(0, state["delay"] - state["age"])
                low = max(low, waiting)
                high = max(high, waiting + 1)
        return dict(
            completed=sum(self.completed[key] for key in self.planned),
            total=sum(self.planned.values()),
            stages=stages,
            current=state,
            estimate_low=None if missing else max(0, low),
            estimate_high=None if missing else max(0, high),
            calibrating=sorted(missing),
        )

    def save_history(self) -> None:
        if self.history_path is not None:
            # Each execution keeps its own history file, avoiding concurrent writers.
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(
                json.dumps(dict(version=1, samples=dict(self.samples)), allow_nan=False),
                encoding="utf-8",
            )

    def _finish_operation(self, key, event):
        self.completed[key] += 1
        self.planned[key] = max(self.planned.get(key, 0), self.completed[key])
        if event.get("cache_hit"):
            self.cached[key] += 1
        if event.get("status") == "skipped":
            self.skipped[key] += 1
        elif event.get("status") == "failed":
            self.failed[key] += 1
        elif not event.get("cache_hit") and event.get("duration", 0) > 0:
            self.samples[key[1]].append(event["duration"])
            self.samples[key[1]] = self.samples[key[1]][-60:]
        self.active = None
        self.state = None


def event_message(event: dict) -> str:
    if event["kind"] in {"plan", "settle"}:
        return ""
    context = [
        f"news {event['news_id']}" if event.get("news_id") else "",
        f"node {event['node_id']}" if event.get("node_id") else "",
        event.get("stage", ""),
        event.get("model", ""),
        event.get("comparison", ""),
        f"repeat {event['repeat']}/{event['repeats']}" if event.get("repeat") else "",
    ]
    kind = event["kind"]
    if kind == "finished":
        detail = "cache hit" if event.get("cache_hit") else event.get("status", "completed")
        detail += f" · {event['duration']:.1f}s"
        if event.get("error"):
            detail += f" · {event['error']}"
        if event.get("reason"):
            detail += f" · {event['reason']}"
    elif kind == "retry_wait":
        detail = (
            f"retry {event['attempt'] + 1}/{event['max_attempts']} "
            f"in {event['delay']:.1f}s · {event['reason']}"
        )
    elif kind == "rate_limit_wait":
        detail = f"waiting for request limit · {event['delay']:.1f}s"
    else:
        detail = kind.replace("_", " ")
    return (
        f"[{event['timestamp']}] " + " · ".join(item for item in context if item) + f" · {detail}"
    )
