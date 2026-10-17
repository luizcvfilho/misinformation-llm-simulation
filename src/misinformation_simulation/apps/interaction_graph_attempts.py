"""Durable run attempts, input snapshots and recovery lineage."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree
from uuid import uuid4

import pandas as pd

from misinformation_simulation.llm.checkpoint import write_json_atomic
from misinformation_simulation.llm.recovery import safe_diagnostics, safe_message
from misinformation_simulation.topic_drift.cluster_comparison import TopicStructurePair
from misinformation_simulation.topic_drift.models import NumericValue, TopicRelation, TopicStructure


def _without_credentials(value):
    if isinstance(value, dict):
        return {
            key: _without_credentials(item)
            for key, item in value.items()
            if not any(part in key.lower() for part in ("api_key", "credential", "token", "secret"))
        }
    if isinstance(value, list):
        return [_without_credentials(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunAttempt:
    def __init__(self, *, job, directory, run_id, graph, df, settings, row=None):
        self.job = job
        self.path = directory / "run_attempt.json"
        directory.mkdir(parents=True, exist_ok=True)
        df.to_json(directory / "input.json", orient="split", force_ascii=False, double_precision=15)
        self.record = dict(
            version=1,
            attempt_id=uuid4().hex,
            run_id=run_id,
            graph=_without_credentials(graph),
            settings=_without_credentials(settings),
            dataset_row=settings.get("_source_dataset_row", row),
            scope="news" if row is not None else "graph",
            state="queued",
            had_error=False,
            errors=[],
            location={},
            created_at=datetime.now(UTC).isoformat(),
            parent_attempt=settings.get("_parent_attempt"),
            recovery_mode=settings.get("_recovery_mode"),
            manifest_path=str(self.path),
        )
        self.save()

    def save(self):
        self.record["updated_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(self.path, self.record)
        self.job.events.put(("attempt", deepcopy(self.record)))

    def observe(self, event):
        if event.get("kind") == "started" and event.get("stage") != "saving":
            self.record["location"] = {
                key: event[key]
                for key in ("stage", "node_id", "news_id", "row", "comparison", "repeat")
                if key in event
            }
        if event.get("exception_type") == "RecoveryCancelled":
            return
        if event.get("kind") == "recovery_wait":
            self.record["state"] = "awaiting_decision"
        if event.get("kind") in {"recovery_wait", "error_permitted", "run_failed"} or (
            event.get("kind") == "finished" and event.get("status") == "failed"
        ):
            message = event.get("error") or event.get("reason")
            if message:
                self.add_error(message, self.record["location"])
                self.save()

    def add_error(self, error, location=None):
        entry = dict(message=safe_message(error), location=safe_diagnostics(location or {}))
        self.record["had_error"] = True
        if not self.record["errors"] or self.record["errors"][-1] != entry:
            self.record["errors"].append(entry)

    def awaiting_recovery(self, request):
        location = {
            key: request.location[key]
            for key in ("stage", "node_id", "news_id", "row", "comparison", "repeat")
            if key in request.location
        }
        if location:
            self.record["location"] = location
        self.add_error(request.message, self.record["location"])
        self.record["state"] = "awaiting_decision"
        self.save()

    def finish(self, state, *, result=None, error=None):
        if error and state != "cancelled":
            self.add_error(error, self.record["location"])
        if result is not None:
            self.record["summary_path"] = str(result.summary_path) if result.summary_path else None
        self.record["state"] = state
        self.record["interruption"] = (
            ("cancelled_after_error" if self.record["had_error"] else "cancelled_by_user")
            if state == "cancelled"
            else state
        )
        self.save()


def result_state(result):
    if result.summary.get("cancelled"):
        return "cancelled"
    if result.summary.get("steps_error", 0):
        return "failed"
    return (
        "failed"
        if any(_has_failed_status(asdict(step)) for step in result.step_results)
        else ("completed")
    )


def _has_failed_status(value):
    if isinstance(value, dict):
        return any(
            "status" in key and isinstance(item, str) and item in {"error", "invalid", "failed"}
            for key, item in value.items()
        ) or any(_has_failed_status(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_failed_status(item) for item in value)
    return False


def load_attempt(path):
    path = Path(path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "graph",
        "settings",
        "state",
        "attempt_id",
        "run_id",
        "scope",
        "created_at",
        "dataset_row",
        "had_error",
        "errors",
        "location",
    }
    if not isinstance(record, dict) or record.get("version") != 1 or not required <= record.keys():
        raise ValueError("Select a valid run attempt manifest.")
    record["manifest_path"] = str(path)
    return record


def find_attempts(directory):
    records, errors = [], []
    for path in Path(directory).rglob("run_attempt.json"):
        try:
            records.append(load_attempt(path))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{path}: {safe_message(error)}")
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True), errors


def recovery_inputs(record, mode):
    if mode not in {"retry", "restart"}:
        raise ValueError("Choose retry or restart.")
    if record["state"] not in {"failed", "cancelled"}:
        raise ValueError("Only failed or cancelled runs can be recovered.")
    path = Path(record["manifest_path"])
    data = pd.read_json(
        path.parent / "input.json",
        orient="split",
        convert_dates=False,
        dtype=False,
        convert_axes=False,
        precise_float=True,
    )
    settings = dict(record["settings"])
    for key in list(settings):
        if key.startswith("_recovery_") or key in {"_parent_attempt", "_source_dataset_row"}:
            settings.pop(key)
    settings.update(
        output_prefix=f"{path.parent.name}_{mode}",
        reuse_previous_results=False,
        _parent_attempt=str(path),
        _recovery_mode=mode,
        _track_news_runs=record["scope"] == "news",
    )
    if mode == "retry":
        checkpoint = path.parent / "call_checkpoint"
        if not checkpoint.is_dir():
            raise ValueError("No call checkpoint is available. Restart this run from zero.")
        settings["_recovery_checkpoint"] = str(checkpoint)
        settings["_recovery_evaluation_cache"] = record.get("evaluation_cache_path")
    if record["scope"] == "news":
        _restore_news_settings(record, settings, path)
    return data, [record["graph"]], settings


def _restore_news_settings(record, settings, path):
    settings.update(max_rows=1, _source_dataset_row=record["dataset_row"])
    settings["max_simultaneous_runs"] = max(1, settings.get("max_simultaneous_runs", 1))
    corpus = Path(record.get("corpus_path") or path.parent.parent.parent / "recovery_corpus.json")
    if corpus.is_file():
        settings["_recovery_corpus"] = str(corpus)
    elif settings.get("stdi_comparison_method", "dual") != "llm":
        raise ValueError("The graph evaluation corpus is not saved yet. Wait for this graph.")


def restore_evaluation_cache(batch_dir, settings):
    source = settings.get("_recovery_evaluation_cache")
    if settings.get("_recovery_mode") == "retry" and source and Path(source).is_dir():
        copytree(source, batch_dir / "evaluation_cache")


def save_corpus(path, groups):
    payload = [
        dict(
            row=row, pairs=[asdict(pair) for pair in pairs if isinstance(pair, TopicStructurePair)]
        )
        for row, pairs in groups
    ]
    write_json_atomic(path, payload)


def _restore_structure(payload):
    payload = dict(payload)
    relations = []
    for item in payload.pop("central_relations"):
        item = dict(item)
        if item.get("numeric_values") is not None:
            item["numeric_values"] = [NumericValue(**value) for value in item["numeric_values"]]
        relations.append(TopicRelation(**item))
    return TopicStructure(**payload, central_relations=relations)


def load_corpus(path, *, exclude_row):
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (
            group["row"],
            [
                TopicStructurePair(
                    pair["pair_id"],
                    _restore_structure(pair["original"]),
                    _restore_structure(pair["modified"]),
                )
                for pair in group["pairs"]
            ],
        )
        for group in groups
        if group["row"] != exclude_row
    ]
