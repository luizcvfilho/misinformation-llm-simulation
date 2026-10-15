from __future__ import annotations

import json

import pytest

from misinformation_simulation.llm.progress import (
    observe_events,
    operation,
    set_progress_context,
)
from misinformation_simulation.llm.retry import generate_and_parse_with_retry
from misinformation_simulation.simulation.progress import ProgressTracker, timing_key


def finished(stage="judge", *, duration=10, model="model", **values):
    return dict(
        kind="finished",
        stage=stage,
        model=model,
        provider="chatgpt",
        duration=duration,
        status="completed",
        **values,
    )


def test_judge_pending_work_is_counted_after_rewrites_and_history_survives_cache(tmp_path):
    tracker = ProgressTracker(tmp_path / "timings.json")
    judge = timing_key("judge", "model", "chatgpt")
    rewrite = timing_key("rewrite", "model", "chatgpt")
    tracker.plan(1, [dict(key=rewrite, total=1), dict(key=judge, total=6)])
    tracker.observe(1, finished("rewrite"))
    before = tracker.snapshot()
    assert before["completed"] == 1 and before["total"] == 7
    assert before["estimate_low"] is None
    assert before["calibrating"] == ["judge"]
    tracker.observe(1, finished())
    snapshot = tracker.snapshot()
    assert snapshot["estimate_low"] == pytest.approx(37.5)
    assert snapshot["estimate_high"] == pytest.approx(75)
    tracker.observe(1, finished(duration=0.001, cache_hit=True))
    tracker.observe(1, finished(duration=0.001) | {"status": "failed"})
    assert tracker.samples[judge] == [10]
    assert tracker.snapshot()["stages"]["judge"]["cache_hits"] == 1
    assert tracker.snapshot()["stages"]["judge"]["failed"] == 1
    tracker.save_history()
    restored = ProgressTracker(tracker.history_path)
    assert restored.samples[judge] == [10]
    restored.plan(1, [dict(key=timing_key("judge", "other", "chatgpt"), total=1)])
    assert restored.snapshot()["estimate_low"] is None


def test_overdue_request_keeps_a_positive_range_and_live_age():
    tracker = ProgressTracker()
    key = timing_key("judge", "model", "chatgpt")
    tracker.samples[key] = [10, 10, 10]
    tracker.plan(1, [dict(key=key, total=1)])
    tracker.observe(
        1, dict(kind="started", stage="judge", model="model", provider="chatgpt", monotonic=100)
    )
    tracker.observe(1, dict(kind="request_started", stage="judge", monotonic=105))
    snapshot = tracker.snapshot(now=200)
    assert snapshot["estimate_low"] == 0
    assert snapshot["estimate_high"] > 0
    assert snapshot["current"]["age"] == 95


def test_settling_blocked_work_keeps_counts_and_failure_visible():
    tracker = ProgressTracker()
    key = timing_key("rewrite", "model", "chatgpt")
    tracker.plan(1, [dict(key=key, total=4)])
    tracker.observe(1, finished("rewrite") | {"status": "failed"})
    tracker.finish_graph(1)
    counts = tracker.snapshot()["stages"]["rewrite"]
    assert counts == dict(completed=4, total=4, failed=1, cache_hits=0, skipped=3)
    assert not tracker.samples[key]


def test_known_rate_limit_wait_bounds_remaining_time():
    tracker = ProgressTracker()
    key = timing_key("judge", "model", "chatgpt")
    tracker.samples[key] = [1]
    tracker.plan(1, [dict(key=key, total=1)])
    tracker.observe(
        1, dict(kind="started", stage="judge", model="model", provider="chatgpt", monotonic=100)
    )
    tracker.observe(1, dict(kind="rate_limit_wait", delay=55, monotonic=100))
    snapshot = tracker.snapshot(now=105)
    assert snapshot["estimate_low"] >= 50
    assert snapshot["estimate_high"] > 50


def test_validation_retry_events_share_context_and_do_not_count_as_new_draw(monkeypatch):
    monkeypatch.setattr("misinformation_simulation.llm.retry.time.sleep", lambda _: None)
    monkeypatch.setattr("misinformation_simulation.llm.retry.random.uniform", lambda *_: 0)
    events = []
    responses = iter(["invalid", "0.5"])
    with observe_events(events.append):
        set_progress_context(news_id="news", node_id="node", comparison="vs original")
        with operation("judge", model="model", repeat=2, repeats=3):
            assert generate_and_parse_with_retry(lambda: next(responses), float) == 0.5
    wait = next(event for event in events if event["kind"] == "retry_wait")
    assert wait["attempt"] == 1 and wait["delay"] == 2
    assert wait["comparison"] == "vs original"
    assert wait["repeat"] == 2
    assert len([event for event in events if event["kind"] == "finished"]) == 1
    assert len({event["operation_id"] for event in events}) == 1
    with observe_events(events.append):
        with operation("rewrite"):
            pass
    assert "news_id" not in events[-1]


def test_corrupt_or_invalid_timing_history_is_ignored(tmp_path):
    path = tmp_path / "timings.json"
    path.write_text("not json", encoding="utf-8")
    assert not ProgressTracker(path).samples
    path.write_text(
        json.dumps(dict(version=1, samples={"judge": [None, -1, "bad", 2]})), encoding="utf-8"
    )
    assert ProgressTracker(path).samples["judge"] == [2]


def test_provider_normalization_preserves_the_same_timing_group():
    assert timing_key("rewrite", "model", " ChatGPT ") == timing_key("rewrite", "model", "chatgpt")
