"""Inspect failures and cancellations and start a linked recovery attempt."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from misinformation_simulation.apps.interaction_graph_attempts import (
    find_attempts,
    load_attempt,
    recovery_inputs,
)
from misinformation_simulation.apps.interaction_graph_credentials import session_api_credentials
from misinformation_simulation.apps.interaction_graph_run_job import start_graph_run_job
from misinformation_simulation.llm.recovery import safe_message
from misinformation_simulation.simulation.io import resolve_project_path


def render_run_history(runner):
    attempts = st.session_state.setdefault("run_attempts", {})
    st.subheader("Run recovery")
    directory = st.text_input(
        "Execution history folder",
        value=str(st.session_state.get("last_run_output_dir", "output/interaction_graph/app_runs")),
        key="recovery_history_folder",
    )
    if st.button("Load saved run history", key="load_run_history"):
        records, errors = find_attempts(resolve_project_path(directory))
        attempts.update({item["manifest_path"]: item for item in records})
        for error in errors:
            st.warning(error)
        if not records:
            st.info("No recovery snapshots found in this folder.")
    st.caption(
        "Failed runs and user cancellations are recorded separately. A cancellation after "
        "an error retains the error and its location. Recovery creates a new attempt and "
        "preserves the original files."
    )
    if not attempts:
        st.info("New executions save recovery snapshots here, including runs without results.")
        st.caption("Older executions without snapshots cannot be recovered from this panel.")
        return
    records = sorted(attempts.values(), key=lambda item: item["created_at"], reverse=True)
    _render_attempt_table(records)
    _render_recovery_selection(records, runner)


def _render_attempt_table(records):
    columns = st.columns(3)
    columns[0].metric("Failed", sum(item["state"] == "failed" for item in records))
    columns[1].metric(
        "Cancelled without error",
        sum(item["state"] == "cancelled" and not item["had_error"] for item in records),
    )
    columns[2].metric(
        "Cancelled after error",
        sum(item["state"] == "cancelled" and item["had_error"] for item in records),
    )
    state_filter = st.multiselect(
        "Run states",
        [
            "failed",
            "cancelled",
            "completed",
            "running",
            "queued",
            "preparing",
            "prepared",
            "evaluating",
            "fitting_graph",
            "awaiting_decision",
            "awaiting_quota",
        ],
        default=["failed", "cancelled"],
        key="recovery_state_filter",
    )
    rows = [
        dict(
            run=item["run_id"],
            graph=item["graph"]["name"],
            news_row=item["dataset_row"],
            scope=item["scope"],
            state=item["state"],
            interruption=item.get("interruption", ""),
            stage=item["location"].get("stage", ""),
            node=item["location"].get("node_id", ""),
            failure_location=item["errors"][-1]["location"] if item["errors"] else {},
            error=item["errors"][-1]["message"] if item["errors"] else "",
            attempt=item["attempt_id"],
            parent=item.get("parent_attempt") or "",
        )
        for item in records
        if item["state"] in state_filter
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_recovery_selection(records, runner):
    recoverable = [item for item in records if item["state"] in {"failed", "cancelled"}]
    if not recoverable:
        st.caption("No failed or cancelled runs are available for recovery.")
        return
    paths = [item["manifest_path"] for item in recoverable]
    lookup = {item["manifest_path"]: item for item in recoverable}
    path = st.selectbox(
        "Run to recover",
        paths,
        key="recovery_selected_attempt",
        format_func=lambda key: (
            f"{lookup[key]['graph']['name']} · {lookup[key]['run_id']} · "
            f"{lookup[key]['state']} · {lookup[key]['created_at']}"
        ),
    )
    record = lookup[path]
    st.write(f"State: **{record['state']}** · Reason: **{record.get('interruption', '')}**")
    if record["errors"]:
        st.dataframe(pd.DataFrame(record["errors"]), hide_index=True, width="stretch")
    if record.get("parent_attempt"):
        st.caption(f"Previous attempt: {record['parent_attempt']}")
    st.caption(
        "Retry replays saved valid responses and continues with failed or unfinished calls. "
        "Restart makes fresh calls for this run. Both use the saved dataset and configuration. "
        "Keys come from the current session or environment; edit them in Configuration."
    )
    active = st.session_state.get("run_job") is not None
    if active:
        st.info("Wait for the active execution to finish before starting a recovery attempt.")
    checkpoint = Path(path).parent / "call_checkpoint"
    retry = st.button(
        "Retry failed / interrupted call",
        key="recover_retry",
        disabled=active or not checkpoint.is_dir(),
    )
    restart = st.button("Restart run from zero", key="recover_restart", disabled=active)
    if not checkpoint.is_dir():
        st.caption("This attempt has no call checkpoint; use Restart run from zero.")
    if retry or restart:
        _start_recovery(path, "retry" if retry else "restart", runner)


def _start_recovery(path, mode, runner):
    try:
        data, graphs, settings = recovery_inputs(load_attempt(path), mode)
        credentials, missing = session_api_credentials(graphs, settings)
        if missing:
            raise ValueError("Configure API keys before recovery: " + ", ".join(missing))
        job = start_graph_run_job(
            df=data,
            graphs=graphs,
            settings=settings,
            runner=runner,
            queue_mode=False,
            api_credentials=credentials,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.error(f"Could not start recovery: {safe_message(error)}")
        return
    st.session_state.run_job = job
    st.session_state.run_progress = None
    st.session_state.run_messages = [f"Recovery started: {mode} · previous attempt: {path}"]
    st.session_state.last_run_output_dir = settings["output_dir"]
    st.rerun()
