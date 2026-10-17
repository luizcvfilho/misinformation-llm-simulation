from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

SCRIPT = """
import streamlit as st
from misinformation_simulation.apps.interaction_graph_run_job import GraphRunJob
from misinformation_simulation.apps.interaction_graph_recovery import render_recovery_dialog
from misinformation_simulation.llm.recovery import RecoveryRequest, RequestContext

if "job" not in st.session_state:
    job = GraphRunJob()
    job.recovery._pending = RecoveryRequest(
        "test", RequestContext("chatgpt", "model", "secondary", ("primary", "secondary")),
        "Both credentials failed", ("signature",), {"stage": "rewrite", "row": 3},
    )
    st.session_state.job = job
job = st.session_state.job
if job.recovery.pending:
    render_recovery_dialog(job, job.recovery.pending)
else:
    st.write("Decision received")
"""


@pytest.mark.parametrize("action", ["retry", "skip", "allow", "cancel"])
def test_dialog_accepts_decision_and_closes_without_exposing_keys(tmp_path, action):
    script = tmp_path / "recovery_app.py"
    script.write_text(SCRIPT, encoding="utf-8")
    app = AppTest.from_file(str(script)).run(timeout=20)
    assert not app.exception
    assert app.code[0].value == "Both credentials failed"
    assert app.radio[0].value == "secondary"
    if action == "retry":
        app.radio[0].set_value("primary").run()
    app.button(key=f"{action}_test").click().run()
    assert not app.exception
    job = app.session_state.job
    assert job.recovery.pending is None
    assert job.recovery._decision.action == action
    if action == "retry":
        assert job.recovery._decision.credential == "primary"
    assert job.cancel_event.is_set() == (action == "cancel")
