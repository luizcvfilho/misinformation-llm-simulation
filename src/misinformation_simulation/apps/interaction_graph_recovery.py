"""Exchange recovery decisions between the UI and its background worker."""

from __future__ import annotations

from threading import Event, Lock

import streamlit as st

from misinformation_simulation.llm.recovery import (
    RecoveryCancelled,
    RecoveryDecision,
    RecoveryRequest,
)


class RecoveryBroker:
    def __init__(self, cancel_event: Event):
        self.cancel_event = cancel_event
        self._lock = Lock()
        self._ready = Event()
        self._pending: RecoveryRequest | None = None
        self._decision: RecoveryDecision | None = None

    @property
    def pending(self) -> RecoveryRequest | None:
        with self._lock:
            return self._pending

    def ask(self, request: RecoveryRequest) -> RecoveryDecision:
        with self._lock:
            self._ready.clear()
            self._decision = None
            self._pending = request
        try:
            while not self._ready.wait(0.1):
                if self.cancel_event.is_set():
                    raise RecoveryCancelled("Execution cancelled during error recovery.")
            with self._lock:
                return self._decision
        finally:
            with self._lock:
                self._pending = None

    def submit(self, request_id: str, decision: RecoveryDecision) -> bool:
        with self._lock:
            if self._pending is None or self._pending.id != request_id or self._decision:
                return False
            if decision.action not in {"retry", "allow", "cancel"}:
                return False
            if decision.credential and decision.credential not in self._pending.context.choices:
                return False
            self._decision = decision
            self._pending = None
            if decision.action == "cancel":
                self.cancel_event.set()
            self._ready.set()
            return True


@st.dialog("Execution paused", dismissible=False)
def render_recovery_dialog(job, request: RecoveryRequest) -> None:
    pending = job.recovery.pending
    if pending is None or pending.id != request.id:
        st.rerun()
    context = request.context
    st.write(f"Provider: **{context.provider}** · Model: **{context.model or '—'}**")
    if request.location:
        st.caption(" · ".join(f"{key}: {value}" for key, value in request.location.items()))
    st.code(request.message, language="text")
    st.warning(
        "Continuing records a failure. Dependent steps may be blocked; missing scores "
        "are not replaced with zero."
    )
    selected = None
    if context.choices:
        selected = st.radio(
            "First OpenAI key on retry",
            context.choices,
            index=context.choices.index(context.credential),
            format_func=lambda slot: "Original primary" if slot == "primary" else "Secondary",
            key=f"recovery_key_{request.id}",
            horizontal=True,
        )
        st.caption("If the selected key fails, the other configured key is also tried once.")
    if st.button("Try again", type="primary", key=f"retry_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("retry", selected))
        st.rerun()
    if st.button("Allow this error in future executions and continue", key=f"allow_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("allow"))
        st.rerun()
    if st.button("Cancel execution", key=f"cancel_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("cancel"))
        st.rerun()
