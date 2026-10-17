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
        self._pending: RecoveryRequest | None = None
        self._decision: RecoveryDecision | None = None
        self._requests: dict[str, tuple[RecoveryRequest, Event]] = {}
        self._decisions: dict[str, RecoveryDecision] = {}
        self._allowed: set[str] = set()

    @property
    def pending(self) -> RecoveryRequest | None:
        with self._lock:
            return self._pending

    @property
    def pending_requests(self) -> list[RecoveryRequest]:
        with self._lock:
            return [item[0] for item in self._requests.values()] or (
                [self._pending] if self._pending else []
            )

    def ask(self, request: RecoveryRequest, *, cancel_check=None) -> RecoveryDecision:
        ready = Event()
        with self._lock:
            if request.signatures and set(request.signatures) <= self._allowed:
                return RecoveryDecision("allow")
            self._requests[request.id] = (request, ready)
            self._pending = self._pending or request
        try:
            while not ready.wait(0.1):
                if self.cancel_event.is_set() or (cancel_check and cancel_check()):
                    raise RecoveryCancelled("Execution cancelled during error recovery.")
            with self._lock:
                return self._decisions.pop(request.id)
        finally:
            with self._lock:
                self._requests.pop(request.id, None)
                self._pending = next((item[0] for item in self._requests.values()), None)

    def submit(self, request_id: str, decision: RecoveryDecision) -> bool:
        with self._lock:
            entry = self._requests.get(request_id)
            request = entry[0] if entry else self._pending
            if request is None or request.id != request_id or request_id in self._decisions:
                return False
            if decision.action not in {"retry", "skip", "allow", "cancel", "cancel_run"}:
                return False
            if decision.credential and decision.credential not in request.context.choices:
                return False
            self._decision = decision
            targets = [request_id]
            if decision.action == "allow":
                self._allowed.update(request.signatures)
                targets = [
                    key
                    for key, (item, _) in self._requests.items()
                    if set(item.signatures) <= self._allowed
                ] or targets
            if decision.action == "cancel":
                self.cancel_event.set()
            for key in targets:
                self._decisions[key] = decision
                if item := self._requests.pop(key, None):
                    item[1].set()
            self._pending = next((item[0] for item in self._requests.values()), None)
            return True


@st.dialog("Runs awaiting recovery", dismissible=False)
def render_recovery_dialog(job, request: RecoveryRequest) -> None:
    pending = job.recovery.pending
    if pending is None or pending.id != request.id:
        st.rerun()
    requests = job.recovery.pending_requests
    if len(requests) > 1:
        selected_id = st.selectbox(
            "Pending errors",
            [item.id for item in requests],
            format_func=lambda key: next(
                f"{item.location.get('run_id', item.id)} · {item.context.provider} · "
                f"{item.location.get('stage', 'execution')} · {item.message[:80]}"
                for item in requests
                if item.id == key
            ),
            key="recovery_pending_selection",
        )
        request = next(item for item in requests if item.id == selected_id)
    st.caption(f"{len(requests)} pending error(s). Other runs continue when possible.")
    context = request.context
    st.write(f"Provider: **{context.provider}** · Model: **{context.model or '—'}**")
    if request.location:
        st.caption(" · ".join(f"{key}: {value}" for key, value in request.location.items()))
    st.code(request.message, language="text")
    _render_validation_diagnostics(request)
    st.warning(
        "Continuing records a failure. Dependent steps may be blocked; missing scores "
        "are not replaced with zero."
    )
    selected = None
    if context.choices:
        selected = st.radio(
            "First Gemini key on retry"
            if context.provider == "gemini"
            else "First OpenAI key on retry",
            context.choices,
            index=context.choices.index(context.credential),
            format_func=lambda slot: "Original primary" if slot == "primary" else slot.title(),
            key=f"recovery_key_{request.id}",
            horizontal=True,
        )
        st.caption(
            "Other Gemini keys are tried on quota errors."
            if context.provider == "gemini"
            else "If the selected key fails, the other configured key is also tried once."
        )
    if st.button("Try again", type="primary", key=f"retry_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("retry", selected))
        st.rerun()
    if st.button("Skip this occurrence and continue", key=f"skip_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("skip"))
        st.rerun()
    if st.button("Allow this error in future executions and continue", key=f"allow_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("allow"))
        st.rerun()
    if request.location.get("run_id") and st.button(
        "Cancel this run", key=f"cancel_run_{request.id}"
    ):
        job.recovery.submit(request.id, RecoveryDecision("cancel_run"))
        st.rerun()
    if st.button("Cancel execution", key=f"cancel_{request.id}"):
        job.recovery.submit(request.id, RecoveryDecision("cancel"))
        st.rerun()


def _render_validation_diagnostics(request):
    if request.diagnostics:
        with st.expander("Validation diagnostics"):
            st.json(request.diagnostics)
