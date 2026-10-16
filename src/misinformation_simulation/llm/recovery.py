"""User-directed recovery shared by LLM calls and execution frontends."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from uuid import uuid4

from dotenv import dotenv_values

from misinformation_simulation.config.settings import PROJECT_ROOT
from misinformation_simulation.llm.progress import emit_event, progress_context

_session: ContextVar[RecoverySession | None] = ContextVar("recovery_session", default=None)
_inside: ContextVar[bool] = ContextVar("inside_recovery", default=False)
_request: ContextVar[RequestContext | None] = ContextVar("recovery_request", default=None)
_manual_slot: ContextVar[str | None] = ContextVar("recovery_manual_slot", default=None)
_attempts: ContextVar[list | None] = ContextVar("recovery_attempts", default=None)
_store_lock = RLock()
_default_session = None
_automatic_session = None
_known_secrets: set[str] = set()


class ErrorPermitted(Exception):
    """The operation failed and the user explicitly allowed execution to continue."""


class RecoveryCancelled(Exception):
    """Recovery was cancelled without accepting the failed output."""


def safe_message(error: Exception | str) -> str:
    message = str(error)
    for value in _known_secrets.copy():
        message = message.replace(value, "[redacted]")
    for name, value in os.environ.items():
        if value and len(value) >= 6 and any(part in name for part in ("API_KEY", "TOKEN")):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"\bsk-[\w-]+", "[redacted]", message)
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", message)
    return message[:6000]


def register_secret(value: str | None) -> None:
    if value and len(value) >= 6:
        _known_secrets.add(value)


@dataclass(frozen=True)
class RequestContext:
    provider: str = "application"
    model: str = ""
    credential: str = "default"
    choices: tuple[str, ...] = ()
    endpoint: str = ""


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    credential: str | None = None


@dataclass(frozen=True)
class RecoveryRequest:
    id: str
    context: RequestContext
    message: str
    signatures: tuple[str, ...]
    location: dict = field(default_factory=dict)


class AttemptFailures(Exception):
    def __init__(self, failures: list[tuple[RequestContext, Exception]]):
        self.failures = failures
        super().__init__(
            "\n\n".join(
                f"{context.credential}: {safe_message(error)}" for context, error in failures
            )
        )


def error_signature(error: Exception, context: RequestContext) -> str:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        body = {}
    body = body.get("error", body)
    if not isinstance(body, dict):
        body = {}
    code = str(getattr(error, "code", None) or body.get("code") or "")
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    message = safe_message(getattr(error, "message", None) or error).casefold()
    message = re.sub(r"\b(?:req[_-]|request[ _-]id[=: ]+)[\w-]+", "request_id", message)
    message = re.sub(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "uuid", message)
    message = re.sub(r"\b\d{4}-\d\d-\d\d[t ][\d:.+z-]+", "timestamp", message)
    message = re.sub(r"\b(?:retry|try again)\s+(?:after|in)\s+[\d.]+s?", "retry later", message)
    message = re.sub(r"[$€£]\s*\d+(?:[.,]\d+)*", "amount", message)
    # Stable codes identify the cause independently of occurrence-specific diagnostics.
    if str(code) in {
        "credit_balance_exhausted",
        "insufficient_quota",
        "organization_usage_limit_exceeded",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "invalid_vad_evidence",
    }:
        message = ""
    payload = [
        context.provider,
        context.model,
        context.credential,
        context.endpoint,
        type(error).__name__,
        str(code),
        str(status),
        " ".join(message.split()),
    ]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def openai_resource_exhausted(error: Exception) -> bool:
    body = getattr(error, "body", None)
    body = body if isinstance(body, dict) else {}
    body = body.get("error", body)
    body = body if isinstance(body, dict) else {}
    code = str(getattr(error, "code", None) or body.get("code") or "")
    kind = getattr(error, "type", None) or body.get("type")
    if code in {"credit_balance_exhausted", "insufficient_quota"}:
        return True
    if code in {
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "rate_limit_exceeded",
        "slow_down",
    }:
        return False
    return kind == "insufficient_quota"


class PermissionStore:
    def __init__(self, path: Path | None = None):
        self.path = path or PROJECT_ROOT / ".cache" / "llm_error_permissions.json"

    def signatures(self) -> set[str]:
        with _store_lock:
            try:
                values = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(values, list):
                    return set()
                return {
                    value
                    for value in values
                    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                }
            except (OSError, ValueError, TypeError):
                return set()

    def allow(self, signatures: tuple[str, ...]) -> None:
        with _store_lock:
            self._write(self.signatures().union(signatures))

    def clear(self) -> None:
        with _store_lock:
            self._write(set())

    def _write(self, values: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(sorted(values), indent=2), encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)


class RecoverySession:
    def __init__(
        self,
        handler: Callable[[RecoveryRequest], RecoveryDecision],
        *,
        store: PermissionStore | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        self.handler = handler
        self.store = store or PermissionStore()
        self.cancel_check = cancel_check or (lambda: False)
        self.active_slots: dict[str, str] = {}
        self.cancelled = False
        self._file_keys = self._read_file_keys()

    @staticmethod
    def _read_file_keys() -> dict[str, str]:
        return {
            key: value
            for key, value in dotenv_values(PROJECT_ROOT / ".env").items()
            if "API_KEY" in key and value is not None
        }

    def reload_keys(self) -> None:
        updated = self._read_file_keys()
        for name in self._file_keys.keys() | updated.keys():
            previous = self._file_keys.get(name)
            # Preserve credentials explicitly set by the caller's environment.
            if os.getenv(name) not in {None, previous}:
                continue
            if name in updated:
                os.environ[name] = updated[name]
            else:
                os.environ.pop(name, None)
        self._file_keys = updated

    def check_cancelled(self) -> None:
        if self.cancelled or self.cancel_check():
            raise RecoveryCancelled("Execution cancelled while waiting for error recovery.")

    def run(
        self, function: Callable, *, context: RequestContext | None = None, isolated: bool = False
    ):
        if _inside.get() and not isolated:
            return function()
        inside_token = _inside.set(not isolated)
        request_token = _request.set(context or RequestContext())
        manual_token = _manual_slot.set(None)
        attempts_token = _attempts.set([])
        transient_attempts = 0
        try:
            while True:
                self.check_cancelled()
                try:
                    return function()
                except (ErrorPermitted, RecoveryCancelled):
                    raise
                except Exception as error:
                    current = _request.get() or RequestContext()
                    history = _attempts.get() or []
                    failures = [(ctx, exc) for ctx, exc in history if exc is not None]
                    if not isinstance(error, AttemptFailures):
                        failures.append((current, error))
                    if not failures:
                        failures = error.failures
                    from misinformation_simulation.llm.rate_limit import get_shared_limiter

                    limiter = get_shared_limiter()
                    cause = failures[-1][1]
                    status = getattr(cause, "status_code", None) or getattr(cause, "status", None)
                    code = str(getattr(cause, "code", ""))
                    temporary = status in {429, 503} or code in {
                        "429",
                        "503",
                        "rate_limit_exceeded",
                    }
                    next_attempt = _cooldown_transient_failure(
                        cause, limiter, temporary, transient_attempts
                    )
                    if next_attempt is not None:
                        transient_attempts = next_attempt
                        continue

                    # Validation can fail after a successful HTTP response. A manual retry
                    # still gives the other key one chance for the complete operation.
                    if _retry_remaining_credential(current, history, failures):
                        continue

                    self._resolve_recovery_decision(
                        current, failures, error, status, cause, limiter
                    )
                    transient_attempts = 0

        finally:
            _attempts.reset(attempts_token)
            _manual_slot.reset(manual_token)
            _request.reset(request_token)
            _inside.reset(inside_token)

    def openai_call(self, client, function: Callable, *, model: str):
        if not callable(getattr(client, "with_options", None)):
            _request.set(RequestContext("chatgpt", model))
            return function(client)
        keys, endpoint, pool, order, history = self._openai_credential_order(client)

        failures = []
        for slot in order:
            self.check_cancelled()
            context = RequestContext("chatgpt", model, slot, tuple(keys), endpoint)
            _request.set(context)
            try:
                request_client = client.with_options(api_key=keys[slot], max_retries=0)
                result = function(request_client)
            except (ErrorPermitted, RecoveryCancelled):
                raise
            except Exception as error:
                failures.append((context, error))
                if history is not None:
                    history.append((context, error))
                if len(failures) == 1 and not (
                    _manual_slot.get() is not None or openai_resource_exhausted(error)
                ):
                    break
                emit_event("credential_failed", credential=slot, provider="chatgpt", model=model)
            else:
                if history is not None:
                    history.append((context, None))
                self.active_slots[pool] = slot
                return result
        raise AttemptFailures(failures)

    def _resolve_recovery_decision(self, current, failures, error, status, cause, limiter):
        signatures = tuple(error_signature(exc, ctx) for ctx, exc in failures)
        message = (
            safe_message(AttemptFailures(failures)) if current.choices else (safe_message(error))
        )
        if set(signatures) <= self.store.signatures():
            emit_event(
                "error_permitted",
                provider=current.provider,
                model=current.model,
                credential=current.credential,
            )
            raise ErrorPermitted(message) from None
        request = RecoveryRequest(uuid4().hex, current, message, signatures, progress_context())
        emit_event("recovery_wait", provider=current.provider, model=current.model)
        resource_error = status in {401, 402, 403} or openai_resource_exhausted(cause)
        if limiter and resource_error:
            limiter.block(current.provider)
        try:
            decision = self.handler(request)
        finally:
            if limiter and resource_error:
                limiter.unblock(current.provider)
        self.check_cancelled()
        if decision.action == "allow":
            self.store.allow(signatures)
            emit_event(
                "error_permitted",
                provider=current.provider,
                model=current.model,
                credential=current.credential,
            )
            raise ErrorPermitted(message) from None
        if decision.action != "retry":
            self.cancelled = True
            raise RecoveryCancelled("Execution cancelled during error recovery.") from None
        self.reload_keys()
        _attempts.set([])
        _manual_slot.set(decision.credential or current.credential)
        emit_event("recovery_retry", credential=_manual_slot.get())

    def _openai_credential_order(self, client):
        from misinformation_simulation.llm.clients import openai_credentials

        keys = openai_credentials(client)
        for value in keys.values():
            register_secret(value)
        endpoint = str(getattr(client, "base_url", ""))
        pool = hashlib.sha256(repr((endpoint, keys)).encode()).hexdigest()
        preferred = _manual_slot.get() or self.active_slots.get(pool, "primary")
        order = [preferred] if preferred in keys else []
        order.extend(slot for slot in keys if slot not in order)
        history = _attempts.get()
        if history is not None:
            tried = {ctx.credential for ctx, _ in history if ctx.endpoint == endpoint}
            order = [slot for slot in order if slot not in tried]
        return keys, endpoint, pool, order, history


def set_request_context(context: RequestContext) -> None:
    _request.set(context)


def discard_successful_attempts() -> None:
    """Allow corrected generation on the working key without retrying failed credentials."""
    if history := _attempts.get():
        _attempts.set([(context, error) for context, error in history if error is not None])


@contextmanager
def use_recovery(session: RecoverySession):
    token = _session.set(session)
    try:
        yield session
    finally:
        _session.reset(token)


def get_recovery() -> RecoverySession | None:
    if session := _session.get():
        return session
    if os.getenv("LLM_ERROR_RECOVERY", "auto").lower() == "off":
        return None
    interactive = sys.stdin is not None and sys.stdin.isatty()
    get_shell = getattr(builtins, "get_ipython", None)
    interactive = interactive or (callable(get_shell) and get_shell() is not None)
    if not interactive:
        return None
    global _default_session
    if _default_session is None:
        from misinformation_simulation.llm.recovery_dialog import desktop_decision

        _default_session = RecoverySession(desktop_decision)
    return _default_session


def automatic_openai_call(client, function: Callable, *, model: str):
    """Keep credential failover available in unattended scripts without opening a dialog."""
    global _automatic_session
    if _automatic_session is None:
        _automatic_session = RecoverySession(lambda request: RecoveryDecision("cancel"))
    token = _attempts.set([])
    try:
        return _automatic_session.openai_call(client, function, model=model)
    except AttemptFailures as error:
        raise error.failures[-1][1] from None
    finally:
        _attempts.reset(token)


def _cooldown_transient_failure(cause, limiter, temporary, transient_attempts):
    if limiter and temporary and not openai_resource_exhausted(cause):
        if transient_attempts < 4:
            delay = min(60, 2 ** (transient_attempts + 1))
            headers = getattr(getattr(cause, "response", None), "headers", {})
            try:
                delay = max(delay, float(headers.get("retry-after", 0)))
            except (ValueError, TypeError):
                pass
            limiter.cooldown(delay)
            transient_attempts += 1
            emit_event(
                "retry_wait",
                delay=delay,
                attempt=transient_attempts,
                max_attempts=5,
                reason=safe_message(cause),
            )
            _attempts.set([])
            return transient_attempts


def _retry_remaining_credential(current, history, failures):
    if _manual_slot.get() is not None and current.provider == "chatgpt":
        tried = {ctx.credential for ctx, _ in history}
        remaining = [slot for slot in current.choices if slot not in tried]
        if remaining:
            _attempts.set(failures)
            _manual_slot.set(remaining[0])
            return True
