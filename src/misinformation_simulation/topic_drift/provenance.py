from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from misinformation_simulation.llm.usage import record_cache_hit


def input_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def request_provenance(
    *,
    model: str,
    provider: str,
    base_url: str | None,
    version: str,
    prompt: str,
    raw_response: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    temperature_supported = provider == "gemini" or not model.lower().startswith(("gpt-5", "gpt-6"))
    return {
        "model": str(model),
        "provider": str(provider),
        "base_url": base_url,
        "prompt_version": version,
        "prompt_sha256": input_hash(prompt),
        "input_sha256": input_hash(inputs),
        "temperature_requested": 0.1,
        "temperature_sent": 0.1 if temperature_supported else None,
        "raw_response": raw_response,
    }


class EvaluationCache:
    """Caches canonical outputs in namespace folders, with legacy flat-cache reads."""

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        shared_directory: Path | str | None = None,
        compatibility: dict[str, dict] | None = None,
    ) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.shared_directory = Path(shared_directory) if shared_directory is not None else None
        self.compatibility = compatibility or {}
        if self.shared_directory is not None and not self.compatibility:
            raise ValueError("Shared caches require explicit stage compatibility contracts.")
        self._records: dict[str, dict[str, Any]] = {}

    def _shared_key(self, namespace: str, inputs: dict) -> str:
        identity = {"inputs": inputs, "compatibility": self.compatibility[namespace]}
        return f"{namespace}_{input_hash(identity)}"

    def _request_identity(self, namespace: str, inputs: dict) -> dict:
        if namespace not in self.compatibility or "provider" not in inputs:
            return inputs
        provider = str(inputs["provider"])
        endpoint = inputs.get("base_url") or {
            "chatgpt": "https://api.openai.com/v1",
            "gemini": "google-genai-default",
            "local": os.getenv("LOCAL_OPENAI_BASE_URL") or "http://127.0.0.1:11434/v1",
            "grok": "https://api.x.ai/v1",
            "deepseek": "https://api.deepseek.com",
            "openrouter": "https://openrouter.ai/api/v1",
        }.get(provider)
        identity = {key: value for key, value in inputs.items() if key != "base_url"}
        identity["endpoint_sha256"] = input_hash(endpoint)
        return identity

    @staticmethod
    def _reusable(namespace: str, output: dict) -> bool:
        if namespace == "extraction":
            return output.get("extraction_status") == "valid"
        if namespace.startswith("vad_"):
            return output.get("status") == "valid"
        if namespace == "rewrite":
            return bool(output.get("text", "").strip())
        return True

    def get(self, namespace: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
        inputs = self._request_identity(namespace, inputs)
        key = f"{namespace}_{input_hash(inputs)}"
        if key in self._records:
            record_cache_hit(namespace, source="memory")
            return self._records[key]
        if self.directory is not None:
            path = self.directory / namespace / f"{key}.json"
            if not path.exists():
                path = self.directory / f"{key}.json"
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    if namespace not in self.compatibility:
                        raise
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                if payload.get("inputs") != inputs:
                    if namespace not in self.compatibility:
                        raise ValueError("Evaluation cache input mismatch")
                    payload = {}
                compatible = namespace not in self.compatibility or (
                    payload.get("compatibility") == self.compatibility[namespace]
                    and isinstance(payload.get("output"), dict)
                    and payload.get("output_sha256") == input_hash(payload["output"])
                    and self._reusable(namespace, payload["output"])
                )
                if compatible:
                    self._records[key] = payload["output"]
                    record_cache_hit(namespace, source="local")
                    return payload["output"]
        if self.shared_directory is not None and namespace in self.compatibility:
            path = self.shared_directory / namespace / f"{self._shared_key(namespace, inputs)}.json"
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    valid = (
                        payload.get("inputs") == inputs
                        and payload.get("compatibility") == self.compatibility[namespace]
                        and isinstance(payload.get("output"), dict)
                        and payload.get("output_sha256") == input_hash(payload["output"])
                        and self._reusable(namespace, payload["output"])
                    )
                except (ValueError, KeyError, TypeError, AttributeError):
                    valid = False
                if valid:
                    self._save_local(namespace, inputs, payload["output"])
                    record_cache_hit(namespace, source="shared")
                    return payload["output"]
        return None

    def put(self, namespace: str, inputs: dict[str, Any], output: dict[str, Any]) -> None:
        inputs = self._request_identity(namespace, inputs)
        self._save_local(namespace, inputs, output)
        if (
            self.shared_directory is not None
            and namespace in self.compatibility
            and self._reusable(namespace, output)
        ):
            self._write(
                self.shared_directory / namespace / f"{self._shared_key(namespace, inputs)}.json",
                self._payload(namespace, inputs, output),
            )

    def _payload(self, namespace: str, inputs: dict, output: dict) -> dict:
        payload = {"inputs": inputs, "output": output}
        if namespace in self.compatibility:
            payload.update(
                compatibility=self.compatibility[namespace], output_sha256=input_hash(output)
            )
        return payload

    def _save_local(self, namespace: str, inputs: dict, output: dict) -> None:
        key = f"{namespace}_{input_hash(inputs)}"
        self._records[key] = output
        if self.directory is not None:
            self._write(
                self.directory / namespace / f"{key}.json", self._payload(namespace, inputs, output)
            )

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
