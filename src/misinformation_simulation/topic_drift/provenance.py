from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4


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
    """Caches canonical saved outputs by complete input/configuration hashes."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self._records: dict[str, dict[str, Any]] = {}

    def get(self, namespace: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
        key = f"{namespace}_{input_hash(inputs)}"
        if key in self._records:
            return self._records[key]
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("inputs") != inputs:
                    raise ValueError("Evaluation cache input mismatch")
                self._records[key] = payload["output"]
                return payload["output"]
        return None

    def put(self, namespace: str, inputs: dict[str, Any], output: dict[str, Any]) -> None:
        key = f"{namespace}_{input_hash(inputs)}"
        self._records[key] = output
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{key}.json"
            temporary = self.directory / f".{key}.{uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(
                    {"inputs": inputs, "output": output},
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
