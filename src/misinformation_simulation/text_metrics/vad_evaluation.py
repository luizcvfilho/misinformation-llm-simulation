from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
from misinformation_simulation.text_metrics.llm_vad import (
    LLM_VAD_RUBRIC,
    LLM_VAD_SYSTEM_INSTRUCTION,
    LLM_VAD_VERSION,
    LLMVADAssessment,
    LLMVADScorer,
)
from misinformation_simulation.text_metrics.vad import (
    DEFAULT_VAD_MODEL_NAME,
    VAD_DIMENSIONS,
    VADScore,
    predict_text_vad,
)

VAD_METHODS = ("model", "llm", "dual")
VAD_EVALUATION_VERSION = "vad_evaluation_v1"


def validate_vad_method(method: str) -> str:
    if method not in VAD_METHODS:
        raise ValueError("'vad_method' must be 'model', 'llm' or 'dual'.")
    return method


def evaluation_score(evaluation: dict, method: str) -> VADScore:
    return VADScore(**evaluation.get(method, {}).get("score", dict.fromkeys(VAD_DIMENSIONS)))


def _score_status(score: VADScore) -> str:
    partial = False
    for value in asdict(score).values():
        if value is None:
            partial = True
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("VAD dimensions must be finite numbers or missing values.")
    return "partial" if partial else "valid"


class VADTextEvaluator:
    """Evaluate selected scorers once per exact text; never average raw VAD vectors."""

    def __init__(
        self,
        *,
        method="model",
        model_bundle=None,
        model_scorer=None,
        llm_scorer=None,
        llm_model=DEFAULT_LLM_MODEL,
        llm_provider=DEFAULT_LLM_PROVIDER,
        api_key=None,
        base_url=None,
        before_request_hook=None,
        cache=None,
    ):
        self.method = validate_vad_method(method)
        self.model_bundle = model_bundle
        self.model_scorer = model_scorer
        self.llm_scorer = llm_scorer
        self.llm_model = str(llm_model)
        self.llm_provider = str(llm_provider)
        self.api_key = api_key
        self.base_url = base_url
        self.before_request_hook = before_request_hook
        if cache is None:
            from misinformation_simulation.topic_drift.provenance import EvaluationCache

            cache = EvaluationCache()
        self.cache = cache
        self.records: dict[str, dict] = {}
        self._llm = None

    def evaluate(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide non-empty text for VAD evaluation.")
        if text in self.records:
            return self.records[text]
        record = {
            "version": VAD_EVALUATION_VERSION,
            "method": self.method,
            "model": {"status": "not_requested"},
            "llm": {"status": "not_requested"},
        }
        for method in ("model", "llm") if self.method == "dual" else (self.method,):
            if method == "model":
                model_name = getattr(self.model_bundle, "model_name", DEFAULT_VAD_MODEL_NAME)
                inputs = dict(
                    text=text,
                    model=model_name,
                    version=VAD_EVALUATION_VERSION,
                    max_length=getattr(self.model_bundle, "max_length", 512),
                )
                custom = self.model_scorer is not None
            else:
                inputs = dict(
                    text=text,
                    model=self.llm_model,
                    provider=self.llm_provider,
                    base_url=self.base_url,
                    version=LLM_VAD_VERSION,
                    rubric=LLM_VAD_RUBRIC,
                    system_instruction=LLM_VAD_SYSTEM_INSTRUCTION,
                )
                custom = self.llm_scorer is not None
            try:
                saved = None if custom else self.cache.get(f"vad_{method}", inputs)
                if saved is not None:
                    _score_status(VADScore(**saved["score"]))
                    record[method] = {**saved, "cache_hit": True}
                    continue
                if method == "model":
                    score = predict_text_vad(
                        text, model_bundle=self.model_bundle, scorer=self.model_scorer
                    )
                    output = dict(
                        score=asdict(score), model="custom_scorer" if custom else model_name
                    )
                else:
                    if self.llm_scorer is not None:
                        assessment = self.llm_scorer(text)
                    else:
                        if self._llm is None:
                            self._llm = LLMVADScorer(
                                model=self.llm_model,
                                provider=self.llm_provider,
                                api_key=self.api_key,
                                base_url=self.base_url,
                                before_request_hook=self.before_request_hook,
                            )
                        assessment = self._llm.assess(text)
                    if isinstance(assessment, LLMVADAssessment):
                        score = assessment.score
                        output = asdict(assessment)
                    else:
                        score = assessment
                        output = {"score": asdict(score)}
                    output.update(
                        model=self.llm_model,
                        provider=self.llm_provider,
                        rubric_version=LLM_VAD_VERSION,
                        attempts=self._llm.last_attempts if self._llm else [],
                    )
                    if any(
                        value is not None and not 1 <= value <= 5
                        for value in asdict(score).values()
                    ):
                        raise ValueError("LLM VAD values must be within [1,5].")
                output["status"] = _score_status(score)
                if not custom:
                    self.cache.put(f"vad_{method}", inputs, output)
                record[method] = {**output, "cache_hit": False}
            except Exception as error:
                record[method] = dict(
                    status="failed",
                    error=str(error),
                    exception_type=type(error).__name__,
                    attempts=self._llm.last_attempts if method == "llm" and self._llm else [],
                )
        self.records[text] = record
        return record


def _drift(original: VADScore, modified: VADScore) -> dict:
    result = {}
    for dimension in VAD_DIMENSIONS:
        a, b = getattr(original, dimension), getattr(modified, dimension)
        valid = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        result[f"{dimension}_delta"] = (b - a) if valid else None
        result[f"{dimension}_drift"] = min(abs(b - a) / 4, 1.0) if valid else None
    complete = all(result[f"{dimension}_drift"] is not None for dimension in VAD_DIMENSIONS)
    result["status"] = "valid" if complete else "partial"
    result["vad_drift"] = (
        sum(result[f"{dimension}_drift"] for dimension in VAD_DIMENSIONS) / 3 if complete else None
    )
    return result


def compare_vad_evaluations(original: dict, modified: dict, *, method: str) -> dict:
    validate_vad_method(method)
    result = {"method": method, "version": VAD_EVALUATION_VERSION}
    for source in ("model", "llm"):
        requested = method == "dual" or method == source
        result[source] = (
            _drift(evaluation_score(original, source), evaluation_score(modified, source))
            if requested
            else {"status": "not_requested", "vad_drift": None}
        )
    dual = {"status": "not_requested", "vad_drift": None}
    if method == "dual":
        dual = {"status": "partial", "vad_drift": None}
        for key in ("vad_drift", *(f"{dimension}_drift" for dimension in VAD_DIMENSIONS)):
            a, b = result["model"].get(key), result["llm"].get(key)
            dual[key] = (a + b) / 2 if a is not None and b is not None else None
        if dual["vad_drift"] is not None:
            dual["status"] = "valid"
        dual["method_gap"] = (
            abs(result["model"]["vad_drift"] - result["llm"]["vad_drift"])
            if dual["status"] == "valid"
            else None
        )
    result["dual"] = dual
    return result


def vad_for_branch(evaluation: dict, branch: str, *, paired: bool = False) -> dict:
    source = {"embedding": "model", "llm_judge": "llm", "dual": "dual"}.get(branch)
    if source is None:
        raise ValueError("Unknown STDI branch for VAD selection.")
    source = source if paired and evaluation["method"] == "dual" else evaluation["method"]
    return {**evaluation[source], "source": source}


def vad_pair_columns(evaluation: dict, *, suffix: str = "") -> dict:
    tail = f"_{suffix}" if suffix else ""
    result = {f"vad_method{tail}": evaluation["method"]}
    for source in VAD_METHODS:
        for key in ("status", "vad_drift", *(f"{dimension}_drift" for dimension in VAD_DIMENSIONS)):
            result[f"{key if key == 'vad_drift' else 'vad_' + key}_{source}{tail}"] = evaluation[
                source
            ].get(key)
    return result
