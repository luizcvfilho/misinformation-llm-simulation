from __future__ import annotations

import ast
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from misinformation_simulation.config import prompts
from misinformation_simulation.topic_drift.provenance import input_hash


def runtime_cache_contracts() -> dict[str, dict]:
    """Fingerprint only each cached stage's dependencies, independently of Git state."""
    root = Path(__file__).resolve().parents[1]

    def source(path: str) -> str:
        return input_hash((root / path).read_text(encoding="utf-8"))

    common = {
        path: source(path)
        for path in (
            "llm/clients.py",
            "llm/retry.py",
            "llm/recovery.py",
            "llm/credentials.py",
        )
    }
    dependencies = {
        "rewrite": ["datasets/selection.py"],
        "extraction": [
            "topic_drift/extraction.py",
            "topic_drift/models.py",
            "topic_drift/qualifiers.py",
        ],
        "judge": [
            "topic_drift/semantic_comparison.py",
            "topic_drift/models.py",
            "topic_drift/manual_evaluation_definitions.py",
        ],
        "vad_llm": [
            "text_metrics/llm_vad.py",
            "text_metrics/vad_evaluation.py",
            "text_metrics/evidence_alignment.py",
        ],
        "vad_model": ["text_metrics/vad.py", "text_metrics/vad_evaluation.py"],
    }
    contracts = {
        namespace: {
            "cache_schema": "stage_cache_v1",
            "implementation": {
                **(common if namespace != "vad_model" else {}),
                **{path: source(path) for path in paths},
            },
        }
        for namespace, paths in dependencies.items()
    }
    for namespace, contract in contracts.items():
        packages = (
            ("torch", "transformers") if namespace == "vad_model" else ("openai", "google-genai")
        )
        contract["libraries"] = {}
        for package in packages:
            try:
                contract["libraries"][package] = version(package)
            except PackageNotFoundError:
                contract["libraries"][package] = "unavailable"
    graph_source = (root / "simulation/graph.py").read_text(encoding="utf-8")
    rewrite_function = next(
        node
        for node in ast.parse(graph_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_rewrite"
    )
    contracts["rewrite"]["generator"] = input_hash(ast.dump(rewrite_function))
    contracts["extraction"]["prompt_version"] = prompts.STRUCTURED_EXTRACTION_VERSION
    contracts["extraction"]["prompt"] = input_hash(
        [
            prompts.STRUCTURED_TOPIC_PROMPT_TEMPLATE,
            prompts.TOPIC_DRIFT_SYSTEM_INSTRUCTION,
        ]
    )
    contracts["judge"]["prompt_version"] = prompts.STRUCTURED_JUDGE_VERSION
    contracts["judge"]["prompt"] = input_hash(
        [
            prompts.STRUCTURED_JUDGE_PROMPT_TEMPLATE,
            prompts.SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
        ]
    )
    return contracts
