"""Audit domain-gate sensitivity using saved structures without new LLM requests."""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from misinformation_simulation.topic_drift.cluster_comparison import TransformerTextEmbedder
from misinformation_simulation.topic_drift.metrics import (
    DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT,
    DEFAULT_STDI_WEIGHTS,
    DEFAULT_VAD_CONTRIBUTION_WEIGHT,
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "output/interaction_graph/app_runs/simulation_ui_20261003_202828"
DESTINATION = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def score(row, suffix, theme):
    content = sum(
        weight * (theme if component == "theme_drift" else row[f"{component}_{suffix}"])
        for component, weight in DEFAULT_STDI_WEIGHTS.items()
    )
    contradiction = row[f"contradiction_drift_{suffix}"]
    vad = row[f"vad_drift_{suffix}"]
    with_contradiction = content + (
        (1 - content) * DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT * contradiction
    )
    return with_contradiction + ((1 - with_contradiction) * DEFAULT_VAD_CONTRIBUTION_WEIGHT * vad)


def normalize(value):
    return " ".join((value or "").casefold().split())


def main():
    inputs = {}
    comparisons = []
    steps = []
    for source_file in sorted(SOURCE.rglob("*steps.jsonl")):
        inputs[str(source_file.relative_to(ROOT))] = hashlib.sha256(
            source_file.read_bytes()
        ).hexdigest()
        previous = {}
        for line in source_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            steps.append(row)
            original = json.loads(row["metadata_original_json"])
            rewritten = json.loads(row["metadata_rewritten_json"])
            for suffix, reference in (
                ("vs_original", original),
                ("incremental", previous.get(row["news_id"], original)),
            ):
                comparisons.append((source_file.parent.name, row, suffix, reference, rewritten))
            previous[row["news_id"]] = rewritten

    topics = list(
        dict.fromkeys(
            structure["main_topic"]
            for _, _, _, reference, rewritten in comparisons
            for structure in (reference, rewritten)
        )
    )
    vectors = dict(zip(topics, TransformerTextEmbedder(MODEL).encode(topics), strict=True))
    results = []
    for scenario, row, suffix, reference, rewritten in comparisons:
        left, right = reference["main_topic"], rewritten["main_topic"]
        similarity = (
            1.0
            if normalize(left) == normalize(right)
            else float(
                np.clip(
                    np.dot(vectors[left], vectors[right])
                    / (np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right])),
                    0,
                    1,
                )
            )
        )
        left_domain = normalize(reference.get("topic_domain"))
        right_domain = normalize(rewritten.get("topic_domain"))
        gate = bool(left_domain and right_domain and left_domain != right_domain)
        saved_theme = row[f"theme_drift_{suffix}"]
        ungated_theme = round(1 - similarity, 6)
        reconstructed = score(row, suffix, saved_theme)
        if abs(reconstructed - row[f"stdi_{suffix}"]) > 0.000002:
            raise ValueError("Saved STDI does not match the current formula.")
        if abs(saved_theme - (1.0 if gate else ungated_theme)) > 0.000002:
            raise ValueError("Saved theme drift does not match the current cached encoder.")
        hypothetical = score(row, suffix, ungated_theme)
        results.append(
            {
                "scenario": scenario,
                "news_id": row["news_id"],
                "step_index": row["step_index"],
                "reference_type": suffix,
                "title": row["metadata_title"],
                "reference_domain": reference.get("topic_domain"),
                "rewritten_domain": rewritten.get("topic_domain"),
                "reference_main_topic": left,
                "rewritten_main_topic": right,
                "domain_gate_applied": gate,
                "topic_embedding_similarity": round(similarity, 6),
                "saved_theme_drift": saved_theme,
                "hypothetical_theme_drift_without_gate": ungated_theme,
                "saved_stdi": row[f"stdi_{suffix}"],
                "hypothetical_stdi_without_gate": round(hypothetical, 6),
                "stdi_reduction": round(reconstructed - hypothetical, 6),
            }
        )
    with (DESTINATION / "domain_gate_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    affected = [r for r in results if r["domain_gate_applied"]]
    summary = {
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "input_sha256": inputs,
        "embedding_model": MODEL,
        "steps": len(steps),
        "distinct_news": len({r["news_id"] for r in steps}),
        "comparisons": len(results),
        "affected_comparisons_by_reference": dict(Counter(r["reference_type"] for r in affected)),
        "formula_and_saved_theme_checks_passed": len(results),
        "scope": "Only theme drift is replaced; all other saved components are preserved.",
        "interpretation": (
            "Sensitivity analysis, not human validation or corrected official scores."
        ),
        "affected_comparisons": affected,
    }
    (DESTINATION / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
