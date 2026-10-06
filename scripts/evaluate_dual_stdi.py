from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402
from misinformation_simulation.topic_drift.comparison_workflow import (  # noqa: E402
    run_comparison_workflow,
    write_comparison_output,
)


def build_validation_report(canonical: pd.DataFrame, repetitions: pd.DataFrame) -> dict:
    report = {
        "pairs": len(canonical),
        "complete_pairs": int((canonical["comparison_status"] == "valid").sum()),
        "status_counts": canonical["comparison_status"].value_counts().to_dict(),
        "parameters": {"polarity": 0.20, "duration": 0.20, "branch_weights": [0.5, 0.5]},
        "interpretation": (
            "Diagnostics of informational change, not factual verification or proof of superiority."
        ),
        "mean_method_gap": pd.to_numeric(canonical["method_gap"], errors="coerce").mean(),
        "human_reviewed_pairs": 0,
    }
    reviewed = (
        canonical.get("manual_review_status", pd.Series("pending", index=canonical.index))
        == "reviewed"
    )
    report["human_reviewed_pairs"] = int(reviewed.sum())
    target = pd.to_numeric(
        canonical.get("manual_expected_stdi", pd.Series(dtype=float)), errors="coerce"
    )
    if reviewed.any() and target.notna().any():
        report["reviewed_mean_absolute_errors"] = {}
        for column in ("historical_embedding_stdi", "embedding_stdi", "llm_judge_stdi", "stdi"):
            predicted = pd.to_numeric(canonical[column], errors="coerce")
            valid = reviewed & target.notna() & predicted.notna()
            report["reviewed_mean_absolute_errors"][column] = {
                "pairs": int(valid.sum()),
                "mae": (predicted[valid] - target[valid]).abs().mean(),
            }
    if not repetitions.empty:
        repetitions = repetitions.copy()
        repetitions["llm_judge_stdi"] = pd.to_numeric(
            repetitions["llm_judge_stdi"], errors="coerce"
        )
        grouped = repetitions.groupby("pair_id")["llm_judge_stdi"]
        summary = grouped.agg(["count", "mean", "std", "min", "max"])
        report["uncached_judge_variability"] = summary.reset_index().to_dict(orient="records")
    qualifier_checks = []
    for _, row in canonical.iterrows():
        serialized = row.get("dual_evaluation_json")
        if not isinstance(serialized, str):
            continue
        evaluation = json.loads(serialized)
        relations = evaluation.get("embedding", {}).get("relations", [])
        if len(relations) != 1:
            continue
        comparison = relations[0]
        for expected_column, actual in (
            ("expected_delta_p", comparison["delta_p"]),
            ("expected_duration_distance", comparison["duration"]["distance"]),
            ("expected_numeric_distance", comparison.get("numeric", {}).get("distance")),
        ):
            expected = row.get(expected_column)
            if expected is not None and pd.notna(expected):
                qualifier_checks.append(
                    {
                        "pair_id": row["pair_id"],
                        "check": expected_column,
                        "expected": float(expected),
                        "actual": actual,
                        "matches": actual is not None and abs(actual - float(expected)) <= 1e-9,
                    }
                )
    report["qualifier_policy_checks"] = qualifier_checks
    # JSON null denotes unavailable statistics, including singleton standard deviations.
    return json.loads(pd.Series(report).to_json())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved rewrites with fixed dual STDI settings."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV pairs or JSON diagnostic fixtures."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "held_out", "all"), default="development"
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--judge-repeats", type=int, default=0, help="Additional uncached judgments on a sample."
    )
    parser.add_argument("--repeat-sample-size", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL.value)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER.value)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    if (
        args.judge_repeats < 0
        or args.repeat_sample_size <= 0
        or (args.max_pairs is not None and args.max_pairs <= 0)
    ):
        parser.error("Pair limits/sample sizes must be positive and judge repeats nonnegative.")
    if args.input.suffix.lower() == ".json":
        source = pd.DataFrame(json.loads(args.input.read_text(encoding="utf-8"))["cases"])
    else:
        source = pd.read_csv(args.input)
    if "split" not in source and args.split == "held_out":
        parser.error("Held-out evaluation requires an explicit split column.")
    if "split" in source and args.split != "all":
        source = source.loc[source["split"] == args.split].copy()
    if args.max_pairs is not None:
        source = source.head(args.max_pairs)
    if source.empty:
        parser.error("No pairs match the requested evaluation selection.")
    options = {
        "method": "dual",
        "extraction_model": args.model,
        "extraction_provider": args.provider,
        "extraction_base_url": args.base_url,
        "cache_dir": args.output_dir / "evaluation_cache",
        "progress_callback": print,
    }
    canonical = run_comparison_workflow(source, **options)
    write_comparison_output(args.output_dir / "canonical", canonical)
    repetitions = []
    for repeat in range(args.judge_repeats):
        workflow = run_comparison_workflow(
            canonical.results.head(args.repeat_sample_size),
            **options,
            uncached_judge=True,
        )
        workflow.results["repeat"] = repeat + 1
        write_comparison_output(args.output_dir / f"repeat_{repeat + 1:02d}", workflow)
        repetitions.append(
            workflow.results[["pair_id", "repeat", "llm_judge_stdi", "llm_judge_status"]]
        )
    repeated = pd.concat(repetitions, ignore_index=True) if repetitions else pd.DataFrame()
    report = build_validation_report(canonical.results, repeated)
    report["selection"] = args.split if "split" in source else "unassigned"
    report["sources"] = str(args.input)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved validation diagnostics to {args.output_dir}.")


if __name__ == "__main__":
    main()
