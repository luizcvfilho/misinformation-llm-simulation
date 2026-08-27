from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.datasets.loading import read_dataset  # noqa: E402
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402
from misinformation_simulation.topic_drift import (  # noqa: E402
    build_manual_stdi_evaluation_dataset,
    fit_manual_stdi_regression,
    generate_metric_rewrites,
    score_manual_stdi_evaluation_pairs,
    summarize_manual_stdi_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, score, and calibrate a manually reviewed STDI evaluation set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "raw" / "newsdata_news.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output") / "stdi_manual_evaluation"
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--article-id-column", default="article_id")
    parser.add_argument(
        "--generate", action="store_true", help="Generate one rewrite per news item."
    )
    parser.add_argument("--score", action="store_true", help="Calculate STDI for generated pairs.")
    parser.add_argument(
        "--fit", action="store_true", help="Fit weights using manual_expected_stdi."
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL.value)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER.value)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-requests-per-minute", type=int, default=450)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--without-vad", action="store_false", dest="include_vad")
    parser.set_defaults(include_vad=True)
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def _write_review_guide(path: Path) -> None:
    path.write_text(
        "# STDI Manual Review\n\n"
        "Each row compares `original_text` with `modified_text`. The `target_metric` identifies "
        "the type of controlled change requested from the rewrite model.\n\n"
        "Fill these columns in `scored_stdi_pairs.csv` after reviewing the pair:\n\n"
        "- `manual_target_metric_score`: intensity from 0 to 1 of the requested metric change.\n"
        "- `manual_expected_stdi`: overall semantic drift from 0 to 1, based on your judgement.\n"
        "- `manual_review_status`: set to `reviewed` when the annotation is complete.\n"
        "- `manual_notes`: explain ambiguity, failed rewrites, or unexpected side effects.\n\n"
        "Use the same rubric across all 50 rows. The regression uses only non-empty "
        "`manual_expected_stdi` values and the STDI component values calculated by the pipeline.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "manual_stdi_pairs.csv"
    generated_path = args.output_dir / "generated_stdi_pairs.csv"
    scored_path = args.output_dir / "scored_stdi_pairs.csv"
    summary_path = args.output_dir / "stdi_evaluation_summary.csv"
    weights_path = args.output_dir / "calibrated_stdi_weights.csv"
    metrics_path = args.output_dir / "calibration_metrics.json"
    guide_path = args.output_dir / "manual_review_guide.md"
    _write_review_guide(guide_path)

    if args.fit:
        if not scored_path.exists():
            raise FileNotFoundError(f"No scored dataset found at {scored_path}.")
        scored = pd.read_csv(scored_path)
        calibration = fit_manual_stdi_regression(
            scored,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        calibration.weights_frame().to_csv(weights_path, index=False)
        metrics_path.write_text(
            json.dumps({"status": "success", **calibration.to_dict()}, indent=2),
            encoding="utf-8",
        )
        print(f"Saved calibrated weights to {weights_path}")
        print(f"Saved calibration metrics to {metrics_path}")
        return

    source_dataset = read_dataset(args.input)
    pairs = build_manual_stdi_evaluation_dataset(
        source_dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
        text_column=args.text_column,
        title_column=args.title_column,
        article_id_column=args.article_id_column,
    )
    pairs.to_csv(pairs_path, index=False)
    print(f"Saved {len(pairs)} manual review pairs to {pairs_path}")
    excluded_source_counts = pairs.attrs.get("excluded_source_counts", {})
    if excluded_source_counts:
        summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(excluded_source_counts.items())
        )
        print(f"Excluded source candidates: {summary}")
    print(f"Saved review guide to {guide_path}")

    generated = pairs
    if args.generate:
        generated = generate_metric_rewrites(
            pairs,
            model=args.model,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            max_requests_per_minute=args.max_requests_per_minute,
            retry_attempts=args.retry_attempts,
            progress_callback=print,
        )
        generated.to_csv(generated_path, index=False)
        print(f"Saved generated rewrites to {generated_path}")

    if args.score:
        if not args.generate:
            if not generated_path.exists():
                raise FileNotFoundError(
                    f"No generated pairs found at {generated_path}. Run with --generate first."
                )
            generated = pd.read_csv(generated_path)
        scored = score_manual_stdi_evaluation_pairs(
            generated,
            model=args.model,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            max_requests_per_minute=args.max_requests_per_minute,
            retry_attempts=args.retry_attempts,
            include_vad=args.include_vad,
            progress_callback=print,
        )
        scored.to_csv(scored_path, index=False)
        summarize_manual_stdi_evaluation(scored).to_csv(summary_path, index=False)
        print(f"Saved scored pairs to {scored_path}")
        print(f"Saved score summary to {summary_path}")


if __name__ == "__main__":
    main()
