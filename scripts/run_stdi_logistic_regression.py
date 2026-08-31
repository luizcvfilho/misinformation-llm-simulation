from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.topic_drift.stdi_logistic_regression import (  # noqa: E402
    run_stdi_logistic_regression_analysis,
)


class ProgressReporter:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.started_at = time.monotonic()

    def rewrite(self, completed: int, total: int, status: str) -> None:
        self._show(completed, total, status)

    def annotation(self, completed: int, total: int) -> None:
        self._show(completed, total, "processed")

    def _show(self, completed: int, total: int, status: str) -> None:
        elapsed = time.monotonic() - self.started_at
        average = elapsed / completed
        remaining = max(total - completed, 0) * average
        print(
            f"[{self.stage}] {completed}/{total} | {status} | "
            f"elapsed: {_format_duration(elapsed)} | ETA: {_format_duration(remaining)}",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    total_seconds = round(max(seconds, 0))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite false news and analyze STDI metrics with logistic regression."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV containing false news.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--true-input", type=Path, help="CSV containing independently true news.")
    parser.add_argument("--text-column", default="original_article_text")
    parser.add_argument("--true-text-column")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--topic-column", default="subject")
    parser.add_argument("--provider", default="chatgpt")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-requests-per-minute", type=int, default=450)
    parser.add_argument("--skip-rewrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewrite_progress = ProgressReporter("Truthification")
    stdi_progress = ProgressReporter("STDI")
    false_news = pd.read_csv(args.input)
    true_news = pd.read_csv(args.true_input) if args.true_input else None
    outputs = run_stdi_logistic_regression_analysis(
        false_news,
        output_dir=args.output_dir,
        true_news=true_news,
        text_column=args.text_column,
        true_text_column=args.true_text_column,
        title_column=args.title_column,
        topic_column=args.topic_column,
        max_rows=args.max_rows,
        skip_rewrite=args.skip_rewrite,
        log=lambda message: print(f"[Flow] {message}", flush=True),
        rewrite_kwargs={
            "provider": args.provider,
            "model": args.model,
            "max_requests_per_minute": args.max_requests_per_minute,
            "progress_callback": rewrite_progress.rewrite,
        },
        annotation_kwargs={
            "provider": args.provider,
            "model": args.model,
            "max_requests_per_minute": args.max_requests_per_minute,
            "progress_callback": stdi_progress.annotation,
        },
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
