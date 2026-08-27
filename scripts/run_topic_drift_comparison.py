from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402
from misinformation_simulation.topic_drift.comparison_workflow import (  # noqa: E402
    SUPPORTED_COMPARISON_METHODS,
    compare_method_outputs,
    load_comparison_input,
    run_comparison_workflow,
    write_comparison_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare shared LLM-extracted topic structures with LLM or clustering."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV containing comparison pairs.")
    source.add_argument("--input-dir", type=Path, help="Existing comparison output directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=SUPPORTED_COMPARISON_METHODS, required=True)
    parser.add_argument("--original-text-column", default="original_text")
    parser.add_argument("--modified-text-column", default="modified_text")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--pair-id-column", default=None)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for a small validation run.",
    )
    parser.add_argument("--extraction-model", default=DEFAULT_LLM_MODEL.value)
    parser.add_argument("--extraction-provider", default=DEFAULT_LLM_PROVIDER.value)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-comparison-model", default=None)
    parser.add_argument("--llm-comparison-provider", default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Optional output directory to join against this run by pair_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = (
        pd.read_csv(args.input) if args.input is not None else load_comparison_input(args.input_dir)
    )
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("'--max-rows' must be greater than zero.")
        source = source.head(args.max_rows).copy()
    workflow = run_comparison_workflow(
        source,
        method=args.method,
        original_text_column=args.original_text_column,
        modified_text_column=args.modified_text_column,
        title_column=args.title_column,
        pair_id_column=args.pair_id_column,
        extraction_model=args.extraction_model,
        extraction_provider=args.extraction_provider,
        extraction_api_key=args.api_key,
        extraction_base_url=args.base_url,
        llm_comparison_model=args.llm_comparison_model,
        llm_comparison_provider=args.llm_comparison_provider,
        embedding_model=args.embedding_model,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    write_comparison_output(args.output_dir, workflow)
    print(f"Saved {len(workflow.results)} comparison rows to {args.output_dir}.")
    if args.compare_with is not None:
        previous = load_comparison_input(args.compare_with)
        previous_method = str(previous.get("comparison_method", pd.Series(["previous"])).iloc[0])
        comparison = compare_method_outputs(
            previous,
            workflow.results,
            left_label=previous_method,
            right_label=args.method,
        )
        comparison_path = args.output_dir / "method_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Saved paired method comparison to {comparison_path}.")


if __name__ == "__main__":
    main()
