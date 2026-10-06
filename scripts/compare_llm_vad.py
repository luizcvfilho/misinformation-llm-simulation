"""Compare current BERT VAD with independent LLM ratings of saved full texts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from misinformation_simulation.analysis.llm_vad_comparison import (  # noqa: E402
    evaluate_comparison,
    load_comparison_pairs,
    prepare_comparison,
    write_comparison_report,
)
from misinformation_simulation.analysis.llm_vad_dashboard import (  # noqa: E402
    export_llm_vad_dashboard,
)
from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_LLM_MODEL))
    parser.add_argument("--provider", default=str(DEFAULT_LLM_PROVIDER))
    parser.add_argument("--scope", choices=("context", "simulation", "both"), default="both")
    parser.add_argument("--news-count", type=int, default=5)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "report", "all"), default="all")
    parser.add_argument(
        "--context-input",
        type=Path,
        default=PROJECT_ROOT / "data/synthetic/vad_context_contrast_news.csv",
    )
    parser.add_argument(
        "--simulation-input",
        type=Path,
        default=PROJECT_ROOT / "output/audit/SimulationVADModelComparison" / "step_comparison.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output/audit/LLMVADComparison_20261006"
    )
    parser.add_argument(
        "--bert-cache",
        type=Path,
        default=PROJECT_ROOT / ".cache/vad_lexicon_comparison" / "current_bert_saved_tests.json",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    pairs = load_comparison_pairs(
        args.context_input, args.simulation_input, news_count=args.news_count, scope=args.scope
    )
    sources = []
    if args.scope in {"context", "both"}:
        sources.append(args.context_input)
    if args.scope in {"simulation", "both"}:
        sources.append(args.simulation_input)
    prepare_comparison(
        pairs, args.output_dir, model=args.model, provider=args.provider, sources=sources
    )
    unique_texts = len(set(pairs.reference_text) | set(pairs.compared_text))
    print(f"Prepared {len(pairs)} pairs and {unique_texts} unique full texts.", flush=True)
    if args.stage in {"evaluate", "all"}:
        try:
            evaluate_comparison(args.output_dir)
        finally:
            manifest = write_comparison_report(args.output_dir, bert_cache_path=args.bert_cache)
            export_llm_vad_dashboard(args.output_dir)
            print(f"Report statuses: {manifest['llm_status_counts']}", flush=True)
    elif args.stage == "report":
        write_comparison_report(args.output_dir, bert_cache_path=args.bert_cache)
        export_llm_vad_dashboard(args.output_dir)


if __name__ == "__main__":
    main()
