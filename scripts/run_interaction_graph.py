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

from misinformation_simulation.datasets.newsdata import QUERY_METADATA_ROW_ID  # noqa: E402
from misinformation_simulation.simulation import (  # noqa: E402
    SimulationEdge,
    SimulationNode,
    run_news_interaction_graph,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a chained LLM interaction graph over news items.",
    )
    parser.add_argument("--input", required=True, help="Input CSV or JSONL file with news rows.")
    parser.add_argument(
        "--graph-config",
        required=True,
        help="Path to a JSON file describing nodes and optional edges.",
    )
    parser.add_argument("--text-column", default="description")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--news-id-column")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-requests-per-minute", type=int)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--allow-title-fallback", action="store_true")
    parser.add_argument("--topic-drift-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--topic-drift-provider", default="gemini")
    parser.add_argument("--output-dir", default="output/interaction_graph")
    parser.add_argument("--output-prefix", default="simulation")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates while the simulation is running.",
    )
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {resolved_path}. Pass a valid path in --input/INPUT."
        )
    if resolved_path.suffix.lower() == ".csv":
        df = pd.read_csv(resolved_path)
        if "article_id" in df.columns:
            article_ids = df["article_id"].fillna("").astype(str).str.strip().str.lower()
            df = df[article_ids != QUERY_METADATA_ROW_ID].copy()
        return df
    if resolved_path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(resolved_path, lines=resolved_path.suffix.lower() == ".jsonl")
    raise ValueError("Unsupported input file. Use CSV, JSON, or JSONL.")


def _load_graph_config(
    path: Path,
) -> tuple[list[SimulationNode], list[SimulationEdge] | None, str | None]:
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Graph config file not found: {resolved_path}. "
            "Pass a valid path in --graph-config/GRAPH_CONFIG."
        )
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    nodes = [SimulationNode(**node_payload) for node_payload in payload.get("nodes", [])]
    raw_edges = payload.get("edges")
    edges = None
    if raw_edges is not None:
        edges = [SimulationEdge(**edge_payload) for edge_payload in raw_edges]
    return nodes, edges, payload.get("start_node_id")


def main() -> None:
    args = _parse_args()
    df = _load_dataframe(Path(args.input))
    nodes, edges, start_node_id = _load_graph_config(Path(args.graph_config))
    progress_callback = print if args.verbose else None

    result = run_news_interaction_graph(
        df=df,
        nodes=nodes,
        edges=edges,
        start_node_id=start_node_id,
        text_column=args.text_column,
        title_column=args.title_column,
        news_id_column=args.news_id_column,
        max_rows=args.max_rows,
        sleep_seconds=args.sleep_seconds,
        max_requests_per_minute=args.max_requests_per_minute,
        retry_attempts=args.retry_attempts,
        allow_title_fallback=args.allow_title_fallback,
        topic_drift_model=args.topic_drift_model,
        topic_drift_provider=args.topic_drift_provider,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        progress_callback=progress_callback,
    )

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    if result.summary_path is not None:
        print(f"summary_path={result.summary_path}")
    if result.steps_path is not None:
        print(f"steps_path={result.steps_path}")


if __name__ == "__main__":
    main()
