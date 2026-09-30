from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from misinformation_simulation.analysis.interaction_graph_visualization import (  # noqa: E402
    create_static_figures,
    export_analysis_tables,
    load_interaction_graph_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible analyses for persisted interaction-graph runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "interaction_graph" / "app_runs",
        help="Directory containing persisted '*_steps.jsonl' files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "interaction_graph" / "analysis",
        help="Directory that receives figures, tables, and the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_interaction_graph_runs(args.runs_dir)
    output_paths = {
        **create_static_figures(runs, args.output_dir),
        **export_analysis_tables(runs, args.output_dir),
    }
    print(f"Loaded {len(runs.source_paths)} run files and {len(runs.steps)} persisted steps.")
    for name, path in output_paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
