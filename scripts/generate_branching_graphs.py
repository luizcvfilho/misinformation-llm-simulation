from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402
from misinformation_simulation.simulation.generation import generate_branching_graphs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prefix-sharing trees from persona sequences."
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True, help="For example: CCCC CCPP PPPP PPCC"
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL.value)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER.value)
    parser.add_argument("--output-dir", default="data/graphs/branching_generated")
    args = parser.parse_args()
    try:
        graphs = generate_branching_graphs(args.sequences, model=args.model, provider=args.provider)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, graph in enumerate(graphs, start=1):
        root = graph["start_node_id"].removeprefix("prefix_").lower()
        path = output_dir / f"{index:02d}_{root}_tree.json"
        if path.exists():
            parser.error(f"Graph config already exists: {path}. Choose another output directory.")
        outputs.append((path, graph))
    for path, graph in outputs:
        with path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
