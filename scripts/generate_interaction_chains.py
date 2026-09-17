from __future__ import annotations

import json
from pathlib import Path

from misinformation_simulation.enums import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DefaultPersonality,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "graphs" / "interaction_chains"

PERSONALITIES = {
    "C": ("Conservative", DefaultPersonality.ConservativeRight),
    "P": ("Progressive", DefaultPersonality.ProgressiveLeft),
    "D": ("Conspiratorial", DefaultPersonality.ConspiracyDenialist),
    "S": ("Investigative skeptic", DefaultPersonality.InvestigativeSkeptic),
    "E": ("Emotional amplifier", DefaultPersonality.EmotionalAmplifier),
    "M": ("Conciliatory communicator", DefaultPersonality.ConciliatoryCommunicator),
}

SCENARIOS = (
    "SSSS",
    "CCCC",
    "PPPP",
    "DDDD",
    "CCPP",
    "PPCC",
    "CPCP",
    "PCPC",
    "DDSS",
    "SSDD",
    "DSDS",
    "SDSD",
    "DDES",
    "DDSE",
    "CMPC",
    "CPMC",
)


def build_graph(sequence: str) -> dict[str, object]:
    nodes = []
    for position, code in enumerate(sequence, start=1):
        label, personality = PERSONALITIES[code]
        nodes.append(
            {
                "node_id": f"p{position}",
                "label": f"{position}. {label}",
                "provider": str(DEFAULT_LLM_PROVIDER),
                "model": str(DEFAULT_LLM_MODEL),
                "personality": str(personality),
            }
        )

    return {
        "start_node_id": "p1",
        "nodes": nodes,
        "edges": [
            {"source": f"p{position}", "target": f"p{position + 1}"}
            for position in range(1, len(sequence))
        ],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for number, sequence in enumerate(SCENARIOS, start=1):
        path = OUTPUT_DIR / f"{number:02d}_{sequence.lower()}.json"
        path.write_text(json.dumps(build_graph(sequence), indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
