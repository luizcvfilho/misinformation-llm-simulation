from __future__ import annotations

import json
import runpy
from pathlib import Path

from misinformation_simulation.apps.interaction_graph_queue import add_graphs_from_directory
from misinformation_simulation.apps.interaction_graph_state import load_initial_graph_nodes
from misinformation_simulation.apps.interaction_graph_ui import build_linear_graph_payload
from misinformation_simulation.config.interaction_chains import (
    DEFAULT_INTERACTION_CHAINS_PATH,
    INTERACTION_CHAINS_ROOT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_and_legacy_sets_import_separately_with_distinct_personas() -> None:
    current_queue = []
    added, errors = add_graphs_from_directory(current_queue, DEFAULT_INTERACTION_CHAINS_PATH)
    assert errors == []
    assert added == [
        "01_nnnn.json",
        "02_cccc.json",
        "03_pppp.json",
        "04_dddd.json",
        "05_ccpp.json",
        "06_ppcc.json",
        "07_ddnn.json",
        "08_nndd.json",
    ]
    for item in current_queue:
        payload = build_linear_graph_payload(item["nodes"])
        payload["nodes"] = [
            {key: value for key, value in node.items() if value is not None}
            for node in payload["nodes"]
        ]
        saved = PROJECT_ROOT / DEFAULT_INTERACTION_CHAINS_PATH / f"{item['name']}.json"
        assert payload == json.loads(saved.read_text(encoding="utf-8"))
        assert len(item["nodes"]) == 4
    assert {node["personality_preset"] for node in current_queue[0]["nodes"]} == {"NeutralRelay"}
    assert build_linear_graph_payload(load_initial_graph_nodes()) == build_linear_graph_payload(
        current_queue[0]["nodes"]
    )

    legacy_queue = []
    legacy_added, legacy_errors = add_graphs_from_directory(
        legacy_queue, f"{INTERACTION_CHAINS_ROOT}/legacy"
    )
    assert len(legacy_added) == 16
    assert legacy_errors == []
    assert {node["personality_preset"] for node in legacy_queue[0]["nodes"]} == {
        "InvestigativeSkeptic"
    }


def test_generator_writes_current_set_without_touching_legacy(tmp_path) -> None:
    generator = runpy.run_path(str(PROJECT_ROOT / "scripts/generate_interaction_chains.py"))
    current = tmp_path / "neutral_relay"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    preserved = legacy / "01_ssss.json"
    preserved.write_bytes(b"preserved legacy config")
    generator["main"].__globals__.update(OUTPUT_DIR=current, PROJECT_ROOT=tmp_path)
    generator["main"]()

    committed = PROJECT_ROOT / DEFAULT_INTERACTION_CHAINS_PATH
    assert sorted(path.name for path in current.glob("*.json")) == sorted(
        path.name for path in committed.glob("*.json")
    )
    for path in current.glob("*.json"):
        assert json.loads(path.read_text()) == json.loads((committed / path.name).read_text())
    assert preserved.read_bytes() == b"preserved legacy config"
    assert not list(tmp_path.glob("*.json"))
