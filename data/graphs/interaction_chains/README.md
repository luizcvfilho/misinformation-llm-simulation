# Interaction chains

Each JSON file is a four-step directed chain compatible with the interaction graph CLI and UI.
Every position is a separate node, including positions that share a personality.

| Code | Personality |
| --- | --- |
| C | Conservative right |
| P | Progressive left |
| D | Conspiracy denialist |
| S | Investigative skeptic |
| E | Emotional amplifier |
| M | Conciliatory communicator |

| Files | Comparison |
| --- | --- |
| `01_ssss.json` | Repeated skeptical rewriting baseline |
| `02_cccc.json`, `03_pppp.json`, `04_dddd.json` | Homogeneous sequences |
| `05_ccpp.json`, `06_ppcc.json` | Ideological blocks, reversed |
| `07_cpcp.json`, `08_pcpc.json` | Alternating ideological perspectives |
| `09_ddss.json`, `10_ssdd.json` | Conspiratorial and skeptical blocks, reversed |
| `11_dsds.json`, `12_sdsd.json` | Alternating conspiratorial and skeptical perspectives |
| `13_ddes.json`, `14_ddse.json` | Emotional amplifier before versus after the skeptic |
| `15_cmpc.json`, `16_cpmc.json` | Conciliatory communicator between perspectives versus after their encounter |

All files use `chatgpt` and `gpt-5.6-luna`, matching the current project defaults. The
skeptical personality requests attention to evidence, but does not perform fact checking.
These chains measure changes in rewritten text; they do not model belief, trust, sharing,
or exposure from multiple neighbors.

Example from the project root:

```powershell
uv run python scripts/run_interaction_graph.py --input data/graphs/graph_news.csv --graph-config data/graphs/interaction_chains/05_ccpp.json --output-prefix ccpp
```

Use a different `--output-prefix` for each scenario so the output files do not overwrite
one another. To regenerate the JSON files from the current personality presets, run:

```powershell
uv run python scripts/generate_interaction_chains.py
```
