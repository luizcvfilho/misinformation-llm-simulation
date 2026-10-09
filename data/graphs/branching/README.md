# Prefix-sharing versions of the linear chain sets

These configurations retain every four-step persona sequence from
`../interaction_chains/`, grouping sequences with the same initial persona into
one rooted tree. Identical complete prefixes share nodes; branches never merge
again after diverging. Models, providers, and full personality prompts match
the corresponding linear configurations. The original linear files remain available.

## Current neutral relay set

Import `data/graphs/branching/neutral_relay` through **Graph queue -> Add graphs
from a folder**. Select this subfolder directly; folder import is not recursive.

| File | Root-to-leaf sequences | Shared prefix | Unique nodes |
| --- | --- | --- | --- |
| `neutral_relay/01_n_tree.json` | NNNN, NNDD | NN | 6 |
| `neutral_relay/02_c_tree.json` | CCCC, CCPP | CC | 6 |
| `neutral_relay/03_p_tree.json` | PPPP, PPCC | PP | 6 |
| `neutral_relay/04_d_tree.json` | DDDD, DDNN | DD | 6 |

The eight original paths use 24 unique nodes instead of 32 independent chain
nodes: eight fewer rewrites per news item (25%). For 50 news items, this is
1,200 instead of 1,600 rewrites per round.

```text
Original -> C -> C +-> C -> C  (CCCC)
                  +-> P -> P  (CCPP)
```

## Legacy set

Import `data/graphs/branching/legacy` to retain the previous 16 sequences,
including the original skeptical, emotional, and conciliatory personas.
The two catalogs are kept separate, even where they contain the same sequence.

| File | Root-to-leaf sequences | Unique nodes |
| --- | --- | --- |
| `legacy/01_s_tree.json` | SSSS, SSDD, SDSD | 9 |
| `legacy/02_c_tree.json` | CCCC, CCPP, CPCP, CPMC, CMPC | 14 |
| `legacy/03_p_tree.json` | PPPP, PPCC, PCPC | 9 |
| `legacy/04_d_tree.json` | DDDD, DDSS, DDSE, DDES, DSDS | 12 |

All 16 original paths use 44 unique nodes instead of 64: 20 fewer rewrites per
news item (31.25%). For 50 news items, this is 2,200 instead of 3,200 rewrites.

These are prefix-sharing trees. Legacy trees can have three children at a node
(for example, DD branches to DDD, DDS, and DDE) to preserve all original paths
without inserting extra transmission steps.

## Reproduce and run

The existing generator reproduces these files from their persona sequences:

```powershell
uv run python scripts/generate_branching_graphs.py --sequences NNNN CCCC PPPP DDDD CCPP PPCC DDNN NNDD --output-dir data/graphs/branching/neutral_relay
uv run python scripts/generate_branching_graphs.py --sequences SSSS CCCC PPPP DDDD CCPP PPCC CPCP PCPC DDSS SSDD DSDS SDSD DDES DDSE CMPC CPMC --output-dir data/graphs/branching/legacy
```

The generator refuses to overwrite existing files. Choose another output
directory when comparing regenerated configurations. It uses the current preset
prompts and model defaults; check them against the source chain JSON files if
those presets or defaults change.

Run one saved tree with the existing CLI:

```powershell
uv run python scripts/run_interaction_graph.py --input data/graphs/graph_news.csv --graph-config data/graphs/branching/neutral_relay/02_c_tree.json --max-rows 5 --output-dir output/interaction_graph/branching_current --output-prefix c_tree --verbose
```

This execution calls the configured models. Importing or generating configurations
makes no model calls. Select the transmission and evaluation settings in the UI
or CLI as for the linear graphs.

Each shared node produces one output per news item for all of its descendants.
The paths preserve the linear persona sequences and node settings, but separate
linear runs produce independent prefix outputs. Shared paths therefore have
dependent results and are not independent simulation repetitions. Rewrite counts
describe logical operations, not evaluation calls, retries, tokens, or money.

See [branching execution and path results](../../../docs/branching_interaction_graphs.md)
for projected linear result files and unique-node accounting. The standalone
`cc_to_cc_or_pp.json` example is retained separately from these catalogs.
