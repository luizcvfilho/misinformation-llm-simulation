# Neutral relay chain set

This is the default set for new simulations. Each JSON file describes four separate
nodes connected in a single directed chain, even when the personalities repeat.

| Code | Personality |
| --- | --- |
| N | NeutralRelay: retransmit the received content and framing with minimal wording changes |
| C | ConservativeRight |
| P | ProgressiveLeft |
| D | ConspiracyDenialist |

| File | Comparison purpose |
| --- | --- |
| `01_nnnn.json` | Minimal-change retransmission reference |
| `02_cccc.json` | Repeated conservative perspective |
| `03_pppp.json` | Repeated progressive perspective |
| `04_dddd.json` | Repeated conspiratorial perspective |
| `05_ccpp.json` | Conservative then progressive blocks |
| `06_ppcc.json` | Progressive then conservative blocks |
| `07_ddnn.json` | Neutral retransmission after conspiratorial interpretation |
| `08_nndd.json` | Conspiratorial interpretation after neutral retransmission |

NeutralRelay preserves the received claims, attribution, qualifications, confidence,
tone, and existing interpretations. It does not add skepticism, fact-check, correct
the account, or make an already biased message neutral. In interpretive mode, its
persona extension explicitly declines the optional omissions and reinterpretations.
These instructions do not guarantee zero measured drift.

InvestigativeSkeptic is unchanged and remains available in the editor and the legacy
set. Do not relabel previous S chains as N chains. Saved results remain historical.

The reversed pairs preserve persona counts but change positions, including the final
persona. They describe whole-chain differences; they do not isolate an effect of the
final persona. Compare the same news items and inspect incremental and original-relative
drift. These simulations do not measure external factual truth or human sharing behavior.

With 50 news items, this set requires 1,600 rewrites per round, compared with 3,200
for the previous 16-chain set. Alternation, emotional amplification, and conciliatory
positioning are outside this reduced set.

In the UI, use **Graph queue -> Add graphs from a folder** with:

```text
data/graphs/interaction_chains/neutral_relay
```

Example from the repository root:

```powershell
uv run python scripts/run_interaction_graph.py --input data/graphs/graph_news.csv --graph-config data/graphs/interaction_chains/neutral_relay/07_ddnn.json --rewrite-mode interpretive --output-prefix batch_07_07_ddnn
```

Use a distinct output prefix or directory for every scenario and round. The JSON files
store the full personality prompts; the UI or CLI selects the transmission and evaluation
settings. Regenerate only this set with `uv run python scripts/generate_interaction_chains.py`.
