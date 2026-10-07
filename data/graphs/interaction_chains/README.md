# Interaction chain sets

Use `neutral_relay/` for new simulations. It contains eight four-step chains:
NNNN, CCCC, PPPP, DDDD, CCPP, PPCC, DDNN, and NNDD. The graph editor opens
`neutral_relay/01_nnnn.json` by default, and its folder import defaults to
`data/graphs/interaction_chains/neutral_relay`.

The previous 16 configurations are preserved in `legacy/`, including their
original personality prompts, models, topology, and filenames. InvestigativeSkeptic
remains available as a separate preset; S still means skeptic and N means NeutralRelay.

Select one subfolder when adding graphs to the UI queue. Folder import reads JSON
files directly inside the selected folder and does not combine the two sets.

- [Current eight-chain set](neutral_relay/README.md)
- [Previous 16-chain set](legacy/README.md)

To regenerate only the current set from the current project defaults:

```powershell
uv run python scripts/generate_interaction_chains.py
```

The generator leaves `legacy/` untouched. Moving these configurations does not
recalculate or relabel saved simulation results.
