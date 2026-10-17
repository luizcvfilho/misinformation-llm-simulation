# Explicit branching interaction graphs

The simulator accepts connected rooted trees as well as linear chains. A node receives
the output of its one parent. Any node can have multiple children, and children can
branch again. There is no configured limit on fanout or path length. Cycles, disconnected
nodes, multiple roots, and merges with multiple parents are rejected before model calls.

## Configure a graph

In **Simulation → Configuration → Graph editor**, the default
**Visual canvas** editor uses [Streamlit Flow](https://github.com/dkapur17/streamlit-flow)
to draw and edit the graph. [NetworkX](https://networkx.org/) supports graph construction,
validation of visual edits, and automatic positioning.
Linear chains and branching trees use the same editor; a linear chain simply gives
each non-final node one child. The canvas is the graph visualization, so Configuration
does not render a second graph preview below it.

- Click a node to configure its persona, model, provider, label, and ID below the canvas.
- Hold **Shift** and drag from empty canvas to draw a selection rectangle around several
  nodes. Release the mouse to send the selected group to **Bulk edit nodes**, which opens
  automatically for multiple nodes. Click empty canvas to clear the selection.
- Choose **New node persona**, then **Add child node** to attach a child to the selected
  node. Repeat while the same parent is selected to create a bifurcation or more branches.
- Drag a node to reposition it. **Arrange automatically** restores a tree layout.
- Drag from a source's right handle to a target's left handle to create a connection.
  Remove the target's previous connection first when changing its parent.
- Right-click a connection to remove it. The canvas also has context menus for node creation,
  editing, and deletion. A node created on the empty canvas gets default model/persona
  settings; connect it to the existing tree before running the simulation.
- Cycles, merges, and removal of the source root are rejected. Disconnected nodes are
  allowed temporarily while editing, but the simulation requires one connected tree.

Canvas positions survive reruns and page navigation and are included in JSON/ZIP exports
under `layout.positions`, keyed by simulation node ID. Import restores these positions.
Canvas movement never changes the graph's transmission order.

Open **Bulk edit nodes** to update several nodes in the current graph at once. Select
nodes directly with **Shift + drag** on the canvas, choose them in **Nodes to edit**, or
use **Select all**. The canvas and the list keep their selections in sync. Check **Change provider**,
**Change model**, and/or **Change personality**, set the replacement values, then choose
**Apply to N selected nodes**. Models accept presets or a custom identifier; personalities
accept predefined presets or a custom prompt. Only checked fields change, so a model update
can preserve each node's existing personality. The panel works with both editing modes and
preserves node identities, connections, and canvas positions. Changes affect the open graph;
existing queue snapshots are updated only through **Save changes to queue** when editing
a queued graph.

The selection bridge adapts the bundled callbacks of the pinned
`streamlit-flow-component==1.6.1` frontend into `.cache/flow-selection/`. It reports completed
selection rectangles and group drags without changing the installed package. Upgrading that
dependency requires checking the adapter against the new frontend bundle.

Select **Node forms** for the original form-based editor. Use
**Receive text from** to set each node's parent. Exactly one node must use
**Source news (root)**. To branch after a shared prefix, choose the same parent for
two or more children. **Add node** initially attaches a new node to the last displayed
node; change its parent to create the desired branch. Reordering cards changes display
order without changing tree connections. **Remove subtree** removes the selected node
and all its descendants; the root is retained.

The canvas shows the configured nodes and connections.
Graph imports, JSON downloads, and queue ZIP downloads preserve the explicit edges.
Graphs without explicit edges still default to a chain in node-list order.

## Inspect and reuse queued graphs

In **Graph queue**, expand **Graph details** below a queued graph to inspect its
topology, complete paths, start node, and per-node connections, providers, models,
and personalities. Select a node under **Personality prompt for node** to read its
full resolved prompt. **Export queued graph JSON** downloads that queued snapshot.

Use **Edit queued graph** to open an existing entry in the editor. Change its nodes,
connections, models, providers, or **Current graph name**, then choose
**Save changes to queue**. Saving validates the graph and updates the same queue entry,
preserving its ID and position even if the queue has been reordered. Edits remain a draft
until saved; queue execution and downloads use the saved version. **Cancel editing**
discards unsaved edits and restores the saved graph in the editor. Importing, generating,
resetting the editor, or loading a copy ends the link to the edited queue entry.
Changes saved during an execution apply to the next execution; an active run keeps its
initial snapshot.

Use **Load copy into editor** to replace the current editor contents with an independent
copy, preserving node IDs, connections, personalities, models, and saved canvas positions.
Change each node's rewrite model and provider, choose a distinct graph name, then use
**Add current graph** to enqueue the variant. Loading or editing a copy does not modify
the original queue entry. Each queued variant produces separate graph results.
Transmission mode, dataset selection, and STDI/VAD evaluation settings apply to the
entire execution. Use the same evaluation settings when comparing rewrite models.

## Generate trees automatically

Open **Generate graphs from sequences**, enter complete persona sequences (one per line
or separated by spaces, commas, or semicolons), and choose the model/provider for generated
nodes. Supported preset codes are C, P, D, S, E, M, and N.

**Generate tree in editor** creates one prefix-sharing tree and opens it for further editing.
For example, `CCCC` and `CCPP` produce six nodes, sharing the first `CC`. A prefix is shared
only at the same position and with the same complete preceding sequence. Repeated input
sequences are deduplicated; suffixes reached through different prefixes remain separate.

**Generate trees in queue** also accepts different first personas. It creates one tree per
first persona, so `NNNN CCCC PPPP DDDD CCPP PPCC DDNN NNDD` produces four trees with 24 total
nodes instead of 32 separate chain nodes. Open a queued tree's **Graph details** and
use **Load copy into editor** to edit a copy. Generation makes no API calls and does not
start a simulation.

The generator requires complete root-to-leaf sequences. If `CC` and `CCPP` are requested
together, it reports that `CC` ends inside the longer path; generate these lengths separately
to retain both endpoints. The visual editor still allows different path lengths and custom
personas, because each branch has its own node identities.

The Python API is `misinformation_simulation.simulation.generation.generate_branching_graphs`.
The command line can save generated configurations without executing them:

```powershell
uv run python scripts/generate_branching_graphs.py --sequences CCCC CCPP PPPP PPCC --output-dir data/graphs/my_generated_trees
```

Generated files can be imported into the editor, added to the graph queue, or passed to
`scripts/run_interaction_graph.py --graph-config`. The generator preserves existing files
instead of overwriting them; select another output directory when regenerating configurations.

The JSON example `data/graphs/branching/cc_to_cc_or_pp.json` defines six nodes:

```text
Original → C → C ┬→ C → C  (CCCC)
                 └→ P → P  (CCPP)
```

The two initial conservative nodes are shared. For each news item the simulator
performs six rewrite operations instead of eight. Separate nodes always generate
separate outputs, even when their persona and input happen to match. Sharing is
determined by the configured topology. Automatic prefix matching is an explicit, opt-in
construction step in the sequence generator; the executor never silently merges separate nodes.

Run the same configuration through the command line:

```powershell
uv run python scripts/run_interaction_graph.py --input data/graphs/graph_news.csv --graph-config data/graphs/branching/cc_to_cc_or_pp.json --max-rows 5 --output-dir output/interaction_graph/branching_example --output-prefix branching_example --verbose
```

This command runs the configured models and incurs API usage. The example preserves
the current presets' model and persona definitions. Set transmission mode, STDI/VAD
methods, and evaluation models through the existing CLI flags or workspace controls.

## Execution and results

Each node is generated and evaluated once per news row in the current graph run.
Each child inherits its parent's text, topic structure, VAD evaluation, and path state.
Incremental metrics compare the child with its actual parent, and cumulative metrics
sum the ancestors of that child. Shared evaluations are copied into the linear views
after scoring; projecting paths makes no additional model calls.

The graph overview stores unique node records. Every root-to-leaf path also receives
its own summary and steps files, compatible with the existing linear analysis:

```text
branching_example_summary.json
branching_example_steps.jsonl
paths/
  01_cccc/
    01_cccc_summary.json
    01_cccc_steps.jsonl
  02_ccpp/
    02_ccpp_summary.json
    02_ccpp_steps.jsonl
```

In **Results → Result view**, select **Graph overview (unique nodes)**, **CCCC**, or
**CCPP**. **Analyze this run** opens the execution containing all its projected paths.
The Analysis page and analysis scripts discover the linear path files and exclude
the tree overview from chain aggregation. Paths belonging to the same saved graph
remain in the same execution. A path has a unique ID even when another path has
the same persona sequence. Custom personas use `X` in the generated chain code;
node IDs, labels, and complete persona prompts are retained. Filename codes are
bounded to 32 characters; the full chain code remains in the summary.

The graph summary reports `planned_unique_rewrites`, `planned_linear_rewrites`,
`planned_rewrites_saved`, and `rewrite_operations_started`. These count logical
rewrite operations, not retry attempts, evaluation requests, billed tokens, or money.
Path summaries attribute a shared rewrite to its first projected path only. Records
include `metadata_graph_step_id`, `metadata_graph_shared_node`,
`metadata_graph_step_reused`, and `metadata_rewrite_operation_attributed` for tracing
shared outputs without adding their cost twice. IDs are scoped to the saved graph.

Shared generation state exists only within the current run. Starting another graph
or batch generates its own outputs; previous rewrite files are never loaded as cache.
The existing, separate evaluation-cache configuration is preserved.

If a branching-tree node fails, its descendants are marked `blocked`; siblings continue
from their own parent state. Legacy linear chains retain their previous failure behavior.
Cancellation saves completed unique records and partial linear paths, marked cancelled.
Projected shared records are independent copies, so editing a path view cannot mutate
another path's results.
