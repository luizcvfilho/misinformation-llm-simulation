# Execution monitoring

The simulation emits structured events for preparation, rewriting, extraction, VAD,
embedding fitting, each judge repetition, configured pacing, and saving. Judge events
identify the news item, node, comparison (`vs original` or `vs previous`), model, and
repetition. API requests, rate-limit waits, and retry delays appear inside the current
operation; retries do not count as additional judge repetitions.

The queue monitor counts all planned operations, including evaluations after generation.
Counts distinguish cached, skipped, and failed operations. Shared graph nodes are counted
once. Work blocked by an earlier failure is marked skipped when its phase closes. A
cancelled run retains its unfinished operations instead of showing successful completion.

## Remaining time

Estimates sum the remaining operations using observed durations for each stage, model,
and provider. Cache hits, identity shortcuts, and failed operations do not train normal
operation timings. Successful operation durations include their API retries and waits.
Configured pauses are included separately. Preparation and saving use small initial
allowances until measured.

The displayed range is an empirical planning range, not a statistical confidence interval.
It uses recent durations and widens around their average. An overdue request retains a
positive upper estimate and shows how long it has been waiting. When a required stage
has no measured timing, the monitor names the stages still being calibrated instead of
extrapolating rewriting speed into judge speed.

Each batch writes `timings.json`; the next batch in the same output directory loads the
latest history and carries its model-specific samples forward. Corrupt history is ignored.
New models calibrate independently. No additional paid requests are made for estimation.

## Logs

- `<batch>/execution.log`: persistent, readable queue history.
- `<graph>/<prefix>_events.jsonl`: timestamped structured operation and request events,
  including partial progress when execution fails.
- The monitor shows recent messages and retains the complete history in session state.
  Expand **Execution history** to filter messages or download the complete log. At most
  500 matching messages render at once; the download contains every message.

The existing `<prefix>_steps.jsonl` remains the scientific result artifact. Operational
events do not change scoring, prompts, repetition counts, cache keys, or aggregation.
The CLI also receives detailed messages through its existing progress callback.
