# LLM error recovery review

Reviewed on 2026-10-15. Scope: OpenAI (`chatgpt`), OpenRouter, DeepSeek, Gemini,
and xAI (`grok`). This is a research and design review, not an implemented feature.
No authenticated provider requests were made. Ollama and NewsData are outside this review.

The subsequent implementation follows the user's revised, generic retry/allow policy.
See [interactive LLM error recovery](llm_error_recovery.md) for implemented behavior;
the proposals below preserve the earlier research context.

## Findings in the current code

- `llm/clients.py` selects one credential per client. For OpenAI, `CHATGPT_API_KEY`
  takes precedence over the alias `OPENAI_API_KEY`; these names do not form a pool.
- `llm/retry.py` recognizes transient generation errors through substrings such as
  `429`, `503`, `resource_exhausted`, and `timeout`. It does not classify billing failures
  or inspect provider retry headers.
- `generate_and_parse_with_retry` retries every ordinary exception from generation or
  parsing, up to four attempts. Extraction, semantic judging, and LLM VAD use it.
  An authentication or billing error can therefore consume the validation retry budget.
- Some rewrite callers retain the SDK's own retry policy in addition to application
  retries; extraction and semantic judging explicitly disable SDK retries. A single
  application attempt is not always a single HTTP attempt.
- `llm/false_to_true.py` can save a CSV checkpoint and skip successful rows on a later
  call. This does not provide arbitrary process-state restoration across the application.
- `simulation/graph.py` records failed nodes and blocks dependent descendants in branching
  execution. Recovery must preserve these dependencies rather than invent a replacement text.
- `apps/interaction_graph_run_job.py` runs graph work in a background thread with an event
  queue and a cancellation event. `interaction_graph_sections.py` polls that queue in the UI.
  This is a useful boundary for a future recovery request and a separate user-response channel.

The installed OpenAI SDK exposes status, response body, error code/type, and request ID.
The installed Google SDK exposes code, status, message, details, and response. Classification
can use these fields before resorting to message matching. SDK exception class alone is
insufficient to distinguish all financial failures.

## Verified provider recovery guidance

### OpenAI

Inspect `error.code` in addition to HTTP status and `error.type`:

- `credit_balance_exhausted`: add prepaid credits. Proposed application behavior: retry
  the identical request with the configured secondary credential before asking the user.
- `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`: review the relevant
  configured cap; repeatedly sending the same request cannot fix it.
- `organization_usage_limit_exceeded`: request a higher approved allowance.
- Legacy or broad `insufficient_quota`: resource/billing restriction, without enough
  information to assert that the balance is zero. Show the uncertainty in the remedy.
- Temporary throttling/overload: paced retries, honoring `Retry-After` when present.
- Invalid authentication, permissions, request arguments, or resource identifiers require
  correcting the corresponding configuration.

The prepaid balance is organization-scoped. Two keys in the same exhausted organization
will not provide independent funds. This application should never change a spending cap
automatically. These remedies follow the [official OpenAI error guide](https://developers.openai.com/api/docs/guides/error-codes).

The SDK retries selected connection/status failures by default. Use one observable retry
budget, accounting for or disabling SDK retries, rather than multiplying nested budgets.
See the [Python SDK reference](https://developers.openai.com/api/reference/python).

### OpenRouter

Read `error.metadata.limit_source` and `reason`, not just HTTP `402`:

- `openrouter_in_flight_budget`: temporary reservation pressure; wait for `Retry-After`.
- `openrouter_credits`: available resources cannot cover the request; adding credits or
  reducing the request's estimated cost may help. It does not always mean a zero balance.
- `openrouter_key_limit`: inspect the key's configured cap and reset window.
- Explicit budget failures can use `403` and identify a workspace or guardrail budget.
  Other `403` responses can mean permission or content restrictions.

See [credit and budget limits](https://openrouter.ai/docs/api_reference/limits).
Reducing inputs or output limits is a user-selected change, not an automatic repair of
an experiment's request.

An upstream failure can appear inside an HTTP `200` response, even without streaming.
Inspect an error envelope before reading `choices`; use its typed error metadata when
available. See [errors and debugging](https://openrouter.ai/docs/api_reference/errors-and-debugging).

### DeepSeek

`402` means insufficient balance: add funds or try an eligible configured backup credential.
`401` requires fixing the key; `400` and `422` require correcting the request.
For `429`, slow down; for `500` or `503`, wait and retry with a bounded budget.
See [DeepSeek error codes](https://api-docs.deepseek.com/quick_start/error_codes/).

### Gemini

Prepay depletion produces `402`; adding funds restores access. All projects linked to an
exhausted billing account are affected, so another key on that account cannot restore funds.
See [Gemini billing](https://ai.google.dev/gemini-api/docs/billing?hl=en).

`429 RESOURCE_EXHAUSTED` is not sufficient evidence of depletion. Inspect the quota and
retry details: minute/token limits, daily quotas, and rolling spend-rate limits have different
remedies. Limits are project-scoped. Waiting, pacing requests, waiting for a daily reset,
or requesting a quota increase may be appropriate. See [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

Use the native error fields from this project's `generate_content` endpoint; do not assume
the Interactions API's string error codes appear identically. Invalid requests, blocked keys,
and permission errors require a targeted fix; transient failures warrant bounded retries.
See [troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting).

### xAI / Grok

The published error guide distinguishes invalid credentials, forbidden operations, missing
models/endpoints, malformed requests, and throttling. A `403` by itself is not proof of
insufficient credit. See [debugging errors](https://docs.x.ai/developers/debugging).

Billing documentation confirms that prepaid exhaustion and invoice limits can reject calls.
Add credits or review the applicable billing limit. See [billing](https://docs.x.ai/console/billing).
The consulted sources do not establish a dedicated, stable inference error code for every
billing case. Until representative sanitized responses establish the mapping, classify
ambiguous responses as unconfirmed and offer diagnostics rather than claiming a zero balance.

## Proposed recovery behavior

Apply this at logical operation boundaries, covering rewrites, extraction, judging, and LLM
VAD, including their response validation. Catching an exception cannot make every program
state safe to resume.

| Failure | Automatic action | User action after recovery is exhausted |
| --- | --- | --- |
| Confirmed credit exhaustion | Try the eligible secondary key with the same model and input | Add funds, reload credentials if changed, and retry; or skip the affected item |
| Explicit spending cap / persistent quota | Explain the limit; no automatic cap change | Review the cap/quota or wait for reset, then retry |
| Throttling / temporary overload | Honor server delay, then bounded backoff with jitter | Retry later, adjust pacing, skip, or stop |
| Connection / timeout | Bounded retries where appropriate | Fix connectivity, retry, skip, or stop |
| Invalid credential / permission | Avoid repeated identical calls | Fix credentials or access and retry |
| Invalid request / context too long / unavailable model | No unchanged automatic retry loop | Review parameters or choose a revised configuration |
| Empty, truncated, malformed, or schema-invalid output | A bounded validation retry when regeneration can help | Retry, revise settings, skip, or stop |
| Content refusal / policy block | Record the refusal; no credential rotation as a remedy | Review the input or skip |
| Unknown exception / local failure | Preserve diagnostics and available state | Retry only at a valid boundary, skip when supported, or stop |

A timeout can leave the remote request's outcome unknown. Retrying may consume additional
tokens and produce a different sample; local deduplication does not guarantee exactly-once
provider execution. Record attempts and observable usage separately.

Proposed interaction:

1. Run only the automatic recovery actions appropriate to the classified cause.
2. Pause the affected execution when a bounded retry budget is exhausted, or immediately
   when manual intervention is required and no eligible backup remains.
3. Show provider/model, stage and affected item, cause/confidence, attempts, proposed remedy,
   and the consequence of skipping. Hide credentials and unnecessary raw request content.
4. Offer **Retry**, **Review configuration**, **Skip this item and continue**, and
   **Stop and preserve available progress**, enabling only actions supported at that boundary.
5. Configuration review may open the relevant dashboard or reload explicitly changed
   settings. It must not silently truncate input, change models/prompts, increase costs,
   or purchase credits. Changing experiment settings must be recorded and reflected in caches.
6. Dismissing the dialog keeps the operation paused. Cancellation must wake a waiting worker
   and must not be swallowed by existing broad exception handlers.

In Streamlit, render the dialog from the UI thread and exchange decisions with the worker.
For scripts and notebooks, provide an environment-appropriate interaction or a clear
noninteractive failure/checkpoint outcome. Do not depend on a native desktop window existing
on the machine hosting a browser session.

## Meaning of continuing after an error

Continuing means recording a missing/failed/skipped result and proceeding with independent
work. It does not mean replacing an absent score with zero, accepting invalid JSON, or
passing a source text along as if a rewrite succeeded. A missing rewrite blocks its dependent
branch. A missing evaluation leaves its metric unavailable; whether independent stages can
continue needs an explicit stage contract. Failure to persist required state or corrupted
state may require stopping rather than exposing a misleading continuation option.

Group repeated failures by operation and normalized cause, excluding changing request IDs
and timestamps. User-selected handling of recurring errors should have a limited, visible
scope and remain resettable. Avoid both endless popups and silently skipping an entire run.

## OpenAI secondary-key requirement

Maintain ordered, distinct credentials and remember the working credential for the current
run across newly constructed clients. On confirmed exhaustion, retry the pending operation
with the secondary key; if successful, continue using it. If both fail, enter the recovery
dialog. A resource restriction with an uncertain subtype must retain that uncertainty in
its user message. Explicit budget limits need a separate decision from credit exhaustion.

Keep model, endpoint, prompts, and validation rules unchanged during credential failover.
Log credential slot labels, never secrets. After manual remediation, reload changed
credentials and re-enable the candidates before retrying the same pending operation.

## Required checks before implementation is accepted

- Simulated native SDK errors for all five providers, including incomplete/non-JSON bodies.
- OpenAI primary exhaustion followed by backup success, both exhausted, and later recovery.
- Temporary `429`, explicit caps, and broad quota errors never misreported as zero balance.
- OpenRouter temporary `402`, budget `403`, and an HTTP `200` error envelope.
- Consistent attempt budgets across SDK, transport, generation, and validation layers.
- Correct skip/dependency outcomes, no invalid successful cache entries or fabricated metrics.
- Dialog dismissal, duplicate/stale decisions, worker cancellation, and noninteractive behavior.
- Provenance for settings changes and no credential leakage in dialogs, events, or logs.

These checks can largely use mocked responses without spending provider credits. Native
samples are still needed where documentation does not establish a reliable mapping,
particularly for xAI billing failures. Process-restart recovery needs a separate checkpoint
contract; pausing a live worker alone does not provide it.
