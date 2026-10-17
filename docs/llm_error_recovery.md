# Interactive LLM error recovery

Graph Studio pauses on LLM request and response-validation errors and opens an
**Execution paused** dialog. This applies to OpenAI, OpenRouter, DeepSeek, Gemini, and
xAI. Local/Ollama calls retain their existing request retry behavior.

The dialog offers:

- **Try again**: repeat the pending operation. If it fails again, ask again.
- **Skip this occurrence and continue**: record only this operation as failed and continue
  independent work. It does not remember a permission, release other pending failures,
  cancel the run, or turn a missing result into a successful evaluation.
  In cluster-only evaluation, comparisons requiring the skipped structure remain failed;
  a later comparison against a valid original can still finish independently.
- **Allow this error in future executions and continue**: retain the failure and proceed
  with independent work. Matching future failures are recorded without another dialog.
- **Cancel execution**: release the waiting worker and request normal graph cancellation.

The dialog shows the failing provider, model, error, and available operation context.
Structured extraction failures also offer **Validation diagnostics**, including the raw
response, validation issues, request hashes, and attempt history. Known credentials are
redacted. Each rejected extraction emits a `validation_failed` event in the run's
`*_events.jsonl` file before any recovery decision, so cancellation does not erase the
diagnostic record. `recovery_wait` events preserve the displayed error and diagnostics too.
It runs in the browser UI; the worker never opens a desktop window on the server.
An uncaught graph execution error uses the same dialog, but retrying that boundary reruns
the current graph. LLM retries resume the pending generation/evaluation rather than the queue.

## Three Gemini credentials

Configure `GEMINI_API_KEY`, plus optional `GEMINI_API_KEY_2` and `GEMINI_API_KEY_3`,
or enter the primary, secondary, and tertiary password fields in **API keys**.
Each slot can independently use an environment value or a session replacement.
Only the primary is required; a selected replacement must be nonempty. Secondary
and tertiary values are private execution-scoped credentials, excluded from
settings, graph exports, caches, and results. Duplicate values are deduplicated.

The wrapper tries another key automatically on HTTP 429 quota errors, replaying
the same pending generation with unchanged model, prompt, and temperature. A
working key remains active across nodes and news workers. The pool is shared
within an execution and isolated from other browser sessions. SDK retries are
disabled for pooled clients, so every credential attempt passes through the
existing request limiter and usage tracker. The quota failure is recorded as
an API attempt, while a successful fallback completes the original operation.

Structured daily quota errors hold that key/model until the next midnight in
`America/Los_Angeles`. Minute/token errors hold the key/model for the API's
`RetryInfo.retryDelay` or `Retry-After`, defaulting to 60 seconds. Available keys
are tried before waiting. When all keys have temporary holds, waiting is
cancellable. Each credential cycle makes at most three API attempts; persistent
temporary failures retain bounded transport recovery. All daily quotas exhausted
open normal recovery immediately, without repeatedly sending requests to those
keys. **Try again** can explicitly retry a selected slot, including after `.env`
credentials have been updated. Non-quota errors and invalid responses retain
normal recovery/validation behavior rather than silently switching models.

Quota state is in memory, per execution for Graph Studio; it is not a persistent
daily request counter and does not know traffic from other applications. Scripts
without an execution credential context use a process-local pool. Raw externally
constructed SDK clients retain their single-client behavior. No extra requests
are sent on checkpoint/cache hits.

Google applies [quotas per project](https://ai.google.dev/gemini-api/docs/rate-limits),
not per key. Three keys in one project cannot multiply its quota. Three independent
projects each with 500 unused RPD provide a nominal total of 1,500 requests;
1,200 logical rewrites leave room for 300 other attempts. Other traffic and retries
can reduce that margin. Keep the shared **Max requests per minute** at `12` for
15 RPM projects, even with five concurrent news runs. That control still includes
OpenAI evaluation calls; this change does not introduce separate provider RPM
budgets or proactive TPM limiting. Restart Graph Studio after updating the code.

## Two OpenAI credentials

Graph Studio's **API keys** section lists every remote provider required by the selected
graph queue (or the current graph), topic extraction, STDI judging, and LLM VAD evaluation.
It identifies the models and stages using each provider, requesting a single password field
per missing primary key. Gemini additionally offers optional secondary and tertiary fields;
OpenAI offers an optional fallback field when none is configured.
Existing environment credentials, including their supported aliases, are reported as configured
without displaying their values. Select **Use a different ... key for this session** to enter a
replacement even when an environment key exists. Uncheck it to use the environment key again.
OpenAI primary and fallback keys have independent replacement options. Local/Ollama providers
and disabled LLM VAD need no input.

The run button remains disabled until all required primary keys are available; the OpenAI
fallback is optional unless its replacement option is selected, which requires a nonempty value.
UI keys stay in the browser's server-side session and are supplied to
the worker through an execution-scoped credential context. They are not written to `.env`,
process-wide environment variables, graph exports, caches, or result settings. Other browser
sessions cannot resolve them. Use **Clear session API keys** to remove the supplied keys and
restore environment defaults. Switching providers or pages preserves entered values and
replacement choices for the current session. Inputs are
disabled while that session's execution is active, and the worker uses a snapshot of its keys.

```dotenv
CHATGPT_API_KEY=your_primary_key
CHATGPT_API_KEY_FALLBACK=your_secondary_key
```

`OPENAI_API_KEY` and `OPENAI_API_KEY_FALLBACK` are aliases. The `CHATGPT_` names take
precedence; they are not additional credential slots. An explicitly passed `api_key`
is the primary credential and takes precedence over session and environment credentials.

When the active key reports credit/resource exhaustion, the other configured key is tried
once. Successful calls make that slot active for subsequent operations and newly created
clients in the current execution queue. If the secondary later becomes exhausted, the
original primary is tried again before prompting. Previously exhausted keys are not banned.

In the dialog, choose **Original primary** or **Secondary** as the first key on retry.
Each manual retry tries that key first and the other key if it fails, including failures
other than resource exhaustion. Each slot is attempted at most once per cycle. A failure
on both slots opens another dialog, unless all those errors were previously allowed.
Duplicate credential values are attempted once, and an empty backup is optional.

Automatic exhaustion detection uses OpenAI's structured `credit_balance_exhausted` and
`insufficient_quota` code/type fields, not the `429` status alone. Explicit spending caps,
usage limits, and temporary throttling are separate errors; they do not trigger automatic
rotation on an ordinary call. No credits are purchased or caps changed by the application.
Two keys sharing an exhausted organization balance cannot provide independent funds.

Retries reload changed API keys from `.env` when those values came from that file;
explicit environment overrides and explicitly passed keys retain precedence. Restart an
already-running application once after installing this change so it loads the new modules
and the initial fallback setting.

## Remembered errors and missing results

Permissions are local to this repository and persist across queues and application restarts
in `.cache/llm_error_permissions.json`, which is ignored by Git. They contain hashes rather
than API keys, error text, prompts, or responses. Identity includes provider, model, endpoint,
credential slot, exception type/code/status, and a normalized message. Changing request IDs,
UUIDs, timestamps, or retry delays does not create a new identity. Specific resource codes
do not depend on changing message amounts. Distinct error causes still require a decision.

When both OpenAI keys fail, allowing the dialog authorizes the displayed failures on both
slots. Credential rotation still runs before remembered permissions are checked. A changed
error is not silently authorized by a permission for another error.

Use **Clear remembered error permissions** beside the run controls to ask again for all
errors. Scripts can also call:

```python
from misinformation_simulation.llm.recovery import PermissionStore

PermissionStore().clear()
```

Allowing a failure does not create a successful output or a zero score. Existing pipeline
failure and dependency handling remains responsible for skipping or blocking dependent work.
Successful outputs still pass existing validation before being cached. Cancellation and
permitted failures bypass validation retry loops, preventing duplicate dialogs.

Structured extraction makes up to four validation attempts (or the smaller configured
budget) before asking for a decision. Corrective attempts append the specific validation
issues to the original prompt, preserving the input text, model, schema, and validation
rules. API failures still enter normal transport recovery. Manual retry starts a new
bounded validation cycle. Empty event-relation arrays remain allowed for opinion-only
texts; the correction must not invent facts to satisfy a field. Outside interactive
recovery, the existing final-partial-result behavior is preserved.

Diagnostics retain the original input hash and the actual prompt hash for every returned
response. Corrected prompts have different hashes, and successful extraction provenance
includes rejected attempts. Prompt and implementation changes affect cache fingerprints
independently of the extraction version: old files are preserved, but incompatible entries are
not silently treated as current results. Recovery of an old experiment should explicitly
preserve its saved rewritten texts rather than generate a new transmission accidentally.

Restart the application after installing these changes, once active work has finished.
Editing source files cannot repair an already-paused worker's loaded functions.

## Scripts and notebooks

Interactive terminals and notebook kernels enable recovery automatically. They use a local
Tk dialog when a desktop display is available, with terminal input as a fallback. A remote
notebook without either requires a custom handler. Provider-specific transport errors
remain subject to the recovery policy; bounded response-validation correction also applies
to structured extraction with other providers.

Unattended scripts do not wait for invisible UI: they retain bounded legacy retries and
exception handling. Automatic OpenAI credential failover remains available. Set
`LLM_ERROR_RECOVERY=off` to retain legacy behavior outside Graph Studio. Graph Studio
always uses its explicit recovery session.

A custom handler can provide another interaction mechanism without modifying LLM callers:

```python
from misinformation_simulation.llm.recovery import (
    RecoveryDecision,
    RecoverySession,
    use_recovery,
)

def decide(request):
    print(request.context.provider, request.message)
    choice = input("Retry [r] or allow this error [a]? ").strip().lower()
    return RecoveryDecision("retry" if choice == "r" else "allow", "primary")

with use_recovery(RecoverySession(decide)):
    # Call the existing rewrite, extraction, judging, or LLM VAD functions here.
    pass
```

SDK retries are disabled for clients created inside interactive recovery sessions. Beyond
bounded validation correction, the user controls further attempts and recovery decisions. Retries
can consume additional tokens and produce a new sample. Pausing a live process does not
restore arbitrary execution state after the process or computer is closed; existing saved
results and compatible caches retain their original recovery guarantees.
