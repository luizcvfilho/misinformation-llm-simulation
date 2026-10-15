# Interactive LLM error recovery

Graph Studio pauses on LLM request and response-validation errors and opens an
**Execution paused** dialog. This applies to OpenAI, OpenRouter, DeepSeek, Gemini, and
xAI. Local/Ollama calls retain their existing request retry behavior.

The dialog offers:

- **Try again**: repeat the pending operation. If it fails again, ask again.
- **Allow this error in future executions and continue**: retain the failure and proceed
  with independent work. Matching future failures are recorded without another dialog.
- **Cancel execution**: release the waiting worker and request normal graph cancellation.

The dialog shows the failing provider, model, error, and available operation context.
It runs in the browser UI; the worker never opens a desktop window on the server.
An uncaught graph execution error uses the same dialog, but retrying that boundary reruns
the current graph. LLM retries resume the pending generation/evaluation rather than the queue.

## Two OpenAI credentials

```dotenv
CHATGPT_API_KEY=your_primary_key
CHATGPT_API_KEY_FALLBACK=your_secondary_key
```

`OPENAI_API_KEY` and `OPENAI_API_KEY_FALLBACK` are aliases. The `CHATGPT_` names take
precedence; they are not additional credential slots. An explicitly passed `api_key`
is the primary credential and takes precedence over the environment.

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

## Scripts and notebooks

Interactive terminals and notebook kernels enable recovery automatically. They use a local
Tk dialog when a desktop display is available, with terminal input as a fallback. A remote
notebook without either requires a custom handler. Other providers are not classified or
automatically repaired: the user sees the error and decides what happens next.

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

SDK retries are disabled for clients created inside interactive recovery sessions. The user
controls repeated attempts, including invalid JSON and schema-invalid responses. Retries
can consume additional tokens and produce a new sample. Pausing a live process does not
restore arbitrary execution state after the process or computer is closed; existing saved
results and compatible caches retain their original recovery guarantees.
