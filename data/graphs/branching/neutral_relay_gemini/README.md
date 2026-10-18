# Gemini free-tier rewrite comparison

These four graphs copy `../neutral_relay/`, changing only each node's provider
to `gemini` and model to `gemini-3.5-flash-lite`. Node IDs, personas, connections,
and layout are preserved. The four trees contain 24 unique nodes across eight
four-step paths: 50 news items require 1,200 logical rewrites before retries.

## Import and comparable settings

Remove the previous entries from **Graph queue**, then use **Add graphs from a
folder** with `data/graphs/branching/neutral_relay_gemini`. Import appends graphs;
it does not replace the queue or restore the dataset and execution settings.
Confirm four graphs and Gemini on all 24 nodes, including branch nodes.

Load `data/graphs/graph_politics_news.csv` through the application and use its
first 50 news rows. The loader removes the `__query_metadata__` row. These 50
IDs, titles, and descriptions were checked against the saved GPT and Grok
outputs on October 17, 2026; all matched. Preserve the input file and row order.

| Setting | Value |
| --- | --- |
| Max rows | `50` |
| Text / title / news ID columns | `description` / `title` / `article_id` |
| Allow title fallback | Enabled |
| Transmission mode | `interpretive` (`interpretive_v2`) |
| STDI comparison | `dual` |
| LLM judge evaluations per pair | `3` |
| Topic drift provider / model | `chatgpt` / `gpt-6-luna` |
| VAD method | `dual` |
| VAD evaluator provider / model | `chatgpt` / `gpt-6-luna` |
| OpenAI processing mode | `flex` |
| Allow temporary fallback to Standard | Enabled; retry Flex after 5 minutes |
| Reuse compatible results from previous executions | Enabled |
| Simultaneous news runs | `5` (`1` also works) |
| Providers with RPM limits | Only `gemini` selected |
| Max requests per minute (gemini) | `12` |
| Sleep between requests (seconds) | `0` |
| Retry attempts | `5` |
| Output prefix | A new execution name identifying Gemini |

The user reported project limits of **15 RPM, 250,000 input TPM, and 500 RPD**
for this model on October 17, 2026. The 12 RPM setting leaves request headroom;
it is shared across Gemini models, keys, and retries in this execution.
OpenAI evaluation calls do not consume this budget or wait for its spacing.
Parallel news processing uses that same cap and does not multiply the per-project rate. Do not use
`Sleep between requests` alone as quota protection.

Keep `GEMINI_API_KEY` and `CHATGPT_API_KEY` (or `OPENAI_API_KEY`) configured in
the environment or the app's session fields. Add `GEMINI_API_KEY_2` and
`GEMINI_API_KEY_3`, or use the secondary and tertiary Gemini password fields,
for automatic quota failover. The user confirmed that these three keys come
from independent projects. A Gemini entry is present in the
local `.env`; validity, project tier, and remaining quota were not verified
with a live API request. OpenAI evaluations retain their existing charges.
Only rewriting nodes change provider, preserving the evaluation setup used
for GPT and Grok. Gemini rewrite cache entries cannot reuse GPT/Grok rewrites
because provider and model participate in cache identity. Compatible evaluations
of identical texts may be reused.

## Daily quota and continuation

With one project at 500 RPD, 1,200 fresh Gemini rewrites require at least three
daily quota windows. With three independent projects each having 500 unused
RPD, the nominal 1,500-request total can cover all 1,200 rewrites in one day.
The application keeps one working key active, switching to an available key
on a quota error without changing the pending message or model. Quota holds
are shared across concurrent news workers. Pilots, retries, rejected calls,
and other clients can affect remaining capacity; use AI Studio's actual usage.
Changing the minute rate does not increase daily capacity. Multiple keys in
the same project share the quota.

There is no proactive RPD counter, TPM limiter, or automatic wait until all
projects' daily quotas reset. The pool learns quota holds from API responses:
daily exhaustion holds that key/model until midnight Pacific; temporary limits
use the API retry delay or 60 seconds. Duplicate key values do not create extra
slots. Inspect AI Studio usage for each project and preserve the headroom for
retries. The shared 12 RPM cap applies to Gemini calls across all configured keys.

If all configured projects' daily quotas are exhausted:

1. Keep the execution paused in **Execution paused**, leaving the application
   process running. Keep the machine awake if you want it available to resume.
2. Wait for the quota reset shown in AI Studio, then choose **Try again**.
   Do not choose **Skip this occurrence** or **Allow this error in future
   executions** to work around a daily quota: those actions record failures
   and can leave dependent work unavailable.
3. If the application has to stop, cancel normally and preserve its output,
   `call_checkpoint`, and shared cache directories. **Run recovery** can replay
   valid saved responses for failed/cancelled attempts. Recovery uses saved
   settings; editing the main form does not change them. Recovery of an
   individual news run under Dual STDI also requires the saved graph-wide
   `recovery_corpus.json`; it can be unavailable if preparation was interrupted.
   For an incomplete graph preparation, rerunning the same complete queue with
   identical inputs/settings and compatible-result reuse enabled can reuse
   valid stage caches while rebuilding unfinished work. Do not select
   **Restart run from zero** when the intention is to retain checkpoint calls.

The documented daily reset is midnight Pacific time. On October 17, 2026,
that is 04:00 in Sao Paulo; it becomes 05:00 when Pacific standard time applies.
Confirm active limits and remaining quota in [Google AI Studio](https://aistudio.google.com/usage?tab=rate-limit).
The [official quota documentation](https://ai.google.dev/gemini-api/docs/rate-limits)
explains RPM, input TPM, RPD, project scope, and the Pacific reset. The
[pricing page](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
lists free-tier input/output for this model; the local cost estimator uses
paid-tier reference rates and is not evidence of a Gemini free-tier charge.

## Readiness and limitations

The recorded GPT and Grok batches both contain four trees, 50 news items per
tree, and 1,200 successful rewrite steps with no rewrite errors or blocked
steps in their graph summaries. Their saved original texts and titles match.
The tree topology and node personas were preserved in these Gemini copies.

The wrapper requests temperature `0.8` for Gemini and Grok rewrites. It omits
temperature for GPT-6 models, leaving their API default; the comparison does
not equalize sampling behavior. Keep the same model throughout the Gemini
round, and record dates and any failures. Do not reduce judge repetitions,
change VAD, split the evaluation corpus, or shorten source texts just to fit
the Gemini quota, since those changes alter the comparison.

No live Gemini generation or full simulation was started during preparation.
A local check imported all four graphs through the application's queue loader,
verified all 24 providers/models, and compared personas and edges with both
saved rounds. All 127 selected tests passed, covering clients, retries, rate
limiting, recovery, parallel execution, model selection, and branching graphs.
A one-news pilot of all four trees requires 24 logical Gemini rewrites plus
OpenAI evaluation calls. With compatible-result reuse enabled, its valid
compatible stages can be reused when running the complete 50-news queue.
