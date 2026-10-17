# Grok rewrite comparison

These four graphs copy `../neutral_relay/`, changing only each node's provider
to `grok` and model to `grok-build-0.1`. Node IDs, personas, connections, and
layout are identical. They contain 24 unique nodes across eight four-step paths:
50 news items require 1,200 logical rewrites, plus evaluation calls and retries.

## Import and execution settings

In the Streamlit app, remove the previous entries from **Graph queue**, then use
**Add graphs from a folder** with `data/graphs/branching/neutral_relay_grok`.
Folder import adds entries; it does not replace existing queued graphs. Confirm
that all four queued graphs show `grok` / `grok-build-0.1` in their details.
Importing these files does not restore execution settings or the news dataset.

Use the same 50 input texts, row order, columns, and news IDs as the GPT run
`simulation_ui_20261016_213859-GPT`, with these recorded settings:

| Setting | Value |
| --- | --- |
| Max rows | `50` |
| Transmission mode | `interpretive` (`interpretive_v2`) |
| STDI comparison | `dual` |
| STDI judge repetitions | `3` |
| Topic extraction / STDI judge provider | `chatgpt` |
| Topic extraction / STDI judge model | `gpt-6-luna` |
| VAD method | `dual` |
| VAD evaluator provider / model | `chatgpt` / `gpt-6-luna` |
| OpenAI processing mode | `flex` |
| Allow temporary fallback to Standard | Enabled; retry Flex after 5 minutes |
| Reuse previous results | Enabled |
| Max simultaneous runs | `10` |
| Execution name | Choose a new name identifying Grok |

Keep both `GROK_API_KEY` (or `XAI_API_KEY`) and `CHATGPT_API_KEY` (or
`OPENAI_API_KEY`) available in the environment or the app's session fields.
OpenAI Flex applies only to OpenAI requests. Rewrite cache inputs include the
provider and model, preventing GPT rewrites from being reused for these Grok
nodes. Compatible evaluations of unchanged original texts can be reused.

The wrapper requests temperature `0.8` for Grok rewrites. It omits temperature
for GPT-6 models, leaving their API default. Preserve and report this difference
when comparing providers; these graphs do not equalize sampling behavior.

## Model choice and verification

Checked on October 17, 2026: `grok-build-0.1` has the lowest short-context text
token rates in the [official xAI pricing table](https://docs.x.ai/developers/pricing):
USD 1.00 input, USD 0.20 cached input, and USD 2.00 output per million tokens.
It is a coding-oriented model. Lower token rates do not establish rewrite quality
or guarantee a lower total bill; generated tokens, reasoning, and retries matter.
Use its explicit model ID rather than retired Fast model IDs, which have different
[redirects and pricing](https://docs.x.ai/developers/migration/may-15-retirement).

Readiness checks confirmed authenticated model retrieval for both
`grok-build-0.1` and `gpt-6-luna`, and 235 relevant local tests passed. No live
text generation or 50-news simulation was started during those checks. Run a
one-news pilot to verify generated output and complete evaluations before the
full batch.
