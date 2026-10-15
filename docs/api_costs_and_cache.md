# API costs and compatible result reuse

In the graph app, **Advanced options > Reuse compatible results from previous executions**
is enabled by default. Runs use `shared_cache/` under the selected output directory, while
retaining their own `evaluation_cache/` copies for audit. This applies to future results
written with compatibility metadata; historical entries without that metadata are not
automatically trusted or migrated.

The switch reuses existing simulation draws. Disable it for a new independent simulation
replicate. Reusing a saved draw is not another independent observation. Within an execution,
the existing evaluation deduplication still applies to identical text pairs.

## Compatibility and invalidation

Each stage has its own compatibility contract, saved in graph summaries and cache entries.
The contract fingerprints relevant implementation files and prompt content, in addition to
the explicit prompt/schema versions. Editing a relevant prompt without incrementing its
version still invalidates the affected cache. Git branch names and commit hashes are not
used as substitutes for content checks.

Inputs include exact texts, titles when applicable, provider, model and endpoint identity.
Rewrite keys also include the effective prompt, system instruction, requested temperature,
news identifier, row position and node lineage. Distinct node occurrences are not merged
just because their texts match. Endpoint identities are hashed; API keys are not part of
cache keys. A changed local endpoint also invalidates reuse.

Examples:

- A judge prompt change recalculates judge draws but can reuse rewrites, extractions and VAD.
- A rewritten source message changes downstream rewrite inputs and affected evaluations.
- Increasing judge repetitions requests only missing draw indices for compatible pairs.
- A formula or embedding change recomputes derived STDI from the available raw evaluations;
  final STDI values and corpus-fitted embeddings are not reused as cached results.

Only compatible, complete outputs are published to the shared cache. Missing metadata,
version mismatches or failed integrity checks require fresh computation. Old shared entries
remain available under their own identities; they are not rewritten as current results.
Local cache copies retain the full extraction provenance, including raw responses.

Compatibility checks cannot detect undisclosed weight changes behind a provider model alias
or changes to local model weights served under the same identifier. Use immutable model
snapshots/revisions and distinct local model identifiers when those distinctions matter.
Custom VAD scorers bypass the VAD cache. Custom Python judge callbacks keep a per-run
identity so their outputs cannot be reused across runs without a verifiable implementation.

## Usage and cost records

Each graph execution writes:

- `<prefix>_api_usage.jsonl`: an append-only record of observable API responses, failed
  requests and cache hits. Entries include a unique `run_id`; they contain usage metadata,
  provider/model identifiers and rates, not prompts, response text or credentials.
- `<prefix>_api_costs.json`: current execution totals by stage: `rewrite`, `extraction`,
  `judge`, `vad_llm` and any local VAD cache hits.
- `<prefix>_summary.json`: the same `api_usage` totals plus the stage cache contracts.

The app also writes `api_costs.json` in the execution's parent folder, summing the graph
runs completed so far (including available partial usage) without counting path projections.

The app displays the selected graph's stage table in **Results > API usage and estimated cost**.
Projected paths refer to their parent graph for execution costs; do not sum path projections
as if they were separately executed graphs. Usage events are written immediately, including
responses later rejected by JSON/evidence validation. They remain available on cancellation
or an interrupted/failed run. SDK-internal retries and provider charges not exposed in usage
cannot be reconstructed; estimates do not replace the provider invoice.

Cache hits contribute zero new API tokens and zero new API cost. Old response costs are not
copied into the current execution. When no provider-reported cost is available, missing
usage, request exceptions, unknown models or service tiers produce an unavailable total
(`null`) with a known-cost subtotal and
an explicit unpriced request count, rather than silently counting those requests as free.

Input tokens, cache reads/writes, output tokens and reasoning tokens are recorded separately.
Reasoning is already included in OpenAI completion-token totals and is not charged twice.
Gemini uses `prompt_token_count` for input and `candidates_token_count` plus
`thoughts_token_count` for generated output. Its cached prompt count is subtracted from
ordinary input before applying the cached-input rate. Standard `generate_content` calls
default to standard traffic when the response does not report a traffic type; explicit
unrecognized traffic types remain unpriced.

The immutable catalog in `src/misinformation_simulation/llm/pricing.py` uses
`ModelPricing` and `TokenRates` dataclasses, keyed by the `Models` enum. It covers every
model preset with a provider, source, verification date and equivalent token-rate fields.
Each tracker receives an isolated JSON-compatible snapshot. Prices were checked on
2026-10-14; they are not fetched automatically at runtime.

Standard text prices in USD per million tokens:

| Provider | Model | Input | Cached input | Output |
| --- | --- | ---: | ---: | ---: |
| OpenAI | `gpt-6-luna` | 0.10 | 0.01 | 0.50 |
| OpenAI | `gpt-5.6-luna` | 0.20 | 0.02 | 1.20 |
| OpenAI | `gpt-5-mini` | 0.25 | 0.025 | 2.00 |
| OpenAI | `gpt-5-nano` | 0.05 | 0.005 | 0.40 |
| OpenAI | `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 |
| Gemini | `gemini-3.5-flash-lite` | 0.30 | 0.03 | 2.50 |
| Grok | `grok-build-0.1` | 1.00 | 0.20 | 2.00 |
| OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` | 0 | 0 | 0 |
| OpenRouter | `stepfun/step-3.5-flash:free` | 0 | 0 | 0 |
| Local | `llama3.1:8b` | 0 | 0 | 0 |

Grok Build 0.1 has the lowest listed text input/output rates in the current
[xAI pricing table](https://docs.x.ai/developers/pricing). It is a coding-oriented model;
this preset is a price-based choice, not evidence of suitability for news evaluation.
Global-endpoint prices double for input, cached input and output when input reaches
200,000 tokens. Provider-reported amounts still take precedence.

[Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
is the current Flash-Lite preset. Its [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
uses explicit per-tier rates: Flex/Batch input 0.15, cache 0.02, output 1.25;
Priority input 0.54, cache 0.05, output 4.50. Cache prices are not derived from a uniform
multiplier. The invite-only Contributor tier requires an account-specific override.
Local zero rates describe provider API charges only, excluding hardware and electricity.
OpenRouter zero rates apply only to the exact free endpoints. Unknown models, tiers or
unsupported cache-write charges remain unpriced.

Gemini estimates use paid-tier text rates, not an assumption about the account's free-tier
eligibility. Supply zero-rate overrides for known free-tier usage. Explicit cache storage,
multimodal input and tool fees are outside this text-only estimator.
The estimator uses the service tier returned by the API and handles the documented
long-context multipliers. It does not enable Flex or change request parameters.
[OpenAI's usage fields](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
are the source of token counts, not character-based estimates.

Grok uses its OpenAI-compatible usage fields (both `prompt_tokens`/`completion_tokens`
and `input_tokens`/`output_tokens` are accepted). When xAI returns `cost_in_usd_ticks`,
that provider-reported amount takes precedence over token-rate estimation: divide by
10,000,000,000 to obtain USD. Raw integer ticks remain in the log, and `cost_source` marks
`provider_reported` versus `token_rates` or `unavailable`. Summaries count requests whose
cost came directly from the provider. This also handles an alias returning a different model
without assuming the alias's historical rates. If the field is absent, a configured Grok
rate can be used; otherwise cost remains unavailable.
See [xAI cost tracking](https://docs.x.ai/developers/cost-tracking).

The retired `gemini-3.1-flash-lite-preview` and `grok-4-1-fast-non-reasoning`
presets were removed from the enum and notebook source configurations. New presets are
`Models.GEMINI35FlashLite` and `Models.GROKBuild01`. Historical outputs and saved run
configurations retain their original identifiers; select the new models for new runs.
Changing the model invalidates reuse for affected cached stages.

Use **API pricing JSON (optional)** or CLI `--api-pricing` to supply dated rate overrides
and additional models. Each `provider::model` record uses USD per million tokens:

```json
{
  "chatgpt::gpt-6-luna": {
    "input": 0.10,
    "cached_input": 0.01,
    "cache_write": 0.125,
    "output": 0.50,
    "long_context_threshold": 272000,
    "long_input_multiplier": 2,
    "long_output_multiplier": 1.5,
    "tier_multipliers": {"default": 1, "flex": 0.5, "fast": 2, "priority": 2},
    "checked_on": "2026-10-14",
    "source": "https://developers.openai.com/api/docs/models/gpt-6-luna"
  }
}
```

Each record may include `tier_rates`, mapping a service tier to a complete set of
`input`, `cached_input`, `cache_write` and `output` rates. These take precedence over
`tier_multipliers` for that tier. Set `cache_write` to `null` when no supported write
rate is known; any reported writes then make the estimate unavailable. The optional
`long_context_threshold_inclusive` flag controls whether the threshold itself incurs
long-context pricing (default `false` preserves the existing override format).

Changing rates affects new cost estimates, not simulation compatibility. Previously saved
cost summaries retain their original estimates and dated rate snapshots in the usage log.

```powershell
uv run python scripts/run_interaction_graph.py --input data/news.csv --graph-config data/graph.json --shared-cache-dir output/interaction_graph/shared_cache --reuse-rewrites
```

The Python API accepts `shared_cache_dir`, `reuse_rewrites` and `api_pricing`.
Explicitly supplied `stdi_cache_dir` remains the per-execution evaluation cache location.
