# Portuguese and English comparison of four VAD scorers

This diagnostic compares NRC v1, MEmoLon MTL_grouped, NRC v2.1, and the current project model RobroKools/vad-bert on the same texts. It is not an accuracy benchmark.

NRC v1 uses its official Portuguese translation and original English lexicon; MEmoLon uses language-specific Portuguese/English files. NRC v2.1 uses its English lexicon in both languages; Portuguese matches are lexical overlap, not validated Portuguese recognition. RobroKools/vad-bert is documented for English and Portuguese inference is an exploratory probe. English news controls use the same manual translations recorded in the previous comparison.

The current BERT model gives the expected direction in 2/3 Portuguese and 3/3 English affect-word contrasts. For simple negation contrasts the counts are 0/3 in Portuguese and 3/3 in English. Its valence gap on the same-word scope contrast is 0.0130 in Portuguese and 1.2514 in English. These chosen controls measure qualitative sensitivity only, not accuracy or general language competence.

## News coverage

| Language | Model | Matched / total tokens | Coverage |
| --- | --- | ---: | ---: |
| pt | nrc_v1 | 103/333 | 30.93% |
| pt | memolon | 326/333 | 97.90% |
| pt | nrc_v2_1 | 37/333 | 11.11% |
| pt | current_model | not applicable | not applicable |
| en | nrc_v1 | 113/317 | 35.65% |
| en | memolon | 306/317 | 96.53% |
| en | nrc_v2_1 | 198/317 | 62.46% |
| en | current_model | not applicable | not applicable |

Coverage is pooled by token count, not averaged across documents. Portuguese input consists of the same three rescue, stadium accident, and hailstorm captions used in the earlier smoke test. English input consists of manual translations. These collected captions are not independently verified full articles. Hashtags, names, numbers, URLs, and function words remain in the token denominator. Lexical coverage is not applicable to the contextual neural model; it is never reported as 100%. Wordpiece lengths, unknown pieces, and truncation flags are in scores.csv.

## Per-sample news scores

| Sample | Language | Model | Coverage | V | A | D |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Flood rescue | pt | nrc_v1 | 28.46% | 3.178 | 3.103 | 3.260 |
| Stadium accident | pt | nrc_v1 | 33.98% | 3.070 | 3.019 | 3.036 |
| Hailstorm | pt | nrc_v1 | 30.84% | 3.118 | 2.587 | 3.017 |
| Poll worker training headline | pt | nrc_v1 | 25.00% | missing | missing | missing |
| Flood rescue | pt | memolon | 98.37% | 3.093 | 2.581 | 3.121 |
| Stadium accident | pt | memolon | 96.12% | 3.066 | 2.586 | 3.102 |
| Hailstorm | pt | memolon | 99.07% | 3.097 | 2.520 | 3.125 |
| Poll worker training headline | pt | memolon | 100.00% | missing | missing | missing |
| Flood rescue | pt | nrc_v2_1 | 8.94% | 3.218 | 2.790 | 2.850 |
| Stadium accident | pt | nrc_v2_1 | 13.59% | 3.060 | 2.848 | 3.058 |
| Hailstorm | pt | nrc_v2_1 | 11.21% | 3.360 | 2.601 | 3.008 |
| Poll worker training headline | pt | nrc_v2_1 | 0.00% | missing | missing | missing |
| Flood rescue (translation) | en | nrc_v1 | 39.84% | 3.333 | 3.137 | 3.263 |
| Stadium accident (translation) | en | nrc_v1 | 31.31% | 3.148 | 2.959 | 3.097 |
| Hailstorm (translation) | en | nrc_v1 | 34.74% | 3.139 | 2.877 | 3.187 |
| Flood rescue (translation) | en | memolon | 98.37% | 3.087 | 2.521 | 3.136 |
| Stadium accident (translation) | en | memolon | 92.93% | 3.036 | 2.493 | 3.105 |
| Hailstorm (translation) | en | memolon | 97.89% | 3.071 | 2.514 | 3.109 |
| Flood rescue (translation) | en | nrc_v2_1 | 64.23% | 3.309 | 3.085 | 3.171 |
| Stadium accident (translation) | en | nrc_v2_1 | 58.59% | 3.086 | 3.057 | 3.116 |
| Hailstorm (translation) | en | nrc_v2_1 | 64.21% | 3.096 | 2.957 | 3.191 |
| Flood rescue | pt | current_model | not applicable | 3.022 | 3.020 | 3.139 |
| Flood rescue (translation) | en | current_model | not applicable | 2.863 | 3.369 | 2.958 |
| Stadium accident | pt | current_model | not applicable | 3.123 | 3.288 | 3.213 |
| Stadium accident (translation) | en | current_model | not applicable | 2.591 | 3.356 | 2.869 |
| Hailstorm | pt | current_model | not applicable | 3.028 | 2.975 | 3.134 |
| Hailstorm (translation) | en | current_model | not applicable | 2.940 | 3.119 | 3.078 |
| Poll worker training headline | pt | current_model | not applicable | missing | missing | missing |

The TSE training headline is assessed for lexical coverage only, and has deliberately suppressed affect scores. No political actors or positions are compared.

## Control summary

| Model | Language | Control | Expected direction / available | Missing |
| --- | --- | --- | ---: | ---: |
| nrc_v1 | pt | affect_word_control | 3/3 | 0 |
| nrc_v1 | pt | negation_control | 3/3 | 0 |
| nrc_v1 | pt | negation_scope_control | 0/1 | 0 |
| nrc_v1 | en | affect_word_control | 3/3 | 0 |
| nrc_v1 | en | negation_control | 1/3 | 0 |
| nrc_v1 | en | negation_scope_control | 0/1 | 0 |
| memolon | pt | affect_word_control | 3/3 | 0 |
| memolon | pt | negation_control | 3/3 | 0 |
| memolon | pt | negation_scope_control | 0/1 | 0 |
| memolon | en | affect_word_control | 3/3 | 0 |
| memolon | en | negation_control | 3/3 | 0 |
| memolon | en | negation_scope_control | 0/1 | 0 |
| nrc_v2_1 | pt | affect_word_control | 0/0 | 3 |
| nrc_v2_1 | pt | negation_control | 0/0 | 3 |
| nrc_v2_1 | pt | negation_scope_control | 0/0 | 1 |
| nrc_v2_1 | en | affect_word_control | 3/3 | 0 |
| nrc_v2_1 | en | negation_control | 1/3 | 0 |
| nrc_v2_1 | en | negation_scope_control | 0/1 | 0 |
| current_model | pt | affect_word_control | 2/3 | 0 |
| current_model | pt | negation_control | 0/3 | 0 |
| current_model | pt | negation_scope_control | 1/1 | 0 |
| current_model | en | affect_word_control | 3/3 | 0 |
| current_model | en | negation_control | 3/3 | 0 |
| current_model | en | negation_scope_control | 1/1 | 0 |

Available excludes missing scores. A result of 0/0 means no evaluable contrast.

## Diagnostic controls

| Control | Language | Dimension | Model | Expected-direction delta | Observed? |
| --- | --- | --- | --- | ---: | --- |
| affect_word_control | pt | valence | nrc_v1 | 2.643 | True |
| affect_word_control | pt | valence | memolon | 1.234 | True |
| affect_word_control | pt | valence | nrc_v2_1 | missing | None |
| affect_word_control | pt | valence | current_model | -0.213 | False |
| affect_word_control | pt | arousal | nrc_v1 | 2.748 | True |
| affect_word_control | pt | arousal | memolon | 0.503 | True |
| affect_word_control | pt | arousal | nrc_v2_1 | missing | None |
| affect_word_control | pt | arousal | current_model | 0.168 | True |
| affect_word_control | pt | dominance | nrc_v1 | 2.891 | True |
| affect_word_control | pt | dominance | memolon | 0.734 | True |
| affect_word_control | pt | dominance | nrc_v2_1 | missing | None |
| affect_word_control | pt | dominance | current_model | 0.101 | True |
| affect_word_control | en | valence | nrc_v1 | 3.401 | True |
| affect_word_control | en | valence | memolon | 1.905 | True |
| affect_word_control | en | valence | nrc_v2_1 | 3.439 | True |
| affect_word_control | en | valence | current_model | 2.653 | True |
| affect_word_control | en | arousal | nrc_v1 | 3.091 | True |
| affect_word_control | en | arousal | memolon | 0.830 | True |
| affect_word_control | en | arousal | nrc_v2_1 | 3.217 | True |
| affect_word_control | en | arousal | current_model | 1.266 | True |
| affect_word_control | en | dominance | nrc_v1 | 2.804 | True |
| affect_word_control | en | dominance | memolon | 0.927 | True |
| affect_word_control | en | dominance | nrc_v2_1 | 2.840 | True |
| affect_word_control | en | dominance | current_model | 0.397 | True |
| negation_control | pt | valence | nrc_v1 | 0.985 | True |
| negation_control | pt | valence | memolon | 0.208 | True |
| negation_control | pt | valence | nrc_v2_1 | missing | None |
| negation_control | pt | valence | current_model | -0.065 | False |
| negation_control | pt | arousal | nrc_v1 | 0.938 | True |
| negation_control | pt | arousal | memolon | 0.069 | True |
| negation_control | pt | arousal | nrc_v2_1 | missing | None |
| negation_control | pt | arousal | current_model | -0.089 | False |
| negation_control | pt | dominance | nrc_v1 | 0.344 | True |
| negation_control | pt | dominance | memolon | 0.038 | True |
| negation_control | pt | dominance | nrc_v2_1 | missing | None |
| negation_control | pt | dominance | current_model | -0.017 | False |
| negation_control | en | valence | nrc_v1 | 0.000 | False |
| negation_control | en | valence | memolon | 0.219 | True |
| negation_control | en | valence | nrc_v2_1 | 0.000 | False |
| negation_control | en | valence | current_model | 2.254 | True |
| negation_control | en | arousal | nrc_v1 | 0.000 | False |
| negation_control | en | arousal | memolon | 0.041 | True |
| negation_control | en | arousal | nrc_v2_1 | 0.000 | False |
| negation_control | en | arousal | current_model | 0.798 | True |
| negation_control | en | dominance | nrc_v1 | 0.131 | True |
| negation_control | en | dominance | memolon | 0.020 | True |
| negation_control | en | dominance | nrc_v2_1 | 0.071 | True |
| negation_control | en | dominance | current_model | 0.624 | True |
| negation_scope_control | pt | valence | nrc_v1 | 0.000 | False |
| negation_scope_control | pt | valence | memolon | 0.000 | False |
| negation_scope_control | pt | valence | nrc_v2_1 | missing | None |
| negation_scope_control | pt | valence | current_model | 0.013 | True |
| negation_scope_control | en | valence | nrc_v1 | 0.000 | False |
| negation_scope_control | en | valence | memolon | 0.000 | False |
| negation_scope_control | en | valence | nrc_v2_1 | 0.000 | False |
| negation_scope_control | en | valence | current_model | 1.251 | True |

Scope controls compare `Estou feliz, mas não estou triste.` with `Estou triste, mas não estou feliz.`, and their English counterparts. These have the same word multiset but different negation scope. Their The table gives their valence differences for each scorer and language. Word-control deltas are high minus low for the target dimension. Negation deltas are affirmative minus negated. A positive delta is a qualitative diagnostic expectation, not a human numeric label. Passing a negation direction check does not establish scope understanding: adding a lexically scored negator can shift an average without changing the value of the negated adjective.

## Method and limits

All three lexicons use identical tokenization, longest-match rules, duplicate aggregation, occurrence weighting, and missing-value handling. NRC Portuguese can have several English source words translated to the same term; their ratings are averaged first to avoid arbitrary row-order choices. Multiword translations are supported. Unsupported lexical forms are counted in the manifest. Only input-relevant entries are retained in memory after streaming each complete lexicon. Case-insensitive normalization also merges repeated and differently cased MEmoLon rows. Their values are averaged by the same rule as NRC translations. The manifest counts input-relevant duplicate rows for both resources. A different case or duplicate policy could produce different scores.

All reported VAD scores use the project's nominal 1-5 scale. NRC v1 uses 1+4*x from [0,1]; NRC v2.1 uses 3+2*x from [-1,1]; MEmoLon uses (x+1)/2 from [1,9]. BERT uses raw regression logits, exactly as the project does. No source or model outputs are clipped. Resource estimates outside their nominal native range are counted rather than silently clipped. Equal numeric scales do not imply equal calibration: the resources use different source norms and rating procedures.

Model deltas and matched-term traces show what changes with the resource. Translation deltas measure sensitivity to this manual wording choice, not cross-language accuracy. News coverage, score agreement, and affect-control separation cannot identify the correct human document score. A held-out, human-rated Portuguese Brazilian and English sample is needed before adopting any scorer as more accurate.
The current model is loaded from a cached, revision-identified Hugging Face snapshot and run with the existing project inference functions, on CPU in evaluation mode. The manifest records model hashes, package versions, batch size, and the 512-token truncation limit. No new translations or model tuning were performed for this extension.

The production scorer, model default, simulation, STDI, and TCC files are unchanged. This audit does not integrate MEmoLon into the application. The model_deltas.csv schema now represents all six pairs; its columns are model_a, model_b, and signed delta_b_minus_a values, replacing the previous MEmoLon-minus-NRC-only columns.

## Sources and reproduction

- [NRC VAD v1 and automatic Portuguese translation](https://saifmohammad.com/WebPages/nrc-vad.html).
- [MEmoLon code and recommended MTL_grouped release](https://github.com/JULIELab/MEmoLon).
- [MEmoLon data record](https://zenodo.org/records/3756607).
- [MEmoLon paper](https://aclanthology.org/2020.acl-main.112/).
- [Current model card](https://huggingface.co/RobroKools/vad-bert).
- [TSE training headline](https://www.tse.jus.br/comunicacao/noticias/2026/Setembro/mesarios-tem-ate-3-de-outubro-para-concluir-treinamento-das-eleicoes-2026).

Public resources are cached locally, not committed or redistributed as project data. Downloads use the official NRC ZIP and selected Portuguese/English members of the official MEmoLon ZIP via HTTP ranges. ZIP member CRCs and SHA256 are recorded; the complete 2.36 GB archive MD5 was not checked.

```powershell
uv run python scripts/download_vad_comparison_resources.py --include-portuguese-pilot
uv --cache-dir .cache/uv run --offline --no-sync python scripts/compare_portuguese_vad_lexicons.py
```

The download step needs network access; the comparison runs offline once the cache exists. NRC v2.1 also requires the existing archive in `.cache/nrc_vad/`, and BERT requires its existing Hugging Face snapshot. `manifest.json` records full texts, source IDs, resource statistics, and hashes. `scores.csv` contains all sample scores; `matched_terms.csv` contains occurrence-level lexical evidence for lexicons only; the remaining CSVs contain coverage and paired deltas. model_deltas.csv contains all six unordered model pairs and signed model_b minus model_a differences. control_summary.csv counts available and expected-direction contrasts separately, so missing is never a pass.
