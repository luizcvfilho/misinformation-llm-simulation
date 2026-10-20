# Chapter 5 evidence and reproduction record

Prepared for the thesis revision dated 2026-10-19 (America/Sao_Paulo).
All numerical findings are this study's results. No simulation, extraction,
structural judgment or affective rating was requested from a model during this task.

## Reproduction

```powershell
uv run --with matplotlib python scripts/analyze_tcc_chapter_5.py
```

The script uses the project's pandas, NumPy and SciPy dependencies and the
additional Matplotlib package. It reads exactly four tree-level JSONL files in
each of the six final run directories listed in `docs/plano_escrita_tcc.md`.
It does not scan historical runs, news-level copies, caches or path exports.

Outputs:

- `output/analysis/tcc_chapter_5/`: CSV tables, case candidates, PNG previews
  and `manifest.json` with source hashes, versions, lexical rules and checks.
- `tcc/figuras/chapter5/`: 14 PDF vector figures, with English chart labels.
- `tcc/elementos-textuais/capitulo-5-tabelas.tex`: seven Portuguese numerical
  table macros placed by the chapter source.
- `tcc/elementos-textuais/capitulo-5.tex`: manually authored Portuguese prose,
  two additional contextual tables and figure placement. The script does not
  overwrite this prose.

The canonical institutional template is unchanged. `tcc/main.tex` already
includes Chapter 5 and was not edited. Bibliographies were not changed.

## Source coverage and checks

The script checks 7,200 unique nodes and reconstructs 9,600 path positions.
Each run contains 1,200 unique nodes, 50 news IDs and eight four-position paths.
The first 50 eligible article IDs and trimmed descriptions of each frozen CSV
match the saved originals. The English metadata record is excluded before
selecting the first 50 rows. Original bodies are identical across generators
and trees within each corpus/article.

All unique-node rewrites and selected original/incremental scores are valid.
Shared prefixes have identical node IDs and scores. The cumulative score equals
the sum of increments along every path. Projected positions are never added to
unique-node counts. A final path contrast has 50 paired differences, not 400
independent articles; a method summary pools 400 final positions from the same
50 articles and is explicitly descriptive.

### Score definitions

- `original`: selected STDI against the input article (English Dual/Dual;
  Portuguese LLM/LLM).
- `incremental`: selected STDI against the actual parent node.
- `delta_original`: within-path difference in selected original-reference STDI,
  defined only for positions 2–4.
- `cumulative`: saved sum of the path increments, checked numerically.
- `llm_original` / `llm_incremental`: full structural LLM scores with LLM VAD,
  used for the common-procedure corpus comparison.
- `cluster_original`: complete Cluster score with BERT VAD in English.
- `cluster_common_original` / `llm_common_original`: saved `dual_metrics`
  scores from the two structural branches, both using Dual VAD. The selected
  Dual equals their arithmetic mean. Incremental versions are checked likewise.

Identity shortcuts sometimes omit `dual_metrics`. They are accepted only when
the actual reference and rewritten strings are identical, the judge explicitly
records the shortcut and every saved branch metric is zero. Their valid zero
scores are retained; absent nonidentity evaluations are never filled with zero.
Floating-point comparison of common affective metrics allows numerical rounding.

The analysis reports means, medians, quartiles, signed differences, absolute
differences and descriptive Spearman associations. It introduces no hypothesis
tests, population confidence intervals, human numerical labels or new simulation
replicates. Interquartile bands show article dispersion, not uncertainty of a mean.

## Numerical evidence map

| Chapter material | Derived source |
| --- | --- |
| Final and incremental tables, trajectory curves, common LLM corpus comparisons | `trajectory_summary.csv`, `path_positions.csv` |
| Six final path contrasts and fraction with positive differences | `paired_contrasts.csv` |
| Paired generator comparisons | `generator_contrasts.csv` |
| Decreasing original-reference scores at positions 2–4 | `transition_counts.csv` |
| Components and internal contradiction counts | `component_summary.csv`, final rows of `path_positions.csv` |
| Signed affective changes | `llm_signed_*` fields in final rows of `path_positions.csv` |
| Method comparisons by reference, iteration and path | `method_comparison.csv` |
| Feature counts and reference extraction | `original_features.csv` |
| Article-level mean outcome across eight final LLM/LLM paths | `article_outcomes.csv` |
| Lexical and structural feature associations | `feature_associations.csv`, `feature_associations_by_chain.csv` |
| Case selection and all four versions with line references | `case_candidates.json` |

All paths in this table are relative to `output/analysis/tcc_chapter_5/`.

### Historical evidence kept separate

- `output/audit/STDIControlledInformationAudit_20261005/change_type_summary.csv`
  and `findings.md`: 36 primary controlled pairs, three scenarios, the selected
  variant means and 6/9 versus 9/9 ordering expectations. Versions are extraction
  v4, judge v3, structured Cluster v3 and Dual v3. BERT VAD was common to both
  branches. These results do not validate final versions or later numeric rules.
- `output/audit/LLMVADComparison_20261006/interpretation.md`: 30 saved-simulation
  pairs from five articles in six historical paths (BERT 0.01986, LLM 0.07167)
  and 20 synthetic pairs (0.05450, 0.06250). The pilot rubric is distinguished
  from final v2. Larger drift is not interpreted as higher accuracy.
- Earlier detector findings are referenced to Chapter 3 without importing their
  outputs into final-run denominators or presenting them as factual validation.

## Exploratory feature protocol

Indicators are derived from the original bodies before inspecting their
associations with outcomes. Fixed bilingual regex lists operationalize
attribution and uncertainty as lexical proxies. Numbers are distinct regex-matched
digit expressions, with internal whitespace removed; written-out numbers,
semantic roles, dates and units are not inferred. Entity/relation counts use
one original extraction from the GPT N tree per article, fixed across generators.
This extraction is a measurement, not a human reference.

Each article/generator outcome is the equal-weight mean of eight final LLM/LLM
scores, keeping 50 observations per corpus/generator. Per-path correlations are
also exported. Length-adjusted rank associations correlate residuals after
separate OLS regressions of feature ranks and outcome ranks on an intercept and
original-word-count ranks. There are no p-values or causal interpretations.

Every Portuguese original contains a digit expression, and 49/50 have an
attribution cue. Presence/absence contrasts in these cases are unavailable or
unbalanced and are explicitly qualified. Only 7/50 English originals contain
an uncertainty cue. Differences in lexical lists, topics, sources and article
length prevent interpreting corpus differences as isolated language effects.
Word-length ratios and lexical numeric retention are post-generation diagnostics;
the latter does not validate quantities or align their referents.

## Inspected cases

Paths below are relative to `output/interaction_graph/app_runs/`; line numbers
refer to physical JSONL records in the named tree file. Every selected case
is also present in `case_candidates.json`, with all four versions and references.

| Role | Corpus / generator / path | Article ID | Final record |
| --- | --- | --- | --- |
| Near the path median; political reframing | Portuguese / Grok / CCPP | `br2026_1105880049043528` | `simulation_ui_20261018_172257-GROK-BR/02_02_c_tree/02_02_c_tree_steps.jsonl:216` |
| Largest selected final drift; loss of country specificity | English / Gemini / CCPP | `0f348499eb9f8af2db8115fd22420501` | `simulation_ui_20261017_221432-Gemini/02_02_c_tree/02_02_c_tree_steps.jsonl:180` |
| High drift; motive and public-memory interpretation | Portuguese / GPT / DDDD | `br2026_1408304840845860` | `simulation_ui_20261018_172257-GPT-BR/04_04_d_tree/04_04_d_tree_steps.jsonl:298` |
| Neutral preservation and whitespace sensitivity | Portuguese / Gemini / DDNN | `br2026_2212221500172755` | `simulation_ui_20261019_124942-GEMINIBR/04_04_d_tree/04_04_d_tree_steps.jsonl:264` |
| Decreased score without clear substantive restoration | Portuguese / Grok / DDDD | `br2026_2181906645728122` | `simulation_ui_20261018_172257-GROK-BR/04_04_d_tree/04_04_d_tree_steps.jsonl:142` |
| Cluster/LLM disagreement and extraction differences | English / GPT / NNNN | `840a13e28d007c39b74217c8e8a096a7` | `simulation_ui_20261016_213859-GPT/01_01_n_tree/01_01_n_tree_steps.jsonl:280` |

The DDNN case has identical whitespace-normalized second/fourth version tokens,
but original-reference scores 0.629895833 and 0.706770833. This is retained and
described as a measurement limitation, without altering source evaluations.
The fuel case drops from 0.62 to 0.401041667 while retaining the hidden-elite
interpretation. The neutral English case has native scores 0.593898948 Cluster,
0.0703125 LLM and selected Dual 0.331101988; extracted entities change from six
to one, with Cluster entity drift 0.833333333 and relation drift 0.912847317.

## Validation and delivery

The complete local project passed the desktop compiler. Native Tectonic is used
only to export a review PDF and retain detailed diagnostics; pdfLaTeX remains
the recommended Overleaf compiler. New Chapter 5 references and assets are
checked against saved files and labels. Ruff checks the standalone analysis
script. Embedded numerical checks cover source inputs, projections and score
composition rather than re-evaluating model outputs.

The review PDF is exported to `output/pdf/capitulo-5.pdf`. Visual QA and final
page counts are recorded in `output/analysis/tcc_chapter_5/validation.json`.
Warnings already present in earlier chapters are outside the Chapter 5 review;
this task does not claim that the complete thesis is ready for final submission.

Exact Overleaf actions are listed in `docs/tcc_chapter_5_overleaf.md`.
`output/tcc/chapter_5_overleaf.zip` packages the three required TeX files and all
14 figure PDFs with paths relative to the Overleaf project root. It excludes
unrelated files, original datasets, numerical analysis outputs and local tools.
Local edits are not synchronized automatically.
