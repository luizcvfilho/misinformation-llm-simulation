# Chapter 4 evidence record

Checked on 2026-10-19 for `tcc/elementos-textuais/capitulo-4.tex`.
This record supports thesis writing; it does not add simulations or recompute
historical model judgments. The institutional template was not modified.

## Final executions

All paths below are relative to `output/interaction_graph/app_runs/`.

| Corpus | Generator | Execution directory | Selected STDI / VAD |
| --- | --- | --- | --- |
| English | gpt-6-luna | `simulation_ui_20261016_213859-GPT` | Dual / Dual |
| English | grok-build-0.1 | `simulation_ui_20261017_160546-GROK` | Dual / Dual |
| English | gemini-3.5-flash-lite | `simulation_ui_20261017_221432-Gemini` | Dual / Dual |
| Portuguese | gpt-6-luna | `simulation_ui_20261018_172257-GPT-BR` | LLM / LLM |
| Portuguese | grok-build-0.1 | `simulation_ui_20261018_172257-GROK-BR` | LLM / LLM |
| Portuguese | gemini-3.5-flash-lite | `simulation_ui_20261019_124942-GEMINIBR` | LLM / LLM |

The four tree-level `*_summary.json` files and four tree-level `*_steps.jsonl`
files were inspected in each execution. Every execution has 1,200 unique-node
records, 1,200 successful rewrites, and 1,200 valid selected scores in each of the
original and incremental comparisons. Every tree contains 50 input rows.
These are final logical-node counts, not API request or retry counts.

All 7,200 node records were checked against the first 50 eligible rows of the
respective CSV: zero original-text mismatches and zero article-ID mismatches.
The metadata record in the English CSV was excluded before taking the first 50.
Path exports were not added to unique-node counts.

All original extraction records identify `structured_extraction_v5`.
Original and rewritten LLM VAD assessments in every node identify
`llm_vad_expressed_tone_v2`. Available structural-judge sample provenance identifies
`structured_judge_v5`. Sample records, cache reuse and identical-text shortcuts
must not be counted as new independent generations.

## Corpus checks

| Complete file | SHA-256 |
| --- | --- |
| `data/graphs/graph_politics_news.csv` | `46fabffefde42ed7a8746d97093f1dab19f5f1524908e4217432aea755cffb64` |
| `data/elections_2026/simulation_news_100_cleaned.csv` | `89582261293f87d494390b4e08e63b5c410b1b1ad2e606fa164d58c351f9abea` |

English selection evidence: `data/graphs/graph_politics_news.selection.md`, the
embedded metadata record, and the NewsData collection implementation in
`src/misinformation_simulation/datasets/newsdata.py`. The source contains 5,519
records; screening produced 2,178 eligible distinct titles; 100 were selected
intentionally, with the first 50 used in the final runs. The selection document's
description of default title context does not override the final runs' explicit
`rewrite_original_title_context=false` and
`topic_extraction_original_title_context=false`.

Portuguese provenance: CSV fields `source_file`, `source_post_id`,
`conta_usuario`, `selection_rank`, `topic`, `event`, `reason`, `watch`, and
`review_status`, plus `notebooks/elections_2026_eda.ipynb` for the underlying
corpus exploration. The first 50 cleaned rows contain 24 source accounts and 45
distinct event labels; their recorded publication dates range from October 5 to
October 10, 2026. Event labels are metadata, not independently verified events.

Comparing the 100 rows of `simulation_news_100.csv` and
`simulation_news_100_cleaned.csv` shows 98 changed descriptions. Only
`description`, `words`, `substantive_words`, and `text_sha256` differ. The first
row removes a site-reading invitation and a photo credit. This observation does
not establish a complete automatic cleaning algorithm for all rows. The frozen
cleaned CSV is the reproducible simulation input. The review status explicitly
does not claim factual verification.

## Operational sources

| Topic | Implementation / documentation |
| --- | --- |
| Rewrite permissions, perspectives and title exclusion | `src/misinformation_simulation/config/prompts.py` |
| Requested versus sent temperature | `src/misinformation_simulation/llm/retry.py`; saved extraction/judge provenance |
| Graph-wide comparison corpus | `src/misinformation_simulation/apps/interaction_graph_parallel.py`, `fit_graph`; `simulation/graph_execution.py` |
| Encoder, matching, K-means and core weights | `src/misinformation_simulation/topic_drift/cluster_comparison.py` |
| Structured Cluster v4, complete formula and repeated judgments | `src/misinformation_simulation/topic_drift/structured_comparison.py` |
| Polarity and numeric adjustment | `src/misinformation_simulation/topic_drift/qualifiers.py`, `numeric_comparison.py`; `docs/dual_stdi.md` |
| Selected VAD / final Dual recomputation | `docs/vad_stdi_methods.md`; `text_metrics/vad_evaluation.py` |
| BERT tokenization and inference | `src/misinformation_simulation/text_metrics/vad.py` |
| Final LLM VAD v2 rubric | `src/misinformation_simulation/text_metrics/llm_vad.py`; final node assessments |

The `cluster_v2` constant in the legacy Cluster module does not replace the
structured `cluster_v4` implementation. Likewise, `dual_stdi_v6` alone does not
identify the selected comparison method. The historical VAD pilot documentation
describes v1 and must not be used to label final v2 assessments.

## Primary references checked for the chapter

- [Sentence-BERT](https://aclanthology.org/D19-1410/): sentence embeddings and cosine similarity.
- [MiniLM](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html): compact-model family; not validation of STDI rules.
- [vad-bert model card](https://huggingface.co/RobroKools/vad-bert): English BERT VAD regression, not a Portuguese validation study.
- [Rating Roulette](https://aclanthology.org/2025.findings-emnlp.1361/): judge variation across runs, not proof that three draws suffice.
- [LLM-as-a-Judge on a Budget](https://proceedings.mlr.press/v300/saha26a.html): repeated-query mean estimation and adaptive allocation; the study uses a fixed count.

All five keys already existed in both bibliographies. No bibliography changes
were necessary. Project-specific weights, rubrics and counts are identified as
this study's operational choices or observations.

## Validation and documentary limits

- The complete working LaTeX project compiled successfully with the desktop compiler.
- Chapter citation keys and cross-references resolve to existing entries/labels.
- There are no duplicate labels in the current working sources.
- Hash strings in the chapter concatenate to the verified 64-character digests.
- `references.bib` and `tcc/elementos-postextuais/referencias.bib` are byte-identical.
- Compilation success does not replace final visual page-by-page review.
- The consulted corpus records do not establish per-text redistribution licenses.
- Final BERT score records name the model but do not explicitly pin an immutable
  weight revision. A current local snapshot or an older audit revision is not
  evidence of which snapshot was loaded in every final execution.
- Editorial cleaning provenance is incomplete beyond the frozen files and their
  observable differences; no missing cleaning protocol was invented.
- No independent human numerical reference, new inferential statistics, new
  simulation or unexecuted sensitivity analysis was introduced.

## Overleaf mapping

Replace `elementos-textuais/capitulo-4.tex` with the complete local chapter.
Keep the existing `main.tex` inclusion of that file. Bibliography files and
formatting files do not require changes for this chapter. This evidence record,
the updated writing plan and `tcc/README.md` have no required Overleaf destination.
Local edits are not synchronized automatically.
