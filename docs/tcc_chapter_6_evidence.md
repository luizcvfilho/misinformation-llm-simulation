# Chapter 6 evidence and review record

Chapter source: `tcc/elementos-textuais/capitulo-6.tex`.
The chapter develops the existing Discussion outline using the completed
Chapters 1–5 and the current thesis writing plan. No model calls, simulations,
extractions, numerical evaluations or statistical analyses were added.
The canonical institutional template and Chapters 1–5 were preserved.

## Argument and internal evidence map

| Discussion topic | Existing evidence | Interpretation boundary |
| --- | --- | --- |
| Subtopics and relations change more than the other content components on average | Chapter 5, `sec:resultados-componentes`; `output/analysis/tcc_chapter_5/component_summary.csv` | Component scores are operational measurements, not proportions of facts lost. |
| Retained names and numbers can coexist with changed motives or explanations | Chapter 5, `sec:resultados-casos`; review/criminal-case and DDNN cases in `case_candidates.json` | Lexical retention does not align semantic roles or establish factuality. |
| Low internal contradiction does not establish textual fidelity | Chapter 5 component analysis; `contradiction_summary.csv` | The 320 positive scores count rewritten versions among 7,200 unique nodes, not distinct contradictions or affected articles. |
| Neutral retransmission can retain an already transformed message | Chapter 5 trajectory/incremental analysis and the Portuguese Gemini DDNN case | N preserves the received message; it does not perform correction against the original or external evidence. |
| Lower original-reference scores do not automatically mean restoration | Chapter 5 fuel case and whitespace-sensitive DDNN case | Score changes can reflect text changes and variation in extraction/judgment. |
| NNDD exceeds NNNN for each observed entry in all six rounds | Chapter 5, `sec:resultados-contrastes`; `paired_contrasts.csv` | This is a descriptive finding under differently permissive prompts, not a claim about people or future inputs. |
| Switching C/P raises continuation means without a universal direction per article | Same contrast sources | The case-based explanation involving replacement of justifications is an interpretation, not a measured mechanism frequency. |
| Reversed orders do not have one universal ranking | Same contrast sources | Order, position and terminal perspective change jointly. Small differences are not an equivalence test. |
| Generator rankings depend on path | Chapter 5 generator contrasts; `generator_contrasts.csv` | Effective sampling parameters differ and generation has no independent replicates. |
| Feature associations vary across corpora | Chapter 5, `sec:resultados-atributos`; `feature_associations.csv` | The length adjustment does not control topic, event or source. Extracted counts are not validated factual density. |
| D paths have lower mean valence and higher mean arousal under LLM VAD | Chapter 5 signed VAD analysis; final rows in `path_positions.csv` | These are textual-tone estimates. They do not measure reader emotion, belief or sharing. |
| Methods disagree even with common VAD | Chapter 5, `sec:resultados-metodos`; `method_comparison.csv` and the English GPT NNNN case | Neither method is a human reference; the branches share extraction and other LLM-derived inputs. |
| Common LLM/LLM procedure does not isolate language | Chapter 5, `sec:resultados-corpora`; Chapter 4 data descriptions | The corpora are not translations and differ in origin, format, event and length. |

CSV and JSON filenames in this table are relative to
`output/analysis/tcc_chapter_5/`. Exact source JSONL records and historical
version boundaries are documented in `docs/tcc_chapter_5_evidence.md`.
All project findings are attributed to this study, not to external literature.

## Primary-source checks

All eight cited keys already exist in `references.bib` and its byte-identical
working copy, `tcc/elementos-postextuais/referencias.bib`.
No new references or metadata changes were necessary.

- [Mohamed et al., ACL 2025](https://aclanthology.org/2025.acl-long.371.pdf),
  introduction, experimental setups in Sections 4.1–4.3 and the rephrasing
  experiments: iterative output-to-input generation, primarily translation,
  with additional rephrasing. This supports a design comparison, not numerical
  equivalence or a claim that this thesis replicated their experiments.
- [Liu et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1330.pdf),
  Sections 3.3 and 4, and Appendix D: FUSE roles and interaction design,
  FUSE-EVAL dimensions and reported human evaluation. The thesis does not
  inherit that validation. No contested evaluator identifier or correlation
  from the paper was imported.
- [Chuang et al., NAACL 2024](https://aclanthology.org/2024.findings-naacl.211/),
  abstract: model tendencies and prompt-induced confirmation bias affect
  simulated opinion dynamics. The thesis does not claim the same mechanism
  was independently measured in its rewrite chains.
- [Frisch and Giulianelli, PERSONALIZE 2024](https://aclanthology.org/2024.personalize-1.9.pdf),
  introduction and conclusion: profile-dependent persona consistency and
  linguistic alignment. NeutralRelay preservation is not interpreted as
  adopting a belief or as a replication of their personality experiment.
- [Buechel and Hahn, EACL 2017](https://aclanthology.org/E17-2092/), abstract:
  writer/reader annotation perspectives. The project's LLM tone rubric is its
  own operational choice, not an EmoBank validation result.
- [Haldar and Hockenmaier, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1361/),
  abstract: repeated judge-score inconsistency. Their findings are not used to
  quantify STDI error or justify a particular repetition count.
- [Saha, Wagde and Kveton, AISTATS 2026](https://proceedings.mlr.press/v300/saha26a.html),
  abstract: mean-score estimation and variance-adaptive query allocation under
  budget. This study uses three fixed evaluations, not their adaptive algorithm.
- [Abdurahman et al., 2025](https://journals.sagepub.com/doi/10.1177/25152459251325174),
  methodological reporting and robustness recommendations, especially Table 5:
  document processing and settings, justify/test prompts, and examine validity.
  Human numerical annotation, prompt sensitivity and independent generation
  replication were not added or represented as completed procedures.

## Initial drafting validation and visual review

- The complete working project compiled successfully using the desktop editor.
- A separate cached native Tectonic run produced a full 86-page review PDF.
  Chapter 6 spans PDF pages 73–82, printed pages 71–80 (ten pages).
- All Chapter 6 citation keys and cross-references resolve. There are no
  duplicate labels in the working LaTeX sources.
- The native log contains no undefined citations/references, missing-character
  warnings, or overfull/underfull boxes. Duplicate `@page.1` destination warnings
  remain associated with the project's existing front-matter pagination.
- All ten Chapter 6 pages were visually reviewed after rendering with Poppler:
  no clipping, overlap, unresolved references or stranded headings were found.
- `git diff --check` passed. Both bibliographies remain byte-identical.
- Compiled PDFs, page renders and the one-off validation script are confined
  to ignored `tmp/` paths and are not retained as repository deliverables.
- This review concerns Chapter 6. The Chapter 7 outline and pending institutional
  elements remain outside this writing task; successful compilation does not
  establish that the full thesis is ready for submission.

## Clarity revision (2026-10-22)

The existing Chapter 6 source was revised in place to keep interpretation central
and reduce material already developed in Chapters 2, 4 and 5. The whitespace-based
word count decreased from 3,934 to 2,245 (42.9%).

- Replaced repeated case narratives, numerical recaps and procedural explanations
  with brief interpretive statements and references to the relevant sections.
- Kept the local distinction between incremental change and original-reference
  drift, the order/terminal-perspective confound, and the meaning of VAD estimates.
- Consolidated measurement and design caveats in the limitations section and
  removed the final section that repeated findings immediately before Chapter 7.
- Preserved all eight citation keys and the interpretation boundaries in the
  evidence map above. No new findings, sources or model calls were introduced.
- Checked all working-source cross-references and label uniqueness; verified
  citation keys and byte-identical bibliography copies. `git diff --check` passed.
- The saved chapter compiled successfully with the desktop compiler as part of
  the existing `tcc/main.tex` project. The editor remained open; no separate PDF
  was compiled or exported. The page count and visual review above describe the
  initial draft, not this shorter revision.

## Overleaf mapping

Replace `elementos-textuais/capitulo-6.tex` in the existing Overleaf project
with the complete local chapter source. Keep the existing line
`\input{elementos-textuais/capitulo-6.tex}` in root `main.tex`.
Recompile with pdfLaTeX after replacing the file.

No change is required to `main.tex`, `formatacao.tex`, the bibliography,
tables or figures. The bibliography mapping remains repository `references.bib`
to Overleaf `elementos-postextuais/referencias.bib`; no upload is needed for
this task. The writing plan, this evidence record and `tcc/README.md` have no
required Overleaf destination. Local edits are not synchronized automatically.
