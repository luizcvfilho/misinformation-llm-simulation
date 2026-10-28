# Chapter 7 evidence and review record

Prepared on 2026-10-22 (America/Sao_Paulo).
Chapter source: `tcc/elementos-textuais/capitulo-7.tex`.

The Conclusion synthesizes the completed Chapters 1–6 and answers the research
question within the scope recorded in `docs/plano_escrita_tcc.md`. It contains
a continuous synthesis of the research-question answer, contributions and study
limits, followed by a single section for future work. No simulations, extractions,
model evaluations, new statistical analyses or numerical human annotations were
added. The institutional template,
previous chapters and bibliography were preserved.

## Evidence map

All empirical claims refer to this study. The conclusion introduces no external
literature claims or new citations; the literature comparison remains in Chapter 6.
Derived filenames below are relative to `output/analysis/tcc_chapter_5/`.

| Conclusion claim | Existing evidence | Interpretation boundary |
| --- | --- | --- |
| Six rounds, two corpora of 50 entries, three generators and 7,200 unique rewrites | Chapter 4 design; Chapter 5 coverage; `manifest.json`; `docs/tcc_chapter_5_evidence.md` | Path projections are dependent views, not additional generations. English uses Dual/Dual; Portuguese uses LLM/LLM. |
| Perspective-conditioned rewriting can preserve the central event and actors while changing motives, justifications and relations | Chapter 5, `sec:resultados-componentes` and `sec:resultados-casos`; political reframing and added-motive cases | The conclusion synthesizes observed transformations; it does not estimate the frequency of each textual mechanism. Detailed component and path contrasts remain in Chapter 5. |
| Switching perspective does not universally increase drift | Chapter 5, `sec:resultados-contrastes`; `paired_contrasts.csv`, CCPP–CCCC and PPCC–PPPP | Positive means in six conditions do not imply a positive difference for every article. Numerical ranges are not repeated in the revised conclusion. |
| Neutral retransmission preserves received interpretations without automatically restoring the original | Chapter 5 trajectory, incremental and case analyses; Portuguese Gemini DDNN case in `case_candidates.json` | N is instructed to preserve the received message, not verify or correct it. |
| Reversed orders do not produce a universal ranking | Chapter 5 sequence contrasts; `paired_contrasts.csv` | Positions and terminal perspective change jointly; order is not causally isolated. Small differences do not establish equivalence. |
| DDDD, DDNN and NNDD have lower mean valence and higher mean arousal | Chapter 5 signed VAD analysis; final LLM VAD fields in `path_positions.csv` | Estimated expressed tone, not reader emotion, belief or sharing behavior. VAD does not exhaust narrative framing. |
| Generator and corpus comparisons do not establish universal rankings or a language effect | Chapter 5 generator comparisons and common LLM/LLM analysis; Chapter 6 interpretation | Sampling parameters differ, generations have no independent replicates, and corpora are not translations. |
| Cluster/LLM disagreement persists with common VAD | Chapter 5 method analysis; `method_comparison.csv`; English GPT NNNN case | Both branches share LLM-derived inputs; neither is an independent human reference. |
| Extraction/judgment variation can change scores without substantive textual change | Chapter 5 cases; whitespace-sensitive DDNN and fuel cases; `docs/tcc_chapter_5_evidence.md` | A case documents a limitation, not its population frequency. A falling STDI is not automatic restoration. |
| The study provides a traceable procedure, decomposed index and bounded empirical assessment | Chapter 1 objectives; Chapter 4 methods; Chapters 5–6 | Recomputing saved aggregates is distinct from reproducing fresh hosted-model outputs. No general calibration or factual verification is claimed. |

Controlled-pair evidence is summarized as part of instrument development and
assessment, not transferred to the final metric versions as independent validation.
Version boundaries and the exact case records remain documented in
`docs/tcc_chapter_5_evidence.md`.

## Future-work boundaries

The chapter proposes possible extensions, not requirements for finishing this
study: sequence comparisons with fixed composition and terminal perspective;
independent generation replicates; extraction/judge stability checks; alternative
evaluators, rubrics and weight sensitivity; external factual verification; new corpora,
additional languages and models; and richer interaction designs.

Portuguese and the comparison of three generators are described as completed
work. Numerical human annotation is not added as a pending task. Factuality,
belief, exposure and sharing remain separate targets from textual drift.

## Validation and presentation

- The research-question answer was revised after the author's feedback:
  numerical contrasts and path-code listings were replaced by a direct synthesis
  of what can be concluded about content, preservation, order and emotional tone.
  The supporting results remain in Chapter 5; no empirical findings were changed.
- The complete working project compiled successfully with the desktop editor.
- The revised project also compiled successfully with the desktop editor.
  A cached native Tectonic run produced an 85-page first-draft review PDF.
  In that review, Chapter 7 occupied PDF pages 79–82, printed pages 77–80
  (four pages). These historical page numbers do not describe the revised layout.
- Cross-references resolve, labels are unique, and the chapter contains no TODOs.
- The native log has no undefined citations/references, missing-character
  warnings or overfull/underfull boxes. The existing duplicate `@page.1`
  destination warnings concern front-matter pagination.
- All four chapter pages were rendered with Poppler and visually inspected:
  no clipping, overlap, unresolved references or stranded headings were found.
- The underlying NNDD–NNNN sample sizes, positive-difference fractions and
  ranges, and the six positive C/P continuation means were checked against the
  existing paired-contrast CSV. These detailed contrasts remain in Chapter 5,
  rather than being repeated in the revised research-question answer.
  No model scores were recalculated.
- The repository and working-copy bibliographies remain byte-identical.
- Review PDFs, page images and the one-off review script are confined to ignored
  `tmp/` paths and are not repository deliverables.
- This completes the first draft of Chapter 7; it does not establish readiness
  of the entire thesis for submission. Abstracts and institutional elements
  remain outside this task.

## Overleaf mapping

Replace the complete contents of `elementos-textuais/capitulo-7.tex` in the
existing Overleaf project with the local chapter source. Root `main.tex`
already contains `\input{elementos-textuais/capitulo-7.tex}`; preserve that line.
Recompile with pdfLaTeX after uploading the source.

No change is required to `main.tex`, `formatacao.tex`, figures or bibliography.
Repository `references.bib` maps to Overleaf
`elementos-postextuais/referencias.bib`; no bibliography upload is needed here.
This record, `docs/plano_escrita_tcc.md` and `tcc/README.md` have no required
Overleaf destination. Local changes are not synchronized automatically.
