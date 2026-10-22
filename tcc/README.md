# Thesis working copy

This directory is the editable thesis workspace copied from
`templates/modelo-tcc-ic-ufrj-2025/source/`. The canonical template remains
unchanged.

Use `main.tex` as the main document and pdfLaTeX as the compiler. The
bibliography used by the document is
`elementos-postextuais/referencias.bib`; keep it synchronized with the
repository-level `references.bib`.

The current draft contains developed chapters 1–7.
Chapter 5 includes descriptive analyses of all six final simulation rounds,
paired path and generator comparisons, Cluster/LLM comparisons with native and
common VAD, a common LLM/LLM corpus comparison, lexical feature diagnostics,
and inspected qualitative cases. Its 14 vector figures are in `figuras/chapter5/`;
its seven generated tables are in `elementos-textuais/capitulo-5-tabelas.tex`.
Independent human numerical annotations remain outside the study scope.
Revision completion is recorded in `docs/plano_revisao_tcc.md`
at the repository root, with the current writing plan in `docs/plano_escrita_tcc.md`.
Chapter 4 evidence checks and remaining provenance limits are recorded in
`docs/tcc_chapter_4_evidence.md`. The complete working project compiled successfully
with the desktop editor after this chapter was written on 2026-10-19.
Chapter 5 evidence and reproduction instructions are recorded in
`docs/tcc_chapter_5_evidence.md`. Reproduce its data, figures and numerical tables
with `uv run --with matplotlib python scripts/analyze_tcc_chapter_5.py`.
Chapter 4 also documents the added descriptive and lexical analysis procedures.
Chapter 6 interprets the trajectory, generator, component and affective findings,
compares the study with iterative-generation and agent-evaluation research, and
discusses measurement, design and generalization limits. Its evidence map and
compilation checks are recorded in `docs/tcc_chapter_6_evidence.md`.
For this chapter, replace only `elementos-textuais/capitulo-6.tex` in the existing
Overleaf project. `main.tex` already includes it; no bibliography, figure or
formatting changes are required. Recompile after uploading the source.
Chapter 7 answers the research question, synthesizes the study's contributions
and limits, and outlines extensions beyond the completed experimental scope.
It uses the existing results without adding simulations or references.
Its evidence and four-page visual review are recorded in
`docs/tcc_chapter_7_evidence.md`. For this chapter, replace only
`elementos-textuais/capitulo-7.tex` in Overleaf. `main.tex` already includes it;
no bibliography, figure or formatting changes are required. Recompile after
uploading the source. The complete project compiled successfully after writing.
For Overleaf, manually upload the updated files from `elementos-textuais/` and
`figuras/chapter5/`, preserving those paths in the existing project. Chapter 5
requires both `capitulo-5.tex` and `capitulo-5-tabelas.tex`, together with its
figure PDFs. Keep the existing chapter inclusion in `main.tex` and recompile
with pdfLaTeX after uploading the files. Local edits are not synchronized
automatically.
Author, advisor, and co-advisor data are intentionally blank in
`elementos-pretextuais/personalizar.tex`. The catalog card, approval page,
Portuguese abstract, and English abstract must be added when their contents
are consolidated.
