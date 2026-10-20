# Thesis working copy

This directory is the editable thesis workspace copied from
`templates/modelo-tcc-ic-ufrj-2025/source/`. The canonical template remains
unchanged.

Use `main.tex` as the main document and pdfLaTeX as the compiler. The
bibliography used by the document is
`elementos-postextuais/referencias.bib`; keep it synchronized with the
repository-level `references.bib`.

The current draft contains developed chapters 1–5 and outlines for chapters 6–7.
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
