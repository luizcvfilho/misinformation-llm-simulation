# Agent Instructions

These instructions apply to coding agents operating in this repository.

## Language and Communication
- Always write code in English.
- Always reply in the same language used by the user.

## Python Tooling
- Prefer `uv` over `pip` for dependency and environment workflows.
- Use `uv add`, `uv sync`, and `uv lock` when applicable.

## Translation Safety
- When translating text, do not change heuristics or algorithms.
- Keep behavior and logic identical unless the user explicitly requests a logic change.

## Code Quality
- Write short, useful comments only when needed.
- Keep naming consistent with existing project conventions.

## Git Safety
- Never use destructive Git commands unless the user explicitly asks for them.

## TCC Writing and Literature
- Before writing, reviewing, or formatting the TCC, read `docs/template_tcc_ic_ufrj.md`, `docs/base_bibliografica_tcc.md`, `docs/plano_escrita_tcc.md`, and `references.bib`.
- Use `templates/modelo-tcc-ic-ufrj-2025/source/` as the canonical LaTeX structure; copy it for working files and never reconstruct or edit the canonical template in place.
- After every TCC task, include an Overleaf synchronization checklist in the user's language that lists every changed or required file, its exact Overleaf path, and the concrete action to reproduce; explicitly state when no file beyond `main.tex` needs changes, and never imply that local changes are synchronized automatically.
- Use `docs/plano_escrita_tcc.md` for the current research question, scope, methodological history, and evidence map.
- Keep software and repository documentation, notebooks, analysis reports, and chart labels in English. Keep TCC writing and presentation documents in Portuguese, and the canonical institutional template in its original language.
- Never invent references or imply that a source supports a claim that was not checked against it.
- Distinguish informational drift, internal contradiction, factual veracity, belief, exposure, and sharing behavior.
- Cite project results as results of this study, not as findings from the external literature.
