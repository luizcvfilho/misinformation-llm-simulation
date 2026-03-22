# Copilot Instructions

These are the default instructions for GitHub Copilot in this repository.

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
