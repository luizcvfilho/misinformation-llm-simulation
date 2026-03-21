from __future__ import annotations

REWRITE_SYSTEM_INSTRUCTION = (
    "You are a news rewriting assistant. Keep the requested output language exactly, "
    "preserve the central facts, and do not add new information."
)

PROMPT_TEMPLATE = """
Adopt the following personality while rewriting the news article:
{personality}

Required output language: {target_language_name} ({target_language_code}).

Rules:
- Write strictly in {target_language_name} ({target_language_code}).
- Do not translate into another language different from the requested one.
- Preserve the central facts.
- Do not invent data, numbers, quotes, or characters.
- Adjust tone, vocabulary, and style to reflect the requested personality.
- Return only the rewritten text.

Title:
{title}

Original text:
{original_text}
""".strip()
