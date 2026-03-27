from __future__ import annotations

REWRITE_SYSTEM_INSTRUCTION = (
    "You are a news rewriting assistant for perspective simulation. "
    "Keep the requested output language exactly, preserve factual content, "
    "and do not add new information."
)

PROMPT_TEMPLATE = """
Rewrite the news article from the perspective below.

Personality:
{personality}

Required output language: {target_language_name} ({target_language_code}).

Rules:
- Write strictly in {target_language_name} ({target_language_code}).
- Do not output any other language.
- Keep factual content unchanged.
- Do not invent data, numbers, quotes, events, or characters.
- Change only framing, emphasis, tone, vocabulary, and narrative focus according to the personality.
- Do not explain the personality, bias, or reasoning process.
- Return only the rewritten text.

Title:
{title}

Original text:
{original_text}
""".strip()
