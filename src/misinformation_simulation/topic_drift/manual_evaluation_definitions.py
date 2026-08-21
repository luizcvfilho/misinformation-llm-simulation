from __future__ import annotations

from dataclasses import dataclass

STDI_COMPONENT_COLUMNS = (
    "theme_drift",
    "subtopic_drift",
    "entity_drift",
    "relation_drift",
    "contradiction_drift",
    "vad_drift",
)
CALCULATED_COMPONENT_COLUMNS = tuple(f"calculated_{column}" for column in STDI_COMPONENT_COLUMNS)
SEMANTIC_COMPONENT_COLUMNS = STDI_COMPONENT_COLUMNS[:5]
SEMANTIC_DRIFT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MANUAL_EXPECTED_STDI_COLUMN = "manual_expected_stdi"
CALCULATED_STDI_COLUMN = "calculated_stdi"
EXCLUDED_SOURCE_TEXT_MARKERS = ("only available in paid plans",)
TRUNCATED_SOURCE_TEXT_MARKERS = (
    "read more",
    "continue reading",
    "full story",
    "click here",
)
TRUNCATED_SOURCE_ENDINGS = ("...", "…", "[...]")
MINIMUM_SOURCE_WORD_COUNT = 50
REWRITE_MINIMUM_WORD_RATIO = 0.85
REWRITE_MAXIMUM_WORD_RATIO = 1.15

MANUAL_REWRITE_SYSTEM_INSTRUCTION = """
You rewrite news articles for a controlled semantic-drift evaluation.
Return only the rewritten news text, with no notes, labels, markdown, or explanation.
Keep the output in the same language as the source article.
""".strip()

MANUAL_REWRITE_PROMPT_TEMPLATE = """
Rewrite the following news item according to the requested controlled change.

Requested controlled change:
{instruction}

Constraints:
- Make the requested change clearly observable.
- Avoid adding an internal contradiction unless the requested change explicitly requires one.
- Keep the result as a coherent news report in a journalistic style.
- The original body has {original_word_count} words. Write between {minimum_word_count} and
  {maximum_word_count} words.
- Do not expand the report with background, examples, recommendations, or other details that
  are not needed for the requested controlled change.

Title: {title}

Original article:
{original_text}
""".strip()

SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION = """
You are an impartial evaluator of semantic drift between two versions of a news report.
Return only valid JSON. Do not add markdown, explanations outside the JSON, or extra keys.
Evaluate semantic meaning, not literal wording. Use the supplied structured extractions as
supporting evidence, but resolve disagreements using the news texts.
""".strip()

SEMANTIC_COMPARISON_PROMPT_TEMPLATE = """
Compare the original and modified versions of this news item. Return one JSON object with
exactly these keys:
- theme_drift: one of 0, 0.25, 0.5, 0.75, 1
- subtopic_drift: one of 0, 0.25, 0.5, 0.75, 1
- entity_drift: one of 0, 0.25, 0.5, 0.75, 1
- relation_drift: one of 0, 0.25, 0.5, 0.75, 1
- contradiction_drift: one of 0, 0.25, 0.5, 0.75, 1
- rationales: object with exactly the five component keys and concise explanations

Scoring scale:
- 0: semantically equivalent for that component
- 0.25: a slight change in emphasis or specificity
- 0.5: a relevant change, still clearly in the same context
- 0.75: a strong change
- 1: essentially different

Component rules:
- theme_drift concerns the primary subject or event. Paraphrase, a different title,
  active/passive voice, or changing a central entity within the same story must not by itself
  change the theme.
- subtopic_drift concerns secondary angles and emphasis. Do not count wording changes as a
  subtopic change.
- entity_drift concerns real-world identity. Aliases, abbreviations, pronouns, and equivalent
  descriptions refer to the same entity. Replacing an actor with a different actor increases it.
- relation_drift concerns factual actions, causality, responsibility, or roles. Equivalent
  active/passive wording has zero drift; reversing roles or changing a factual assertion raises it.
- contradiction_drift concerns internal contradiction introduced in the modified text. Score 0
  for no contradiction, 0.25 for slight/peripheral tension, 0.5 for partial contradiction,
  0.75 for a strong contradiction in an important claim, and 1 for a central contradiction.

Title: {title}

Original text:
{original_text}

Modified text:
{modified_text}

Original structured extraction:
{original_structure}

Modified structured extraction:
{modified_structure}
""".strip()


@dataclass(frozen=True, slots=True)
class MetricRewritePrompt:
    metric: str
    label: str
    instruction: str


METRIC_REWRITE_PROMPTS = (
    MetricRewritePrompt(
        metric="theme_drift",
        label="Main theme",
        instruction=("Change the main theme to a clearly different but plausible subject. "),
    ),
    MetricRewritePrompt(
        metric="subtopic_drift",
        label="Subtopic",
        instruction=(
            "Keep the main theme and central actors, but shift the article's focus to a "
            "different relevant subtopic or angle."
        ),
    ),
    MetricRewritePrompt(
        metric="entity_drift",
        label="Central entity",
        instruction=(
            "Keep the main theme and the key relations, but replace one central person, "
            "organization, place, or group with a different plausible entity."
        ),
    ),
    MetricRewritePrompt(
        metric="relation_drift",
        label="Central relation",
        instruction=(
            "Keep the main theme and central entities, but change one key causal, action, "
            "or responsibility relation between them."
        ),
    ),
    MetricRewritePrompt(
        metric="contradiction_drift",
        label="Internal contradiction",
        instruction=(
            "Preserve the overall topic, but introduce one explicit internal contradiction "
            "about a central fact, number, action, or outcome."
        ),
    ),
    MetricRewritePrompt(
        metric="vad_drift",
        label="Emotional framing (VAD)",
        instruction=(
            "Preserve the factual claims, topic, entities, and relations. Make the framing "
            "clearly more negative and urgent: lower valence and increase arousal through "
            "alarming but journalistic wording. Do not add, remove, or alter facts."
        ),
    ),
)
