from __future__ import annotations

from dataclasses import dataclass

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
- Do not add titles, headlines, markdown headings, labels, or prefixes.
- Return only the rewritten text.

Title:
{title}

Original text:
{original_text}
""".strip()


PERSONALITY_PROMPTS = {
    "ConservativeRight": (
        "You are a strongly right-wing, socially conservative commentator. "
        "Frame events through tradition, authority, nationalism, family values, "
        "and skepticism of progressive institutions. "
        "Prioritize social order, personal responsibility, "
        "and cultural continuity in how you interpret and rewrite the news."
    ),
    "ProgressiveLeft": (
        "You are a strongly left-wing, progressive commentator. "
        "Frame events through social justice, inequality, anti-discrimination, "
        "labor rights, and institutional reform. "
        "Prioritize structural causes, historical inequality, "
        "and protection of vulnerable groups in how you interpret and rewrite the news."
    ),
    "ConspiracyDenialist": (
        "You are a conspiratorial, denialist persona that rejects official explanations. "
        "Interpret events as coordinated manipulation by hidden elites and institutions. "
        "Prioritize suspicion, hidden motives, and narrative inversion "
        "when interpreting and rewriting the news."
    ),
    "InvestigativeSkeptic": (
        "You are an investigative skeptic focused on evidence quality and narrative construction. "
        "Question source credibility, missing context, selective framing, and rhetorical bias. "
        "Maintain a critical but neutral tone, "
        "emphasizing uncertainty and competing interpretations "
        "when rewriting the news."
    ),
    "EmotionalAmplifier": (
        "You are an emotionally expressive news communicator. "
        "Emphasize the human stakes and emotional significance already present in the news. "
        "Use vivid, engaging language while keeping the intensity proportional to the source. "
        "Preserve the facts, uncertainty, and context without inventing details "
        "or exaggerating claims."
    ),
    "ConciliatoryCommunicator": (
        "You are a conciliatory communicator addressing readers with different worldviews. "
        "Use respectful, accessible language and highlight shared concerns supported by the news. "
        "Present disagreements fairly while preserving the strength of the evidence "
        "and uncertainty. "
        "Do not invent consensus, create false equivalence, or change the facts."
    ),
}


TOPIC_DOMAINS = (
    "arts_culture_entertainment_and_media",
    "conflict_war_and_peace",
    "crime_law_and_justice",
    "disaster_accident_and_emergency_incident",
    "economy_business_and_finance",
    "education",
    "environment",
    "health",
    "human_interest",
    "labour",
    "lifestyle_and_leisure",
    "politics",
    "religion_and_belief",
    "science_and_technology",
    "society",
    "sport",
    "weather",
)

TOPIC_DOMAIN_VALUES = ", ".join(TOPIC_DOMAINS)

TOPIC_DOMAIN_CLASSIFICATION_RULE = (
    "Select the article's primary domain from the top-level IPTC Media Topics "
    f"controlled vocabulary: {TOPIC_DOMAIN_VALUES}."
)

TOPIC_DRIFT_SYSTEM_INSTRUCTION = """
You extract the semantic structure of a news report for topic-drift analysis.
Return only valid JSON.
Do not add markdown fences, explanations, or extra keys.
Use concise, factual phrases grounded in the provided text.
If a field is unavailable, use null or an empty array.
""".strip()

TOPIC_DRIFT_PROMPT_TEMPLATE = f"""
Analyze the following news item and return a JSON object with exactly these keys:
- main_topic: string or null
- topic_domain: string or null
- subtopics: array of strings
- central_entities: array of strings
- central_relations: array of objects with keys subject, action, object
- narrative_frame: string or null
- has_internal_contradiction: boolean
- internal_contradiction_score: number between 0 and 1

Extraction rules:
- main_topic must capture the primary subject of the article.
- topic_domain must be exactly one of the controlled vocabulary values.
  {TOPIC_DOMAIN_CLASSIFICATION_RULE}
  Use null only when no domain can be determined.
- subtopics must list secondary themes or angles.
- central_entities must include the most important people, organizations, places, or groups.
- central_relations must describe core factual relations in (subject, action, object) form.
- narrative_frame is optional and should summarize the dominant framing if present.
- has_internal_contradiction must be true only when the text contradicts itself internally.
- internal_contradiction_score must grade the severity/centrality of internal contradiction:
  0 means none, 0.25 means slight or peripheral tension, 0.5 means partial contradiction,
  0.75 means strong contradiction in an important claim, and 1 means a central contradiction.
- Keep outputs short and normalized.
- Do not invent facts beyond the text.

Title: {{title}}

Text:
{{text}}
""".strip()

TOPIC_DOMAIN_ONLY_SYSTEM_INSTRUCTION = """
You classify the primary domain of a news report.
Return only one domain label, with no JSON, punctuation, explanation, or markdown.
""".strip()

TOPIC_DOMAIN_ONLY_PROMPT_TEMPLATE = f"""
Classify the following news item into exactly one primary domain from this closed list:
{", ".join(TOPIC_DOMAINS)}.

These are the top-level IPTC Media Topics categories. Do not return null.

Title: {{title}}

Text:
{{text}}
""".strip()

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

DEFAULT_FALSE_TO_TRUE_SYSTEM_INSTRUCTION = (
    "You are a careful news verification and rewriting assistant. "
    "Rewrite false or unsupported news text into a truthful, neutral news "
    "article about the same topic. "
    "Preserve the approximate length, structure, and journalistic style. "
    "Correct or remove unsupported claims. Do not invent sources, "
    "quotes, numbers, dates, or events. "
    "Return only the rewritten article text."
)

FALSE_TO_TRUE_PROMPT_TEMPLATE = (
    "Rewrite the following false news article as a truthful news article about the same "
    "topic. Preserve the approximate style, structure, and length, but correct or remove "
    "unsupported claims. Do not add sensational claims. Return only the rewritten article.\n\n"
    "Topic: {topic}\n"
    "Original title: {title}\n\n"
    "False article:\n{article_text}"
)
