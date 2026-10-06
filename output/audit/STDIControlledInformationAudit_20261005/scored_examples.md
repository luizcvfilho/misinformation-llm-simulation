# Scored synthetic news examples

Fictional English examples. LLM comparison is a sensitivity analysis, not a human gold standard. All methods retain the existing STDI weights.

## government_identity

Structured cluster (polarity/duration) STDI: 0.000000; cluster relation drift: 0.000000; Structured LLM judge STDI: 0.000000.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Intended change:** None; exactly identical text.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions describe the same postponed government negotiations with the teachers union.",
    "subtopic_drift": "Both include the two-day postponement and the union's request for a new meeting.",
    "entity_drift": "The government and teachers union are the same entities in both versions.",
    "relation_drift": "The actions, roles, and duration are unchanged.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_paraphrase

Structured cluster (polarity/duration) STDI: 0.056849; cluster relation drift: 0.080222; Structured LLM judge STDI: 0.001420.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

Negotiations with the teachers union were postponed by the government for two days. A new meeting was requested by the union.

**Intended change:** Wording and voice only; preserve both events and their actors.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions report the government postponing negotiations with the teachers union.",
    "subtopic_drift": "Both include the two-day postponement and the union's request for a new meeting.",
    "entity_drift": "The government and teachers union are the same actors in both versions.",
    "relation_drift": "The modified text preserves the same actions and roles using passive phrasing.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed for two days",
    "object": "Negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed for two days",
    "base_action": "postponed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "A new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_normative_opinion

Structured cluster (polarity/duration) STDI: 0.389189; cluster relation drift: 0.533999; Structured LLM judge STDI: 0.378848.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government should negotiate promptly and treat workers fairly.

**Intended change:** Add a recommendation, without asserting a new past action or motive.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The central story remains the government’s two-day postponement and the union’s request, though the added appeal slightly broadens the emphasis.",
    "subtopic_drift": "The modified text adds a secondary angle: a recommendation for prompt negotiations and fair treatment of workers.",
    "entity_drift": "The government and teachers union remain; workers are newly mentioned, without replacing a central actor.",
    "relation_drift": "The original actions are retained, but the modified text adds recommendations that the government negotiate promptly and treat workers fairly.",
    "contradiction_drift": "The recommendation does not contradict the stated two-day postponement or the union’s meeting request."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The government",
    "action": "postponed negotiations",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postpone",
    "negation_scope": null,
    "signed_action": "postponed negotiations",
    "base_action": "postponed negotiations",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "The government",
    "action": "should negotiate promptly",
    "object": "",
    "polarity": "affirmed",
    "predicate": "negotiate",
    "negation_scope": null,
    "signed_action": "should negotiate promptly",
    "base_action": "negotiate promptly",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  },
  {
    "subject": "The government",
    "action": "should treat workers fairly",
    "object": "workers",
    "polarity": "affirmed",
    "predicate": "treat",
    "negation_scope": null,
    "signed_action": "should treat workers fairly",
    "base_action": "treat workers fairly",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  }
]
```

## government_motive_tentative

Structured cluster (polarity/duration) STDI: 0.262751; cluster relation drift: 0.333333; Structured LLM judge STDI: 0.252765.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government may have delayed the talks to weaken the union.

**Intended change:** Add an unsupported possible intention; preserve tentative modality.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions center on the government's two-day postponement of negotiations with the teachers union.",
    "subtopic_drift": "The modified version adds a possible motive—weakening the union—not present in the original.",
    "entity_drift": "The government and teachers union are the same entities in both versions.",
    "relation_drift": "The modification adds a speculative claim that the government delayed talks in order to weaken the union.",
    "contradiction_drift": "The added possible motive does not conflict with any claim within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "government",
    "action": "may have delayed the talks to weaken the union",
    "object": "the talks",
    "polarity": "affirmed",
    "predicate": "delayed",
    "negation_scope": null,
    "signed_action": "may have delayed the talks to weaken the union",
    "base_action": "delayed the talks to weaken the union",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "hypothesis"
  }
]
```

## government_motive_asserted

Structured cluster (polarity/duration) STDI: 0.146321; cluster relation drift: 0.012753; Structured LLM judge STDI: 0.315446.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government deliberately delayed the talks to weaken the union.

**Intended change:** Add the same unsupported intention as a categorical assertion.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions center on the government postponing negotiations with the teachers union; the added motive slightly changes the framing.",
    "subtopic_drift": "The modified text adds a significant secondary claim that the delay was intended to weaken the union.",
    "entity_drift": "The government and teachers union are the same entities in both versions.",
    "relation_drift": "The modified text attributes a deliberate purpose to the government's delay that the original does not state.",
    "contradiction_drift": "The added claim does not contradict another claim within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union; deliberately delayed the talks to weaken the union",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postpone/delay",
    "negation_scope": null,
    "signed_action": "postponed; deliberately delayed",
    "base_action": "postponed negotiations with the teachers union; deliberately delayed the talks to weaken the union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "attributed_intention"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_opposite_motive

Structured cluster (polarity/duration) STDI: 0.166259; cluster relation drift: 0.004358; Structured LLM judge STDI: 0.315257.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government deliberately delayed the talks to protect the union.

**Intended change:** Add an intention opposite to weakening the union; compare directly with motive_asserted.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions center on the government's two-day postponement of negotiations with the teachers union; the added motive slightly shifts emphasis.",
    "subtopic_drift": "The modified text adds the government's deliberate intent to protect the union, a new secondary angle.",
    "entity_drift": "The government and teachers union remain the same entities.",
    "relation_drift": "The postponement and meeting request remain, but the modified text adds an asserted purpose and deliberate intent for the government's delay.",
    "contradiction_drift": "The added motive does not contradict any claim within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days; deliberately delayed the talks to protect the union",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postpone",
    "negation_scope": null,
    "signed_action": "postponed",
    "base_action": "postponed negotiations with the teachers union to protect the union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "attributed_intention"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_negation

Structured cluster (polarity/duration) STDI: 0.104532; cluster relation drift: 0.123563; Structured LLM judge STDI: 0.439332.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government did not postpone negotiations with the teachers union for two days. The union requested a new meeting.

**Intended change:** Deny the original postponement claim; this is a between-text conflict, not internal contradiction.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both texts concern government negotiations with the teachers union, but the modified text reverses whether the central postponement occurred.",
    "subtopic_drift": "The two-day postponement is replaced by a denial of that postponement; the union's request for a new meeting remains.",
    "entity_drift": "The government and teachers union are the same actors in both versions.",
    "relation_drift": "The government is said to have postponed negotiations in the original and not to have postponed them in the modified version.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The government",
    "action": "did not postpone negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "negated",
    "predicate": "postpone",
    "negation_scope": "postpone",
    "signed_action": "did not postpone negotiations with the teachers union for two days",
    "base_action": "postpone negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "the teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "request a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_actor_reassignment

Structured cluster (polarity/duration) STDI: 0.121160; cluster relation drift: 0.119417; Structured LLM judge STDI: 0.187926.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The teachers union postponed negotiations with the government for two days. The union requested a new meeting.

**Intended change:** Preserve actors but reverse responsibility for the postponement.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern a two-day postponement of negotiations and a request for a new meeting.",
    "subtopic_drift": "The postponement duration and request for a new meeting are unchanged.",
    "entity_drift": "Both versions identify the government and teachers union; neither entity is replaced.",
    "relation_drift": "Responsibility for postponing negotiations shifts from the government to the teachers union.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "teachers union",
    "action": "postponed negotiations with the government for two days",
    "object": "negotiations with the government",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the government for two days",
    "base_action": "postponed negotiations with the government",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_quantity_change

Structured cluster (polarity/duration) STDI: 0.136528; cluster relation drift: 0.188332; Structured LLM judge STDI: 0.250350.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for twenty days. The union requested a new meeting.

**Intended change:** Change the delay from two to twenty days without changing the actors.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern the government postponing negotiations with the teachers union.",
    "subtopic_drift": "The postponement duration changes materially from two days to twenty days; the union's request remains the same.",
    "entity_drift": "The government and teachers union are the same actors in both versions.",
    "relation_drift": "The government still postpones negotiations, but the asserted duration changes from two days to twenty days.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed",
    "base_action": "postpone",
    "duration_status": "exact",
    "duration_value": 20.0,
    "duration_unit": "days",
    "duration_expression": "twenty days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_new_action

Structured cluster (polarity/duration) STDI: 0.293571; cluster relation drift: 0.399156; Structured LLM judge STDI: 0.379859.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting. The government secretly ordered surveillance of the union.

**Intended change:** Add a concrete action and concealment attributed to an existing actor.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The negotiation postponement remains, but adding covert surveillance broadens the report to a materially different government action.",
    "subtopic_drift": "The modified text adds secret surveillance as a new angle alongside the postponement and meeting request.",
    "entity_drift": "The government and teachers union remain the actors; no entity is replaced.",
    "relation_drift": "The original actions are retained, and the government is additionally said to have ordered surveillance of the union.",
    "contradiction_drift": "The added surveillance claim does not conflict with any other claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postpone",
    "negation_scope": null,
    "signed_action": "postponed",
    "base_action": "postpone",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "government",
    "action": "secretly ordered",
    "object": "surveillance of the teachers union",
    "polarity": "affirmed",
    "predicate": "order",
    "negation_scope": null,
    "signed_action": "secretly ordered",
    "base_action": "secretly order",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## government_omission

Structured cluster (polarity/duration) STDI: 0.348514; cluster relation drift: 0.506905; Structured LLM judge STDI: 0.257182.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days.

**Intended change:** Remove the union's request; preserve the postponement.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern the government's two-day postponement of negotiations with the teachers union.",
    "subtopic_drift": "The modified text omits the union's request for a new meeting, removing a secondary angle.",
    "entity_drift": "The government and teachers union are the same entities in both versions.",
    "relation_drift": "The postponement relation is preserved, but the union's request for a new meeting is omitted.",
    "contradiction_drift": "The modified text introduces no internally contradictory claims."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postpone",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  }
]
```

## government_internal_contradiction

Structured cluster (polarity/duration) STDI: 0.387727; cluster relation drift: 0.342966; Structured LLM judge STDI: 0.451006.

**Original**

The government postponed negotiations with the teachers union for two days. The union requested a new meeting.

**Rewrite**

The government postponed negotiations with the teachers union for two days. The government did not postpone those negotiations. The union requested a new meeting.

**Intended change:** Affirm and deny the same postponement inside the rewritten text.

**Original extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postponed negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "requested a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern government negotiations with the teachers union and a possible postponement.",
    "subtopic_drift": "The modified version adds a conflicting claim about whether the postponement happened.",
    "entity_drift": "The government and teachers union are the same actors in both versions.",
    "relation_drift": "The modified version adds the opposite action claim while retaining the asserted two-day postponement.",
    "contradiction_drift": "The modified text directly says both that the government postponed the negotiations and that it did not."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with the teachers union for two days",
    "object": "negotiations with the teachers union",
    "polarity": "affirmed",
    "predicate": "postponed",
    "negation_scope": null,
    "signed_action": "postponed negotiations with the teachers union for two days",
    "base_action": "postpone negotiations with the teachers union",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "government",
    "action": "did not postpone those negotiations",
    "object": "those negotiations",
    "polarity": "negated",
    "predicate": "postpone",
    "negation_scope": "postpone those negotiations",
    "signed_action": "did not postpone those negotiations",
    "base_action": "postpone those negotiations",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "a new meeting",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a new meeting",
    "base_action": "request a new meeting",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_identity

Structured cluster (polarity/duration) STDI: 0.000000; cluster relation drift: 0.000000; Structured LLM judge STDI: 0.000000.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Intended change:** None; exactly identical text.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions describe the same temporary clinic closure and request for review.",
    "subtopic_drift": "Both include the two-day closure and workers' review request.",
    "entity_drift": "The health department, clinic, and clinic workers are unchanged.",
    "relation_drift": "The same actors perform the same actions with the same duration.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_paraphrase

Structured cluster (polarity/duration) STDI: 0.097066; cluster relation drift: 0.056735; Structured LLM judge STDI: 0.003036.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

A clinic was closed for two days by the health department. A review of the decision was requested by the clinic workers.

**Intended change:** Wording and voice only; preserve both events and their actors.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern a two-day clinic closure and workers requesting a review.",
    "subtopic_drift": "The same closure duration and review request are emphasized.",
    "entity_drift": "The health department, clinic, and clinic workers are unchanged.",
    "relation_drift": "The passive wording preserves who closed the clinic and who requested review.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "review of the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_normative_opinion

Structured cluster (polarity/duration) STDI: 0.314828; cluster relation drift: 0.529422; Structured LLM judge STDI: 0.252132.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision. The health department should protect access to care and listen to workers.

**Intended change:** Add a recommendation, without asserting a new past action or motive.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The clinic closure and workers’ review request remain central, with a brief recommendation added.",
    "subtopic_drift": "The modified text adds a secondary emphasis on protecting care access and listening to workers.",
    "entity_drift": "The same health department, clinic, and clinic workers are involved.",
    "relation_drift": "The original actions are unchanged; the modification adds recommendations directed at the department.",
    "contradiction_drift": "The added recommendation does not conflict with any claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "should protect",
    "object": "access to care",
    "polarity": "affirmed",
    "predicate": "should protect",
    "negation_scope": null,
    "signed_action": "should protect",
    "base_action": "should protect",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  },
  {
    "subject": "health department",
    "action": "should listen",
    "object": "clinic workers",
    "polarity": "affirmed",
    "predicate": "should listen",
    "negation_scope": null,
    "signed_action": "should listen",
    "base_action": "should listen",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  }
]
```

## health_motive_tentative

Structured cluster (polarity/duration) STDI: 0.205637; cluster relation drift: 0.372562; Structured LLM judge STDI: 0.318149.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision. The health department may have closed the clinic to punish the workers.

**Intended change:** Add an unsupported possible intention; preserve tentative modality.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The clinic closure and workers’ review request remain central, with a possible motive added.",
    "subtopic_drift": "The modified text adds a possible punitive motive as a secondary angle.",
    "entity_drift": "The same health department, clinic, and clinic workers are involved.",
    "relation_drift": "The modified text introduces a possible purpose for the closure—punishing workers—not stated in the original.",
    "contradiction_drift": "The added motive is presented as possible and does not conflict with another claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "may have closed the clinic to punish the workers",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "may have closed the clinic to punish the workers",
    "base_action": "closed the clinic to punish the workers",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "hypothesis"
  }
]
```

## health_motive_asserted

Structured cluster (polarity/duration) STDI: 0.216581; cluster relation drift: 0.371157; Structured LLM judge STDI: 0.318344.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision. The health department deliberately closed the clinic to punish the workers.

**Intended change:** Add the same unsupported intention as a categorical assertion.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The clinic closure and workers’ review request remain the central event, though the added punitive framing slightly shifts its presentation.",
    "subtopic_drift": "The modified version adds a significant secondary claim that the closure was intended to punish workers.",
    "entity_drift": "The same health department, clinic, and clinic workers are involved.",
    "relation_drift": "The original actions are preserved, but the modified text adds an attributed punitive purpose for the department’s closure.",
    "contradiction_drift": "The modified text introduces no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "deliberately closed to punish the clinic workers",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "deliberately closed to punish the clinic workers",
    "base_action": "deliberately closed to punish the clinic workers",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "attributed_intention"
  }
]
```

## health_opposite_motive

Structured cluster (polarity/duration) STDI: 0.249148; cluster relation drift: 0.125534; Structured LLM judge STDI: 0.316130.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision. The health department deliberately closed the clinic to protect the workers.

**Intended change:** Add an intention opposite to punishing the workers; compare directly with motive_asserted.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The temporary clinic closure and workers’ review request remain central, with a new motive added.",
    "subtopic_drift": "The modified text adds the meaningful secondary angle that the closure was intended to protect workers.",
    "entity_drift": "The health department, clinic, and clinic workers are the same entities.",
    "relation_drift": "The original actions and roles are preserved, but the text adds that the department deliberately closed the clinic to protect workers.",
    "contradiction_drift": "The added protective purpose does not conflict with any other claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "deliberately closed to protect the workers",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "deliberately closed to protect the workers",
    "base_action": "deliberately closed to protect the workers",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_negation

Structured cluster (polarity/duration) STDI: 0.179029; cluster relation drift: 0.167795; Structured LLM judge STDI: 0.438610.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department did not close a clinic for two days. The clinic workers requested a review of the decision.

**Intended change:** Deny the original closure claim; this is a between-text conflict, not internal contradiction.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern the health department's clinic decision and workers' review request, but they reverse whether the clinic was closed.",
    "subtopic_drift": "The closure detail becomes a non-closure, while the workers' review request remains.",
    "entity_drift": "The health department, clinic, and clinic workers are the same entities in both versions.",
    "relation_drift": "The department's action is reversed: it closed the clinic for two days in the original but did not close it in the modified text.",
    "contradiction_drift": "The modified text contains no internal contradiction; its conflict with the original is not an internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "did not close",
    "object": "clinic",
    "polarity": "negated",
    "predicate": "close",
    "negation_scope": "close a clinic",
    "signed_action": "did not close",
    "base_action": "close",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request a review of",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_actor_reassignment

Structured cluster (polarity/duration) STDI: 0.195445; cluster relation drift: 0.186638; Structured LLM judge STDI: 0.313116.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The clinic workers closed a clinic for two days. The health department requested a review of the decision.

**Intended change:** Preserve actors but swap responsibility for both actions.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern a two-day clinic closure and a request to review the decision.",
    "subtopic_drift": "The secondary emphasis shifts from workers responding to the department's closure to the department responding to workers' closure.",
    "entity_drift": "The same entities appear in both versions; none are replaced.",
    "relation_drift": "The versions reverse which actor closed the clinic and which requested a review.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "clinic workers",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_quantity_change

Structured cluster (polarity/duration) STDI: 0.128976; cluster relation drift: 0.157535; Structured LLM judge STDI: 0.250107.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for twenty days. The clinic workers requested a review of the decision.

**Intended change:** Change closure duration from two to twenty days without changing actors.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions focus on the clinic's closure and workers' request for a review.",
    "subtopic_drift": "The stated closure duration changes substantially from two days to twenty days.",
    "entity_drift": "The health department, clinic, and clinic workers are unchanged.",
    "relation_drift": "The department still closes the clinic and workers still request a review, but the duration of the closure changes.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 20.0,
    "duration_unit": "days",
    "duration_expression": "twenty days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_new_action

Structured cluster (polarity/duration) STDI: 0.319302; cluster relation drift: 0.372562; Structured LLM judge STDI: 0.380285.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The clinic workers requested a review of the decision. The health department secretly ordered surveillance of the workers.

**Intended change:** Add a concrete action and concealment attributed to an existing actor.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The clinic closure and workers’ review request remain, but the added surveillance order gives the story a substantial new focus.",
    "subtopic_drift": "Secretly ordering surveillance introduces a significant secondary angle absent from the original.",
    "entity_drift": "The health department, clinic, and clinic workers are the same entities in both versions.",
    "relation_drift": "The original actions remain, while the modified text adds the health department’s order to surveil the workers.",
    "contradiction_drift": "The modified text contains no internally conflicting claims."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "secretly ordered surveillance of",
    "object": "clinic workers",
    "polarity": "affirmed",
    "predicate": "order",
    "negation_scope": null,
    "signed_action": "secretly ordered surveillance of",
    "base_action": "secretly ordered surveillance of",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## health_omission

Structured cluster (polarity/duration) STDI: 0.454440; cluster relation drift: 0.506540; Structured LLM judge STDI: 0.380205.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days.

**Intended change:** Remove the workers' request; preserve the closure.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions center on the health department's two-day clinic closure, though the modified text narrows the focus by omitting the review request.",
    "subtopic_drift": "The workers' request for a review, a distinct secondary angle in the original, is absent.",
    "entity_drift": "Clinic workers appear in the original but are omitted from the modified text; no entity is replaced.",
    "relation_drift": "The modified text omits the asserted action of clinic workers requesting a review.",
    "contradiction_drift": "The modified text introduces no internally contradictory claims."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  }
]
```

## health_internal_contradiction

Structured cluster (polarity/duration) STDI: 0.465401; cluster relation drift: 0.505641; Structured LLM judge STDI: 0.500946.

**Original**

The health department closed a clinic for two days. The clinic workers requested a review of the decision.

**Rewrite**

The health department closed a clinic for two days. The health department did not close that clinic. The clinic workers requested a review of the decision.

**Intended change:** Affirm and deny the same closure inside the rewritten text.

**Original extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed a clinic for two days",
    "object": "a clinic",
    "polarity": "affirmed",
    "predicate": "closed",
    "negation_scope": null,
    "signed_action": "closed for two days",
    "base_action": "closed",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested a review of the decision",
    "base_action": "requested a review of the decision",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both texts concern the clinic closure, but the modified text reframes it as conflicting claims about whether it happened.",
    "subtopic_drift": "The modified text adds a substantial secondary angle: the department's denial of the closure.",
    "entity_drift": "The health department, clinic, and clinic workers are the same entities in both versions.",
    "relation_drift": "The modified text adds an assertion that the department did not close the clinic, opposing its stated closure action.",
    "contradiction_drift": "The modified text directly asserts both that the department closed the clinic and that it did not."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic",
    "polarity": "affirmed",
    "predicate": "close",
    "negation_scope": null,
    "signed_action": "closed",
    "base_action": "close",
    "duration_status": "exact",
    "duration_value": 2.0,
    "duration_unit": "days",
    "duration_expression": "for two days",
    "assertion_type": "asserted"
  },
  {
    "subject": "health department",
    "action": "did not close",
    "object": "clinic",
    "polarity": "negated",
    "predicate": "close",
    "negation_scope": "close",
    "signed_action": "did not close",
    "base_action": "close",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_identity

Structured cluster (polarity/duration) STDI: 0.000000; cluster relation drift: 0.000000; Structured LLM judge STDI: 0.000000.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Intended change:** None; exactly identical text.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions report the removal of voter-register names and the affected voters’ appeal request.",
    "subtopic_drift": "The same details and emphasis appear in both versions.",
    "entity_drift": "The election commission, voter register, and affected voters are unchanged.",
    "relation_drift": "Both versions state that the commission removed 120 names and affected voters requested an appeal.",
    "contradiction_drift": "The modified text introduces no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_paraphrase

Structured cluster (polarity/duration) STDI: 0.211505; cluster relation drift: 0.027774; Structured LLM judge STDI: 0.000692.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

A total of 120 names were removed from the voter register by the election commission. An appeal was requested by the affected voters.

**Intended change:** Wording and voice only; preserve both events, actors, and number.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions report the removal of 120 names from the voter register and an appeal request.",
    "subtopic_drift": "The same removal and appeal details are included in both versions.",
    "entity_drift": "The election commission and affected voters remain the actors in both versions.",
    "relation_drift": "The passive wording preserves who removed the names and who requested an appeal.",
    "contradiction_drift": "The modified text introduces no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_normative_opinion

Structured cluster (polarity/duration) STDI: 0.311792; cluster relation drift: 0.513887; Structured LLM judge STDI: 0.256701.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal. The election commission should protect voting rights and treat voters fairly.

**Intended change:** Add a recommendation, without asserting a new past action or motive.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The voter-register removal and appeal remain the primary subject, with a brief added emphasis on fair treatment.",
    "subtopic_drift": "The modified text adds voting rights and fair treatment as a secondary angle.",
    "entity_drift": "The same election commission and affected voters are involved.",
    "relation_drift": "The original removal and appeal relations are unchanged; the modified text adds recommendations for the commission.",
    "contradiction_drift": "The recommendations do not contradict any claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "election commission",
    "action": "should protect",
    "object": "voting rights",
    "polarity": "affirmed",
    "predicate": "should protect",
    "negation_scope": null,
    "signed_action": "should protect",
    "base_action": "protect",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  },
  {
    "subject": "election commission",
    "action": "should treat fairly",
    "object": "voters",
    "polarity": "affirmed",
    "predicate": "should treat",
    "negation_scope": null,
    "signed_action": "should treat fairly",
    "base_action": "treat fairly",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "recommendation"
  }
]
```

## election_motive_tentative

Structured cluster (polarity/duration) STDI: 0.265051; cluster relation drift: 0.351850; Structured LLM judge STDI: 0.313646.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal. The election commission may have removed the names to silence the voters.

**Intended change:** Add an unsupported possible intention; preserve tentative modality.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The report remains about voter-register removals and the affected voters’ appeal, with a slight shift in framing from the added possible motive.",
    "subtopic_drift": "The modified text introduces a new angle: that the removals may have been intended to silence voters.",
    "entity_drift": "The election commission and voters are the same entities in both versions.",
    "relation_drift": "The modified text adds a speculative purpose for the commission’s action, not present in the original.",
    "contradiction_drift": "The added possible motive does not contradict any claim within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "election commission",
    "action": "may have removed the names to silence the voters",
    "object": "the voters",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "may have removed the names to silence the voters",
    "base_action": "may have removed the names to silence the voters",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "hypothesis"
  }
]
```

## election_motive_asserted

Structured cluster (polarity/duration) STDI: 0.203090; cluster relation drift: 0.119484; Structured LLM judge STDI: 0.314363.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal. The election commission deliberately removed the names to silence the voters.

**Intended change:** Add the same unsupported intention as a categorical assertion.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern removal of voter-register names and the affected voters’ appeal; the added motive changes the framing somewhat.",
    "subtopic_drift": "The modified text adds a significant secondary claim that the removal was intended to silence voters.",
    "entity_drift": "The same election commission, voter register, and affected voters are involved.",
    "relation_drift": "The removal and appeal remain, but the modified text newly attributes deliberate silencing intent to the commission.",
    "contradiction_drift": "The added motive does not internally contradict any other claim in the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "deliberately removed",
    "object": "120 names from the voter register to silence the voters",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "deliberately removed to silence",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "attributed_intention"
  },
  {
    "subject": "voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_opposite_motive

Structured cluster (polarity/duration) STDI: 0.168543; cluster relation drift: 0.101862; Structured LLM judge STDI: 0.315031.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal. The election commission deliberately removed the names to protect the voters.

**Intended change:** Add an intention opposite to silencing the voters; compare directly with motive_asserted.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The voter-register removal and affected voters’ appeal remain the main event, with a new purpose added.",
    "subtopic_drift": "The modified text introduces a substantial secondary angle: the commission allegedly acted to protect voters.",
    "entity_drift": "The same election commission, affected voters, and voter register are involved.",
    "relation_drift": "The removal and appeal relations remain, but the modified text adds deliberate intent and a protective purpose.",
    "contradiction_drift": "The added intention does not conflict with any other claim within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "deliberately removed 120 names from the voter register to protect the voters",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "deliberately removed the names to protect the voters",
    "base_action": "deliberately removed the names to protect the voters",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "attributed_intention"
  },
  {
    "subject": "affected voters",
    "action": "requested an appeal",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested an appeal",
    "base_action": "requested an appeal",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_negation

Structured cluster (polarity/duration) STDI: 0.202370; cluster relation drift: 0.109562; Structured LLM judge STDI: 0.438438.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission did not remove 120 names from the voter register. The affected voters requested an appeal.

**Intended change:** Deny the original removal claim; this is a between-text conflict, not internal contradiction.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both texts concern the voter register and affected voters’ appeal, but they assert opposite outcomes for the central register event.",
    "subtopic_drift": "The focus shifts from removing 120 names to explicitly stating that they were not removed; the appeal request remains.",
    "entity_drift": "The election commission, affected voters, and voter register are the same entities.",
    "relation_drift": "The commission’s central action is reversed: the original says it removed 120 names, while the modified text says it did not.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "did not remove",
    "object": "120 names from the voter register",
    "polarity": "negated",
    "predicate": "remove",
    "negation_scope": "remove 120 names from the voter register",
    "signed_action": "did not remove",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_actor_reassignment

Structured cluster (polarity/duration) STDI: 0.124162; cluster relation drift: 0.205837; Structured LLM judge STDI: 0.375525.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The affected voters removed 120 names from the voter register. The election commission requested an appeal.

**Intended change:** Preserve actors but swap responsibility for both actions.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern names being removed from the voter register and an appeal request.",
    "subtopic_drift": "The secondary emphasis shifts: voters are said to remove the names and the commission to request the appeal, rather than the reverse.",
    "entity_drift": "The same entities appear in both versions; their identities are not replaced.",
    "relation_drift": "Responsibility for both actions is reversed: the commission and voters exchange roles in the removal and appeal request.",
    "contradiction_drift": "The modified text does not contradict itself internally; its claims conflict with the original, which is not counted here."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "affected voters",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "election commission",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_quantity_change

Structured cluster (polarity/duration) STDI: 0.127704; cluster relation drift: 0.040662; Structured LLM judge STDI: 0.187969.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 1200 names from the voter register. The affected voters requested an appeal.

**Intended change:** Change the quantity from 120 to 1200 without changing actors or actions.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern voter-register removals and affected voters requesting an appeal.",
    "subtopic_drift": "The stated number removed changes from 120 to 1200, altering a detail but not the secondary angles.",
    "entity_drift": "The election commission, voter register, and affected voters are the same entities.",
    "relation_drift": "The commission still removes names and the voters still request an appeal, but the number removed changes substantially.",
    "contradiction_drift": "The modified text contains no internal contradiction."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "1200 names from the voter register",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested an appeal",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_new_action

Structured cluster (polarity/duration) STDI: 0.415052; cluster relation drift: 0.379717; Structured LLM judge STDI: 0.439584.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The affected voters requested an appeal. The election commission secretly ordered surveillance of the voters.

**Intended change:** Add a concrete action and concealment attributed to an existing actor.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The voter-register removal and appeal remain, but the added claim of secret surveillance substantially broadens the story’s focus.",
    "subtopic_drift": "Secret surveillance introduces a major secondary angle absent from the original.",
    "entity_drift": "The same election commission and voters are involved; no actor is replaced.",
    "relation_drift": "The modified text adds the commission’s action of secretly ordering surveillance of voters.",
    "contradiction_drift": "The added surveillance claim does not contradict any other assertion within the modified text."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "removed",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "requested",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "requested",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "election commission",
    "action": "secretly ordered surveillance of",
    "object": "voters",
    "polarity": "affirmed",
    "predicate": "ordered",
    "negation_scope": null,
    "signed_action": "secretly ordered surveillance of",
    "base_action": "secretly ordered surveillance of",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_omission

Structured cluster (polarity/duration) STDI: 0.560986; cluster relation drift: 0.522145; Structured LLM judge STDI: 0.376910.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register.

**Intended change:** Remove the voters' appeal request; preserve the removal.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "Both versions concern the commission removing 120 names from the voter register; the modified version omits the appeal request.",
    "subtopic_drift": "The secondary angle of affected voters seeking an appeal is removed.",
    "entity_drift": "Affected voters are omitted, but no entity is replaced and the election commission remains.",
    "relation_drift": "The removal action is preserved, while the voters' request for an appeal is omitted.",
    "contradiction_drift": "The modified text contains no internally contradictory claims."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "removed",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

## election_internal_contradiction

Structured cluster (polarity/duration) STDI: 0.333533; cluster relation drift: 0.333333; Structured LLM judge STDI: 0.500665.

**Original**

The election commission removed 120 names from the voter register. The affected voters requested an appeal.

**Rewrite**

The election commission removed 120 names from the voter register. The election commission did not remove those names. The affected voters requested an appeal.

**Intended change:** Affirm and deny the same removal inside the rewritten text.

**Original extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```

**LLM rationale**

```json
{
  "rationales": {
    "theme_drift": "The item remains about voter-register removals and the affected voters’ appeal, but the added denial shifts the framing to conflicting claims.",
    "subtopic_drift": "The modified text adds a consequential secondary angle: the commission denies carrying out the removal.",
    "entity_drift": "The same election commission, affected voters, and voter register are involved.",
    "relation_drift": "The modified text retains the removal claim but also asserts that the commission did not remove those same names.",
    "contradiction_drift": "The modified text directly contradicts itself on the central claim of whether the commission removed the names."
  }
}
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register",
    "polarity": "affirmed",
    "predicate": "remove",
    "negation_scope": null,
    "signed_action": "removed",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "election commission",
    "action": "did not remove",
    "object": "those names",
    "polarity": "negated",
    "predicate": "remove",
    "negation_scope": "remove those names",
    "signed_action": "did not remove",
    "base_action": "remove",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal",
    "polarity": "affirmed",
    "predicate": "request",
    "negation_scope": null,
    "signed_action": "requested",
    "base_action": "request",
    "duration_status": "absent",
    "duration_value": null,
    "duration_unit": null,
    "duration_expression": null,
    "assertion_type": "asserted"
  }
]
```
