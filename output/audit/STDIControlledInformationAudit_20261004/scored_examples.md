# Scored synthetic news examples

Fictional English examples. LLM comparison is a sensitivity analysis, not a human gold standard. All methods retain the existing STDI weights.

## government_identity

Current STDI: 0.000000; current relation drift: 0.000000; LLM-comparator STDI: 0.000000.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

## government_paraphrase

Current STDI: 0.215154; current relation drift: 0.162974; LLM-comparator STDI: 0.001420.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "government"
  }
]
```

## government_normative_opinion

Current STDI: 0.319959; current relation drift: 0.368676; LLM-comparator STDI: 0.316733.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations for two days",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  },
  {
    "subject": "government",
    "action": "should treat fairly",
    "object": "workers"
  }
]
```

## government_motive_tentative

Current STDI: 0.245879; current relation drift: 0.374254; LLM-comparator STDI: 0.252765.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The government",
    "action": "postponed negotiations for two days",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  },
  {
    "subject": "The government",
    "action": "may have delayed the talks to weaken",
    "object": "teachers union"
  }
]
```

## government_motive_asserted

Current STDI: 0.240086; current relation drift: 0.441983; LLM-comparator STDI: 0.253214.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting",
    "object": "government"
  },
  {
    "subject": "government",
    "action": "delayed talks to weaken",
    "object": "teachers union"
  }
]
```

## government_opposite_motive

Current STDI: 0.150556; current relation drift: 0.333333; LLM-comparator STDI: 0.253007.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  },
  {
    "subject": "government",
    "action": "deliberately delayed",
    "object": "the talks to protect the teachers union"
  }
]
```

## government_negation

Current STDI: 0.050166; current relation drift: 0.036764; LLM-comparator STDI: 0.439332.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "did not postpone",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

## government_actor_reassignment

Current STDI: 0.221769; current relation drift: 0.192423; LLM-comparator STDI: 0.187926.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "teachers union",
    "action": "postponed negotiations for two days with",
    "object": "government"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

## government_quantity_change

Current STDI: 0.121229; current relation drift: 0.051262; LLM-comparator STDI: 0.250350.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

## government_new_action

Current STDI: 0.231781; current relation drift: 0.444237; LLM-comparator STDI: 0.379859.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed negotiations with",
    "object": "teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested a new meeting with",
    "object": "government"
  },
  {
    "subject": "government",
    "action": "secretly ordered surveillance of",
    "object": "teachers union"
  }
]
```

## government_omission

Current STDI: 0.406778; current relation drift: 0.561500; LLM-comparator STDI: 0.195280.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "the government",
    "action": "postponed negotiations for two days",
    "object": "the teachers union"
  }
]
```

## government_internal_contradiction

Current STDI: 0.394524; current relation drift: 0.346011; LLM-comparator STDI: 0.500914.

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
    "action": "postponed",
    "object": "negotiations with the teachers union for two days"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "government",
    "action": "postponed for two days",
    "object": "negotiations with the teachers union"
  },
  {
    "subject": "government",
    "action": "did not postpone",
    "object": "negotiations with the teachers union"
  },
  {
    "subject": "teachers union",
    "action": "requested",
    "object": "a new meeting"
  }
]
```

## health_identity

Current STDI: 0.000000; current relation drift: 0.000000; LLM-comparator STDI: 0.000000.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

## health_paraphrase

Current STDI: 0.172338; current relation drift: 0.009359; LLM-comparator STDI: 0.003036.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "decision"
  }
]
```

## health_normative_opinion

Current STDI: 0.310085; current relation drift: 0.418617; LLM-comparator STDI: 0.127487.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of the decision",
    "object": "health department"
  },
  {
    "subject": "health department",
    "action": "should protect access to care and listen to",
    "object": "clinic workers"
  }
]
```

## health_motive_tentative

Current STDI: 0.310460; current relation drift: 0.333333; LLM-comparator STDI: 0.318149.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  },
  {
    "subject": "health department",
    "action": "may have closed the clinic to punish",
    "object": "clinic workers"
  }
]
```

## health_motive_asserted

Current STDI: 0.363744; current relation drift: 0.333333; LLM-comparator STDI: 0.318344.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  },
  {
    "subject": "health department",
    "action": "closed to punish",
    "object": "clinic workers"
  }
]
```

## health_opposite_motive

Current STDI: 0.219328; current relation drift: 0.333333; LLM-comparator STDI: 0.253960.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  },
  {
    "subject": "health department",
    "action": "closed to protect",
    "object": "the workers"
  }
]
```

## health_negation

Current STDI: 0.171092; current relation drift: 0.070466; LLM-comparator STDI: 0.438610.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "did not close",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "decision"
  }
]
```

## health_actor_reassignment

Current STDI: 0.196562; current relation drift: 0.217499; LLM-comparator STDI: 0.313116.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "clinic workers",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "health department",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

## health_quantity_change

Current STDI: 0.113062; current relation drift: 0.025550; LLM-comparator STDI: 0.250107.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for twenty days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

## health_new_action

Current STDI: 0.387662; current relation drift: 0.369540; LLM-comparator STDI: 0.442257.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "decision"
  },
  {
    "subject": "health department",
    "action": "secretly ordered surveillance of",
    "object": "workers"
  }
]
```

## health_omission

Current STDI: 0.401911; current relation drift: 0.547102; LLM-comparator STDI: 0.318225.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed",
    "object": "clinic for two days"
  }
]
```

## health_internal_contradiction

Current STDI: 0.508103; current relation drift: 0.380495; LLM-comparator STDI: 0.500946.

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
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested a review of",
    "object": "the decision"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "health department",
    "action": "closed for two days",
    "object": "clinic"
  },
  {
    "subject": "health department",
    "action": "did not close",
    "object": "clinic"
  },
  {
    "subject": "clinic workers",
    "action": "requested",
    "object": "a review of the decision"
  }
]
```

## election_identity

Current STDI: 0.000000; current relation drift: 0.000000; LLM-comparator STDI: 0.000000.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

## election_paraphrase

Current STDI: 0.356553; current relation drift: 0.003218; LLM-comparator STDI: 0.000692.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed 120 names from",
    "object": "the voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

## election_normative_opinion

Current STDI: 0.397930; current relation drift: 0.533576; LLM-comparator STDI: 0.194759.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  },
  {
    "subject": "election commission",
    "action": "should protect",
    "object": "voting rights"
  },
  {
    "subject": "election commission",
    "action": "should treat fairly",
    "object": "voters"
  }
]
```

## election_motive_tentative

Current STDI: 0.407681; current relation drift: 0.378102; LLM-comparator STDI: 0.251250.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  },
  {
    "subject": "election commission",
    "action": "may have removed the names to silence",
    "object": "affected voters"
  }
]
```

## election_motive_asserted

Current STDI: 0.256538; current relation drift: 0.333333; LLM-comparator STDI: 0.314363.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  },
  {
    "subject": "election commission",
    "action": "deliberately removed names to silence",
    "object": "voters"
  }
]
```

## election_opposite_motive

Current STDI: 0.328434; current relation drift: 0.341936; LLM-comparator STDI: 0.252761.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The election commission",
    "action": "removed 120 names from",
    "object": "the voter register"
  },
  {
    "subject": "The affected voters",
    "action": "requested",
    "object": "an appeal"
  },
  {
    "subject": "The election commission",
    "action": "removed the names to protect",
    "object": "the voters"
  }
]
```

## election_negation

Current STDI: 0.244031; current relation drift: 0.070350; LLM-comparator STDI: 0.438438.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The election commission",
    "action": "did not remove",
    "object": "120 names from the voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

## election_actor_reassignment

Current STDI: 0.135889; current relation drift: 0.223753; LLM-comparator STDI: 0.313077.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "affected voters",
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "election commission",
    "action": "requested",
    "object": "an appeal"
  }
]
```

## election_quantity_change

Current STDI: 0.064627; current relation drift: 0.016313; LLM-comparator STDI: 0.063041.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed 1200 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

## election_new_action

Current STDI: 0.423479; current relation drift: 0.374373; LLM-comparator STDI: 0.377316.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "The election commission",
    "action": "removed 120 names from",
    "object": "the voter register"
  },
  {
    "subject": "voters",
    "action": "requested",
    "object": "an appeal"
  },
  {
    "subject": "The election commission",
    "action": "secretly ordered surveillance of",
    "object": "voters"
  }
]
```

## election_omission

Current STDI: 0.518918; current relation drift: 0.567153; LLM-comparator STDI: 0.314601.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register"
  }
]
```

## election_internal_contradiction

Current STDI: 0.514372; current relation drift: 0.378102; LLM-comparator STDI: 0.400798.

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
    "action": "removed 120 names from",
    "object": "voter register"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```

**Rewritten extracted relations**

```json
[
  {
    "subject": "election commission",
    "action": "removed",
    "object": "120 names from the voter register"
  },
  {
    "subject": "election commission",
    "action": "did not remove",
    "object": "those names"
  },
  {
    "subject": "affected voters",
    "action": "requested",
    "object": "an appeal"
  }
]
```
