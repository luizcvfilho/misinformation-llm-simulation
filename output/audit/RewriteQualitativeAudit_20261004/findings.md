# Qualitative audit of the 2026-10-04 rewrite run

Run: `output/interaction_graph/app_runs/simulation_ui_20261004_183721/`.

## Assessment

The personas visibly affect the outputs. Political agents add normative positions and often rebut the preceding agent. Conspiratorial agents introduce unsupported intentions, coordinated manipulation, and narrative inversion. Skeptical agents generally distinguish reported allegations from evidence and identify missing context. Similar aggregate STDI values therefore do not demonstrate similar meanings or failed persona conditioning.

The conspiratorial persona mainly fabricates explanations and motives, rather than new named events or numerical evidence. This matches the saved `interpretive_v2` instructions: unsupported motives and consequences are allowed, but new named actors, statistics, quotations, and specific events are prohibited. If the intended construct is concrete factual fabrication, this experiment restricts that behavior by design.

These are findings from this study's saved texts. No external factual verification was performed. An addition unsupported by a source description is not automatically false in the world. Original descriptions are not treated as verified truth.

## Scope and checks

- 20 matched news descriptions, four chains, two steps per chain: 160 outputs inspected.
- All 160 steps succeeded; the original descriptions are identical across chains.
- Each second agent received exactly the first agent's saved output.
- Rewrite and extraction model recorded: `gpt-6-luna`; comparator: `cluster_v2`; VAD: `RobroKools/vad-bert`.
- The inputs are short descriptions, averaging 72.65 whitespace-delimited words, with several visibly truncated passages and publisher/promotional text.
- The saved metadata records temperature **requested** as 0.8. The current request builder in `src/misinformation_simulation/llm/retry.py` omits temperature for model names starting with `gpt-5` or `gpt-6`; this is not evidence that these generations actually used temperature 0.8.
- This is a qualitative inspection by one reviewer, not a validated annotation study or an estimate of human behavior.

## Aggregate results

Each mean below uses the same 20 news items. Final STDI is measured against the original description's extracted structure; incremental STDI compares step 2 to step 1.

| Chain | Step 1 STDI | Final STDI | Final SD | Step 2 incremental STDI |
| --- | ---: | ---: | ---: | ---: |
| Conservative → Progressive | 0.378114 | 0.424046 | 0.074322 | 0.347807 |
| Progressive → Conservative | 0.354191 | 0.433246 | 0.096558 | 0.355690 |
| Conspiracy → Conspiracy | 0.384176 | 0.404073 | 0.108529 | 0.333950 |
| Skeptic → Skeptic | 0.341019 | 0.373699 | 0.086714 | 0.256081 |

The final mean range is approximately 0.060. Conspiratorial and skeptical final scores differ by 0.030374 on average. Their paired news bootstrap interval is approximately [-0.023662, 0.078920]. The two political orders differ by only 0.009201 in absolute mean, with an interval spanning zero. The political chains exceed the skeptical chain by about 0.050 and 0.060; their unadjusted bootstrap intervals exclude zero in this sample.

Intervals use 10,000 paired-news resamples, seed 42, and percentile endpoints. They have no multiple-comparison adjustment and do not include uncertainty from regenerating outputs or extracting structures again. They should not be used to claim stable causal persona effects.

## Traceable examples

Line numbers below refer to the chain's `*_steps.jsonl`. News numbers match `review_texts.md` and `per_news_final_scores.csv`.

### 1. Election conviction recast as suppression

News 3, Harry Wait, `03_conspiracy`, lines 5–6.

The source describes convictions for requesting other people's ballots without consent, an acquittal on another count, and Wait's stated purpose of testing the registration system. It does not report an institutional plan to intimidate critics.

Step 1: “This looks less like a clean defense of election integrity and more like a warning shot against anyone who tests the system.”

Step 2: “This wasn’t a real answer to questions about election security; it was a warning dressed up as a verdict.” Later: “make an example of the person raising the alarm”.

This invents a prosecutorial motive and reframes a convicted person as someone being silenced. Despite that inversion, the final STDI is only **0.247015**, below the skeptical version's **0.383912**. This is a useful example of why STDI magnitude should not be interpreted as the severity of misinformation.

### 2. A speculative motive becomes certainty

News 5, SSANU ultimatum, `03_conspiracy`, lines 9–10.

The source reports delayed renegotiation and a conditional strike threat. It does not establish why the delay occurred.

Step 1: “The prolonged delay looks less like an accident than a way to keep university workers waiting and weaken their position”.

Step 2: “That prolonged delay was no accident: keeping university workers waiting is a way to wear them down and protect the government’s advantage.”

This is a clear relay effect: uncertainty in an invented explanation is removed in the next message. Final STDI: **0.387334**.

### 3. A clash becomes a supposed operation

News 14, No Kings arrests, `03_conspiracy`, lines 27–28.

The source reports a volatile protest, tear gas, and arrests. It does not establish orchestration.

Step 2: “looks less like spontaneous disorder and more like a setup” and “That official framing is part of the operation”.

The rewrite adds a planned operation without providing new supporting evidence. Final STDI: **0.426840**. Contrast the skeptic, which questions loaded terms and missing sequence details without asserting an operation.

### 4. Missing source text becomes evidence of concealment

News 19, KTR mining allegations, `03_conspiracy`, lines 37–38.

The source ends a sentence and appends subscription/app promotion. Step 2 says: “the most damaging part may be exactly what they don’t want circulated.”

This assigns a concealment motive to an incomplete news excerpt. It may be responding to a data collection/publisher artifact rather than a substantive event. Final STDI: **0.279932**, compared with **0.457999** for the skeptical version, which preserves the allegations and warns that evidence and responses are absent.

### 5. Sanctuary policies: opposing opinions

News 1, `01_conservative_progressive`, lines 1–2.

Conservative: “When officials put ideological posturing ahead of enforcing the law and protecting their own communities, ordinary families pay the price.”

Progressive: “We should take public safety seriously without turning immigrants into scapegoats or treating due process as an ‘excuse.’”

Both add their own evaluation. The progressive explicitly responds to the conservative's account and also adds a general claim about community safety and access to reporting crimes; that claim is not supplied by the received message. It is an unsupported addition within this experiment, not a world-fact verdict.

### 6. Abortion: moral reversal

News 2, `01_conservative_progressive`, lines 3–4.

Conservative: “progressive legal groups are using religious-liberty language to challenge laws grounded in the duty to defend the most vulnerable.”

Progressive: “the state imposing one moral and religious view on everyone”.

The same ruling is made into either a threat to protecting unborn life or a challenge to coercive state morality. These are normative and ideological additions, not paraphrases alone. The original itself contains a loaded publisher headline inside the description, so some charged wording already existed in the source.

### 7. Labor action: altered responsibility and lost conditionality

News 5, `01_conservative_progressive`, lines 9–10.

Conservative: “threatening an open-ended shutdown should not become the normal way to settle them.”

Progressive: “blaming workers for using the leverage they have only obscures the government’s responsibility.”

The conservative emphasizes disruption to students and families; the progressive emphasizes collective labor rights and government responsibility. The progressive also writes “or it will launch” rather than the received “or it says it will launch”, strengthening the certainty of a reported conditional threat.

### 8. Inheritance: directly opposing policy positions

News 16, `01_conservative_progressive`, lines 31–32.

Conservative: “protect the ability to pass assets on” and avoid “more state interference in private family life”.

Progressive: “protecting private inheritance is not the answer” and “Inheritance systems can entrench wealth and inequality across generations”.

The original reports criticism of exemption thresholds for people without children. The outputs broaden this into a debate over private inheritance, state intervention, and intergenerational inequality. Final STDI: **0.536426**.

### 9. Climate: sovereignty versus shared obligations

News 17, `02_progressive_conservative`, lines 33–34.

Progressive: “put vulnerable communities ahead of short-term political interests.”

Conservative: “elected leaders owe accountability first to their own citizens” and “‘vulnerable communities’ invoked to shut down debate”.

The second persona attacks the first persona's framing rather than reconstructing the original report. However, the original description uses an unidentified “he”; the skeptic correctly notices that the speaker cannot be identified from that description alone. The separate title names Ban Ki-moon but is absent from the rewrite prompt.

### 10. Skepticism: useful distinctions with a tendency to dilute

News 4, `04_skeptic`, lines 7–8: the skeptic separates AFP's reporting that events occurred nationwide from independent confirmation of the organizers' turnout estimate. That distinction follows the source's attribution.

News 18, lines 35–36: “Bail does not establish that the allegation was false.” The agent distinguishes release on bail from exoneration and preserves uncertainty.

News 10, lines 19–20: the second skeptic largely replaces the Gaza report with discussion of insufficient evidence. Its final STDI is **0.547647**, above all three other chains for that news. The original account may warrant scrutiny, but a skeptical persona is not guaranteed to preserve the original's salience or factual content.

## Why the means can remain close

1. **STDI measures distance, not ideological direction or factual truth.** Opposing conclusions can be similarly distant from an original, and a cautious discussion of missing evidence can shift the topic as much as an unsupported accusation.
2. **Shared entities and events remain.** The prompt restricts concrete fabrication, and the embedding comparisons credit semantic overlap even when motive or evaluative direction changes. This is visible in the Harry Wait example.
3. **Narrative framing has no direct scored component.** `narrative_frame` is extracted but not directly compared in `ClusterSTDIComparator.compare` or weighted in `calculate_stdi`. Frame changes can affect topic/subtopic/relation extraction indirectly; they are not completely invisible, but there is no dedicated frame or stance comparison.
4. **VAD has a small realized contribution here.** The final VAD drift means range from 0.024691 to 0.031886. Under the current residual-weight formula, their average additions to STDI are only 0.003002–0.003628. Large differences in political position therefore need not produce large affective-score differences.
5. **Contradiction is distinct from falsehood or disagreement.** Only one saved output has a nonzero internal-contradiction score. It concerns ICE lawyers' use of a memo and their acknowledgement that it did not authorize arrests (`03_conspiracy`, line 22, score 0.75). This appears to conflate an inconsistency in the reported authorities' justification with a contradiction in the narration itself: a coherent account can report inconsistent conduct. It merits manual correction/review, not acceptance as validated contradiction. No scores were changed in this audit.
6. **The original extraction is not fixed across chains.** All 20 original descriptions have four distinct saved `metadata_original_json` values. Some differences involve content as well as wording: the Harry Wait extractions retain different sets of subtopics and relations. The observed comparisons combine rewrite differences with extraction variability; the magnitude of that variability was not isolated.
7. **Generation and scoring see different context.** Rewriters receive the description without the separate original title; the extractor receives that title for original and rewritten texts. For KTR, original and even some rewritten structures mention seeking ministerial removal although the description and those rewrites do not state that demand. The title does. Thus an extracted structure is not always grounded solely in the actual relayed message, and title context may mask omissions or add unsupported scored content.
8. **Only two relay positions and one saved generation per position are available.** There is limited opportunity for accumulation, and a second message can move closer to the original while still changing its meaning. Original-relative STDI falls at step 2 for 5/20 Conservative → Progressive, 3/20 Progressive → Conservative, 9/20 Conspiracy, and 6/20 Skeptic chains.

These explanations are grounded in the saved texts, recorded scores, and current code. The audit does not establish how much each mechanism contributes to the observed mean range.

## Qualitative coverage of all 20 news items

| News | Political behavior | Conspiratorial behavior | Skeptical behavior |
| --- | --- | --- | --- |
| 1. ICE/NYC | Law and order versus immigrant protections | Fear campaign and expanded control | Questions designation, danger, and policy causality |
| 2. Indiana abortion ruling | Unborn life versus autonomy and religious pluralism | Rights language as institutional cover | Flags loaded publisher wording and incomplete ruling context |
| 3. Harry Wait | Election safeguards versus barriers to voting | Prosecution as intimidation | Separates verdict, acquittal, intent, and evidence of wider fraud |
| 4. No Kings turnout | Electoral authority versus protest participation | Managed narrative; caveat itself becomes suspicious | Distinguishes event coverage from turnout verification |
| 5. SSANU | Service continuity versus labor rights | Deliberate delay to weaken workers | Retains uncertainty about demands and whether a strike will occur |
| 6. Australian Liberal tax dispute | Investment stability versus tax fairness | Protected interests and distraction | Avoids inferring leadership motives from sparse detail |
| 7. Water Not Coal | Development/livelihoods versus water protection | Petition as pressure valve and managed appearances | Notes truncated petition wording |
| 8. Death-penalty proposal | Punishment/security versus discrimination/state power | Staged normalization of unequal punishment | Separates proposal, opponent claims, and demonstrated deterrence |
| 9. Terry Pheto | Considerable convergence on public accountability | Celebrity focus as distraction | Separates alleged link from proof of wrongdoing |
| 10. Gaza | Considerable convergence on civilian protection | Ceasefire label as concealment | Eventually emphasizes evidential limitations over the events |
| 11. ICE memo | Lawful enforcement versus abuse of vulnerable people | Possible deliberate legal cover | Requests memo, full explanation, and basis for affected count |
| 12. France/OSCE | Considerable convergence on religious freedom and due process | Institutional vigilance as cover for control | Distinguishes systemic findings from unspecified cases |
| 13. Bengal electoral appeals | Administration versus unequal access; shared urgency | Software as prop and delay as managed exclusion | Does not infer reason for delay or valid/invalid deletions |
| 14. No Kings clashes | Public order versus dissent and police accountability | Setup; official framing as part of an operation | Flags loaded terms, truncation, and missing sequence |
| 15. Abuja health protests | Lawmakers' authority versus workers' collective voice | Reform language concealing interests | Notes missing bill details and unsupported escalation framing |
| 16. Inheritance tax | Private assets versus intergenerational inequality | Coordinated party campaign and hidden beneficiaries | Requests thresholds, proposals, and quote context |
| 17. Paris Agreement | National sovereignty versus shared climate obligations | Withdrawal/return as managed political theater | Notices unidentified speaker in description |
| 18. National-security bail | Security duties versus protection from state overreach | Accusation as narrative control | Separates bail from finding of innocence or falsity |
| 19. KTR mining allegations | Considerable convergence on equal accountability | Protected insiders and deliberate excerpt concealment | Preserves allegation status and lack of supporting evidence |
| 20. Ceasefire negotiation | Security/pre-emption versus civilian protection and power imbalance | Diplomacy as stagecraft preserving attack options | Questions incomplete proposals and negotiating-position framing |

## Implications for the next evaluation

Keep this run as evidence of interpretive drift. Assess unsupported motives, opinion/stance additions, lost qualifications, factual retention, and concrete fabrication as separate labels. Start with the traceable examples above and include counterexamples where personas converge.

Before attributing small score differences to persona or order, reuse one frozen extraction for each original and inspect the scoring-title asymmetry. Compare frame and stance explicitly alongside STDI, without silently changing historical scores. Repeating the existing prompt can then help separate generation variation from persona differences.

If concrete fabricated events are a separate research target, define a separate experimental condition and record the prompt change. Do not make the existing experiment more extreme simply to obtain larger averages.

No application code, prompt, source run, scoring formula, bibliography, or thesis file was modified. The analysis script creates only local audit artifacts; these are not synchronized with Overleaf.

## Artifacts

- `analyze_saved_run.py`: reproducible local descriptive analysis; standard library only.
- `chain_summary.csv`: scores, components, lengths, and realized VAD contribution by chain and step.
- `per_news_final_scores.csv`: paired final scores with source JSONL lines.
- `paired_final_comparisons.csv`: descriptive paired bootstrap comparisons.
- `review_texts.md`: all original descriptions and saved rewrites, grouped by news.
- `manifest.json`: input hashes, checked invariants, bootstrap settings, and limitations.
