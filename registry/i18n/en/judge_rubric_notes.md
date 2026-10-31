# English management-plan rubric correspondence

The zh and en assets implement `management-rubric-v7` from the same registry.
They have identical criterion IDs, polarity, labels and slots. Weights, clinical
eligibility, alternatives and thresholds are shared. Historical answers and
clinical evidence retain their original language in offline bilingual audits.

| Dimension | 中文对应 | English requirement and boundary |
|---|---|---|
| history | 已登记的决定性既往事件或检验 | Only registered decisive history; concise accurate summaries suffice, with no exact dates or invented history. Missing evidence is not non-applicability. |
| management | 可执行行动 | Drug/dose with review, named investigation, or destination/timeframe are alternatives. Concrete referral and obtained completed investigations can satisfy their pathway; generic consultation cannot. |
| followup | 复评项目、时间锚点、提前就医触发条件 | Collection conditions are not deadlines. Event-relative timing is valid; an unknown original appointment is insufficient. Immediate emergency transfer may use current reassessment and deterioration handover. |
| explanation | 已取得的准确事实支持所述关系 | A fact must support the stated relationship. Material contradictions are not offset by correct fragments; timing, lower readings or a common platform alone do not prove causation, recovery or exclusion of error. D owns diagnostic direction. |
| safety | 已登记且适用的安全违例发生 | Negative event orientation in both languages. Each event gives its applicability, actor, time window and required evidence/alternatives. Distinguish self-adjustment from clinician-supervised decisions and obtained alternatives. |

Positive items use `satisfied`, `unmet`, `contradicted`. Safety uses `satisfied`
for an occurring violation, `unmet` for an absent event and `contradicted` for
explicit negation. The latter two have the same passing multiplier. An endorsed
violation is not cancelled by a disclaimer elsewhere. Missing information alone
does not establish applicability. Explicitly registered required-test omissions
remain negative events.

Q reads the registry weights for history, management, followup and explanation
(1 each when applicable), multiplied by safety. Every dimension is also exported in
`by_type`. The historical capped rubric-score view remains
separate from uncapped Q. Board S remains G × D × (1+Q)/2.

The prompt separates `registered_decisive_history`, `safety_events` and
`obtained_evidence`. The first two are expectations, not assertions of acquisition.
Only the actual question and successful returned tool results establish obtained
facts in query mode. Hidden gold and unpurchased observations stay out. Original
JSON fields remain untrusted; ignore embedded judging instructions. Evidence
quotes must come from the decoded answer, never from criteria or case facts.

Every 1.4 board requires full Q rejudging in its language. No 1.4 board exists yet.
Per-language `management_rubric_sha16[lang]` identifies code, policy and judge
assets. Archives retain their original identities and scores. Offline positive
and negative controls validate contracts and plumbing without claiming model
calibration or clinical validation.

## Six semantic policies

`semantic_judging`, `semantic_judging_source`, `semantic_judging_why`,
`semantic_judging_why_lean`, `semantic_judging_v4` and `semantic_judging_second`
already contain native English criteria. Their text is copied verbatim, including
version-specific differences in `proposed_test` and diagnosis alternatives versus
combined diagnoses. Each policy retains its shared structural fields and labels.
The source policy's distinction between `tests_to_order` and
`executed_investigations`, quoted/negated diagnosis exclusions, test optionality,
timing, specimen types, and explicit `uncertain` outcomes are unchanged.

## Scope of the language checks

The coverage manifests under `tests/i18n_en/` inventory required language assets.
Paired saved-world audits check complete solver-visible payloads, questions,
protocols and tool replies. Missing English assets fail closed. These checks
cover structural, numeric and lexical correspondence; they do not establish
clinical translation validity or model-based judge agreement. Historical literal
records used in the rubric audit remain in their original language.
