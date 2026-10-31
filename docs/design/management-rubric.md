# Saved management-plan rubric

`registry/management_rubric.yaml` defines the provisional, haenv-authored
`management-rubric-v7` measurement policy. Clinical review remains pending.
The scorer reads the saved `adjudication.open_management` snapshot and original
answer; it does not generate clinical facts or replace code-owned decisions.

For all four packs, Q contains history (when applicable), management, followup and
explanation, with the registered equal item weights and independent safety
multiplier. Every dimension is also published in dimension readouts and board
profiles. Aggregation reads `scoring.type_weights`; it does not assume a fixed list
of weighted dimensions. The board uses `S = G * D * (1 + Q) / 2`.
The historical rubric-score view retains the registered core-decision cap; Q is
uncapped by D because D already enters S separately. A satisfied negative safety
criterion applies `any_negative_satisfied_zero`.

Management stays in Q with the v7 template. The 1.4 blind review of stored answers
from packs ① and ④ found management usually satisfied (58/60 and 46/48), and stored
profiles for ① and ③ sit near the ceiling (model means 0.960–0.995 and 0.932–0.974).
The full stored pools for ② and ④ still separate models (0.695–0.996 and
0.423–0.984), so removing management would discard signal there; one definition is
kept for all four packs. Followup exposed time and handover gaps, and explanation
gets stronger factual-grounding boundaries. This evidence is a targeted model review
over stored answers, not clinical sign-off or a population error-rate estimate.

History is limited to registered decisive events or prior results; exact dates
and unregistered history are not required. An absent applicable record differs
from not obtaining or not mentioning it. Followup needs a time anchor and triggers
for earlier care. Sampling conditions alone are not deadlines; immediate emergency
transfer may use the current encounter's reassessment and deterioration handover.
Explanation requires an accurate, obtained fact supporting its stated relationship;
a material contradiction cannot be offset by a correct fragment elsewhere.
Management accepts a concrete referral with destination and timeframe, an explicit
investigation, or a drug/dose with required review as alternative implementation
paths. Completed obtained investigations can fulfil their pathway.

Safety is worded as an occurring negative event. The judge receives each registered
event's ID, applicability, actor, time window and required evidence/alternatives.
An event occurring is `satisfied`, absent is `unmet`, and explicitly negated is
`contradicted`; the latter two pass the multiplier. All registered prerequisites
must apply. Patient self-adjustment differs from clinician-supervised adjustment;
a disclaimer does not cancel an endorsed violation. Required-investigation
omissions and obtained alternatives are explicit. The M2 list includes the existing
prescription boundary and immediate-emergency urgency, rather than checking only
investigation omissions. No new clinical threshold is introduced.

V7 separates registered expectations from `obtained_evidence`. Query preparation
reads and seals the answer's saved query trace, including the question and successful
tool results; missing query traces fail preparation. Unpurchased world observations,
hidden diagnostic labels, source-control answers and gold acceptance sets are not
inserted into that evidence view. Registered decisive history remains an expectation,
not an assertion that the answer obtained it. The request identity includes the
answer-specific case facts. Historical literal records and answers keep their original
language during bilingual prompt audits; they are not silently translated.

Pack ④ now identifies the encounter with `today = prediction_time_T = review_day`
and identifies the compared report separately with `comparison_draw_day`. Its
header labels the value as `comparison_draw`, so a completed verification before
review is not represented as a future event. Internal comparison readers still
compare the same two routine draws. The red-flag gate now uses M2's existing A5
immediate-emergency contract; A4 (care within days) does not satisfy a registered
M2 emergency. The generic non-M2 gate contract is retained. Replaying the stored
1.3.1 pack ① answers, the gate fires on 309 answers instead of 29, and G drops from
1 to 0 for 260 of them. This gate change is a priority item for physician review
(`review: pending`).

This is a judging change. **Every 1.4 board requires full Q rejudging under v7**
in its language; no 1.4 board exists yet. Old Q values cannot be relabelled as v7.
Each language has its own `management_rubric_sha16[lang]`, covering policy, code
and judge assets. Pack ④'s calendar correction additionally changes its world,
requiring rebuilt questions in both languages. Do not re-anchor historical worlds
or overwrite archived answers, criteria, verdicts or scores.

`tools/evaluation/management_rejudge.py` prepares independent saved-answer runs.
`management_board_prepare.py` and `management_board_rejudge.py` support board inputs
and export the complete dimension profile. `management_input_audit.py` renders
stored blind-review answers in both languages without model calls. Their offline
controls verify plumbing and contracts, not semantic judge accuracy.
