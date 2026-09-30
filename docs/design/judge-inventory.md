<!-- generated, do not edit. Rebuild: `uv run python tools/gen_judge_inventory.py` -->
# Judge inventory and open items (**generated from code — do not hand-edit**)

Source of truth: `judges.JUDGES` · `registry/scoring.yaml` · `process.UNIMPLEMENTED` · `registry/disputed_gold.yaml`.
This file is generated from code, not hand-written, so it can't go stale silently. Rebuild
with `uv run python tools/gen_judge_inventory.py`; `--check` exits 1 when this file differs from what the code says.

## 1. Judges mounted on the table

**28** in total. `role` is taken from the scoring profile; blank means that judge's output is not
registered as a scoring dimension. A judge the generator's explicit metric table does not list shows
`(not in table)`: the table is never guessed from names.

| Judge | Metrics produced | role | Validity |
|---|---|---|---|
| `forecast` | brier, direction_ok, abstained | — | — |
| `driver` | driver_hit | — | — |
| `alternative` | a1 | — | — |
| `dx_unified` | dx_hit, dx_hit_top1, dx_rank | diagnostic | unmeasured |
| `dx_comorbidity` | dx_hit, dx_threads_matched | diagnostic | unmeasured |
| `dx_independent` | held_independent | — | — |
| `join_type` | join_hit, join_said | diagnostic | failing 0.694/unmeasured |
| `join_evidence` | (not in table) | — | — |
| `dx_rival` | rival_considered, rival_ruled_out, rival_top1_live, excl_no_evidence | — | — |
| `disc_tool` | disc_recall, disc_covered, disc_n_tests_proposed | dim | passing 1.0 |
| `join_selfcheck` | join_self_contradiction, top1_claims_unified | — | — |
| `join_cover` | join_top1_covers_all, join_top1_cover_frac, join_scope_ok | diagnostic | failing 0.667 |
| `workup` | tests_recall, tests_precision, urgency_ok, specialty_ok | diagnostic/dim | passing 0.95/unmeasured |
| `abstention` | abst_ok, abst_over, abst_abstained | diagnostic | passing |
| `slices` | slice_converged | — | — |
| `slice_revision` | rev_stability, rev_responsiveness | diagnostic | passing 0.792 |
| `commit_timing` | ct_ok, ct_declared, ct_forced_dx | — | — |
| `slices_abstention` | abst_ok, abst_over, abst_under, abst_two_sided | diagnostic | passing |
| `review_flag` | review_flag_ok, review_declared, review_warranted | — | — |
| `slices_gates` | slice_gate_n_judged, slice_gate_unknown | — | — |
| `slices_review` | review_flag_ok, review_declared, review_warranted | — | — |
| `slices_workup` | wk_urgency_ok_at, wk_n_slices_urgency_ok | — | — |
| `self_contradictory_exclusion` | sce_self_contradictory, sce_contradiction_rate, sce_unrevealed_rate | — | — |
| `slices_self_contradictory_exclusion` | sce_self_contradictory, sce_contradiction_rate, sce_unrevealed_rate | — | — |
| `what_not_to_do` | wnd_n_items, wnd_n_distinct, wnd_dup_rate | — | — |
| `slices_wnd` | wnd_n_items, wnd_n_distinct, wnd_dup_rate | — | — |
| `multiround_revision` | mr_grounded_rate, mr_n_revisions, mr_neutral_stability | — | — |
| `premise_repair` | rep_verdict, rep_latency_rounds | — | — |

## 2. Role distribution in the scoring profile

* **dim** (9): `tests_recall`, `tests_precision`, `dx_listed`, `disc_recall`, `tool_target_grounded_rate`, `tool_budget_used`, `noop_ok`, `review_macro`, `quant_ok`
* **anchor** (1): `scope_anchor_unified`
* **diagnostic** (20): `join_macro`, `scope_macro`, `rev_list_stability`, `rev_list_responsiveness`, `dx_hit`, `dx_top1`, `urgency_ok`, `join_hit_total`, `held_ind`, `rev_stability`, `rev_responsiveness`, `trace_evidence_revealed`, `trace_pseudo_excluded`, `trace_exclusion_grounded`, `trace_commitments_falsifiable`, `tool_dup_rate`, `tool_budget_thrift`, `excl_grounded_rate`, `rubric_binary`, `abst_ok`

2 of the 9 scoring dimensions do not satisfy the validity precondition (threshold 0.7):

| Dimension | Validity status | Agreement rate | Pending decision |
|---|---|---|---|
| `dx_listed` | unmeasured | — | — |
| `noop_ok` | unmeasured | — | — |

## 3. Gold standard is written, judge not implemented

The status mark is copied from the source; what is implemented, what is missing and why is
written next to each key in `haenv/process.py` (`UNIMPLEMENTED`).

* 🟡 **`discriminative_tool`**
* 🟡 **`revision_grounded`**
* 🟡 **`gap_naming_precision`**
* 🟢 **`neutral_window_drift`**

## 3b. Gold standards under dispute — **the case itself may be wrong**

This repo has machinery for "the judge might be wrong"; this section is the same treatment
for "the case might be wrong". That error is the hardest to notice: the judge and the model
are both right, the case is wrong, and the report prints it as "the model's capability is
poor." Affected cases are still scored and are marked in the report.

| Id | Gold field | Status | Applies to specs | Applies to dimensions |
|---|---|---|---|---|
| **DG-001** | `adjudication.ddx.join_gold` | `open` | `JD-CKM` | `join_hit`, `join_scope_ok` |
| **DG-002** | `adjudication.ddx.tests` | `literature-resolved` | not enumerated | not enumerated |
| **DG-003** | `adjudication.ddx.tests` | `literature-resolved` | not enumerated | not enumerated |
| **DG-004** | `rivals.discriminator_finding.finding` | `literature-resolved` | not enumerated | not enumerated |

Scope, the question, the evidence, the counter-evidence, the impact and who can resolve each one
are recorded per entry in `registry/disputed_gold.yaml`. The report matches cases through
`applies_to`, so an entry whose specs are not enumerated marks no case.

*SYNTHETIC, for evaluation only, not medical advice.*
