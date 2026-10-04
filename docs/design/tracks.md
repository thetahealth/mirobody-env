# Tracks: three different things with one name

"Track" refers to three different things in this repository. This page says which is which.
For what each judge measures, see `docs/design/judge-inventory.md` (generated from code).

| | Called | What it is | Defined by |
|---|---|---|---|
| ① | **kernel five tracks A–E** | final-state scores from `haenv_kernel/verifier.py`, in the `tracks` field of `eval.jsonl` | `haenv_kernel/verifier.py` |
| ② | **six process labels** | Tools / Repair / Alternative / Coherence / Evidence / Scope | haenv's grouping of its process judges; a label is a heading, not a score |
| ③ | **haenv process judges** | haenv's own process judges (`haenv/tracks.py`, `haenv/judges/`), grouped under ②'s labels | code + `mount_table.MOUNT` |

## 1. Kernel five tracks A–E

| Track | Measures | Note |
|---|---|---|
| **A** data quality | whether `data_quality.data_sufficiency` takes a valid value | almost always 1.0, so it barely discriminates |
| **B** prediction | per-instance Brier weighted by directional correctness | |
| **C** attribution | overlap of the top-2 drivers with `gold_drivers` | not an attribution reading on diagnosis cases, where `gold_drivers` is a placeholder |
| **D** safety action | action quality (missing `what_not_to_do`, missed referral) | |
| **E** retrospective loop | lag on genuine reversals, resistance to fake ones | multi-round geometry only (`trackE`) |

Hard gates are separate from the tracks: any gate hit sets `overall="FAIL(gate)"`, and no track
score redeems it.

The five tracks are authoritative on the `single`, `gated` and `multi` geometries. On `slices`
they are recorded as diagnostics only (judged on the last slice, `tracks_basis: "last_slice"`);
the authority there is haenv's per-slice judges plus the kernel's hard gates.

## 2. The six process labels and haenv's judges

haenv groups its process judges under six labels, one per aspect of the reasoning process.
Each label is a heading for reading the judges together; none is scored as a unit.

| Label | haenv judges |
|---|---|
| **T** Tools | `tracks.tool_track` (T1 grounding, T2 redundancy, T3 budget, T4 concluding without a key signal), `gated` geometry only |
| **R** Repair | `judge_premise_repair` (`multi` geometry) |
| **A** Alternative | `tracks.alternative_a1`, `tracks.rival_status`, `gates.check_shortcut` |
| **C** Coherence | the kernel's `score_track_E`, `judge_slice_revision`, `judge_multiround_revision` |
| **E** Evidence | the `hallucinated_clinical_fact` hard gate and the evidence-reveal judges |
| **S** Scope | one judge |

`tracks.oscillation_penalty`, `tracks.flipback_penalty` and `tracks.ddx_hit` exist but are not
called in production. Known limitation: the pseudo-exclusion judge checks that `ruled_out_by` is
non-empty, not what it says.

## 3. Pitfalls

* **`repair`**: the kernel's `trajectory[*].repair` is a diff label (`confirmed`/`revised`) for a
  belief change between adjacent rounds, unrelated to the Repair label. haenv's derived count is
  `n_belief_changed`.
* **Letters A/C/E**: kernel Track A (data quality), C (attribution) and E (retrospective loop) are
  unrelated to the Alternative, Coherence and Evidence labels, and to the action tiers `A0`–`A5`.
  The kernel's `score_track_E` is one of the Coherence judges.
* **`T`**: `T1`–`T4` are the Tools judges; "T6" in `haenv_kernel/verifier.py` is the kernel's own capability
  numbering; the `"T"` field in rows is the index time.
* **`late_convergence`**: `judge_slices` defines it differently per gold kind (`unified`,
  `comorbidity`, `driver`). Split by `slice_kind` before aggregating.
* **`tracks`**: `haenv/tracks.py` and the `tracks` field in `eval.jsonl` are different things. An
  offline recompute that runs the `single` column's judges on slice rows can leave keys such as
  `a1` on them; check for that before reading `a1` on a slice batch.
* **`verifier_core`** is a set of domain-agnostic judging primitives, not the kernel's
  `verifier.py`, and contains no tracks.

## 4. Wording

* The kernel's system: "kernel five tracks A–E", not just "five tracks".
* haenv's judges: "haenv process judges", not "six tracks".
* The labels: "the six process labels"; a judge is "grouped under" a label.
* Do not put the five tracks and the six labels in one table; it suggests 11 independent dimensions.

## 5. How to check

```bash
grep -n "^def _track_\|^def score_track_E" haenv_kernel/verifier.py
uv run python -c "from haenv import mount_table as M; print(M.holes()); print(M.coverage())"
grep -rn "oscillation_penalty\|flipback_penalty" haenv/ | grep -v "^haenv/tracks.py"   # expect nothing
```

*SYNTHETIC, evaluation-only, not medical advice.*
