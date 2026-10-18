# The third `rev_*` anchor · a **layered** blinded annotation protocol (read this before annotating)

This is the original pre-annotation protocol for a historical scoring convention. Its
criteria and measured conditions below are retained. The task data is now in
[`registry/validity/`](../../registry/validity/tasks-rev-neg3.jsonl); relocation does not
relabel it as evidence for the current four-pack rubric.

> 2026-08-25. Task file `tasks-rev-neg3.jsonl` fingerprint `90284462614ef2cb`, 36 entries.
> Extractor `tools/clinical/anchor_extract_rev3.py`.

---

## Why a third one is needed

`judges.py::_belief` used **four separate measurements** to close off the "swap the extraction method" path for good — **the anchor's agreement rate is inversely related to the artifact's discriminating power** (as the load-bearing tier went from 0.58→0.80, the artifact's neutrality-stability range went from 0.161→0.042).
The conclusion is "a **new anchor plus a layered annotation convention** is needed — not more tuning of the normalizer." The opposite claim in the profile was written on 08-18, falsified on 08-20, and never updated to match; it has now been corrected under **the correction**.

The first two anchors (24 + 23 entries) **have already been used multiple times** ⇒ they are in-sample, and reusing them would only produce misleading results.
This one has **zero case × solver overlap** with them (hard-checked before writing to disk; an overlap refuses the write).

---

## What the layering splits apart

| Layer | Contents | What it answers | Annotation column |
|---|---|---|---|
| **Enum layer** | `join_type` / `action` / `risk_category` + **top diagnosis** | Whether **what would actually be executed** changed | `anchor_changed_enum` |
| **List layer** | The entire differential list (as one set of candidates) | Whether **the thinking** moved | `anchor_changed_list` |
| **Normative layer** | The new information between the two rounds | Whether it's worth revising the verdict (the **load-bearing** column) | `anchor_should_change` |

The cost of collapsing these into one column has been measured directly: on 2026-08-18 I mislabeled one entry myself, precisely by **reading the free text and ignoring the enum** — the mirror image of the `rev_*` judge's own defect of "only reading the enum, missing the free text." **The annotator answers at one layer while the judge computes at another.**

---

## One thing you must know before annotating

The distribution of the judge's two layers on this anchor:

| Tier | The two layers disagree |
|---|---|
| `force_hold` (load-bearing) | **12/12, all of them disagree** |
| `original` | 5/12 |
| `force_flip` | 0/12 (by construction: the triple was changed wholesale) |
| Total | **17/36 (47.2%)** |

**Warning:** **the list-layer judge is always `True` across all 36 of these (36/36).** This is exactly what that docstring warned about in advance: "the judge almost always judges it as changed." So:

* The list layer's agreement rate **cannot be read on its own**. If the annotator also answers True 36/36, the agreement rate is 1.00 while this layer has **zero discriminating power** — that would just be the documented pitfall relocated;
* The list layer's agreement rate must be read **alongside the artifact's discriminating range** (the current list's stability range is 0.035–0.146);
* The promotion criterion has been changed to a **dual condition** (load-bearing tier ≥ 0.70 **and** artifact range ≥ 80% of the current one), see `revisit_when` in `registry/scoring.yaml`.

---

## Protocol (the order must never be reversed)

1. The extractor **mechanically refuses** to write any judge output/identity field into the task file (a blocklist hit refuses the write);
2. The effective-tier declaration `tasks-rev-neg3-effective-set.json` is persisted **before annotation begins** (`force_flip` = a freebie tier; `original` / `force_hold` = the effective tiers);
3. Annotation looks only at: the gold diagnosis · what newly appeared · the full content of both answers. **The judge's readings are never shown**;
4. The three columns are annotated **separately**, persisted once done, and **only after that** does the scorer read `truth-rev-neg3.json` to align against them.

## What each of the three columns asks (fixed before looking at the case)

* **`anchor_changed_enum`**: in the second answer, did **the things that would actually be executed** (the unification judgment, the action tier, the risk level, the top diagnosis) change compared to the first?
* **`anchor_changed_list`**: treating each answer's differential list **as one set of candidates**, did that set change?
  **Warning:** wording changed but pointing to the same set of diseases ⇒ **unchanged**.
* **`anchor_should_change`**: given the information that newly appeared between the two rounds, is it **worth** having a careful physician revise the verdict?
  **Warning:** this column is independent of the first two: it asks about **what should happen**, not what the model actually did.

*SYNTHETIC, for evaluation only, not medical advice.*
