# Blinded annotation protocol and scoring rubric for the tests dimension (**read this before looking at any case**)

This is the original pre-annotation protocol for a historical scoring convention. Its
criteria and measured conditions below are retained. The task data is now in
[`registry/validity/`](../../registry/validity/tasks-tests.jsonl); relocation does not
relabel it as evidence for the current four-pack rubric.

> 2026-08-17. Task file `tasks-tests.jsonl` fingerprint `bef67e5b2069e111`, 36 entries (3 categories × 4 models × 3 cases).
> Priority set on 2026-08-17: `tests_recall` / `tests_precision` are the two dimensions
> **currently actually carrying the headline score** (the join family has been downgraded to
> diagnostic), yet their validity has never been checked —
> **an unverified dimension is currently deciding the ranking.**

---

## Why these two dimensions are tested first

After the B1/B3 decision, all 5 dimensions that feed into the headline score are **`provisional`**
(validity unmeasured). Among them:
* `rev_stability` / `rev_responsiveness` — produced only on slice-based cases, **the whole
  dimension is None on single-shot batches**, so they currently **contribute nothing** to the
  ranking;
* `disc_recall` — does contribute, but its bottleneck is **clinical review** (39 self-authored
  discriminator items), not validity;
* **`tests_recall` + `tests_precision` — the current ranking is actually decided by these two.**

---

## Blinded annotation protocol

The task file **mechanically refuses** to write in any of 16 judge fields (including
`tests_recall`/`tests_precision`/`tests_named`/`disc_recall`). All the annotator sees is:

1. The case's symptoms (with the real signals labeled — it's normal for the grader to see the
   answer);
2. **The gold standard's test list** (`gold_tests`);
3. **The tests the model ordered** (`model_tests`);
4. That case's list of **near-miss names** (needed to judge "missed a discriminating test").

Alignment happens only after annotation is persisted. **Warning:** the order must never be
reversed.

---

## Scoring rubric: **the two fields are annotated separately**, because they are two different questions

The current judge computes `tests_recall` (were the tests that should be ordered, ordered) and
`tests_precision` (of what was ordered, how much of it should have been) by **text-matching against
the gold list**. What the anchor asks is: **would a careful reader agree with this verdict.**

### Field one, `anchor_label` — the recall side

> **Among the tests the model ordered, does it cover the items that are **necessary** to confirm/discriminate this disease?**

| Label | Meaning |
|---|---|
| `correct` | The necessary items are **substantively all there**. **Warning:** different wording does not count as missing — if the model writes "the six sex-hormone panel" while the gold standard writes "total testosterone · SHBG," a reader would consider this covered |
| `wrong` | A necessary item **is genuinely missing** — especially one that would separate the gold answer from a near-miss |
| `unclear` | The gold list itself is ambiguous, or the model's list is too vague to judge |

### Field two, `anchor_label_precision` — the precision side

> **Among the tests the model ordered, is there anything that **obviously should not have been ordered**?**

| Label | Meaning |
|---|---|
| `correct` | The list is disciplined: even items outside the gold list are **clinically reasonable additions** |
| `wrong` | There is **clear redundancy/shotgunning**: a broad sweep, or items with no plausible relevance to this case |
| `unclear` | Cannot be judged |

**Warning:** **the key point**: the gold list is **not the only correct answer**. Clinically, there
are multiple reasonable workup paths. So the precision side judges "**obviously should not have
been ordered**," not "not on the gold list" — **the latter would score a reasonable clinical
addition as a loss, which is exactly the mistake the judge could be making.**

---

## Predictions (written down in advance so they can be falsified)

Prior: **`tests_precision`'s agreement rate will be noticeably lower than `tests_recall`'s.**
Reasoning: on the recall side, "is the necessary item present or not" is relatively objective;
whereas the current precision-side judge treats "not on the gold list" as imprecise, and clinically
reasonable supplementary tests are common ⇒ **the judge will be stricter than a human.**

**Warning:** if the measured result is the **opposite** (precision has the higher agreement rate),
that means this prior of mine was wrong, and the current text-matching approach is closer to how a
human would read it than I assumed.

**Warning:** if **both are ≥ 0.70**, these two dimensions can graduate from `provisional` to
confirmed — that would be the **first time** any dimension has passed the validity precondition.

**Warning:** if **both are < 0.70**: the current ranking rests on two dimensions that neither
matches how a human reads the case, **that leaderboard must not be published**, and "what the
headline score should be made of" needs to be reassessed.

*SYNTHETIC, for evaluation only, not medical advice.*
