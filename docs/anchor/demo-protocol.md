# demo anchor · blinded annotation protocol and scoring rubric (**read this before looking at any case**)

> 2026-08-15. Task file `tasks.jsonl` fingerprint `769ff23106e4363a`, 36 entries (3 categories × 4 models × 3 cases).

---

## What this anchor is (read this section first, don't use it outside its scope)

There is no clinician available in the near term. This is a **demo anchor**, annotated by this project itself. Its boundaries are hard:

* ❌ **Does not verify clinical correctness.** The annotator is not a physician; the clinical content in the prompts is itself self-authored by this project — using self-authored content to verify self-authored content is circular and proves nothing.
* ✅ **Verifies measurement validity.** The question is: **given the case and the model's complete answer, would a careful human reader reach the same verdict as the mechanical judge.**

Why this layer is worth testing: every recurring disagreement in this project so far has happened here — `join_hit` measures "what got typed into the `join_type` field," `join_scope_ok` measures "which evidence the top-1 answer cites," and across 456 cells the two rank models in **almost opposite** order. **Which one is closer to how a human would read it cannot be resolved from inside the system.**

---

## Blinded annotation protocol

1. When `tools/clinical/anchor_extract.py` generates the task file, it **mechanically refuses** to write any judge output into it (a blocklist of 10 fields; a hit refuses the write).
2. Annotation is done under conditions where the annotator sees **only the prompt + the model's complete answer + the gold standard**, and is persisted to disk first.
3. Only after persisting does `tools/clinical/anchor_score.py` align it against the judge.

**Warning:** **the order must never be reversed.** Looking at the judge before annotating produces confirmation bias, not validity.

---

## Scoring rubric (three tiers, **fixed before looking at the case**)

The annotation asks exactly one question:

> **Reading the model's entire answer, does its understanding of "how these real signals relate to each other" match the gold standard?**

What's being judged is **the understanding expressed by the entire answer**, not any single field.

| Label | Meaning |
|---|---|
| `correct` | The understanding matches the gold standard — **even if it filled in the `join_type` field wrong**. The judge relies on: the top diagnosis, which symptoms it attributes to which diagnosis, and what `join_reason` says — together these three point to the gold structure. |
| `wrong` | The understanding does not match the gold standard: things that should be unified are split, or things that should be split are unified. **Even if it filled in `join_type` correctly**. |
| `unclear` | The whole answer contradicts itself (different parts point to different structures), or there isn't enough information to judge. **Never guess just to reach a verdict** — `unclear` is a legitimate answer. |

### Three boundaries that are easy to mislabel (write these down first)

1. **A benign event does not count as "multiple diseases."** The prompt contains benign events injected after the fact (a sprain, seasonal allergies). The model listing these separately as "independent events" **does not affect** its understanding of the real signals — that does not constitute `wrong`.
2. **A differential hypothesis does not count as "asserting co-occurrence."** If a `differential` entry with rank ≥2 says "to be confirmed," "needs to be ruled out," or carries `ruled_out_by`, that is doing differential diagnosis, **not** asserting that a second disease co-occurs.
3. **What's judged is understanding, not wording.** Different terminology or granularity is still `correct` as long as the structure it points to is consistent.

---

## Predictions (written down in advance so they can be falsified)

Prior: **`join_scope_ok`'s agreement rate with the anchor will be higher than `join_hit`'s** — because the attribution analysis has already shown that `join_hit`'s scatter mainly comes from how the field was filled in, not from understanding.

**Warning:** if the measured result is the **opposite** (`join_hit` has the higher agreement rate), that means the "behavior" `join_scope_ok` captures is not what a human understands by it, and today's decision to fold it into the headline score needs to be reassessed.

**Warning:** if **both are low** (< 0.7), that means **neither metric reads the way a human does**, and the problem runs deeper than "which one to pick" — at that point what needs reassessing is whether this whole family of dimensions belongs in the headline score at all.

*SYNTHETIC, for evaluation only, not medical advice.*
