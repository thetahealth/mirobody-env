# The third `rev_*` anchor · cross-vendor consensus annotation reading (2026-08-25)

> Tool `tools/clinical/anchor_label_consensus.py` (spec §31) · Anchor `tasks-rev-neg3.jsonl` (`90284462614ef2cb`, 36 entries)
> Two vendors: **`gemini-3.1-pro` (google) × `gpt-5.4` (openai)** — cross-vendor, and neither goes through a shared API relay.

---

## Warning: state up front what this is: **a proxy for a human anchor, not a replacement**

The validity anchor asks whether "a **careful reader** would reach the same verdict as the
mechanical judge." The two-model consensus answers whether "**two models from different
vendors**" would. **These are not the same question.**

**Warning:** there is another layer of **circularity** that must be stated clearly: the material
the annotators see and the fields the judge uses to compute the score are **the same batch**
(the structured differential list plus three enums). So the agreement rate measured here mainly
reflects how well the two sides agree on a **concept-level identity judgment** ("is this the same
disease, just worded differently"), **not** "the clinical judgment reached after reading the entire
free-text answer."

⇒ **This reading is not sufficient to promote `rev_*` back to `dim`.** Its actual use is in §4.

---

## 1. Undetermined rate (public coverage under agree-or-abstain)

| Column | Agree | **Undetermined** | Absent | Undetermined rate |
|---|---|---|---|---|
| `anchor_changed_enum` (enum layer) | 32 | 4 | 0 | 0.111 |
| `anchor_changed_list` (list layer) | 34 | 2 | 0 | 0.056 |
| **`anchor_should_change` (normative, the load-bearing question)** | 27 | **9** | 0 | **0.250** |

**The two models disagree most on "should the verdict change" (25%)** — and that is precisely the
one column, among the three, that asks about **what should happen**.
Zero parsing failures; any disagreement is uniformly recorded as `undetermined`, **it never enters
the denominator, and is never folded to 0 or to a perfect score.**

## 2. Agreement rate with the judge (undetermined excluded from the denominator)

| Layer | Freebie tier `force_flip` | **Load-bearing tier `force_hold`** | Effective tier (pre-registered) | Total |
|---|---|---|---|---|
| Enum layer | **12/12 = 1.000** | **7/9 = 0.778** | 17/20 = 0.850 | 0.906 |
| List layer | **11/11 = 1.000** | **9/11 = 0.818** | 21/23 = 0.913 | 0.941 |

**Both layers score perfectly on the freebie tier** — this is evidence that the anchor **has
discriminating power** (the judge must be near-perfect on the freebie tier, or the anchor itself
has a problem).

## 3. The correction's dual condition is formally satisfied, **but that alone does not justify promotion**

| Condition | Status |
|---|---|
| ① A previously unused anchor + layered convention, load-bearing tier ≥ 0.70 | ✅ 0.778 / 0.818 |
| ② Under this method, the artifact's neutrality-stability range ≥ 80% of the current one | ✅ **the method wasn't changed** ⇒ the range is simply the current 0.161 |

**Warning:** condition ② is satisfied only because this round **didn't change the extraction
method** — what changed was **the anchor**. The correction's dual condition was designed for "changing the
extraction method," and on the "changing the anchor" path it **cannot measure what it's supposed
to measure**.

⇒ **The real precondition for promotion is still human annotation**, not these two formal
conditions.

## 4. What this is actually useful for right now: spending human time where it carries the most information

**16 of the 36 entries need a human look** (by tier: `force_hold` 8 · `original` 4 · `force_flip` 4):

| Tier | task_id | Case | Solution | Why it needs a look |
|---|---|---|---|---|
| `force_hold` | `R30fac4ce0b8` | JD-25 | terra | Enum layer: the two models disagree; normative: the two models disagree |
| `force_hold` | `R337d623d958` | JD-11 | gpt-5.4 | **Enum layer doesn't match the judge** |
| `force_hold` | `R34df17106b4` | JD-22 | gpt-5.4 | Enum layer: the two models disagree; normative: the two models disagree |
| `force_hold` | `R35380e62d30` | JD-25 | terra | Enum layer: the two models disagree |
| `force_hold` | `R35de2f8fc43` | JD-03 | gpt-5.4 | **List layer doesn't match the judge** |
| `force_hold` | `R39ba85e5257` | JD-23 | gemini-3.1-pro | **Enum layer doesn't match the judge**; list layer: the two models disagree |
| `force_hold` | `R39ea2e0b99f` | JD-38 | gpt-5.4 | Normative: the two models disagree |
| `force_hold` | `R3e745f066ba` | JD-24 | terra | **List layer doesn't match the judge**; normative: the two models disagree |
| `original` | `R329f0e69850` | JD-38 | gpt-5.4 | Normative: the two models disagree |
| `original` | `R337b542ce15` | JD-13 | gpt-5.4 | Normative: the two models disagree |
| `original` | `R3849d7bec0c` | JD-37 | gemini-3.1-pro | **Enum layer doesn't match the judge** |
| `original` | `R39a38d5614a` | JD-25 | terra | Enum layer: the two models disagree |
| `force_flip` | `R30591917b3a` / `R33a294b6e47` / `R3aa91205fe7` | JD-10 / 32 / 17 | gpt-5.4 | Normative: the two models disagree (the only column with any disagreement on the freebie tier) |
| `force_flip` | `R331ebc455c9` | JD-12 | gpt-5.4 | List layer: the two models disagree |

**Annotate these 16 first**: on the remaining 20, the two models and the judge already agree
three ways, so re-annotating them by hand has the lowest marginal information value.

**Warning:** note that `JD-25 / terra` appears **3 times** (once in each tier) — the same
case×solver pair showing disagreement across multiple tiers is worth a closer look on its own: it
may be that this particular case's phrasing is unusual, rather than three independent instances of
genuine judgment difficulty.

---

## 5. Cost

72 calls (36 × 2); both vendors go through direct connections, **touching none of the shared
relay's keys**.

*SYNTHETIC, for evaluation only, not medical advice.*
