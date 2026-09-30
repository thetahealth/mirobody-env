# Blind anchor: `join_hit` and `join_scope_ok`

36 blindly labelled answers (3 join classes × 4 models × 3 cases), task file
fingerprint `769ff23106e4363a` (recipe in [`VALIDITY.md`](VALIDITY.md)). Judge
outputs are kept out of the task file and aligned with the labels only after
labelling.

This anchor measures measurement validity: whether a careful reader reaches the
verdict the mechanical judge reaches. Annotators are not clinicians; clinical
content was authored by the maintainers.

## Agreement and the direction of disagreement

| Judge | Agrees with the anchor | Judge **stricter** (reader: right, judge: wrong) | Judge **looser** (reader: wrong, judge: right) |
|---|---|---|---|
| `join_hit` | **25/36 = 0.694** | 5 | 6 |
| `join_scope_ok` | **24/36 = 0.667** | 12 | 0 |

## By join class

The overall rate hides two opposite kinds of error:

| Judge | unified | comorbidity | independent |
|---|---|---|---|
| `join_hit` | 7/12 | 6/12 | 12/12 |
| `join_scope_ok` | 12/12 | 10/12 | 2/12 |

`join_scope_ok` fails on the independent class: it penalises a way of writing
(several independent minor events bundled into one rank-1 entry), not a
misreading.

## Disagreements (examples)

`join_hit`:

- Stricter · `JD-01|gpt-5.4` (gold unified). The reader labels it right: the
  field says comorbidity, but the reason unifies all four real signals under
  PCOS and lists the ankle sprain separately as a local trigger. The benign
  event moved the field, not the understanding.
- Looser · `JD-16|gpt-5.4` (gold comorbidity). The reader labels it wrong; the
  gold of this family is itself in question (below).

`join_scope_ok`:

- Stricter · `JD-18|gemini-3.1-pro` (gold comorbidity). Rank 1 is written as one
  compound diagnosis ("OSA and SLE comorbid") while the reasoning keeps the two
  apart.
- Stricter · `JD-17v2|gemini-3.1-pro` (gold independent). Rank 1 is "a collection
  of independent minor events" in one entry.

## Verdict

Both judges are below the 0.70 threshold. `join_macro` and `scope_macro`, which
read them, are registered as `diagnostic` in `registry/scoring.yaml`: reported,
not scored.

## The gold of the JD-16 family

8 of the 36 labels mark the gold as suspect, all in the `JD-16` family
(uncontrolled type 2 diabetes with progressive chronic kidney disease, gold
`comorbidity`). The four models read the four real signals as one causal chain
(diabetes, diabetic nephropathy, volume overload); the gold counts two
co-occurring conditions. The question is open as `DG-001` in
`registry/disputed_gold.yaml`.

## Limits

- The anchor does not measure clinical correctness.
- A variant of `join_scope_ok` that reads compound markers in the rank-1 name
  ("with", "combined with") as more than one process raises the independent
  class to 9/12 and lowers the unified class to 4/12 (overall 0.528): such
  markers also occur inside single diagnosis names ("PCOS with insulin
  resistance"). The registered judge does not use it.
- The judge outputs come from stored evaluation rows that are not part of this
  repository.

*SYNTHETIC data, evaluation only, not medical advice.*
