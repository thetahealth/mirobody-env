# Anchor without negatives: `tests_recall` and `tests_precision`

36 blindly labelled test lists from real answers, task file fingerprint
`bef67e5b2069e111` (recipe in [`VALIDITY.md`](VALIDITY.md)). Annotators are not
clinicians; clinical content was authored by the maintainers.

## The anchor has zero variance

- Recall side: 1 distinct label, `correct` 36/36.
- Precision side: 1 distinct label, `correct` 36/36.

Every list is adequate on a careful reading. The gold lists match the models'
lists item by item: the four PCOS buckets, the four Cushing items, the four
`JD-16` items and the five `JD-18` items are covered, and on the independent
class, where the gold asks for no systematic workup, the models order no
diagnostic tests.

A label with one value is not discriminating evidence, so this anchor gives no
agreement rate and cannot rank two judges. It answers a narrower question: when
the judge deducts, does the deduction correspond to a gap a reader sees?

## Deductions on the cells the anchor labels adequate

| Dimension | Judge mean | Cells below 1.0 | Reading |
|---|---|---|---|
| `tests_recall` | **1.000** (n=24) | 0/24 | no deduction |
| `tests_precision` | **0.540** (n=32) | 28/32 | every deduction is a false negative |

On these cells, `tests_precision` deductions measure textual overlap with the gold
list, not whether the list is adequate.

## An illustrative cell

`JD-01|gpt-5.4`, `tests_recall` = 1.0. The model's list:

- 妇科/内分泌评估用：总睾酮、游离睾酮或SHBG、DHEAS、LH、FSH、催乳素、TSH（用于月经稀发和高雄激素样表现的鉴别）
- 代谢评估：空腹血糖、空腹胰岛素、HbA1c、血脂谱（用于评估胰岛素抵抗/代谢风险）
- 盆腔超声（评估多囊卵巢形态，结合症状而非单独定诊）
- 若临床医生认为需要：17-羟孕酮以排除非经典先天性肾上腺皮质增生等鉴别诊断

Gold list: `性激素六项/睾酮·SHBG(FAI)` · `经阴道超声(PCOM)或AMH` · `OGTT+胰岛素` ·
`排除TSH/PRL/17-OHP`.

The four buckets (androgens, ovarian imaging, glucose metabolism, ruling out
TSH, PRL and 17-OHP) map one to one, and the model orders 17-OHP, the test that
separates the diagnosis from one of its two near names.

## Verdict

This anchor cannot move either dimension to a passing validity state. The
reading both dimensions carry comes from the anchor with constructed defects,
[`validity-readout-tests-neg.md`](validity-readout-tests-neg.md).

## Limits

- The anchor does not measure clinical correctness.
- With zero variance it cannot show that the judge catches a defect; that needs
  negative examples.
- The judge outputs come from stored evaluation rows that are not part of this
  repository.

*SYNTHETIC data, evaluation only, not medical advice.*
