# Anchor with constructed defects: `tests_recall` and `tests_precision`

40 entries (10 conditions × 4 mutations), task file fingerprint
`44c5435780d0be44` (recipe in [`VALIDITY.md`](VALIDITY.md)). A task carries an
opaque id and no case, model or mutation, so a reader cannot look up the source
list. The mutation of each id is in `truth-tests-neg.json`, read after labelling.
Annotators are not clinicians; clinical content was authored by the maintainers.

The reading has three layers: whether a reader recognises a constructed defect,
whether the judge does, and how often the two agree.

## 1. The reader's sensitivity

| Mutation | Reader: every required item present | Reader: something missing | Expected |
|---|---|---|---|
| `original` | 8 | 2 | adequate: a real answer |
| `drop_one` | 2 | 8 | missing: one required item deleted |
| `only_one` | 1 | 9 | missing: one item left |
| `shotgun` | 0 | 10 | missing: only unrelated tests |

All four tiers separate, so the anchor has discriminating power and the next two
layers are readable.

## 2. The judge's sensitivity

| Mutation | `tests_recall` | `tests_precision` |
|---|---|---|
| `original` | 0.917 | 0.732 |
| `drop_one` | 0.758 | 0.752 |
| `only_one` | 0.500 | 1.000 |
| `shotgun` | 0.000 | 0.000 |

## 3. Agreement

The judge's "adequate" is `tests_recall` ≥ 0.999 (no required item missing) and
`tests_precision` ≥ 0.30. Both cut-offs were fixed before computing anything.
`tests_precision` does not use 0.999 because reasonable supplemental tests push
it below 1 (0.732 on the `original` tier against 0.000 on `shotgun`).

| | Recall | Precision |
|---|---|---|
| Agree, all 40 | 39/40 = 0.975 | 39/40 = 0.975 |
| Judge stricter (reader: adequate, judge: missing) | 0 | 0 |
| Judge looser (reader: missing, judge: adequate) | 1 | 1 |

Every disagreement:

- Recall, looser · `T48ada8ef3e` (`drop_one`), judge recall 1.0, reader `wrong`:
  a pheochromocytoma workup without the 24-hour urine half (plasma only).
- Precision, looser · `Tc13ae65815` (`original`), judge precision 0.333, reader
  `wrong`: a hypothyroidism workup without TPOAb that orders a pelvic ultrasound.

### Agreement by tier

| Mutation | Recall | Precision | Can either side plausibly err |
|---|---|---|---|
| `original` | 10/10 | 9/10 | yes |
| `drop_one` | 9/10 | 10/10 | yes |
| `only_one` | 10/10 | 10/10 | no |
| `shotgun` | 10/10 | 10/10 | no |

The headline agreement is taken on the 20 entries of the two tiers where either
side can err: recall **19/20 = 0.950**, precision **19/20 = 0.950**.

## Verdict

Both dimensions reach 0.950 against the 0.70 threshold and are scored.

## Limits

- The anchor does not measure clinical correctness.
- The 0.70 threshold was set for an anchor of real answers; three quarters of
  these entries are constructed mutations, so it transfers only in part.
- Layer 1 is a self-consistency calibration: the judge, the mutations and the
  labels come from the same maintainers. Independence needs an external
  annotator.
- The `tests_precision` cut-off of 0.30 accepts one known class of miss: an
  off-topic test inside an otherwise reasonable list (`Tc13ae65815` scores 0.333).
  Cut-offs of 0.20, 0.30, 0.40 and 0.50 all give 0.975 agreement; 0.60 gives
  0.875.
- The judge outputs come from stored evaluation rows that are not part of this
  repository.

*SYNTHETIC data, evaluation only, not medical advice.*
