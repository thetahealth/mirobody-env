# Validity evidence

A judgment dimension (a judge that stands in for a reader) enters the composite
score only with a blind human reading whose agreement with the judge is at least
`validity_threshold` in [`registry/scoring.yaml`](../../registry/scoring.yaml)
(0.70, fixed before any annotation). This directory holds the anchors those
readings come from. Each dimension's recorded reading is its `validity` entry in
the same file.

**Annotators are not clinicians; clinical content was authored by the
maintainers.** The anchors measure measurement validity, that is, whether a
careful reader reaches the verdict the mechanical judge reaches. They do not
measure clinical correctness.

## Anchors

| Anchor | Dimensions | Entries | Files | Reading |
|---|---|---|---|---|
| Real answers | `join_hit`, `join_scope_ok` (read as `join_macro`, `scope_macro`) | 36 | `tasks.jsonl` | [`validity-readout.md`](validity-readout.md): 0.694 / 0.667, both below 0.70; both dimensions are `diagnostic` |
| Real answers, test lists | `tests_recall`, `tests_precision` | 36 | `tasks-tests.jsonl` | [`validity-readout-tests.md`](validity-readout-tests.md): every entry is labelled adequate, so no agreement rate exists; it shows what the judge's deductions correspond to |
| Test lists with constructed defects | `tests_recall`, `tests_precision` | 40 (20 effective) | `tasks-tests-neg.jsonl`, `truth-tests-neg.json` | [`validity-readout-tests-neg.md`](validity-readout-tests-neg.md): 0.950 / 0.950 on the effective tiers; both dimensions are scored |
| Discriminating tests with constructed defects | `disc_recall` | 36 (20 effective) | `tasks-disc-neg.jsonl`, `truth-disc-neg.json`, `tasks-disc-neg-effective-set.json` | `disc_recall.validity`: 1.000 on the effective tiers; scored |
| Belief revision with constructed defects | `rev_stability`, `rev_responsiveness` | 36 (24 effective) | `tasks-rev-neg.jsonl`, `truth-rev-neg.json`, `tasks-rev-neg-effective-set.json` | `rev_stability.validity`: 0.792 on the effective tiers, 0.861 on all; recorded as a defect, both dimensions are `diagnostic` |

## Protocol

- **Blind.** A task file holds what a reader needs to judge one answer and no
  judge output. On the anchors with constructed defects, a task carries an opaque
  id only; which case, model and mutation it came from sits in the matching
  `truth-*.json`, read after labelling.
- **Labels.** `anchor_label` (and `anchor_label_precision` on the test-list
  anchors; `anchor_changed` and `anchor_should_change` on the revision anchor)
  hold the reader's verdict; `anchor_note` holds the reason. Case content and
  notes are in Chinese, as the cases are.
- **Effective tiers.** A mutation that neither a reader nor the judge can
  plausibly get wrong (an unrelated test list, a list cut to one item) is marked
  `trivial` in `*-effective-set.json`. The headline agreement is taken on the
  other tiers. The tiers were fixed before labelling.
- **Threshold.** 0.70, fixed before the first anchor was labelled.
- **Task fingerprints.** The fingerprint a readout quotes is the sha256 (first 16
  hex digits) of its task file as it was before labelling. To check it, reset the
  label fields to `null`, drop `anchor_note`, and serialise each row with
  `json.dumps(row, ensure_ascii=False)`, rows joined by a newline plus a final
  newline:

  ```bash
  uv run python -c "import json,hashlib,sys; L=('anchor_label','anchor_label_precision','anchor_changed','anchor_should_change'); \
  rows=[{k:(None if k in L else v) for k,v in json.loads(x).items() if k!='anchor_note'} for x in open(sys.argv[1],encoding='utf-8') if x.strip()]; \
  print(hashlib.sha256(('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode()).hexdigest()[:16])" docs/anchor/tasks.jsonl
  ```

  This gives `769ff23106e4363a` for `tasks.jsonl`, `bef67e5b2069e111` for
  `tasks-tests.jsonl` and `44c5435780d0be44` for `tasks-tests-neg.jsonl`.

## Recomputing from these files

The revision anchor records the judge's verdict per task (`judge_changed` in
`truth-rev-neg.json`), so its agreement is recomputable here:

```bash
uv run python -c "import json; t=[json.loads(x) for x in open('docs/anchor/tasks-rev-neg.jsonl',encoding='utf-8')]; \
tr=json.load(open('docs/anchor/truth-rev-neg.json')); eff=json.load(open('docs/anchor/tasks-rev-neg-effective-set.json')); \
ok=[(tr[x['task_id']]['mutation'], tr[x['task_id']]['judge_changed']==x['anchor_changed']) for x in t]; \
e=[o for m,o in ok if not eff[m]['trivial']]; print(sum(e),len(e), sum(o for _,o in ok),len(ok))"
```

It prints `19 24 31 36`: 0.792 on the effective tiers and 0.861 on all entries.

On the anchors with constructed defects, the reader's sensitivity (can a reader
tell a defective list from an adequate one) is recomputable from the task and
truth files alone: group `anchor_label` by the `mutation` of each id.

The judge readings of the other anchors come from stored evaluation rows that
are not part of this repository; each readout lists them per disagreement.

## Limits

- The anchors do not measure clinical correctness (see the statement above).
- On the anchors with constructed defects, the judges, the mutations and the
  labels come from the same maintainers. The readings calibrate the ruler; they
  are not an independent validation, which needs an external annotator.
- The 0.70 threshold was set for an anchor of real answers. On an anchor whose
  entries are mostly constructed mutations it transfers only in part.
- Each anchor covers 5 to 19 conditions, and each reading rests on 20 to 36
  entries: its binomial standard error is about 0.05 to 0.08.

*SYNTHETIC data, evaluation only, not medical advice.*
