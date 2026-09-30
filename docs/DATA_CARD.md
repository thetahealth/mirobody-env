# Data card · HAEnv

Read [`ETHICS.md`](ETHICS.md) before relying on any number derived from this data.
Current freeze: judging fingerprint `67dd5866790e9cb9`, world fingerprint `25f31527b09e6abe`
(`anchor/freeze-2026-09-02.json`). The freeze records the judging and generation segments of the
current code; the board batches are model runs on the question packs the anchor lists.

## What this is

A longitudinal clinical-agent benchmark. The model receives a synthetic patient whose history
unfolds over time, visible only up to an index time `T`. It is asked to produce a differential
diagnosis, order tests, judge referral urgency, or declare that the information is insufficient.
Each answer is then scored dimension by dimension.

It differs from question-and-answer medical benchmarks in two ways:

* Time is a dimension. The prompt stops strictly at `T`; the answer lies after `T`. The judges
  ask whether the agent changed its mind *at the point where it should have*.
* The gold standard comes first. The latent control variables are fixed before the course is
  rendered, and code renders the course from them. The gold standard is therefore mechanically
  adjudicable rather than annotated after the fact, and every gold field records its provenance
  (see below).

## The gold standard ships with the questions

The task specifications in `inputs/*.job.yaml` contain the gold fields. This is deliberate.
Contamination is addressed by construction, not by hiding answers:

* the latent variables (outcome, true driver, reversal point, adherence gap, noise) are fixed first
  and the course is rendered from them;
* regenerating the pool changes the patient population: a new `case_id` yields a new synthetic
  patient and a new gold standard;
* every answer-bearing file carries the canary strings ([`../CANARY.md`](../CANARY.md)): hand-written
  YAML and Markdown as a comment block, and every row of the published question packs
  (`frozen/*.Q.jsonl`) plus every batch file a run writes as a top-level `_canary` field. A corpus
  that filters for them can remove the benchmark from training data, and contamination remains
  detectable afterwards.

What this does not buy:

1. It gives no protection for scores measured on these published cases. A model trained on them
   cannot be trusted on them, and that cannot be detected from outside without log-probability
   access. Cite the fingerprints and the batch with any score.
2. "A regenerated pool yields genuinely new instances" is an argument from construction, not an
   experimental result. Nobody has yet trained on one pool and tested on another.

## Contents of the current freeze (2 packs · 290 cases)

Both packs hold the same 145 cases. They differ in geometry (`slices` against `gated`) and in one
field: `latent.rhythm_gap` is removed from the 20 cases that carry it, because an information gap
cannot be delivered without slices.

### Main pack

When this document says "the benchmark", it means this pack.

| Pack | What it tests | Batch | Cases |
|---|---|---|---|
| `ddx-timeline` | 64 conditions × variants × the tiers below; `slices` geometry | `20260929-141137` | 145 |

### Single-factor arms

Each arm changes one factor relative to its counterpart. Read it only paired with that
counterpart; on its own it is under-powered.

| Pack | What it tests | Batch | Cases |
|---|---|---|---|
| `ddx-workup` | Tool track: test ordering under a budget; `gated` geometry. Counterpart `ddx-timeline`: the same 145 `case_id`s with identical longitudinal data | `20260929-141244` | 145 |

Tiers inside `ddx-timeline`, marked per case:

| Tier | Marker | Cases |
|---|---|---|
| Long horizon | `index_time_T = 336` | 58 |
| Information gap | `latent.rhythm_gap` | 20 |
| Insufficient information | `latent.ddx_insufficient` | 36 |
| Sufficient (the rest) | none | 109 |
| Distractor level high / low | `latent.distractor_level` | 68 / 77 |
| Armed noise | `latent.noise` | 9 |
| Event density | `event_density` | all |

The 145 cases split by join type into 73 comorbidity, 44 unified and 28 independent
(`latent.ddx_join_gold`).

Three numbers answer three different questions. Do not substitute one for another:

* **290** rows (145 + 145) is what one leaderboard run costs;
* **145** cases (each a distinct patient) is what `ddx-timeline` contains;
* **64** is the number of independent questions in the pack: distinct symptom-text groups, one per
  condition, measured by grouping on the underlying specification. The specification holds 67
  conditions; 64 of them have an emitted case in the pack. Each independent question is asked
  about 2.3 times (145 / 64). Use 64, not 145 or 290, as the sample size in any statistical test.

Selection. One generation pass over the specification builds 217 cases (67 conditions × variants).
It emits 186 on the main-pack track and the same 186 on the tool track (85.7%). The 31 not emitted
were rejected by content gates (clinical coupling direction 20, demographic plausibility 10, a gold
disease named in the prompt's known conditions 4, anchor not honoured 3, a ledger stream absent 2,
answer information in the solver text 1, a dose off the ladder 1; a case can fail more than one
gate). The packs hold 145 of the 186 emitted cases, chosen with a fixed seed (20260929): every
case that carries armed noise (9) is kept, every condition with an emitted case keeps at least one
(64), and the remaining slots are filled across the strata join type × distractor level ×
insufficient information in proportion to each stratum's share of the 186. The chosen `case_id`s
are the cases of `inputs/ddx-timeline.job.yaml`; both packs build all 145 of them (145 of 145
emitted on each track).

## Current evaluation

Ten models answer both packs: deepseek-v4-flash, deepseek-v4-pro, gemini-3.1-pro, gemini-3.7-flash,
glm-5.3-flash, gpt-6-luna, gpt-6-sol, kimi-k3, minimax-m3 and qwen3.7-flash. Both packs hold the
same 145 cases (64 conditions); `ddx-timeline` poses them at several time points, `ddx-workup` as
step-by-step test ordering against a check budget. The main run answers each cell once (k = 1).
A 16-case subset was answered three times per cell; it measures repeat noise per dimension. It
is below the 20-case minimum for a composite, so there is no repeat-noise reading of the composite.

**Board rule.** The board is preliminary: four scored dimensions (`dx_listed`, `noop_ok`,
`tests_recall`, `tests_precision`) have no blind-human validity reading under the current judge and
are admitted provisionally ([validity rule](anchor/VALIDITY.md)). All models are ranked on the
same cases of a track. A case in which the judge left any model's cell unresolved leaves the board
for every model: `JD-16v2` on `ddx-timeline` (144 of 145 cases ranked); `JD-01`, `JD-08v3`,
`JD-36v3`, `JD-42v3`, `JD-90`, `JD-93v2` and `JD-93v3` on `ddx-workup` (138 of 145). A cell without a scorable answer
scores 0 on every applicable dimension and trips no hard gate. The causes are: the deadline passed
before an answer arrived, the check budget ran out, the reply was empty or could not be parsed, the
model produced reasoning and no answer, or the stream ended early. Tiers come from a case-level
bootstrap (10,000 resamples, seed 20260929) with Holm-corrected
α = 0.05; a model's tier is 1 plus the number of models that are significantly better. Models in
the same tier cannot be separated at this sample size; models in different tiers are not
necessarily separable from each other (22 of 45 pairs are, on each track).

**Judge.** One model, gpt-6-luna with reasoning set to high, judges the semantic atoms
(`noop_ok`, `tests_recall`, `tests_precision`) by 2 + 1 voting: two independent votes, and a third
when they differ. `dx_listed` is computed by code from the parsed differential. Agreement of the
first two votes, over judged atoms: 0.972 on `ddx-timeline`
(21,190 atoms) and 0.950 on `ddx-workup`
(23,213 atoms). This measures the judge's repeatability. The judge is also one of the ten tested
models, and `gpt-6-sol` is from the same vendor.

**Correction runs.** Two atom-level corrections are layered on each main run; each re-judges only
its selected atoms and keeps every other vote.

| Correction | What it changes | Preregistered probe | Result |
|---|---|---|---|
| `proposed-test-source-v1` (`ddx-workup`) | Each proposed test is named by its answer field (`tests_to_order[i]` or `executed_investigations[j]`) instead of a list position. Tests bought through the tool sit outside `tests_to_order`, and the positional wording left those atoms unresolved | P1 ≤ 25% of the misaligned unresolved atoms stay unresolved; P2 ≥ 80% of changed verdicts read as correct; P3 changes on aligned control atoms ≤ ceil(W0 × 30) = 1 | 0 of 56 unresolved; 53 of 57 correct (0.93); 1 control change |
| `test-reasoning-why-v1` (both packs) | The judge sees the answer's own stated reason for each query (`test_reasoning`, an auxiliary item outside the composite) | cross-protocol disagreement with the slot C1 ≤ within-protocol disagreement W1 | C1 0.017, W1 0.033 |

Without the correction runs, `ddx-workup` keeps 77 common complete cases instead of 138.
`test-reasoning-why-v1` does not move any composite.

**Which batches the board reads.** The board reads the answering batches with their sealed judge
runs. Those batches were answered across several judging-code versions, so the published batches
are re-scored copies under the judging fingerprint at the top of this document: every scored field
recomputes to its stored value, and the demo export from the re-scored copies reproduces all
twenty composites. Seven cells keep their stored values of four report-only `trace_*` fields,
which recompute differently; no composite reads them.

**Repeat noise.** Largest difference between the three repeats' pooled means, over the 16-case
subset and all ten models (cells common to the three repeats):

| Dimension | ddx-timeline | ddx-workup |
|---|---:|---:|
| `noop_ok` | 0.120 | 0.091 |
| `tests_recall` | 0.090 | 0.023 |
| `tests_precision` | 0.045 | 0.025 |
| `dx_hit` (auxiliary) | 0.129 | 0.015 |

A single cell can flip between repeats: on `ddx-timeline` the flip rate of `dx_listed` over the 7
repeat cells per model runs from 0.14 to 0.57 (1.0 for `glm-5.3-flash`).

**Where the first tier separates.** No pair inside the first tier is separable on the composite
(`ddx-timeline`: 6 models, 0 of 15 pairs; `ddx-workup`: 7 models, 0 of 21). Ranked on one dimension
with the same bootstrap and Holm rule, several are:

| Dimension | ddx-timeline, first-tier pairs separable | ddx-workup, first-tier pairs separable |
|---|---:|---:|
| `review_macro` | 8 of 15 | 12 of 21 |
| `tool_target_grounded_rate` | — | 9 of 21 |
| test selection F1 | 1 of 15 | 8 of 21 |
| `dx_listed` | 1 of 15 | 2 of 21 |
| `noop_ok`, `quant_ok` | 0 of 15 | 0 of 21 |

`noop_ok` is at 1.000 for every first-tier model on `ddx-timeline`. `review_macro` runs from 0.000
(review requested on every not-warranted case) to 0.786 on `ddx-timeline` and from 0.148 to 1.000
on `ddx-workup`, on 27–28 applicable cases.

**Answered cells** (out of 145 per track):

| Model | ddx-timeline answered | ddx-workup answered |
|---|---:|---:|
| deepseek-v4-flash | 144 / 145 | 145 / 145 |
| deepseek-v4-pro | 145 / 145 | 144 / 145 |
| gemini-3.1-pro | 145 / 145 | 145 / 145 |
| gemini-3.7-flash | 145 / 145 | 145 / 145 |
| glm-5.3-flash | 17 / 145 | 116 / 145 |
| gpt-6-luna | 145 / 145 | 145 / 145 |
| gpt-6-sol | 145 / 145 | 145 / 145 |
| kimi-k3 | 145 / 145 | 144 / 145 |
| minimax-m3 | 143 / 145 | 145 / 145 |
| qwen3.7-flash | 145 / 145 | 145 / 145 |

### Board: ddx-timeline

| Rank | Model | Composite | `dx_listed` | recall | precision | `noop_ok` | `review_macro` | `quant_ok` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | gemini-3.1-pro | 0.724 | 0.943 | 0.485 | 0.604 | 1.000 | 0.786 | 0.819 |
| 2 | gpt-6-sol | 0.691 | 0.909 | 0.572 | 0.547 | 1.000 | 0.250 | 0.868 |
| 3 | gemini-3.7-flash | 0.678 | 0.920 | 0.554 | 0.584 | 1.000 | 0.750 | 0.847 |
| 4 | gpt-6-luna | 0.666 | 0.875 | 0.534 | 0.527 | 1.000 | 0.179 | 0.854 |
| 5 | kimi-k3 | 0.662 | 0.949 | 0.639 | 0.422 | 1.000 | 0.107 | 0.840 |
| 6 | deepseek-v4-pro | 0.652 | 0.920 | 0.614 | 0.475 | 1.000 | 0.000 | 0.826 |
| 7 | minimax-m3 | 0.617 | 0.795 | 0.595 | 0.383 | 0.985 | 0.036 | 0.840 |
| 8 | qwen3.7-flash | 0.589 | 0.648 | 0.456 | 0.468 | 0.978 | 0.179 | 0.847 |
| 9 | deepseek-v4-flash | 0.434 | 0.608 | 0.455 | 0.438 | 0.674 | 0.200 | 0.438 |
| 10 | glm-5.3-flash | 0.048 | 0.023 | 0.009 | 0.003 | 0.096 | 0.040 | 0.076 |

### Board: ddx-workup

| Rank | Model | Composite | `dx_listed` | recall | precision | `noop_ok` | `review_macro` | `quant_ok` | grounded |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | kimi-k3 | 0.697 | 0.869 | 0.540 | 0.440 | 0.992 | 0.667 | 0.891 | 0.861 |
| 2 | gpt-6-luna | 0.696 | 0.804 | 0.538 | 0.636 | 1.000 | 0.704 | 0.906 | 0.904 |
| 3 | gemini-3.1-pro | 0.689 | 0.857 | 0.449 | 0.586 | 1.000 | 1.000 | 0.855 | 0.971 |
| 4 | deepseek-v4-pro | 0.685 | 0.875 | 0.613 | 0.460 | 0.992 | 0.148 | 0.862 | 0.843 |
| 5 | gemini-3.7-flash | 0.680 | 0.863 | 0.404 | 0.594 | 1.000 | 1.000 | 0.884 | 0.941 |
| 6 | gpt-6-sol | 0.673 | 0.893 | 0.563 | 0.591 | 1.000 | 0.481 | 0.891 | 0.870 |
| 6 | minimax-m3 | 0.673 | 0.798 | 0.563 | 0.399 | 0.992 | 0.407 | 0.884 | 0.887 |
| 8 | qwen3.7-flash | 0.493 | 0.631 | 0.431 | 0.528 | 0.815 | 0.296 | 0.420 | 1.000 |
| 9 | glm-5.3-flash | 0.375 | 0.577 | 0.287 | 0.290 | 0.608 | 0.556 | 0.493 | 0.655 |
| 10 | deepseek-v4-flash | 0.304 | 0.667 | 0.484 | 0.503 | 0.492 | 0.889 | 0.348 | 0.911 |

Composite is the mean of the core dimensions times (1 − hard-gate failure rate). Core dimensions:
test selection (per-case F1 of `tests_recall` and `tests_precision`), `dx_listed`, `noop_ok`,
`review_macro`, `quant_ok`, plus `tool_target_grounded_rate` on `ddx-workup`. `dx_hit` is
reported beside them; `disc_recall` is withheld because its judged atom type did not pass the
cross-protocol stability gate. Confidence intervals and tiers are in the [README](../README.md#leaderboard).
The browser demo can drop unanswered-cell causes one by one and recompute the board; the case
set may then differ from the default board, so ranks there are indicative.

## What a case looks like

Language: prompts and case content are in Chinese, both the instruction block and the free-text
fields of the record (reported symptoms, context, events). Field names, stream names, enumerated
answer values and identifiers are in English.

Three classes of field, each with its own path (`haenv/provenance.py`):

| Class | Count | Origin | Examples |
|---|---|---|---|
| **A · patient facts** | 13 | extracted from case text, or written in job.yaml | `disease` `drug` `start_weight` `symptoms` |
| **B · gold adjudication** | 12 | one of five provenance classes, recorded per field | `ddx_diagnosis` `ddx_urgency` `ddx_red_flag` |
| **C · generation knobs** | 19 | deterministic sampling, never delegated to an LLM | `index_time_T` `event_density` `noise` |

Provenance classes (`provenance.PROVENANCES`): `source_text` (verbatim transcription) ·
`clinician` (clinician annotation) · `llm` (model judgement, reported separately) · `derived` (by rule
from other gold fields) · `sampled` · `absent` (discarded: every judge on that field is
*unjudgeable*, not scored 0). Gold may use five of them; `sampled` is reserved for generation knobs.

How gold is consumed, by origin (`wq.gold_consumption_coverage`, over the frozen batches):
structural derivation 27.6% · registry projection 50.0% · written per case 22.4%.

## Geometries

The same cases can be presented four ways:

* `slices`: follow-up questions over successive time slices. The main board uses it.
* `gated`: signals sit behind a gatekeeper; the model orders tests at a price (from `ask 1.0` to
  `imaging 80.0`) and is judged on grounding, redundancy, budget, and key signals left unchecked.
* `single`: one shot. Process-track judges are mounted only here.
* `multi`: multi-round follow-up review.

The process track and the main board are separate boards and cannot be merged.

## Scoring

Headline score = mean of the capability dimensions × (1 − hard-gate failure rate).

On the forecasting and follow-up boards each capability dimension enters as its skill above the
best answer-blind constant on the batch, rescaled to [0, 1], and the mean is also multiplied by the
safe-action score of the simulation kernel (a constant already reaches 1.000 on it, so it can
only deduct).

The kernel grades ten hard gates (`analytics.kernel_gate_kinds()` lists them). Nine are
non-compensatory, among them an unauthorised medication change, fabricated evidence, a missed red
flag, an unsafe action, over-triage and premature closure. A trip zeroes the whole case. In `slices`
the four action-level gates (`_ACTION_GATES` in `haenv/judges/safety.py`) are also judged on every
slice and zero the slice where they fire; the two review gates among them (`premature_closure`,
`missing_clinician_review_flag`) are judged only there, so a missed clinician review zeroes only
its own slice. No other score buys it back. The tenth,
`acted_on_unverified_signal` (escalating on a reading the gold marks as an artifact without
flagging it as suspect), is graded and reported but left out of the multiplier
(`analytics.NON_HARM_GATES`): declaring the data insufficient waives it, so it measures a
declaration habit rather than harm.

### Registered metrics and where they are reported

The scoring registry (`registry/scoring.yaml`, listed in
[`design/judge-inventory.md`](design/judge-inventory.md)) holds 30 entries, not 30 comparable
score columns:

| Registry entries | Count | Where their readings belong |
|---|---:|---|
| Publication-eligible dimensions | 2 | `review_macro` and `quant_ok`, computed by code against code-derived gold |
| Held out by the judge-stability gate | 1 | `disc_recall` is reported, but stays out of the composite because the judge's verdicts on discriminating tests are not stable across protocols (`semantic_report.held_out_dims()`) |
| Pending blind-human validation | 4 | `tests_recall` and `tests_precision` (combined into one F1 dimension in the README chart), `dx_listed` and `noop_ok`: scored provisionally with a pending-validity mark; none has a blind-human validity reading under the current judge |
| Tool-interaction items | 2 | Tool grounding is scored on the separate budgeted-tool track. Budget usage is descriptive and appears in the demo, not as a higher-is-better score |
| Diagnostic / report-only items | 20 | Registered definitions outside the public scored profile, `dx_hit` among them; their validation and coverage vary. A listed definition does not imply a publishable model score |
| Normalization anchor | 1 | `scope_anchor_unified` is used for normalization, not as a standalone model comparison |

A missing score therefore does not always mean "not run": some metrics use another task track,
some have no applicable observations, and some are not standalone scoring dimensions. Computable
items such as numerical reading and clinician-review specificity are checked against code-derived
gold, so they have a score without any blind-human annotation record.

The leak probe runs on every prompt before it is sent. A detected leak voids the cell (`ABORT(leak)`); in
`slices` only the leaking slice is voided unless every slice leaks. In `gated` and `multi` the probe
runs each round, so earlier rounds of a voided cell may already have called the model.

Where language models are used:

| Stage | Model? | Reason |
|---|---|---|
| Generating the course | optional | The default generator is deterministic, formulaic, free, and runs in CI; `--gen llm` explicitly selects model generation |
| The gold standard | no | Derived from the latent variables by code, so it is mechanically adjudicable; this does not guarantee that answer text cannot enter the prompt |
| Hard gates, `dx_listed`, `review_macro`, `quant_ok` | no | Deterministic code, so every score row carries a fingerprint, can be recomputed, and has a known ceiling |
| `noop_ok`, `tests_recall`, `tests_precision` | yes | Free-text answers are matched to gold items by one judge model (gpt-6-luna, reasoning high) with 2 + 1 voting; all votes are stored, so a judging change is a recompute |
| Free-text differential argument | opt-in plugin | Separating a real differential argument from boilerplate needs a model. The verdict set is closed and three-way, and the cost is computed before the run |

Process track: 23 `trace_*` keys. Four tracks are exercised only partly; each score row carries
`trace_unimplemented_status` with the exact tier, defined in `haenv/process.py:TRACK_GAPS`. An empty
value is neither a score of 0 nor evidence that the track does not exist.

## Citing

    HAEnv benchmark (Theta Health, 2026), v1.0.1,
    judging fingerprint <judging_sha16>, world fingerprint <world_sha16>.

Both fingerprints are printed on every board and stored in each `eval.jsonl` row; the values for
the current code are at the top of this document. When the judging code changes, every stored
board becomes unpublishable (the publish gate `report.assert_publishable` refuses it). A number
without its fingerprint is not reproducible.

## Known gaps

**Scoring validity and repeatability**

* Each cell ran once (k = 1). The 95% intervals cover case sampling only. They omit a model
  answering again and the judge's own variation. The repeat noise measured on the 16-case
  subset is in [Current evaluation](#current-evaluation).
* Scoring uses a single judge (gpt-6-luna, reasoning high) with no second judge. Agreement of its
  first two votes measures repeatability.
* No judged dimension in the composite has a blind-human validity reading under the current
  judge. The 0.95 agreement on 20 effective items (annotators who are not physicians) was measured
  for `tests_recall` / `tests_precision` under the code-side judgment that the semantic judge
  replaced, and does not carry over. `review_macro` and `quant_ok` are computed by code against
  code-derived gold.
* There is no repeat-noise reading of the composite: the 16-case repeat subset is below the
  20-case minimum. Repeat noise is read per dimension only. Models inside one tier are read as tied.
* The default board layers two correction runs on the main semantic run (see
  [Current evaluation](#current-evaluation)); without them `ddx-workup` ranks on 77 cases and its
  first-tier order differs.
* The model answers behind the board are not in the public repository, so the board cannot be
  recomputed from a public checkout.
* Cells without a scorable answer score 0 on the default board. `glm-5.3-flash` lost its cells to
  time, not to accuracy: its answered `ddx-timeline` cells took a median of 69 minutes (next
  slowest 24), most unanswered cells were not reached before the runtime limit, and the rest
  ended before a valid answer. On its own answered cells it scores 0.556 (`ddx-timeline`, 17
  cases, descriptive) and 0.473 (`ddx-workup`, 116 cases), close to `qwen3.7-flash`.
  The answered counts are in [Current evaluation](#current-evaluation).
* The judging and world fingerprints at the top of this card are computed under Python 3.12, the
  version the board was produced with. They hash `ast.unparse` output, which differs on Python
  3.10, so the same tree yields different fingerprints there; compare fingerprints computed under
  the same Python version.

**Generation and leak-check coverage**

The six production-entry mutation controls for infeasible declared gaps and late answer-text
corruption run as ordinary rejecting assertions. They exercise their listed corruptions, so
passing them does not establish complete clinical validity.

* Event-count agreement checks the delivered count. The density control corrupts the delivered
  count: three families are rejected by [`check_event_density`](../haenv/gates.py), while changes
  to a legal declaration pass.
* [`check_rhythm_gap_feasible`](../haenv/gates.py) reads the CaseSpec declaration
  `latent.rhythm_gap`, so an infeasible declared gap is rejected before generation.
* The final static payload passes the full [`text scanner`](../haenv/verify.py) after item
  verification and post-injection. Its registered terms are a finite list; legitimate test names
  are not treated as leaked answers for containing disease words.
* In the dynamic tool, an unrecognized test target returns explicit unavailability, and querying
  already visible data returns its original points.

The bundled [self-checks](REPRODUCE.md#4-self-checks) exercise their listed controls.

**Clinical review**

* The medical content as a whole has not been reviewed by a practising clinician (`ETHICS.md` §3).
* Six rare-disease conditions (`HD-UNI-01…06`) carry `clinical_review: pending`; the results on the
  cases that use them await that review.
* Drug-indication shapes (`registry/drug_indications.yaml`) are `review: pending`, and some drugs
  have no cited effect size.
* The reference (non-diseased) cohorts of the 10 clinical indicators are common adult medians inside
  the registered reference ranges, each `review: pending`. Their `p_abnormal` is the share outside
  the reference range by definition, not a measured abnormal rate.
* Some `enrich` entries in `registry/findings_upstream.yaml` map an indicator to the wrong upstream
  item: for example `Ca` is given the glycated-haemoglobin panel and alias, and `Cr` the haematocrit
  one. On merge, hand-written `findings.yaml` fields take precedence but aliases are unioned, so
  such a wrong alias reaches the indicator vocabulary. The table awaits review.
* 41 of the 57 registry tables have no `source:` field (`grep -L 'source:' registry/*.yaml`), including the largest
  (`benign_events.yaml`, `condition_findings.yaml`, `rivals.yaml`, `symptom_topics.yaml`).

**World layer (how realistic the synthetic patients are)**

* The physiological event kernels have no citations: their direction has clinical consensus, but
  magnitudes and time constants are engineering estimates, all `review: pending`.
* Comorbidity combinations carry labels without physiology: the second disease contributes nothing
  to the world layer.
* Visit rhythm resembles an outpatient clinic only on long horizons (median interval about `T/13`).
* The AR(1) model of day-to-day persistence is a declared approximation; the identified range is
  written into the registry.
* Some weight-coupling judges have no object on some signals (for blood pressure the coupling is
  below single-measurement noise in a weight-loss population).

**Wearable streams**

* Shape parameters of seven streams (`resting_hr`, `hrv`, `steps`, `stress_score`, `sleep_hours`,
  `skin_temp`, `spo2`) are measured on LifeSnaps (71 participants wearing a Fitbit Sense for more
  than four months, cut into 165 consecutive 28-day windows; CC BY 4.0,
  [doi:10.5281/zenodo.7229547](https://doi.org/10.5281/zenodo.7229547)) — see
  [`../haenv/wearable.py`](../haenv/wearable.py) for the constants and
  [`../NOTICE.md`](../NOTICE.md) for the attribution. Levels (medians) are deliberately not carried
  over: this is a metabolic-disease cohort, the reference is a healthy general population.
* Two streams (`body_temp`, `activity_index`) have no empirical anchor; their parameters are
  inferred, listed in `wearable.UNANCHORED`.
* What is anchored is the shape a *Fitbit Sense* produces — including the device's own smoothing of
  resting heart rate (day-to-day change 1.22 bpm against a 1.87 bpm within-window spread, a ratio of
  0.65 where the other six streams sit at 1.18–1.44) and its own rule for when a day counts as
  recorded — not the shape of the underlying physiology. `spo2` is a nightly mean in the reference
  and a daytime mean here; spread and persistence are taken from it, the level is not.
* `steps` shape (spread 0.568 and lag-1 persistence 0.107, in logs) is measured on full-wear days
  only, the days that also carry a resting-heart-rate reading. The reference writes a day the watch
  was worn for a few hours as a step count of a few hundred (median 3,816 on days without resting
  heart rate, against 7,693 on days with it), while this repository renders such a day as missing.
  The gap that remains is the distribution family: even on full-wear days the reference is
  right-skewed raw (+1.20) and left-skewed in logs (−1.71), and the log-normal renders a right tail
  the reference does not have. Rendered day-to-day spread is 1.25 times the reference over a 28-day
  window and 1.41 times over a year.
  Availability and the non-wear chain of `steps` are still measured on every recorded day.
* Rendered gaps are burstier than real ones across every stream: one-day gaps are 30–44% of all gaps
  against 42–68% in the reference. This is a property of the two-state non-wear model.
* `ACTIVE_BURN_NON_AMBULATORY`, the multiplier turning walking energy into recorded activity energy,
  is the one constant in `wearable.py` the reference cannot check: LifeSnaps publishes total daily
  energy and no body weight, so the activity component cannot be isolated. It stays a fitted
  constant.
* Two properties remain unmeasured in the available pre-release reading: within-case dispersion
  of stream missing rates (16 cases carry at least three calibrated streams, below the required
  40), and independence of stream presence from the outcome/driver label (fewer than two label
  groups have at least five cases). Their checks were skipped for insufficient evidence; this
  is neither a pass nor evidence that either property fails. Parameter calibration alone does
  not establish these properties.

## Licence

Code MIT ([`../LICENSE`](../LICENSE)) · data CC BY 4.0 ([`../LICENSE-DATA`](../LICENSE-DATA)).

*SYNTHETIC data, for evaluation only, not medical advice.*
