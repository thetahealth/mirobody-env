---
name: haenv-synth
description: >-
  Generate patients and generate questions -- from a set of settings (`job.yaml`) or a piece of case-description text, produce a whole longitudinal trajectory that satisfies those constraints plus a machine-adjudicable gold standard, verifying item by item that "does this question even hold together" before it's handed to a model. Covers two entry points (hand-written job / text extraction), the three kinds of thing that must never be mixed (patient facts / gold standard / generation knobs), whether generating questions costs money (the deterministic tier is free, the LLM tier is cached), and how to read a question blocked by the emission gate. Triggers when the user says things like "generate a batch of questions", "synthesize data", "generate from this case", "question generation got blocked", "generate a patient".
---

# haenv-synth — generate patients, generate questions: longitudinal data and gold standards from settings or case text

**One line**: given a set of constraints (or a real case description), produce a whole longitudinal
trajectory satisfying those constraints plus a machine-adjudicable gold standard, verifying item by item that
"does this question even hold together" before it's handed to a model.

Companion skills: `haenv-bench` (running models against a question pack to produce a board) · `haenv-extend`
(adding your own judge).

Warning: this file is an index, not the sole source of truth. How artifacts are organized, and what is free to
reproduce, is in `docs/REPRODUCE.md`; the packs and their gaps are in `docs/DATA_CARD.md`. Every command here has actually
been run, with the reading noted next to it.

---

## 0. Pick an entry point first: there are two, not one

| what you have | which one to use |
|---|---|
| a set of settings (age range/medication/start-end weight/event density...) | A · hand-write a `job.yaml` |
| a piece of case-description text (`.md` / `.txt` / a directory) | B · text extraction |

After either entry point, the path is identical: the same emission gates, the same scoring, the same batch
archiving.

---

## A · Hand-writing a job.yaml

```bash
uv run haenv build  inputs/example-ew.job.yaml --gen deterministic   # generate questions (formulaic, free)
uv run haenv verify inputs/example-ew.job.yaml --gen deterministic   # a per-item verification report
```

The reading is the last line of the report `verify` writes:
`[haenv] per-item verification: <N> item(s) · failed <F> · text leaked in <L> case(s)`.

### A case's three parts have completely different natures

`job.yaml`'s `cases[i]` has only three fields, `case_id` / `raw` / `latent`, but `latent` holds two kinds of
thing that must never be mixed (`job.LATENT_REGISTRY` lists the keys):

| kind | what it is | example | Important: discipline |
|---|---|---|---|
| A. patient facts | `raw.*` -- who this person is | `age_range` · `sex` · `drug` · `dose_steps` · `devices` · `start_weight` · `nadir_weight` · `symptoms[day,text,context]` | it's in the text => extract it |
| B. gold-standard adjudication | `outcome`, `driver` and the `ddx_*` keys of `latent` | `ddx_diagnosis` · `ddx_join_gold` · `ddx_urgency` · `ddx_red_flag` · `ddx_tests` · `outcome` · `driver` | transcribe only, never infer (see section B below) |
| C. generation knobs | every other key of `latent` | `index_time_T` · `event_density` · `noise` · `distractor_level` · `course_end_day` · `reversal_week` · `missingness` | never in the case text at all => sampled, not extracted |

> Important: why this is the single most important rule: having an LLM "extract" `symptom_rate: 0.38` is
> the same as having it make up a generation control parameter; having an LLM infer `ddx_join_gold` is the
> same as having the LLM produce the gold standard -- that breaks ground-truth isolation: the solver
> only ever sees the `<=T` payload, and ground truth, premises and future data are never in it (see
> `docs/design/llm-io-and-scoring.md`), and it's circular reasoning
> (a question produced by the model, used to test the model). How all three are handled is documented at the
> top of `haenv/extract.py`.

---

## B · Generating questions from case text

```bash
uv run python tools/text_to_job.py \
  --in raw_cases/joint_dx/rc-jd-*.md \
  --job-id my-job --task-type joint_dx \
  --gold text --input-kind retrospective \
  --out my-job.job.yaml
```

What you get is an ordinary job file: build it like any other (section A), and the extracted gold
rides along in its `_provenance` block.

### Four switches, each with a "what happens if you get it backwards"

| switch | values | what happens if you get it backwards |
|---|---|---|
| `--gold` (required) | `absent` discard / `text` transcribe / `llm` adjudicate / `clinician` physician-annotated | not having a default is deliberate -- where the gold standard comes from decides how trustworthy this batch is, and that should never be decided for you by a default. The `llm` tier flags it field by field and lists it separately in the batch summary |
| `--input-kind` | `prospective` (default, strict tier) / `retrospective` | mis-declaring a prospective case as retrospective makes the extractor force-find an outcome in text that has none, and stamp `source_text` on it -- that's fabricated provenance |
| `--cut-at <heading>` | truncates the input at that heading | our own raw case markdown has verifier-only sections; feeding the whole thing in means showing the extractor the answer |
| `--extract-offline` | only allows replaying a fixture or an existing cache | this is the path for zero-cost reproduction; if neither is available it fails, it never goes online or falls back to an empty extraction (an empty `raw` would get padded by downstream defaults into a patient that doesn't come from this text) |

The output carries a `_provenance` block, propagated all the way to the report -- "how many questions in this
batch had a model-adjudicated gold standard" is a number in the report, not something you have to go
archaeology to find.

---

## B' · The reverse direction: trajectory -> a case-report text

Entry point B is "text -> job.yaml"; this one is its reverse, and together the two form a loop:

```
real case chart --extract--> job.yaml --build--> trajectory+gold --render--> synthetic case report
                    ^                                                          |
                    +----------------- extract again <------------------------+
                       compare the two job.yaml files: how much was recovered
```

```python
from haenv.record_render import render_record, leak_words
text, origin = render_record(sp)        # sp = the first return value of build_instance
assert not leak_words(text)             # the prose modality's own leak gate
```

Important: `render_record`'s signature only takes `sp` -- `vp`, `latent`, and `case` cannot be passed in at
all. That's not "forgot to pass them," it's structurally impossible. Without this constraint, a round-trip
test would just be measuring the LLM's transcription noise: recovery would sit near 100% regardless of the
synthetic patient's quality.

**Measured (n=12, deduplicated, across three question packs, `gemini-3.8-flash`)**:

| arm | recovery rate |
|---|---|
| a. oracle (bypasses prose) | 0.833 |
| b. prose | 0.786 |
| c. negative control (someone else's trajectory) | 0.179 |

`b - c = 0.607` => the recovery rate genuinely measures this patient, not a population prior.
Leak-word hits: 0/12.
`b ~ a` => the round trip through prose loses almost nothing; the remaining gap is expression loss
(the biggest one measured: the solver payload has no drug name at all, only the dose ladder).

The per-case rows behind these summary values are written by the probe itself; it is
`docs/scripts/record_roundtrip_probe.py` (`--offline` only consumes the cache, zero cost).

Warning: the denominator is fixed first: `record_render.ROUNDTRIP_FIELDS` contains only class-A patient
facts. Class-C generation knobs (`index_time_T` / `event_density` / `noise`) don't describe the patient and
have no business being in the case report -- put them in the denominator, and this number becomes something
you can tune at will.

---

## 1. Does it cost money

| path | cost |
|---|---|
| `--gen deterministic` (default, `config.yaml:synth.generator`) / `--offline` | 0 (formulaic generation) |
| `--gen llm` | a GV-1 round plus an event-planning round per case; the model is `config.yaml:synth.gen_model` |
| re-running the same batch | 0, as long as the prompt hasn't changed -- `cases/_llm_cache/` hits on `sha256(model+argv+prompt)`; `--regen` forces a fresh run |
| text extraction | one call/case (`--extract-model`); 0 with `--extract-offline` |

The generation model's calls and cache hits for a batch are in its `batch.json`
(`gen_usage`): read them there rather than trusting a number quoted in prose.

## 2. Important: the free path can be blocked by the batch-level gate

The batch-level gate decides on the generated data, not on which generation tier produced it. It fires
when the injected noise is separable from the real symptoms by a trivial rule, and says so in the log:

```
[haenv] blocked by the batch-level emission gate: ['footprint_discriminates_real_symptom']
[haenv]    the batch is persisted and marked blocked_by_batch_gate -- `run`/`report` will refuse it
           (add --override-batch-gate explicitly to force it)
```

This is not a bug, it's the gate doing its job: that batch of questions could be answered correctly without
any reasoning at all. A formulaic (`--gen deterministic`) pack can trip it, because its noise is drawn
without the case-aware checks the LLM tier applies; some jobs pass it at either tier, so treat it as a
property of the pack's data rather than of the tier. Reproduce a blocked pack with `--gen llm`, or change
`latent` until the noise stops being separable.

## 3. What to do when question generation gets blocked

The emission gate has two levels, read differently:

* **per-case gate** -- that one case isn't emitted, the rest proceed as normal. The log shows
  `[build] <case> blocked by the emission gate: conflicts=[...] leak=[...]` plus a reason kind;
  common ones: premise validation failed · GV-1 didn't converge · a conflict after noise injection · a leak;
* **batch-level gate** -- the whole batch is marked `blocked_by_batch_gate`, and `run`/`report` refuse it
  (forcing it through requires an explicit `--override-batch-gate`, which is written into `batch.json` as a
  record).

Important: don't tune parameters to route around the gate. What the gate blocks is "this question can be
answered correctly without reasoning" or "the answer has already leaked into the prompt" -- routing around it
just produces a batch of scores with no visible problem. The correct fix is to change `latent` so the question
actually holds up.


## 3b. What counts as "irrelevant noise" to inject

`event_density` + `raw.devices` decide what gets injected, but whether it passes depends on two groups of
judges -- consistent with the case's baseline facts, and does not change the answer's conclusion. The five
easiest to get wrong:

* **must be backed by a device**: step count needs `wearable` · `scale_qc_flag` needs a scale ·
  `diet_carb_pct` needs `cgm`;
* **the baseline must match this person**: adjusted for BMI / age / comorbidities; if the case text states a
  measured vital sign, that value takes precedence (a patient documented with "resting heart rate 102" cannot
  also have a stream sitting at 64 bpm);
* **must not be a proxy for the true driver or a comorbidity**: step count when `driver=activity_decline`,
  oxygen saturation/sleep for an OSA patient -- none of these are ever injected;
* **must not explain the predicted outcome itself**: under a weight outcome, a one-off diet/exercise event
  (a big meal, starting to work out) falls in this category, even if it isn't this case's true driver -- it
  hands the solver a ready-made attribution.
  Warning: this rule only bans discrete events; a flat, non-drifting step-count stream can't explain a
  weight change, so it may stay;
* **must not use a red-flag symptom as noise**: injecting chest pain or dyspnea when the ground truth has
  `red_flag_present=False` fabricates an un-adjudicable missed-diagnosis trap.

Rejection reasons are traceable item by item in the `verify` report's "What the generate-verify loop
dropped" table, and machine-readably in the batch's `audit.jsonl` (one line per injected item). The
reason kinds include `device_backed` · `baseline_matches_raw_facts` · `not_driver_or_comorbid_proxy` ·
`plausible_for_profile` · `not_red_flag_symptom` · `neutral_across_reversal` ·
`no_collision_with_real_symptom` · `no_duplicate_topic`.

Important: the generator and the verifier share the same spec -- so "a normal batch passes clean on the
first try" proves nothing about whether the verifier works; only a negative control can. Anyone changing
`events.py` / `verify.py` must run `tools/verify_selftest.py`.

## 4. Acceptance tests (must run after changing anything on the question-generation side)

| what changed | what to run | time |
|---|---|---|
| `events.py` / `verify.py` / any injection logic | `uv run python tools/verify_selftest.py` | 0.2s |
| the payload's shape | `uv run python tools/leak_probe_selftest.py` | 0.8s |

Warning: how to cite that self-test's item count: quote its **genuine-negative-control** count, not the
total. The total also counts positive controls, so deleting 5 negative controls and adding 5 positive
controls would leave it unmoved -- and it is "no negative control silently deleted" that the number
guards. `verify_selftest.py` prints the breakdown it reports.

Warning: don't pipe it -- `| tail` turns a crash into exit code 0; redirect to a file and read that instead.

## 5. Iron rules (breaking any of these voids the batch)

1. **Ground truth never enters the solver's input.** The order can't be reversed: `build_instance(<=T)` ->
   `leakage_probe` -> `solve` -> `grade`;
2. **The gold standard is never handed to the model.** Under `--gen llm` the numeric sequences come from the
   model, but `outcome_label` / `gold_drivers` / `adjudication` / `reversal_points` / `label_rule` are always
   given deterministically from the hidden control variables;
3. **Every injected observation must be consistent + non-leaking + gold-standard-preserving**; anything that
   fails is discarded and reinjected; if a case fails, it isn't emitted;
4. **Emission means it's saved.** A case that passes the gate has its content written into the batch's
   `cases.jsonl`, with a per-case sha256 recorded; `run`/`report` only read that copy, they never regenerate
   the questions.

## 6. When *not* to use this skill

* you already have a question pack and just want to run models against it => `haenv-bench`;
* you want to add your own judge => `haenv-extend`;
* you're just editing comments or docs => edit them directly.

*SYNTHETIC data, for evaluation only, not medical advice.*
