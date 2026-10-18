<h1 align="center">Health Agent Environment</h1>

<p align="center">
  <strong>Generation and Evaluation in One Pipeline</strong><br>
  Generate synthetic patients and clinical scenarios, then evaluate health agents against code-derived ground truth.
</p>

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code: MIT"></a>
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-green.svg" alt="Data: CC BY 4.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pypi.org/project/haenv/"><img src="https://img.shields.io/pypi/v/haenv.svg" alt="PyPI"></a>
  <a href="https://github.com/thetahealth/mirobody-env/actions/workflows/ci.yml"><img src="https://github.com/thetahealth/mirobody-env/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://thetahealth.github.io/mirobody-env/"><img src="https://img.shields.io/badge/demo-live-black.svg" alt="Live demo"></a>
</p>

<p align="center">
  <strong>English</strong> &middot; <a href="https://github.com/thetahealth/mirobody-env/blob/main/README.zh-CN.md">简体中文</a>
  &nbsp;|&nbsp;
  <a href="https://thetahealth.github.io/mirobody-env/">Live demo</a> &middot;
  <a href="#the-four-packs">Packs</a> &middot;
  <a href="#synthetic-patients">Patients</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#how-it-works">How it works</a> &middot;
  <a href="#scoring">Scoring</a> &middot;
  <a href="#leaderboard">Results</a> &middot;
  <a href="#evaluate-your-own-agent">Your agent</a> &middot;
  <a href="#agent-skills">Skills</a> &middot;
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_hero.png" alt="HAEnv in five steps: define a hidden patient state, render years of records, run release checks, run the agent under test on the questions, and score its answer with code and an LLM judge against hidden gold derived from the same state" width="100%">
</p>
<p align="center"><sub>The agent sees records up to <code>T</code>; hidden gold goes only to the judges, which are code plus one LLM judge for management-plan quality <code>Q</code> (see <a href="#scoring">Scoring</a>).</sub></p>

## What it measures

HAEnv (Health Agent Environment) lives in the `mirobody-env` repository and installs as the `haenv`
Python package and command-line tool.

HAEnv grades clinical decisions on a patient's record over time. A synthetic patient's record
grows over months; the agent sees it up to an index time `T` and makes one decision. The world
that rendered the record fixes the right answer before the record is written, and the answer is
hidden from the agent.

HAEnv generates the patients, sets their difficulty and derives their gold standard. To run an
existing system against published health benchmarks such as ESL-Bench, use
[mirobody-eval](https://github.com/thetahealth/mirobody-eval).

### The four packs

The benchmark is four task packs. Each item asks for one core decision.

| Pack | Decision at `T` | Answer | Judged by |
|---|---|---|---|
| ① Diagnosis (budgeted differential) | which conditions explain the record, when a second condition hides behind findings the first one explains; whether a clinician should review | a differential, the tests to order, a review flag | code, against the world's truth: registered aliases match the listed diagnoses to the gold, with no model call |
| ② Triage | where the patient should go now | `ed_now` · `within_24h` · `routine_followup` · `watchful_waiting` | code, against the world's truth |
| ③ Medication adjustment | what to do with the drug at this follow-up visit | `uptitrate` · `downtitrate` · `maintain` · `switch` · `check_adherence_or_adverse_effect` | code, against the dose line, refills and the control target the world records |
| ④ Lab-change attribution | whether the change since the last result is real | `true_change` · `analytic_biological_noise` · `preanalytical` · `method_difference`, plus the size of the true change | code, against the world's true values and its per-reading perturbation records |

The two diagnosis tracks of the 1.x board, `ddx-timeline` and `ddx-workup`, stay in the
repository as bridge tracks: pack ① carries bridge items drawn from them, and they are the only
link between the 1.x board and the packs.

- **Gold standard first.** Disease, drug response, adherence, true lab values and measurement
  noise are set before the course is rendered, and each pack's gold is derived from them by code.
- **Coherent patients.** Weight, laboratory values, medication, wearable streams and life events
  are rendered by one set of rules on one timeline. A known comorbidity enters the world tables, so
  in packs ③ and ④ it changes the right answer where it should.
- **Records read like records.** Complaints are in the patient's own words, and the record lists
  the patient's known conditions with the medication for each.
- **Difficulty as a parameter.** Measurement artifacts, distractor events and the timing of the
  clinical turn are settings in the job file.
- **Safety failures are not averaged away.** Hard gates, among them an unauthorised medication
  change, fabricated evidence, a missed emergency and premature closure, zero the item whatever
  else scored well. Two propensity gates, over-triage and a missed clinician review, are graded and
  reported but do not enter the multiplier.

| Term | Meaning |
|---|---|
| index time `T` | the cut: the prompt holds the record up to `T`; grading uses what follows or what the world fixed |
| latent variables | the hidden state the world fixes before rendering: disease, drug response, adherence, true values, noise |
| pack | one task's question set, built from a job file by its generator and checked by the pack audit |
| seed | orders which cases and answer classes a pack draws; a pack records only the seed's sha256 |
| emission gate | the checks a generated case must pass to be released: premise check, per-item verification, leak probe |
| hard gate | a safety failure that zeroes its scoring unit |
| batch | one run directory: its cases, answers, scores and fingerprints |

### Packs are pipelines

A pack is not a fixed file. You choose its size and seed, and one command writes the job, builds
the pack with the deterministic generator and runs the generation-time audit that every pack
shares. It makes no model call:

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out packs/p4
```

`--pack` is `m2` (①), `p2` (②), `p3` (③) or `p4` (④) (`pack2` is the triage pack of the 1.3.0 board); `--n` is the item count (50, 100, …);
`--seed` (or `HAENV_PACK_SEED`) sets the seed. The pack is the same for the same pack, size and
seed. The job carries a header `pack: {n_items, seed_sha256}`, and `batch.json` records the size,
the seed's hash and the job, never the seed. Class shares follow each pack's proportion table at
every size.

| Pack | Example job (current defaults) | Items at N = 50 | Answer classes at N = 50 |
|---|---|---:|---|
| ① Diagnosis (budgeted differential) | `benchmarks/inputs/m2-pack1.job.yaml` | 50 | 34 new items (24 with a hidden condition, 10 without) and 16 bridge items |
| ② Triage | `benchmarks/inputs/pack2-triage.job.yaml` | 50 | `ed_now` 13 · `within_24h` 13 · `routine_followup` 12 · `watchful_waiting` 12 |
| ③ Medication adjustment | `benchmarks/inputs/p3-meds.job.yaml` | 50 | `maintain` 12 · `downtitrate` 12 · `check_adherence_or_adverse_effect` 12 · `uptitrate` 8 · `switch` 6 |
| ④ Lab-change attribution | `benchmarks/inputs/p4-followup.job.yaml` | 50 | `true_change` 13 · `analytic_biological_noise` 13 · `preanalytical` 12 · `method_difference` 12 |

- **Example jobs, not the evaluated sets.** The job files in this table are example jobs at the current
  defaults (public seed, N = 50). Building them gives packs of the same design, not the items of the
  1.4.0 board, which ship as frozen pack files (see [Leaderboard](#leaderboard)). Row ② is the 1.3.0
  triage job `pack2-triage`; the 1.4.0 triage task is built with `--pack p2`.
- **Seeds.** A board's batch records only the seed's hash. The 1.4.0 board uses the public seed
  (see [Leaderboard](#leaderboard)). Earlier boards used a private seed, so their items cannot be
  rebuilt from the repository; when a board is replaced, its seed is published and the retired
  board becomes reproducible.
- **The audit is a release gate.** `tools/validation/pack_audit.py` checks the built batch: every cell the job
  planned is present, gold re-derived from the records equals the stored gold, and question-blind
  stubs and surface features do not predict the answer. `haenv run` refuses a pack's items without
  a passing audit.

The bridge tracks and two smaller example tasks ship as plain job files:

| Task | Job file | Case specifications | Format |
|---|---|---|---|
| Differential diagnosis at several time points (bridge) | `benchmarks/inputs/ddx-timeline.job.yaml` | 145 | questions at several time points |
| Budgeted test ordering (bridge) | `benchmarks/inputs/ddx-workup.job.yaml` | 145 | the agent orders tests against a budget |
| Weight-regain forecast and driver attribution | `benchmarks/inputs/early_warning-20.job.yaml` | 20 | single question at `T` |
| Multi-round follow-up review | `benchmarks/inputs/tracking_review-20.job.yaml` | 20 | rounds; the agent may revise |

A specification becomes a case only if it passes the emission gate. The question packs, their case
counts and the known gaps are in the [data card](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md).

**One pipeline, several tasks.** A task is a gold-standard shape plus an answer contract. The same
world renders every pack's patients, the same emission gate checks them item by item, and each pack
registers its gold and its scoring as a judge group. Adding a task type of your own is a package
outside this repository, not a change to it: [Evaluate your own agent](#evaluate-your-own-agent) and
[`docs/design/external-task-contract.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md).

**Language.** Prompts and case content are in Chinese: the instruction block and the free-text
fields of the record (complaints, context, events). Field names, stream names, enumerated answer
values and identifiers are in English.

## Synthetic patients

The [live demo](https://thetahealth.github.io/mirobody-env/) renders a patient in the browser; its knobs
change the course, the labs and adherence.

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_cohort.png" alt="Three panels: individual drug response for 60 maintainers on tirzepatide and 60 low responders on semaglutide; change in HbA1c against change in fasting glucose over 90 days; weight trajectories of 12 patients, aligned on each index time T, that separate into regain and maintenance after T" width="100%">
</p>
<p align="center"><sub>(a) Two specifications, 60 case ids each. Every patient draws an individual
response to the drug inside the band its declared driver allows (shaded). (b) Over the first 90
days, maintainers' fasting glucose and HbA1c fall together; low responders barely move on either.
(c) The 12 of the 20 <code>early_warning-20</code> specifications that pass the emission gate, each
aligned on its own <code>T</code>: regain and maintenance mostly separate after <code>T</code>, so a
forecast at <code>T</code> must rest on earlier signals.</sub></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_difficulty.png" alt="The same patient at three settings: default; with two transient weight spikes recorded as traps; and with spikes plus the high distractor level, which brings seven distractor events before T instead of two" width="92%">
</p>
<p align="center"><sub>The same patient at three difficulty settings. The underlying series is
identical (grey dots mark the default where it differs), and outcome and driver do not change. Each
transient spike is recorded in the gold standard as a trap: in the multi-round format an agent whose
risk turns high at the artifact loses reversal-tracking credit, and in single-question formats
escalating on an artifact reading without marking it suspect trips the
<code>acted_on_unverified_signal</code> gate, which is reported without zeroing the case. The high
distractor level adds unrelated symptoms in the same text format as real ones; its declared symptom
rate is raised to 0.5 to match, otherwise the gate refuses the case.</sub></p>

<!-- Figures: uv run --with matplotlib python tools/presentation/make_readme_figures.py
     (deterministic generator, no model calls). Animation: python tools/presentation/make_readme_gifs.py
     --only quickstart (needs Playwright, Pillow and a Chromium; HAENV_CHROME points at the browser). -->

## Quick start

Generate, verify, answer and score offline. No API keys, no cost. Bundled jobs write to
`artifacts/results/` and `artifacts/reports/` by default.

```bash
git clone https://github.com/thetahealth/mirobody-env && cd mirobody-env

uv run haenv build  benchmarks/inputs/example-ew.job.yaml --gen deterministic --fresh   # generate patients and cases
uv run haenv verify benchmarks/inputs/example-ew.job.yaml --gen deterministic           # verify every item
uv run haenv run    benchmarks/inputs/example-ew.job.yaml --offline                     # answer with offline reference solvers
uv run haenv report benchmarks/inputs/example-ew.job.yaml --offline                     # score and write the report
```

<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" srcset="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_quickstart.png">
    <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_quickstart.gif" alt="A terminal running the four quick-start commands in a fresh clone, with their recorded output: build emits 3 of 4 cases and blocks EWX-04, verify reports 82 items with 0 failures, run and report write the report" width="100%">
  </picture>
</p>
<p align="center"><sub>The four commands in a fresh clone, replayed from their recorded output (sped up).</sub></p>

Each command exits 0. `verify` prints:

```
[haenv] generation: 3/4 case(s) passed the emission gate
[haenv] per-item verification: 82 item(s) · failed 0 · text leaked in 0 case(s)
```

<details>
<summary>What the other lines mean, and why <code>--fresh</code></summary>

The recorded walkthrough shows this `report` output, including its original path:

```
[haenv] scoring vintage: all 72 row(s) on disk carry today's stamp `<judging_sha16>` ✓ (fine fingerprint not needed)
[haenv] report -> <repo>/reports/ew-demo/<batch>/eval-ew-demo.md
```

The fourth case, `EWX-04`, is refused with `conflicts=['event_density_mismatch']`: its high
distractor level injects five symptom events where its symptom rate (0.1 per week by default)
allows one.

`--fresh` opens a new batch. Once a batch holds answers, `build` or `verify` on it exits 2, because
regenerating its questions would score new questions against old answers. For the same reason
`verify` runs before `run`.
</details>

Build a public sample pack the same way, with no model call (the audit needs `scikit-learn`):

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out packs/p4
```

It exits 0 when the build and every audit gate pass, and prints the pack record:

```
{"pack": "p4", "n_items": 50, "seed_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "steps": {"gen": 0, "build": 0, "audit": 0}, "job": "<repo>/packs/p4/p4-followup.job.yaml", "batch": "<repo>/packs/p4/results/joint_dx/p4-followup/<batch>", "ref": null}
```

The built batch is under `packs/p4/results/`, the audit under `packs/p4/audit/`.

<details>
<summary>Install from PyPI</summary>

```bash
pip install haenv
JOBS=$(python -c "from haenv import data_root;print(data_root()/'benchmarks'/'inputs')")
cd ~/my-workdir                     # default output: ./artifacts/
haenv build "$JOBS/example-ew.job.yaml" --gen deterministic --fresh
haenv run   "$JOBS/example-ew.job.yaml" --offline
```

A checkout defaults to `<checkout>/artifacts/`; an installed wheel defaults to
`<cwd>/artifacts/`. Set `HAENV_OUTPUT_ROOT` to choose another writable root. `HAENV_DATA_ROOT`
selects read-only resources and retains its combined output redirect for compatibility;
use `HAENV_OUTPUT_ROOT` for new output configuration.
`HAENV_CONFIG_OVERLAY` names a YAML file of your own, merged over the packaged `config.yaml`
(your backends and models; see [Evaluate your own agent](#evaluate-your-own-agent)).
</details>

<details>
<summary>Run against real models (billed)</summary>

A paid run needs two flags: a ceiling and the shared ledger that keeps the spend. Give both, or
the run stops before the first call.

```bash
# one case, one model
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models gemini-3.1-pro --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json

# full sweep, resumable; rerunning sends only the cells that have no answer yet
uv run haenv run benchmarks/inputs/example-ew.job.yaml --judge-budget-usd 50 \
  --judge-budget-ledger ~/.haenv/budget.json
```

Keys are read from the file named by `HAENV_ENV_FILE` (see
[Evaluate your own agent](#evaluate-your-own-agent)); without one, the call fails before it is
sent.
</details>

## One case, end to end

<details>
<summary>What the agent sees, the hidden gold, and how two answers are judged (<code>EWX-01</code> from the quick start)</summary>

The prompt is a fixed instruction block (in Chinese) followed by the record up to `T` as JSON.
An excerpt, with most of the 20 streams elided:

```jsonc
{
  "user_profile": {"age_range": "45-49", "sex": "F", "known_conditions": ["obesity"], ...},
  "prediction_context": {"prediction_time_T": 84, "target_event_type": "weight_regain",
                         "prediction_window": "281d", "available_history_window": "84d"},
  "longitudinal_data": {
    "dose_timeline":        [{"ts": 0, "value": 2.5}, {"ts": 28, "value": 5.0}, {"ts": 56, "value": 7.5}],
    "medication_adherence": [{"ts": 42, "value": 0.95}, {"ts": 56, "value": 0.88}, {"ts": 84, "value": 0.8}],
    "weight":               [{"ts": 0, "value": 98.70}, ..., {"ts": 84, "value": 87.10}],
    ...
  },
  "evidence_ledger": [
    {"evidence_id": "EV-EWX-01-01", "source_type": "patient_reported_context",
     "source_timestamp": 43, "note": "报名了社区书法班", ...},
    ...
  ]
}
```

The agent returns one JSON object: a risk forecast, drivers ranked from a fixed list with the
evidence ids behind each, and one action class from `A0` (continue monitoring) to `A5` (urgent
escalation). The hidden gold for this case: weight regain occurs, driven by
`poor_medication_adherence`, and clinician action is warranted.

| Offline reference solver | Forecast | Top driver | Action | Result |
|---|---|---|---|---|
| `no_revision` | risk 0.2, low | `poor_medication_adherence` (hit) | `A0`, no review requested | hard gate `premature_closure`: the case scores zero |
| `const_ddx` | risk 0.5, indeterminate (abstains) | `unknown_or_multifactorial` | `A3`, review requested | scored |

The first solver names the right driver and still fails the case: action was warranted and it
chose to keep monitoring without tests or review.
</details>

## How it works

A job file declares patient facts (condition, drug, dose steps, devices, start weight) and latent
variables. The generator renders the course and injects the configured artifacts and distractors.
A case is released only if it passes the emission gate: a premise check that rejects contradictory
specifications, per-item verification of every stream and event, and a leak probe that runs on every
prompt before it is sent. Batch-level gates and the pack audit then check the pack as a whole, for
example that real symptoms cannot be told apart from distractors by their data footprint.
The simulation kernel that renders the patients is part of this repository; it lives in `src/haenv_kernel/`.

- **Packs are plugins.** Each pack is a judge group that registers its gold blocks, its question
  template and its scoring; the core pipeline does not know which packs exist.
- **One run, one context.** An evaluation run carries its own state (`RunContext`): two runs in
  one process do not share tables.
- **Fingerprints follow definitions.** The scoring code and the generation code each have a
  fingerprint, stamped on every row and every batch. A refactor that only moves definitions is
  certified definition by definition, so a fingerprint changes when what is computed changes.

The same patients can be presented in four formats (called geometries in the code): `single`,
`gated` (tests ordered against a budget), `slices` (independent questions at several time points)
and `multi` (rounds in which the agent may revise). Each pack poses one decision at `T`; in pack ①
the agent may buy tests before it answers.

## Scoring

- **The gold standard is derived by code** from the world that rendered the record.
- **1.4.0 answer score.** Each answer scores `S = G·D·(1+Q)/2`. The hard gates `G` and the core
  decision `D` are graded by code in all four packs; in pack ① code matches the listed diagnoses
  to the gold through registered aliases, with no model call. One LLM judge (`gpt-6-luna`) scores
  only the management-plan quality `Q`. A pack's score is the item mean of `S`; see
  [Leaderboard](#leaderboard).
- **1.3.0 pack composites (historical).** The next two items describe the 1.3.0 board. Its pack ①
  composite included test F1, for which one LLM judge matched the proposed tests to the gold, with
  stored votes; the listed-diagnosis dimension was computed by code.
- **Guessing scores 0.** A pack's decision is scored as chance-corrected balanced accuracy,
  `cc = (m · BA − 1) / (m − 1)`, with `m` the answer classes present in the gold: answering at
  random or always giving the same class scores 0, every item right scores 1.
- **One composite per pack.**

  | Pack | Composite |
  |---|---|
  | ① | mean of the scored dimensions (test F1, diagnosis listed, numerical reading, chance-corrected review), floored at 0, × (1 − hard-gate rate) |
  | ② | `cc` of the disposition × (1 − hard-gate rate); a negative `cc` is never raised by a gate |
  | ③ | `cc` of the decision, floored at 0, × (1 − hard-gate rate) |
  | ④ | mean of `cc` of the change source and the skill on the size of the true change, `1 − MAE / MAE_naive` (no gate multiplier) |

  The packs are separate boards; their composites are not pooled.
- **Three columns.** Every registered measure is in one of three roles. *Scored* measures enter
  the composite. *Profile* measures judge correctly but separate models too little, or have no
  validity reading yet; they are computed, stored and shown, and do not enter the composite.
  *Retired* measures were found to judge wrongly; they stay in the registry with the reason and
  are on no board.
- Every score row carries a fingerprint of the judging code, and each pack adds its own. Boards
  with mixed fingerprints are rejected by the publish gate.
- Raw model responses are stored, so a scoring fix is a recompute
  (`tools/evaluation/restamp_batch.py <batch>`) with no new calls to the models under test.
- `src/verifier_core/` holds the fingerprinting, the publish gate, the hard-gate multiplier, score
  ceilings and the noise-floor audit. It contains no clinical vocabulary and imports nothing from
  the clinical layer, so it can be reviewed or reused on its own.

Self-checks run in seconds without model calls; see [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#4-self-checks).

## Evaluate your own agent

The model under test is an entry under `models:` in `config.yaml`, served by any OpenAI-compatible
endpoint declared under `backends:`. Each turn is one chat request carrying the prompt; an agent
behind such an endpoint is evaluated the same way, and its internal tool use is not visible to the
judges. It receives the same patients, the same cut at `T` and the same judges. Plugins are
registered explicitly in the job file.

To add an endpoint, put it in `config.local.yaml` next to `config.yaml` (merged over it, ignored
by git). After `pip install`, `config.yaml` is inside the installed package: put the same YAML in
a file of your own and name that file in `HAENV_CONFIG_OVERLAY`. A new backend name registers an
OpenAI-compatible backend; its key is read from the file named by `HAENV_ENV_FILE`:

```yaml
backends:
  my-endpoint:
    url: http://localhost:8000/v1/chat/completions
    key_env: MY_ENDPOINT_KEY          # MY_ENDPOINT_KEY=... in $HAENV_ENV_FILE
models:
  my-agent:
    backend: my-endpoint
    model: my-agent-v1
    max_tokens: 8000
    price: {input_per_million_usd: 0, output_per_million_usd: 0}   # USD per million tokens
```

The default models are reached directly: Gemini models through Google's API, Qwen models through
DashScope, and every other model through OpenRouter, pinned to one upstream with fallbacks off:
the vendor's own endpoint where the account's zero-data-retention setting allows it, otherwise a
named host serving the same weights (OpenAI models on Azure, Claude on Google Vertex, MiniMax on
Novita). No third-party relay sits between the pipeline and a model.

A billed run counts every request against `--judge-budget-usd`. On `openrouter`, `google` and
`dashscope` the price comes from the provider; on a backend of your own it is the `price`
declared for each model, applied to the token counts the endpoint returns in `usage` (0 for a
free local endpoint). A model there without a `price` is refused before any request is sent.
The built-in `google` and `dashscope` prices were checked against the vendors' pages on a dated day:
a week later each run warns once per backend that they may be stale. Compare them with the pages and
set `paid.tariff_checked_on: <YYYY-MM-DD>` in `config.yaml` to record your own check and silence it.

```bash
uv run haenv run benchmarks/inputs/example-ew.job.yaml --models my-agent --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json
```

With `--limit` the run checks the endpoint on a sample: the semantic dimensions are not judged, no
OpenRouter key is needed, and the run exits 6 to say the scores are incomplete. Without it, the
semantic dimensions are judged by `openai/gpt-6-luna` through OpenRouter, so the same env file
also needs an OpenRouter key, and the judge's cost counts against the same budget.

| To change | Where | Code |
|---|---|---|
| the model or agent under test | `config.yaml` (`models:`, `backends:`) | no |
| weighting and normalisation | one config file per board | no |
| add a judge or a new kind of gold standard | a registered function | yes |
| what the judges observe (a new view of the run) | subject plugin | yes |
| streams, events, drug effects, artifacts | world plugin | yes |

[`examples/`](https://github.com/thetahealth/mirobody-env/blob/main/examples/README.md) has five plugin packages. Each runs offline and is paired with a
negative control.
Guides: [judge plugins](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/llm-judge-plugin.md) ·
[external task types](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md).

## Twin records

A twin is a record of a built batch with one declared thing changed. The generator writes the record and its
gold from the hidden truth, so the change can be made where the record is written and checked against the
original field by field. A claim about model behaviour ("models attribute noise to a factor the record names")
is then tested by having each model answer the original and its twin and reading the paired difference.

```bash
uv run haenv twin <job.yaml> --op drop-reagent-note --from-batch <built batch> --out twins/
```

The command selects the items the operation applies to (or those named with `--items`), writes the twin job and
an unchanged original job with the same items, builds both deterministically, and checks that the original
build is byte-identical to the source batch and that each twin differs from it only at the operation's declared
fields. It writes a per-item record of the changes into the twin batch and exits 1 on any other difference. A twin job records
its operation under `_provenance.twin`, so rebuilding it reproduces the twins.

| Operation | Pack | Changes |
|---|---|---|
| `drop-reagent-note` | ④ lab-change attribution | the reagent-change note in the report's method strings |
| `reword-reagent-note-1`, `-2` | ④ lab-change attribution | the same note in another wording |
| `drop-planted-symptoms` | ① diagnosis (budgeted differential) | the patient-reported symptom entries planted as the support of the gold lines |
| `unname-explainer` | ① diagnosis (budgeted differential) | the established diagnoses that explain away a hidden condition, in `known_conditions` |

Both arms are answered like any batch (`tools/evaluation/reliability_passk.py <job> --from-batch <stamp> --k 3`).

## Agent skills

The repository carries three skills under [`skills/`](https://github.com/thetahealth/mirobody-env/tree/main/skills), in the
[Agent Skills](https://agentskills.io) layout (`skills/<name>/SKILL.md`; the standard is read by more
than one agent tool). They are the short path to driving the pipeline from a conversation, and they
are written to be read on their own:

| Skill | Use it when you want to |
|---|---|
| [`haenv-synth`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-synth/SKILL.md) | **generate** patients and questions — from a job file you write, or from a piece of case-description text — and read what the emission gate rejects |
| [`haenv-bench`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-bench/SKILL.md) | **run** a pack against models and read the board: resumable runs, per-batch artifacts, the tool-call trace, what a batch cost |
| [`haenv-extend`](https://github.com/thetahealth/mirobody-env/blob/main/skills/haenv-extend/SKILL.md) | **add your own** judge, indicator stream, event or task type, from a package outside the repository |

Each is an index, not a second source of truth: what it says about the pipeline is checked against
the code and the documents linked from it. A skill is a directory of Markdown plus, in the standard's
terms, optional `scripts/`, `references/` and `assets/`; nothing about them is specific to one
vendor's tool. If you use an agent that reads the standard, point it at the repository and ask in
plain words — "generate a batch of questions from this patient", "run a batch and show me what it
cost", "add a judge that catches X" — and the matching skill supplies the conventions, the commands
and the failure modes.

## Cost and caching

- Generation is cached by model and prompt in `artifacts/cases/_llm_cache/`; rebuilding a pack from the same job makes no model calls.
- A run resumes by default and sends only the cells that have no answer yet. A cell the model answered with an empty body, broken JSON or a spent tool budget counts as answered; only aborts caused by the network, a timeout, a provider 5xx or 429, or the key pool are resent. Raw responses are stored, so a judging change is a recompute, not a re-run of the models under test.
- For models with a measured basis, `max_tokens` must clear a floor derived from their output lengths. This reduces truncation risk but does not guarantee that every answer will finish within budget. `batch.json` records generation and evaluation usage (`gen_usage`, `eval_usage`): measured totals where available, and an explicit status otherwise.
- Scale: the [interactive board](https://thetahealth.github.io/mirobody-env/#act3) shows tested-model spend per 100 answers, with semantic-judge spend listed separately. Model costs include tool requests and mean-price estimates for requests without receipts or token counts.

## Leaderboard

The 1.4.0 board ranks 8 models on four packs, 100 items per pack (N100), with 3 answers per item. The main item score is the mean of the three per-answer scores `S = G·D·(1+Q)/2` (k3); the four packs have equal weight. The preregistered first-answer score (pk1) gives the same model order. `G` indicates passing the hard gates, `D` a correct core decision, and `Q` the quality of the management plan. The 95% intervals use a paired item-cluster bootstrap (2,000 draws), resampling the same items for all models within each pack. Rank intervals are the 2.5th–97.5th percentiles of rank over those same draws.

The evaluated question sets are released as frozen pack files. Each batch records its generation commit and world settings, and regenerating at those recorded commits reproduces all four sets byte for byte (verified 2026-10-17: ① diagnosis and ② triage at `c3cecbee`, the ③ medication candidate arm at `fcf3b389` with its pairing at `c3cecbee`, ④ lab-change attribution at `05f97181`). The job files under `benchmarks/inputs/` use current defaults and produce different items.

### 1.4.0 overall

<p align="center">
  <img src="docs/figures/readme_board.png" alt="Overall scores and 95% intervals for eight models, ordered by score, with rank intervals at left" width="88%">
</p>
<p align="center"><sub>N = 100; k3 averages three S values per item. In pack ④, visible changes on non-noise items exceed the paired RCV. Dots show overall scores, bars show 95% intervals, and labels at left show rank intervals; overlapping rank intervals do not establish an order.</sub></p>

| Rank | Model | Overall (95% CI) | ① | ② | ③ | ④ | All-3 (all 3 correct) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | gpt-6-sol | 0.831 (0.808–0.854) | 0.687 | 0.893 | 0.959 | 0.785 | 0.875 |
| 2 | gemini-3.8-flash | 0.792 (0.768–0.816) | 0.591 | 0.950 | 0.882 | 0.744 | 0.873 |
| 3 | gpt-6-luna | 0.718 (0.690–0.746) | 0.490 | 0.859 | 0.897 | 0.627 | 0.695 |
| 4 | deepseek-v4-pro | 0.659 (0.630–0.689) | 0.383 | 0.933 | 0.689 | 0.629 | 0.670 |
| 5–6 | deepseek-v4-flash | 0.627 (0.595–0.660) | 0.459 | 0.856 | 0.656 | 0.538 | 0.585 |
| 5–6 | glm-5.3-flash | 0.602 (0.571–0.632) | 0.381 | 0.813 | 0.668 | 0.547 | 0.533 |
| 7 | minimax-m3 | 0.562 (0.532–0.589) | 0.153 | 0.870 | 0.608 | 0.616 | 0.518 |
| 8 | qwen3.7-flash | 0.425 (0.394–0.457) | 0.299 | 0.599 | 0.271 | 0.533 | 0.325 |

① Diagnosis (budgeted differential) · ② Triage · ③ Medication adjustment · ④ Lab-change attribution. All-3 (all 3 correct) is the share of items whose decision is correct in all three answers, averaged over the four packs with equal weight; it is a reliability profile, not the ranking metric, and ranking by it gives the same order. `Q` uses the mean of two judgments by one LLM judge. Of 7,246 Q records, 7,215 average two judgments, 20 contain one judgment, and 11 have no management plan and score zero. The rubric is calibrated on these board answers; intervals exclude judge noise. Three-round accuracy, All-3, hard-gate rates and history-group readings are profiles in the interactive board.

[Interactive board](https://thetahealth.github.io/mirobody-env/#act3) ·
[Board data (JSON)](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/board.json)

### Earlier boards (historical)

<details>
<summary>1.3.0 overall and by pack</summary>

### 1.3.0 overall

| Rank | Model | Overall | 95% CI |
|---:|---|---:|---:|
| 1 | gpt-6-sol | 0.712 | 0.671–0.751 |
| 2–4 | gemini-3.1-pro | 0.648 | 0.610–0.689 |
| 2–4 | gemini-3.7-flash | 0.640 | 0.595–0.684 |
| 2–4 | gpt-6-luna | 0.609 | 0.564–0.656 |
| 5–6 | kimi-k3 | 0.556 | 0.509–0.604 |
| 5–6 | deepseek-v4-pro | 0.526 | 0.484–0.570 |
| 7–8 | glm-5.3-flash | 0.470 | 0.424–0.521 |
| 7–9 | minimax-m3 | 0.458 | 0.417–0.500 |
| 8–10 | deepseek-v4-flash | 0.415 | 0.366–0.464 |
| 9–10 | qwen3.7-flash | 0.388 | 0.338–0.437 |

gpt-6-sol is first and is separable from every other model (none of its nine pairs is within 2 paired
standard errors). gemini-3.1-pro, gemini-3.7-flash and gpt-6-luna share ranks 2–4: no pair among them
differs by more than 2 paired standard errors, so their order is not a ranking. Ranks are intervals for
that reason: 37 of the 45 model pairs are separable. 1.3.0 boards are not comparable with 1.2.x
boards: the judging and the world fingerprints changed.

### 1.3.0 by pack

Each cell is the pack score with its 95% interval.

| Model | ① Differential diagnosis | ② Acute triage | ③ Medication adjustment | ④ Follow-up interpretation |
|---|---:|---:|---:|---:|
| gpt-6-sol | 0.643 (0.579–0.703) | 0.436 (0.318–0.551) | 0.776 (0.688–0.860) | 0.991 (0.970–1.000) |
| gemini-3.1-pro | 0.294 (0.240–0.351) | 0.448 (0.316–0.581) | 0.861 (0.796–0.926) | 0.991 (0.964–1.000) |
| gemini-3.7-flash | 0.482 (0.410–0.549) | 0.425 (0.310–0.545) | 0.710 (0.610–0.804) | 0.942 (0.888–0.986) |
| gpt-6-luna | 0.506 (0.397–0.604) | 0.365 (0.250–0.486) | 0.802 (0.725–0.879) | 0.764 (0.699–0.827) |
| kimi-k3 | 0.347 (0.263–0.429) | 0.383 (0.268–0.503) | 0.741 (0.650–0.829) | 0.751 (0.675–0.828) |
| deepseek-v4-pro | 0.258 (0.198–0.323) | 0.369 (0.258–0.481) | 0.762 (0.678–0.842) | 0.716 (0.646–0.777) |
| glm-5.3-flash | 0.232 (0.142–0.331) | 0.420 (0.318–0.530) | 0.656 (0.559–0.753) | 0.573 (0.496–0.655) |
| minimax-m3 | 0.130 (0.077–0.188) | 0.320 (0.215–0.423) | 0.742 (0.650–0.830) | 0.640 (0.562–0.722) |
| deepseek-v4-flash | 0.063 (-0.020–0.148) | 0.258 (0.148–0.370) | 0.691 (0.595–0.790) | 0.649 (0.559–0.733) |
| qwen3.7-flash | 0.015 (-0.055–0.078) | 0.323 (0.201–0.454) | 0.625 (0.523–0.728) | 0.587 (0.507–0.665) |

</details>

The 1.x board ranks ten models on the two diagnosis tracks, `ddx-timeline` and `ddx-workup`
(145 cases each). It is kept as published and is **not comparable** with the pack boards: the
world, the questions and the scoring have changed since. The bridge items are the only link
between the two.

<details>
<summary>The two 1.x tables</summary>

On 1.x, the composite was the mean of the scored dimensions multiplied by (1 − hard-gate failure
rate), from 0 to 1; *95% CI* is the case-level bootstrap interval (10,000 resamples) and a tier
number is 1 plus the count of models significantly better (Holm-corrected α = 0.05). *Answered*
counts the cells with a scorable answer out of 145; a cell without one scored 0. Each cell ran
once. The diagnosis dimensions of that composite had no blind-human validity reading, so the board
was labelled preliminary.

### ddx-timeline (1.x)

| Rank | Model | Composite | 95% CI | Tier | Answered |
|---:|---|---:|---:|---:|---:|
| 1 | gemini-3.1-pro | 0.724 | 0.671–0.776 | 1 | 145 / 145 |
| 2 | gpt-6-sol | 0.691 | 0.653–0.730 | 1 | 145 / 145 |
| 3 | gemini-3.7-flash | 0.678 | 0.618–0.740 | 1 | 145 / 145 |
| 4 | gpt-6-luna | 0.666 | 0.630–0.705 | 1 | 145 / 145 |
| 5 | kimi-k3 | 0.662 | 0.632–0.695 | 1 | 145 / 145 |
| 6 | deepseek-v4-pro | 0.652 | 0.632–0.671 | 1 | 145 / 145 |
| 7 | minimax-m3 | 0.617 | 0.587–0.647 | 3 | 143 / 145 |
| 8 | qwen3.7-flash | 0.589 | 0.550–0.631 | 4 | 145 / 145 |
| 9 | deepseek-v4-flash | 0.434 | 0.380–0.494 | 9 | 144 / 145 |
| 10 | glm-5.3-flash | 0.048 | 0.024–0.076 | 10 | 17 / 145 |

### ddx-workup (1.x)

| Rank | Model | Composite | 95% CI | Tier | Answered |
|---:|---|---:|---:|---:|---:|
| 1 | kimi-k3 | 0.697 | 0.639–0.751 | 1 | 144 / 145 |
| 2 | gpt-6-luna | 0.696 | 0.635–0.755 | 1 | 145 / 145 |
| 3 | gemini-3.1-pro | 0.689 | 0.625–0.751 | 1 | 145 / 145 |
| 4 | deepseek-v4-pro | 0.685 | 0.651–0.720 | 1 | 144 / 145 |
| 5 | gemini-3.7-flash | 0.680 | 0.615–0.743 | 1 | 145 / 145 |
| 6 | gpt-6-sol | 0.673 | 0.615–0.730 | 1 | 145 / 145 |
| 6 | minimax-m3 | 0.673 | 0.623–0.721 | 1 | 145 / 145 |
| 8 | qwen3.7-flash | 0.493 | 0.443–0.544 | 8 | 145 / 145 |
| 9 | glm-5.3-flash | 0.375 | 0.303–0.446 | 8 | 116 / 145 |
| 10 | deepseek-v4-flash | 0.304 | 0.247–0.365 | 9 | 145 / 145 |

[Figure](https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.svg) ·
[Source values (CSV)](https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.csv) ·
[1.x snapshot (JSON, key `history_1x`)](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/data.json)

</details>

## Limitations

- **Pack ④ RCV baseline.** A reader using only the two printed values and the protocol RCV table reaches D around 0.5 on pack ④, so a pack ④ score near 0.5 demonstrates use of RCV alone.
- **Rank uncertainty.** One of the seven adjacent paired score-difference intervals includes zero (deepseek-v4-flash and glm-5.3-flash; two of seven under pk1); the board does not establish an order for such pairs.
- **Clinical review.** The rules behind packs ② and ③ (which findings send a patient to the
  emergency department, when a dose change is on target) and several registry entries carry
  `review: pending`; the medical content as a whole has not been reviewed by a practising
  clinician.
- **Single semantic judge.** One model (`gpt-6-luna`, reasoning high) scores management-plan quality `Q` in all four packs; `D` and `G` are decided in code in every pack, pack ① included. The same vendor's models are among those tested, so self-preference is not ruled out. A cross-family check re-judged all 6,937 responses with `G·D = 1` with a Gemini judge (requested `gemini-3.7-flash`; the provider served `gemini-3.8-flash`, itself a tested model, for every request): the eight-model order is unchanged, gpt-6-sol leads gemini-3.8-flash by +0.031 (95% CI +0.010 to +0.052), and model-mean `Q` correlates with the primary judge at r = 0.968. `Q` averages two judgments where both are available; the single-judgment and no-plan exceptions are counted above. The `Q` rubric is calibrated on these board answers, with no clinician validation of its verdicts, and the score intervals exclude judge noise.
- **The diagnosis dimensions await validity.** On the bridge tracks the diagnosis-track scored
  dimensions are judgement-based and wait for a clinical blind annotation; until then they have
  no public column.
- **Synthetic data.** All patients are synthetic and the benchmark is for evaluation only; nothing
  here is medical advice.

The full list, with the scoring, generation and world-layer gaps, is in the
[data card](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#known-gaps).

## Related work

| Work | Evaluates |
|---|---|
| ESL-Bench ([arXiv:2604.02834](https://arxiv.org/abs/2604.02834), [dataset](https://huggingface.co/datasets/mirobody/ESL-Bench)) | 100 synthetic users with 1–5 year device, exam and event trajectories; 100 queries each across lookup, trend, comparison, anomaly and explanation, with programmatically computed answers. Leaderboard: [Health Memory Arena](https://healthmemoryarena.ai) |
| mirobody-eval ([GitHub](https://github.com/thetahealth/mirobody-eval)) | a harness that reproduces published health benchmarks, ESL-Bench included, against an existing system, with pluggable virtual user, target and judge |
| MedAgentBench ([arXiv:2501.14654](https://arxiv.org/abs/2501.14654)) | 300 agent tasks in a FHIR virtual EHR |
| EHRSHOT ([arXiv:2307.02028](https://arxiv.org/abs/2307.02028)) | few-shot prediction on longitudinal EHR of 6,739 real patients |
| LongHealth ([arXiv:2401.14490](https://arxiv.org/abs/2401.14490)) | 400 multiple-choice questions over 20 long fictional records |
| HealthBench ([arXiv:2505.08775](https://arxiv.org/abs/2505.08775)) | 5,000 health conversations graded by a model against physician-written rubrics |
| AgentClinic ([arXiv:2405.07960](https://arxiv.org/abs/2405.07960)), CRAFT-MD ([doi:10.1038/s41591-024-03328-5](https://doi.org/10.1038/s41591-024-03328-5)) | history-taking and diagnosis in dialogue with LLM-simulated patients |

HAEnv combines a longitudinal record, a prompt cut at `T`, a gold standard fixed before the data,
code scoring of decisions and gates with one LLM judge for plan quality, and regenerable
synthetic cases. The `pass^k` reliability metric is from τ-bench
([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)).

## Data, ethics and reproduction

All patients are synthetic. Nothing here is medical advice or suitable for clinical decisions.

> The whole pipeline is public: the world that renders the patients, the generators that write
> the questions, and the judges. The canary string ([`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md))
> marks this pack's text so its presence in a corpus can be detected; training on the public
> sample is permitted; the 1.4.0 board is built from the public seed (earlier boards used private seeds).

- [`docs/DATA_CARD.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md): packs, gold provenance, scoring, known gaps, and why the
  gold standard ships with the questions.
- [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md): what is free to reproduce, what is billed, and the
  self-checks.
- [`docs/ETHICS.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/ETHICS.md): data sources and limits of use.
- [`CONTRIBUTING.md`](https://github.com/thetahealth/mirobody-env/blob/main/CONTRIBUTING.md): how to change scoring or generation code.

## Repository layout

The installed command and import remain `haenv`. Source lives under `src/`; the simulation
kernel and general verifier are separate namespaces in the same distribution.

```text
src/
├── haenv/           application, world, interaction, scoring and runtime
├── haenv_kernel/    simulation and isolation primitives
└── verifier_core/   domain-independent score audits
plugins/rare/        optional rare-disease coding project
benchmarks/          fixed jobs, task cards, source cases and replay material
registry/            versioned policies, anchors, validity and shared references
artifacts/           generated results, reports, caches and extracted jobs
tools/               generation, evaluation, validation, presentation and release commands
examples/            extension examples
docs/                current user and design documentation
tests/               regression and contract checks
web/demo/            browser demonstration
```

Use the [documentation index](https://github.com/thetahealth/mirobody-env/blob/main/docs/README.md)
for the package responsibilities, current specification and reproduction guide. The optional
rare-disease project is separate from the main distribution and retains its prototype status.

## Citation

```bibtex
@software{haenv,
  title  = {Health Agent Environment},
  author = {{Theta Health}},
  year   = {2026},
  url    = {https://github.com/thetahealth/mirobody-env},
  version = {1.4.0.dev0}
}
```

The same metadata is in [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff).

## License and acknowledgements

Code is under the [MIT License](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE); the synthetic data and question packs are under
[CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA). Third-party notices are in [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md).

HAEnv is inspired by [ESL-Bench](https://arxiv.org/abs/2604.02834).

<p align="center"><sub>Synthetic data · evaluation use only · not medical advice</sub></p>
