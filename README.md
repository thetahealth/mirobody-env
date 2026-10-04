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

> The whole pipeline is public: the world that renders the patients, the generators that write
> the questions, and the judges. The canary string ([`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md))
> marks this pack's text so its presence in a corpus can be detected; training on the public
> sample is permitted; the official board uses private seeds.

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_patient.png" alt="One synthetic patient with type 2 diabetes on tirzepatide: daily home weight and clinic weight, steps and resting heart rate, dose and adherence, and reported events; the record after index time T is shaded as hidden" width="92%">
</p>
<p align="center"><sub>One synthetic patient. Left of <code>T</code> is the prompt; right of <code>T</code>
is hidden and used for grading. Adherence falls before <code>T</code>, and the weight regain it drives
begins after <code>T</code> at the latent reversal point.</sub></p>

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
| ① Differential diagnosis | which conditions explain the record, when a second condition hides behind findings the first one explains; whether a clinician should review | a differential, the tests to order, a review flag | code against the world's truth; one LLM judge matches the free-text diagnoses and tests to the gold |
| ② Acute triage | where the patient should go now | `ed_now` · `within_24h` · `routine_followup` · `watchful_waiting` | code, against the world's truth |
| ③ Chronic medication adjustment | what to do with the drug at this follow-up visit | `uptitrate` · `downtitrate` · `maintain` · `switch` · `check_adherence_or_adverse_effect` | code, against the dose line, refills and the control target the world records |
| ④ Follow-up interpretation | whether the change since the last result is real | `true_change` · `analytic_biological_noise` · `preanalytical` · `method_difference`, plus the size of the true change | code, against the world's true values and its per-reading perturbation records |

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
uv run --with scikit-learn --with joblib python tools/make_pack.py --pack p4 --n 50 --out packs/p4
```

`--pack` is `m2` (①), `pack2` (②), `p3` (③) or `p4` (④); `--n` is the item count (50, 100, …);
`--seed` (or `HAENV_PACK_SEED`) sets the seed. The pack is the same for the same pack, size and
seed. The job carries a header `pack: {n_items, seed_sha256}`, and `batch.json` records the size,
the seed's hash and the job, never the seed. Class shares follow each pack's proportion table at
every size.

| Pack | Job file | Items at N = 50 | Answer classes at N = 50 |
|---|---|---:|---|
| ① Differential diagnosis | `inputs/m2-pack1.job.yaml` | 50 | 34 new items (24 with a hidden condition, 10 without) and 16 bridge items |
| ② Acute triage | `inputs/pack2-triage.job.yaml` | 50 | `ed_now` 13 · `within_24h` 13 · `routine_followup` 12 · `watchful_waiting` 12 |
| ③ Chronic medication adjustment | `inputs/p3-meds.job.yaml` | 50 | `maintain` 12 · `downtitrate` 12 · `check_adherence_or_adverse_effect` 12 · `uptitrate` 8 · `switch` 6 |
| ④ Follow-up interpretation | `inputs/p4-followup.job.yaml` | 50 | `true_change` 13 · `analytic_biological_noise` 13 · `preanalytical` 12 · `method_difference` 12 |

- **Public sample packs.** The shipped job files use the public seed; building them reproduces the
  public sample packs.
- **Official boards use a private seed.** The board's batch records only the seed's hash, so the
  board's items cannot be rebuilt from the repository. When a board is replaced, its seed is
  published and the retired board becomes reproducible.
- **The audit is a release gate.** `tools/pack_audit.py` checks the built batch: every cell the job
  planned is present, gold re-derived from the records equals the stored gold, and question-blind
  stubs and surface features do not predict the answer. `haenv run` refuses a pack's items without
  a passing audit.

The bridge tracks and two smaller example tasks ship as plain job files:

| Task | Job file | Case specifications | Format |
|---|---|---|---|
| Differential diagnosis at several time points (bridge) | `inputs/ddx-timeline.job.yaml` | 145 | questions at several time points |
| Budgeted test ordering (bridge) | `inputs/ddx-workup.job.yaml` | 145 | the agent orders tests against a budget |
| Weight-regain forecast and driver attribution | `inputs/early_warning-20.job.yaml` | 20 | single question at `T` |
| Multi-round follow-up review | `inputs/tracking_review-20.job.yaml` | 20 | rounds; the agent may revise |

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

<!-- Figures: uv run --with matplotlib python docs/scripts/make_readme_figures.py
     (deterministic generator, no model calls). Animation: python docs/scripts/make_readme_gifs.py
     --only quickstart (needs Playwright, Pillow and a Chromium; HAENV_CHROME points at the browser). -->

## Quick start

Generate, verify, answer and score offline. No API keys, no cost.

```bash
git clone https://github.com/thetahealth/mirobody-env && cd mirobody-env

uv run haenv build  inputs/example-ew.job.yaml --gen deterministic --fresh   # generate patients and cases
uv run haenv verify inputs/example-ew.job.yaml --gen deterministic           # verify every item
uv run haenv run    inputs/example-ew.job.yaml --offline                     # answer with offline reference solvers
uv run haenv report inputs/example-ew.job.yaml --offline                     # score and write the report
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

`report` prints:

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
uv run --with scikit-learn --with joblib python tools/make_pack.py --pack p4 --n 50 --out packs/p4
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
JOBS=$(python -c "import haenv,pathlib;print(pathlib.Path(haenv.__file__).parent/'_data'/'inputs')")
cd ~/my-workdir                     # artifacts go to the current directory
haenv build "$JOBS/example-ew.job.yaml" --gen deterministic --fresh
haenv run   "$JOBS/example-ew.job.yaml" --offline
```

`HAENV_DATA_ROOT` sets the read-only resource root and `HAENV_OUTPUT_ROOT` the artifact root.
`HAENV_CONFIG_OVERLAY` names a YAML file of your own, merged over the packaged `config.yaml`
(your backends and models; see [Evaluate your own agent](#evaluate-your-own-agent)).
</details>

<details>
<summary>Run against real models (billed)</summary>

A paid run needs two flags: a ceiling and the shared ledger that keeps the spend. Give both, or
the run stops before the first call.

```bash
# one case, one model
uv run haenv run inputs/example-ew.job.yaml --models gemini-3.1-pro --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json

# full sweep, resumable; rerunning sends only the cells that have no answer yet
uv run haenv run inputs/example-ew.job.yaml --judge-budget-usd 50 \
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

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_how_it_works.png"><img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_how_it_works.png" alt="HAEnv synthetic evaluation: patient facts and latent variables produce a checked history; the agent sees records up to T, while code-derived hidden gold is used only to score its response. Cases that fail emission checks are withheld." width="100%"></a>
</p>
<p align="center"><sub>Conceptual workflow. The agent sees records up to <code>T</code>; hidden gold goes only to the judges, which are code plus one LLM judge for the semantic dimensions (see <a href="#scoring">Scoring</a>). Click to view full size.</sub></p>

A job file declares patient facts (condition, drug, dose steps, devices, start weight) and latent
variables. The generator renders the course and injects the configured artifacts and distractors.
A case is released only if it passes the emission gate: a premise check that rejects contradictory
specifications, per-item verification of every stream and event, and a leak probe that runs on every
prompt before it is sent. Batch-level gates and the pack audit then check the pack as a whole, for
example that real symptoms cannot be told apart from distractors by their data footprint.
The simulation kernel that renders the patients is part of this repository; it lives in `haenv_kernel/`.

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

- **The gold standard is derived by code** from the world that rendered the record. Packs ②, ③
  and ④ are scored by code alone. Pack ① is scored by code, and one LLM judge (`gpt-6-luna`)
  matches its free-text diagnoses and tests to the gold, with stored votes.
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
  (`tools/restamp_batch.py <batch>`) with no new calls to the models under test.
- `verifier_core/` holds the fingerprinting, the publish gate, the hard-gate multiplier, score
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

```bash
uv run haenv run inputs/example-ew.job.yaml --models my-agent --limit 1 \
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

- Generation is cached by model and prompt in `cases/_llm_cache/`; rebuilding a pack from the same job makes no model calls.
- A run resumes by default and sends only the cells that have no answer yet. Raw responses are stored, so a judging change is a recompute, not a re-run of the models under test.
- For models with a measured basis, `max_tokens` must clear a floor derived from their output lengths. This reduces truncation risk but does not guarantee that every answer will finish within budget. `batch.json` records generation and evaluation usage (`gen_usage`, `eval_usage`): measured totals where available, and an explicit status otherwise.
- Scale: a pack item is one request. At N = 50 one pass costs, estimated from each prompt's token count at fitted provider prices, $0.09–$9.55 per model on pack ② (mean $3.20 over the ten 1.x models), $0.09–$8.19 on pack ③ (mean $2.65) and $0.09–$8.62 on pack ④ (mean $2.82); packs ②–④ have no judge cost. Details and the recompute command are in [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#tokens-and-caching).

## Leaderboard

The 1.2.0 board ranks 10 models on the four packs, 50 items each, every item answered 3 times.
The overall score is the equal-weight mean of the four pack scores, each from 0 to 1. The 95%
intervals come from a case × round bootstrap (4,000 draws that resample items and rounds
together); ranks follow the unrounded scores. The board packs are drawn with a private seed; jobs
and batches record only its sha256, `8f90c70f61fe7a53860e9a21576c0393e26515ba7eede0467a534360dfa7f9b9`.

### 1.2.0 overall

| Rank | Model | Overall | 95% CI |
|---:|---|---:|---:|
| 1 | gemini-3.1-pro | 0.732 | 0.674–0.788 |
| 2 | gemini-3.7-flash | 0.677 | 0.602–0.742 |
| 3 | gpt-6-sol | 0.671 | 0.617–0.729 |
| 4 | deepseek-v4-pro | 0.560 | 0.484–0.634 |
| 5 | gpt-6-luna | 0.547 | 0.470–0.616 |
| 6 | kimi-k3 | 0.547 | 0.470–0.622 |
| 7 | minimax-m3 | 0.521 | 0.439–0.600 |
| 8 | glm-5.3-flash | 0.483 | 0.408–0.564 |
| 9 | deepseek-v4-flash | 0.426 | 0.347–0.500 |
| 10 | qwen3.7-flash | 0.389 | 0.317–0.465 |

### 1.2.0 by pack

Each cell is the pack score with its 95% interval.

| Model | ① Differential diagnosis | ② Acute triage | ③ Medication adjustment | ④ Follow-up interpretation |
|---|---:|---:|---:|---:|
| gemini-3.1-pro | 0.834 (0.746–0.908) | 0.504 (0.364–0.645) | 0.799 (0.684–0.902) | 0.790 (0.691–0.871) |
| gemini-3.7-flash | 0.755 (0.646–0.854) | 0.532 (0.389–0.673) | 0.667 (0.518–0.801) | 0.754 (0.581–0.890) |
| gpt-6-sol | 0.607 (0.539–0.697) | 0.408 (0.265–0.554) | 0.785 (0.662–0.894) | 0.886 (0.815–0.944) |
| deepseek-v4-pro | 0.566 (0.490–0.664) | 0.457 (0.311–0.598) | 0.677 (0.537–0.804) | 0.538 (0.366–0.693) |
| gpt-6-luna | 0.547 (0.485–0.608) | 0.362 (0.234–0.493) | 0.743 (0.579–0.879) | 0.536 (0.339–0.712) |
| kimi-k3 | 0.612 (0.517–0.726) | 0.468 (0.319–0.615) | 0.694 (0.556–0.818) | 0.411 (0.205–0.609) |
| minimax-m3 | 0.552 (0.486–0.632) | 0.421 (0.277–0.569) | 0.806 (0.673–0.919) | 0.305 (0.099–0.491) |
| glm-5.3-flash | 0.594 (0.489–0.717) | 0.497 (0.356–0.631) | 0.566 (0.421–0.703) | 0.277 (0.068–0.476) |
| deepseek-v4-flash | 0.406 (0.292–0.523) | 0.306 (0.175–0.441) | 0.589 (0.460–0.710) | 0.403 (0.203–0.581) |
| qwen3.7-flash | 0.428 (0.341–0.533) | 0.329 (0.187–0.487) | 0.542 (0.413–0.655) | 0.258 (0.050–0.464) |

[Interactive board](https://thetahealth.github.io/mirobody-env/#act3) ·
[Board data (JSON)](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/board.json)

### 1.x board (historical)

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

- **Pack ② does not separate the leading models.** Among the six best models on pack ② no pair
  differs by more than 1.96 standard errors (0 of 15 pairs); pack ③ separates 3 of 15, pack ①
  8 and pack ④ 10. On the overall board 9 of the 15 top-six pairs are separable.
- **Clinical review.** The rules behind packs ② and ③ (which findings send a patient to the
  emergency department, when a dose change is on target) and several registry entries carry
  `review: pending`; the medical content as a whole has not been reviewed by a practising
  clinician.
- **Single judge in pack ①.** One model (`gpt-6-luna`, reasoning high) matches free-text
  diagnoses and tests to the gold, with two votes and a third on disagreement. The same vendor's
  models are among those tested; self-preference is not ruled out, and no second judge or physician
  labelling checks its verdicts. Packs ②–④ have no LLM judge.
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
code scoring for every dimension except the semantic ones (one LLM judge) and regenerable
synthetic cases. The `pass^k` reliability metric is from τ-bench
([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)).

## Data, ethics and reproduction

All patients are synthetic. Nothing here is medical advice or suitable for clinical decisions.

- [`docs/DATA_CARD.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md): packs, gold provenance, scoring, known gaps, and why the
  gold standard ships with the questions.
- [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md): what is free to reproduce, what is billed, and the
  self-checks.
- [`docs/ETHICS.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/ETHICS.md): data sources and limits of use.
- [`CONTRIBUTING.md`](https://github.com/thetahealth/mirobody-env/blob/main/CONTRIBUTING.md): how to change scoring or generation code.

## Citation

```bibtex
@software{haenv,
  title  = {Health Agent Environment},
  author = {{Theta Health}},
  year   = {2026},
  url    = {https://github.com/thetahealth/mirobody-env},
  version = {1.2.0}
}
```

The same metadata is in [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff).

## License and acknowledgements

Code is under the [MIT License](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE); the synthetic data and question packs are under
[CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA). Third-party notices are in [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md).

HAEnv is inspired by [ESL-Bench](https://arxiv.org/abs/2604.02834).

<p align="center"><sub>Synthetic data · evaluation use only · not medical advice</sub></p>
