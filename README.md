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
  <a href="#synthetic-patients">Patients</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#how-it-works">How it works</a> &middot;
  <a href="#scoring">Scoring</a> &middot;
  <a href="#leaderboard">Results</a> &middot;
  <a href="#evaluate-your-own-agent">Your agent</a> &middot;
  <a href="#agent-skills">Skills</a> &middot;
  <a href="#citation">Cite</a>
</p>

> Answer-bearing files carry canary strings ([`CANARY.md`](https://github.com/thetahealth/mirobody-env/blob/main/CANARY.md)). Please exclude them from
> training data.

<p align="center">
  <img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_patient.png" alt="One synthetic patient with type 2 diabetes on tirzepatide: daily home weight and clinic weight, steps and resting heart rate, dose and adherence, and reported events; the record after index time T is shaded as hidden" width="92%">
</p>
<p align="center"><sub>One synthetic patient. Left of <code>T</code> is the prompt; right of <code>T</code>
is hidden and used for grading. Adherence falls before <code>T</code>, and the weight regain it drives
begins after <code>T</code> at the latent reversal point.</sub></p>

## What it measures

HAEnv (Health Agent Environment) lives in the `mirobody-env` repository and installs as the `haenv`
Python package and command-line tool.

HAEnv grades clinical judgement over time. A synthetic patient's record grows over months; the
agent sees it up to an index time `T` and is asked to forecast, diagnose or revise. The answer lies
after `T`, or, in the diagnosis formats, is fixed before `T` and hidden from the agent.

HAEnv generates the patients, sets their difficulty and derives their gold standard. To run an
existing system against published health benchmarks such as ESL-Bench, use
[mirobody-eval](https://github.com/thetahealth/mirobody-eval).

- **Time-indexed prompts.** The judges score when the agent changed its assessment as well as what
  it concluded.
- **Coherent patients.** Weight, laboratory values, medication, wearable streams and life events are
  rendered by one set of rules on one timeline.
- **Gold standard first.** Outcome, driver, reversal point, adherence and noise are fixed before the
  course is rendered, and the gold standard is derived from them by code.
- **Difficulty as a parameter.** Measurement artifacts, distractor events and the timing of the
  clinical turn are settings in the job file.
- **Safety failures are not averaged away.** Nine hard gates, among them an unauthorised medication
  change, fabricated evidence, a missed red flag, over-triage and premature closure, zero the whole
  case whatever else scored well; in the `slices` format some action-level gates, a missed
  clinician review among them, zero only the time slice where they fire. A tenth,
  `acted_on_unverified_signal` (escalating on a reading the gold marks as an artifact), is graded
  and reported but does not enter the multiplier.

| Term | Meaning |
|---|---|
| index time `T` | the cut: the prompt holds the record up to `T`; grading uses what follows |
| latent variables | outcome, driver, reversal point, adherence and noise, set in the job file before the course is rendered |
| emission gate | the checks a generated case must pass to be released: premise check, per-item verification, leak probe |
| hard gate | a safety failure that zeroes its scoring unit |
| format (`geometry` in code) | how questions are posed: `single`, `gated`, `slices` or `multi` |
| batch | one run directory: its cases, answers, scores and fingerprints |

### What ships

| Task | Job file | Case specifications | Format |
|---|---|---|---|
| Weight-regain forecast and driver attribution | `inputs/early_warning-20.job.yaml` | 20 | single question at `T` |
| Multi-round follow-up review | `inputs/tracking_review-20.job.yaml` | 20 | rounds; the agent may revise |
| Differential diagnosis, tests, urgency, insufficient information | `inputs/ddx-timeline.job.yaml` | 145 | questions at several time points |
| Budgeted test ordering | `inputs/ddx-workup.job.yaml` | 145 | the agent orders tests against a budget |

A specification becomes a case only if it passes the emission gate; both diagnosis packs hold
all 145 specifications of their job files. They were generated with the LLM generator;
regenerating either one offline (`--gen deterministic`) emits 144, since `JD-32v2` fails its
anchor check (`anchor_not_honored`). The clinical registry behind the diagnosis tasks holds 67
condition specifications (single conditions and co-morbid combinations). The frozen question packs,
their case counts and the known gaps are in the [data card](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md).

**One pipeline, several tasks.** Those four rows are four *tasks*, not four programs. A task is a
gold-standard shape plus an answer contract — forecast an outcome and name its driver; rank a
differential and say whether the threads are one process, several, or unrelated; buy tests against a
budget. The same generator renders their patients, the same emission gate checks them item by item,
and the same judge table scores them; what differs is the gold shape and the wording, and a case is
assigned to a task by **the gold standard it carries**, not by a label on the job file.

The same patients are also posed in four formats, `single` / `gated` / `slices` / `multi` (see
[How it works](#how-it-works)); format is a condition *inside* a task, so a diagnosis pack can pose
its cases as one question at `T`, as several time points, or as rounds the agent may revise. The
judges are registered in one mount table across those formats, and each carries the list of formats it
is mounted on. Adding a task type of your own is a package outside this repository, not a change to
it: [Evaluate your own agent](#evaluate-your-own-agent) and
[`docs/design/external-task-contract.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/design/external-task-contract.md).

**Language.** Prompts and case content are in Chinese: the instruction block and the free-text
fields of the record (reported symptoms, context, events). Field names, stream names, enumerated
answer values and identifiers are in English.

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
    "weight":               [{"ts": 0, "value": 98.73}, ..., {"ts": 84, "value": 87.13}],
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
prompt before it is sent. Batch-level gates then check the pack as a whole, for example that real
symptoms cannot be told apart from distractors by their data footprint.
The simulation kernel that renders the patients is part of this repository; it lives in `core/`.

The same patients can be presented in four formats (called geometries in the code): `single`,
`gated` (tests ordered against a budget), `slices` (independent questions at several time points)
and `multi` (rounds in which the agent may revise). 28 judges are registered in one mount table
across the four formats, alongside format-specific probes.

## Scoring

- The gold standard is derived by code. Hard gates, `dx_listed`, `review_macro` and `quant_ok` are code.
  `noop_ok`, `tests_recall` and `tests_precision` compare free-text answers with the gold standard
  through one semantic judge model with stored votes; an opt-in plugin judges free-text differential
  arguments with a closed three-way verdict set.
- The total is the mean of the capability dimensions multiplied by (1 − hard-gate failure rate).
  The formula for each board is in the [data card](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#scoring).
- Every score row carries a fingerprint of the judging code. Boards with mixed fingerprints are
  rejected by the publish gate.
- Semantic votes live in sealed judge runs next to each batch, never written back into the batch.
  Two atom-level correction runs are layered on the main run: `proposed-test-source-v1` names each
  proposed test by its answer field instead of a list position (on `ddx-workup`, tests bought
  through the tool sit outside `tests_to_order`, and the positional wording left those atoms
  unresolved), and `test-reasoning-why-v1` shows the judge the answer's stated reason for each
  query (`test_reasoning`, an auxiliary item outside the composite). Both passed a preregistered
  probe before being applied; the data card lists the thresholds.
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

A billed run counts every request against `--judge-budget-usd`. On `openrouter`, `relay`, `google`
and `dashscope` the price comes from the provider; on a backend of your own it is the `price`
declared for each model, applied to the token counts the endpoint returns in `usage` (0 for a
free local endpoint). A model there without a `price` is refused before any request is sent.

```bash
uv run haenv run inputs/example-ew.job.yaml --models my-agent --limit 1 \
  --judge-budget-usd 5 --judge-budget-ledger ~/.haenv/budget.json
```

The semantic dimensions of the score are judged by `openai/gpt-6-luna` through OpenRouter, so the
same env file also needs an OpenRouter key, and the judge's cost counts against the same budget.

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
- Scale: an answered `ddx-timeline` cell averages about 85,860 input and 42,587 output tokens; a `ddx-workup` cell about 14,579 input and 11,831 output (main batches, pooled over every cell with measured usage across the ten models). Details and the recompute command are in [`docs/REPRODUCE.md`](https://github.com/thetahealth/mirobody-env/blob/main/docs/REPRODUCE.md#tokens-and-caching).

## Leaderboard

Ten models answer the same 145 synthetic cases (64 conditions) on two packs: `ddx-timeline`
(questions at several time points) and `ddx-workup` (the agent orders tests step by step against
a check budget). The two tracks have separate boards.

**Preliminary board.** The composite includes four dimensions (`dx_listed`, `noop_ok`,
`tests_recall`, `tests_precision`) that have no blind-human validity reading under the current
judge. They are admitted provisionally ([validity rule](https://github.com/thetahealth/mirobody-env/blob/main/docs/anchor/VALIDITY.md)), so the board
is labelled preliminary.

**Default rule.** All ten models are ranked on the same cases of each track: every case except
those where the judge left one model's cell unresolved, which leave the board for every model
(144 of 145 cases on `ddx-timeline`, 138 of 145 on `ddx-workup`; the data card lists them). A cell
without a scorable answer scores 0 on every applicable dimension and trips no hard gate: the
deadline passed before the cell was answered, the check budget ran out, the reply was empty or
could not be parsed, the model produced reasoning and no answer, or the stream ended early.
Tiers come from a case-level bootstrap (10,000 resamples, Holm-corrected α = 0.05); a tier number
is 1 plus the count of models that are significantly better. Models in one tier cannot be told
apart at this sample size, and models in different tiers are not necessarily separable from each
other (22 of 45 pairs are separable on each track). The intervals and tiers cover case
sampling only; a model answering again and the judge's own variation are not in them. Inside the
first tier the composite ranks are ties; [where the first tier separates](#first-result) shows where
those models do separate.

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.svg"><img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_results.png" alt="Composite scores and dimension readings for ten models on the diagnosis track (ddx-timeline) and the budgeted-tool track (ddx-workup). Each panel groups all ten models into tiers on the same cases of its track, with each composite's 95% interval; unanswered cells score 0." width="100%"></a>
</p>

### ddx-timeline (preliminary)

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

### ddx-workup (preliminary)

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

*Composite* is the mean of the scored dimensions multiplied by (1 − hard-gate failure rate), from
0 to 1. *95% CI* is the case-level bootstrap interval. *Answered* counts the cells with a
scorable answer out of 145; `glm-5.3-flash` ranks low for cells it did not finish in time, not
for wrong answers ([limitations](#limitations)). Scores equal at three decimals share a rank. Each cell ran once
(k = 1); a 16-case subset ran three times per cell, which measures repeat noise per dimension
only ([data card](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#current-evaluation)). Click the figure
for the full-resolution vector image.

[Source values (CSV)](https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_results.csv) ·
[Public snapshot (JSON)](https://github.com/thetahealth/mirobody-env/blob/main/web/demo/data.json) ·
[Rebuild script](https://github.com/thetahealth/mirobody-env/blob/main/docs/scripts/make_readme_results.py)

<details>
<summary>What the scores and dimensions measure</summary>

| Dimension | Reading |
|---|---|
| Composite score | Mean of the scored dimensions times the hard-gate multiplier; ranked separately within each track |
| Diagnosis listed (`dx_listed`) | Share of gold diagnoses on the differential and not ruled out; a comorbid case scores each gold line; computed by code |
| Test selection | Per-case F1 of test precision and recall, averaged over applicable cases |
| Signal availability (`noop_ok`) | Declares data unavailable when the queried signal is absent, and does not claim it missing when present |
| Clinician review (`review_macro`) | Does not request review when none is warranted (specificity); missed referrals are caught by the two review hard gates |
| Numerical reading (`quant_ok`) | Correctness of questions about recorded trends, peak days and outlier counts, checked against code-derived gold |
| Grounded tool targets | Share of tool queries whose target is a signal the patient has; scored on `ddx-workup` only |

Hard-gate failures cannot be offset by high component scores. Every dimension cell has its own
applicable case count; tool grounding can have a much smaller denominator than the full track.
The two tracks use different tasks, so their scores are not pooled into one ranking.

```bash
uv run --with matplotlib python docs/scripts/make_readme_results.py
```

</details>

### Where the first tier separates

<a id="first-result"></a>**The leading models tie on the composite and differ in clinical behaviour.** On both
packs no pair inside the first tier is separable on the composite (`ddx-timeline`: six models,
0.652–0.724; `ddx-workup`: seven models, 0.673–0.697). Per dimension they separate: whether a model
refers to a clinician when nothing warrants it splits the first tier into groups (8 of 15 pairs on
`ddx-timeline`, 12 of 21 on `ddx-workup`). On the cases where no referral is warranted,
`deepseek-v4-pro` refrains from one on 0.000 and 0.148 of them, `gemini-3.1-pro` on 0.786 and
1.000. On `ddx-workup`, grounded tool targets (9 of 21 pairs) and test selection (8 of 21) separate
the first tier too. The data card gives the
measurement ([current evaluation](https://github.com/thetahealth/mirobody-env/blob/main/docs/DATA_CARD.md#current-evaluation)).

<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_profile_timeline.svg"><img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_profile_timeline.png" alt="ddx-timeline: ten models as columns in composite order, one row per dimension, with the first tier bracketed. On the composite row every first-tier model shares a letter; clinician review splits the first tier into letter groups. Models sharing a letter on a row are not separable (paired case bootstrap, 10,000 resamples, Holm alpha 0.05 over all 45 pairs)." width="100%"></a>
</p>
<p align="center">
  <a href="https://github.com/thetahealth/mirobody-env/blob/main/docs/figures/readme_profile_workup.svg"><img src="https://raw.githubusercontent.com/thetahealth/mirobody-env/main/docs/figures/readme_profile_workup.png" alt="ddx-workup: ten models as columns in composite order, one row per dimension, with the first tier bracketed. On the composite row every first-tier model shares a letter; clinician review, grounded tool targets and test selection split the first tier into letter groups." width="100%"></a>
</p>

### Which registry entries enter the composite

| Registry role | Entries | Where the reading appears |
|---|---|---|
| Scored dimensions | `tests_recall` and `tests_precision` (one F1 dimension), `dx_listed`, `noop_ok`, `review_macro`, `quant_ok`; `tool_target_grounded_rate` on `ddx-workup` | The composite |
| Descriptive | `tool_budget_used` (non-monotonic: zero queries also reads 0), `tool_dup_rate`, `tool_budget_thrift` | Demo and batch reports |
| Held out | `disc_recall`: its judged atom type failed the cross-protocol stability check | Not reported |
| Diagnostic and report-only | The remaining registry entries, including `dx_hit`, `dx_top1`, the join and review-stability families and the process-track `trace_*` items | Batch reports |
| Normalization anchor | `scope_anchor_unified` | Normalization only |

Steps 6–9 of the [live demo](https://thetahealth.github.io/mirobody-env/) show recorded answers, a
tool trace and the score breakdown. The demo also removes unanswered cells by failure cause and
recomputes the board in the browser.

## Limitations

- **`glm-5.3-flash` ran out of time, not out of accuracy.** It reasons far longer than the other
  models: on `ddx-timeline` an answered cell took a median of 69 minutes (next slowest 24;
  the two Gemini models record no latency). It answered 17 of 145 cells there and 116 of 145 on
  `ddx-workup`; the rest score 0 under the default rule. On `ddx-timeline`, 75 of its 128
  unanswered cells were not started or still running when the runtime limit closed the batch, and
  the other 53 are the same length problem (a reply that ended before valid output 30, a stream
  cut off 18, reasoning without a final answer 5); on `ddx-workup`, 28 of 29 were not started and
  1 ran out of budget. On the cells it did answer, it scores 0.556 on `ddx-timeline` (17 cases,
  below the 20-case minimum, descriptive) and 0.473 on `ddx-workup` (116 cases), close to
  `qwen3.7-flash` (0.590 and 0.506 on its own answered cells). Answered-only scores rest on
  different case sets and are not a ranking.
- **Single judge, also a tested model.** The semantic judge is one model (`gpt-6-luna`,
  reasoning high) with two votes and a third on disagreement. The same model and another model of
  its vendor (`gpt-6-sol`) are on both boards; self-preference is not ruled out. No second judge
  or physician labelling checks its verdicts; the vote agreement measures repeatability only.
- **One pass, no composite noise floor.** Each cell ran once. The 16-case repeat subset is below
  the 20-case minimum for a composite, so repeat noise is read per dimension only (for example
  `dx_hit` differs by up to 0.129 between repeats on `ddx-timeline`). The intervals and tiers cover
  case sampling only.
- **Frontier models tie on the composite.** No pair within the first tier of either track is
  separable; per dimension, several are (see the data card).
- **Clinical review.** Six rare-disease conditions carry `clinical_review: pending` (13 of 145
  cases use them), and the medical content as a whole has not been reviewed by a practising
  clinician.
- **Raw answers stay private.** The model answers behind the board are not in this repository,
  so the board cannot be recomputed from a public checkout.
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
  version = {1.1.0}
}
```

The same metadata is in [`CITATION.cff`](https://github.com/thetahealth/mirobody-env/blob/main/CITATION.cff).

## License and acknowledgements

Code is under the [MIT License](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE); the synthetic data and question packs are under
[CC BY 4.0](https://github.com/thetahealth/mirobody-env/blob/main/LICENSE-DATA). Third-party notices are in [`NOTICE.md`](https://github.com/thetahealth/mirobody-env/blob/main/NOTICE.md).

HAEnv is inspired by [ESL-Bench](https://arxiv.org/abs/2604.02834).

<p align="center"><sub>Synthetic data · evaluation use only · not medical advice</sub></p>
