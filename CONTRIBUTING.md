# Contributing to HAEnv

## 1. Getting it running

```bash
git clone <repo> mirobody-env && cd mirobody-env
uv run haenv --help                 # uv sets up the environment on first run; the only runtime dependency is pyyaml
```

Four commands run the whole pipeline (zero model calls, no cost):

```bash
uv run haenv build  inputs/example-ew.job.yaml --gen deterministic --fresh   # generate the cases in a new batch
uv run haenv verify inputs/example-ew.job.yaml --gen deterministic   # check every item
uv run haenv run    inputs/example-ew.job.yaml --offline             # answer with the offline stub
uv run haenv report inputs/example-ew.job.yaml                       # score and render the report
```

Two self-checks feed every check a deliberately broken input and require it to be caught:

```bash
uv run python tools/verify_selftest.py
uv run python tools/leak_probe_selftest.py
```

CI runs these on Python 3.10 and 3.12, plus the tests below, a wheel install into a clean environment and
the five example plugin packages under `examples/`, each with its negative control
([`examples/README.md`](examples/README.md)).

The checks above validate product behavior; installing or importing HAEnv does not run them.
The separate Pages workflow checks the demo in a browser and does not run during package use.

### Tests

`tests/` holds the unit tests and the world-layer contract tests (`tests/contracts/`). They make
no model calls and need no API key and no evaluation results. CI runs both tiers on Python 3.10
and 3.12:

```bash
uv run --group test pytest -n auto --dist worksteal   # unit tests, in parallel
uv run --group test pytest --fast -n auto             # the edit-test loop: skips `integration` tests
uv run --group test pytest -m world                   # world-layer contract tests
```

The `test` dependency group installs pytest and the two example plugin packages the plugin tests
load (`examples/plugin_demo`, `examples/llm_judge_demo`). The wheel test builds with `uv build`
and reports a skip when `uv` is not on `PATH`. The maintainers' governance checks (guard budgets,
commit and style rules, runtime ledgers, the export gate) are not part of this repository.

---

## 2. Two rules this code base follows

**Skipped is not the same as passed.** A check that finds nothing to judge never turns green silently.
It either fails or reports a visible "unmeasured" line.

**Every check gets a negative control.** A check that is only shown to fire when it should, and never
shown to stay quiet when it should, has not been tested. When you add a check, add both halves.

---

## 3. After you change scoring or generation code

`docs/anchor/freeze-2026-09-02.json` records which version of the code the frozen question packs were
produced under. Files in `haenv/` and parts of `registry/` fall into two segments (`generation` and
`judging`); changing either one moves that segment's fingerprint. Report which behavior changed
and rerun the relevant self-checks or offline example.

When intentionally recording a new source version, the snapshot can be refreshed without
evaluation artifacts:

```bash
uv run python tools/make_freeze.py --revision <current+1> --segments-only \
  --supersedes <current> --why "one sentence describing what you changed"
```

Current revision:
`python -c "import json,pathlib;print(json.loads(pathlib.Path('docs/anchor/freeze-2026-09-02.json').read_text())['revision'])"`

It only recomputes the source fingerprints; the question packs do not change by one byte.

> Re-anchoring cannot make a stale leaderboard publishable. The publish gate
> (`analytics.assert_publishable`) compares each score row's judging fingerprint with the current code,
> independently of this snapshot.

Comments and docstrings do not enter the fingerprint; changing only a comment needs no re-anchor.

---

## 4. Commits and pull requests

### One commit, one thing

A commit should be describable in one sentence that names what changed. If the sentence needs an
"and", it is two commits. Mechanical sweeps (formatting, renaming, regenerated figures) go in their
own commit so they never sit between a semantic change and its evidence.

Titles follow `type(scope): what changed`, imperative, lower case, no trailing period, 72 characters
at most. `type` is one of `feat`, `fix`, `docs`, `ci`, `build`, `refactor`, `test`, `chore`; `scope`
names the directory or component touched (`build`, `cli`, `judge`, `demo`, `readme`). The commit body
carries the *why* — the diff already carries the what.

Changes to anything a reader of the published scores relies on — `README.md`, `docs/DATA_CARD.md`,
`docs/REPRODUCE.md`, the leaderboard data, the frozen packs — name that surface in their scope, so
that "which commit moved a published reading" stays answerable later.

### How releases reach `main`

Each release reaches `main` as one self-contained commit with a written summary, and the history
of `main` is not rewritten: a release commit is what a downstream project rebases onto, and a
published tag points at the commit its package was built from. Work in progress does not land on
`main`.

### Reviews

* A change should carry its reading: what changed, what was measured, and which control pins it down.
* Run it yourself before citing a number. If a documented number does not reproduce, open a
  number-dispute issue — correcting it is the expected outcome.
* Comments and docstrings are in English, without emoji; quoting a Chinese data value is fine.

---

## 5. Data and licensing

* All data is synthetic; no record describes a real patient. `docs/ETHICS.md` states that it must
  not be used for clinical decisions.
* Code is MIT (`LICENSE`); data is CC BY 4.0 (`LICENSE-DATA`).
* The gold standard is published alongside the questions. Contamination is handled in two ways:
  1. every answer-bearing file carries the canary strings ([`CANARY.md`](CANARY.md)): hand-written
     YAML and Markdown as a comment block, and every row of the published question packs
     (`frozen/*.Q.jsonl`) plus every batch file a run writes as a top-level `_canary` field. A corpus
     that filters for them can remove the benchmark from training data, and contamination remains
     detectable afterwards;
  2. regenerating the pool changes the patient population: a new `case_id` yields a new synthetic
     patient and a new gold standard.

  A canary makes contamination visible; a holdout is what makes it worthless. See `docs/DATA_CARD.md`.

---

## 6. Where to start reading

| what you want to know | read |
|---|---|
| what the data is, and where its gaps are | [`docs/DATA_CARD.md`](docs/DATA_CARD.md) |
| how to reproduce the numbers | [`docs/REPRODUCE.md`](docs/REPRODUCE.md) |
| whether it can be used clinically | [`docs/ETHICS.md`](docs/ETHICS.md) (no) |
| how to add a data stream | the header of [`registry/streams.yaml`](registry/streams.yaml): one entry there plus a dossier in `registry/indicators.yaml` |

*SYNTHETIC data, for evaluation only, not medical advice.*
