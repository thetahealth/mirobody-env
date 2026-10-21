# Contributing to HAEnv

## 1. Getting it running

```bash
git clone <repo> mirobody-env && cd mirobody-env
uv run haenv --help                 # uv sets up the environment on first run; the only runtime dependency is pyyaml
```

Four commands run the whole pipeline (zero model calls, no cost):

```bash
uv run haenv build  benchmarks/inputs/example-ew.job.yaml --gen deterministic --fresh   # generate the cases in a new batch
uv run haenv verify benchmarks/inputs/example-ew.job.yaml --gen deterministic   # check every item
uv run haenv run    benchmarks/inputs/example-ew.job.yaml --offline             # answer with the offline stub
uv run haenv report benchmarks/inputs/example-ew.job.yaml                       # score and render the report
```

Two self-checks feed every check a deliberately broken input and require it to be caught:

```bash
uv run python tools/testing/verify_selftest.py
uv run python tools/testing/leak_probe_selftest.py
```

CI runs these on Python 3.10 and 3.12, plus the tests below, a wheel install into a clean environment and
the five example plugin packages under `examples/`, each with its negative control
([`examples/README.md`](examples/README.md)).

The checks above validate product behavior; installing or importing HAEnv does not run them.
The separate Pages workflow checks the demo in a browser and does not run during package use.

### Tests

Contracts live in `tests/<component>/`, mirroring `src/haenv/<component>/`;
release and paper-script contracts live in `tests/release/` and `tests/paper/`.
Add a regression row with a stable ID to the table beside its contract and
parametrize the invariant over it. Component fixtures live beside contracts; only shared helpers stay at the test root.
World artifacts use the frozen world identity and their audited inputs as cache
keys; readers take private copies. Ordinary test edits and commits reuse them.
Use the component gate while working, the public surface for code changes,
and the full default suite once before a push:
```bash
export UV_NO_SYNC=1 UV_FROZEN=1 PYTHONPATH=src:.:plugins
.venv/bin/python tools/dev/affected_tests.py --run src/haenv/p2/rules.py
.venv/bin/python tools/dev/affected_tests.py --public --run
.venv/bin/python -m pytest tests -n 64 --ff --durations=30
```

---

## 2. Two rules this code base follows

**Skipped is not the same as passed.** A check that finds nothing to judge never turns green silently.
It either fails or reports a visible "unmeasured" line.

**Every check gets a negative control.** A check that is only shown to fire when it should, and never
shown to stay quiet when it should, has not been tested. When you add a check, add both halves.

---

## 3. After you change scoring or generation code

`registry/anchors/freeze-2026-09-02.json` records which version of the code the frozen question packs were
produced under. Files in `haenv/` and parts of `registry/` fall into two segments (`generation` and
`judging`); changing either one moves that segment's fingerprint. Report which behavior changed
and rerun the relevant self-checks or offline example.

When intentionally recording a new source version, the snapshot can be refreshed without
evaluation artifacts:

```bash
uv run python tools/release/make_freeze.py --revision <current+1> --segments-only \
  --supersedes <current> --why "one sentence describing what you changed"
```

Current revision:
`python -c "import json,pathlib;print(json.loads(pathlib.Path('registry/anchors/freeze-2026-09-02.json').read_text())['revision'])"`

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
at most. `type` is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`; CI, packaging and
release changes are `chore` (`chore(ci): …`, `chore(release): …`). `scope` names the directory or
component touched (`cli`, `judge`, `demo`, `readme`, `ci`). The commit body carries the *why* — the
diff already carries the what.

Changes to anything a reader of the published scores relies on — `README.md`, `docs/DATA_CARD.md`,
`docs/REPRODUCE.md`, the leaderboard data, the frozen packs — name that surface in their scope, so
that "which commit moved a published reading" stays answerable later.

### How releases reach `main`

A release reaches `main` as a short series of commits, one per change and each titled as above.
The last one sets the version and the changelog; the release tag points at it, the commit the
package is built from. The history of `main` is not rewritten, since downstream projects rebase
onto it. Work in progress does not land on `main`.

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
  1. every answer-bearing file carries the canary string ([`CANARY.md`](CANARY.md)): hand-written
     YAML and Markdown as a comment block, and every row of the published question packs
     (`frozen/*.Q.jsonl`) plus every batch file a run writes as a top-level `_canary` field. The
     canary string marks this pack's text so its presence in a corpus can be detected; training
     on the public sample is permitted; the official board uses private seeds;
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
