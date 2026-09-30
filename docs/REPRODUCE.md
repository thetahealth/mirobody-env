# Reproduction guide

For anyone who has this repository and wants to run the numbers. Commands run from the
repository root; `uv` builds the environment from `pyproject.toml`, so no venv needs to be set
up first.

## 0. One minute to see that it is alive (free, no model calls)

```bash
uv run python -c 'import haenv.report; print("kernel ok")'   # (1) does the kernel mount? (seconds)
uv run haenv run inputs/example-ew.job.yaml --offline --fresh # (2) end to end
```

(1) is the cheapest possible check: `haenv/report.py` imports the L0 kernel at module level, so a
mount failure surfaces here instead of halfway through a run.

(2) `--offline` uses formula-driven generation and deterministic baseline stubs: reproducible and
free. Exit code 0 means it passed. Artifacts land in `results/<task_type>/<job_id>/<timestamp>/`
and reports in `reports/`.

## 1. Prerequisites

| What | Notes |
|---|---|
| Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/) | `uv` manages dependencies and the environment. The only runtime dependency is `pyyaml` |
| L0 kernel | `core/`, part of this repository. Its path comes from `config.yaml:kernel_path` and is resolved by `haenv.kernel_path()`. To point at another checkout: `HAENV_KERNEL_PATH=/path/to/kernel uv run haenv ...` |
| API key (model runs only) | Read as `KEY=VALUE` from `config.yaml:env_file` and injected into the subprocess environment; never logged, never written to an artifact. Point it at your own file with `HAENV_ENV_FILE=/path/to/env` |
| An OpenAI-compatible relay (optional) | Models declared with `backend: relay` need the gateway's url, set as `backends.relay.url` in a `config.local.yaml` next to `config.yaml` (merged over it, ignored by git; see the comment above `backends:` in `config.yaml`). Without it those models are skipped, and naming one with `--models` fails |

## 2. What can be reproduced for free

Stored responses can be re-judged. A change to the judges is one recompute, not a re-run of
every model:

```bash
# Re-judge a batch under the current judging code and prove it passes the publish gate
uv run python tools/restamp_batch.py results/joint_dx/ddx-timeline/<batch>

# Re-render the report (reads existing JSONL only; no regeneration, no model calls)
uv run haenv report inputs/ddx-timeline.job.yaml --batch <batch>
```

`restamp_batch` drops the stubs, recomputes from the stored responses, re-runs the stubs offline,
and then proves the batch passes the publish gate. A failed gate exits non-zero.

### The judging version is part of the citation

Changing the scoring code invalidates every stored leaderboard at once. A cited reading must
carry its `judging_sha16` and the freeze revision; see [`DATA_CARD.md`](DATA_CARD.md).

## 3. The step that costs money

```bash
uv run haenv run inputs/ddx-timeline.job.yaml --models <model> --limit 1   # one cell first
uv run haenv run inputs/ddx-timeline.job.yaml                              # full run; resumes the latest batch
uv run haenv run inputs/ddx-timeline.job.yaml --fresh                      # starts a new batch
```

* Resuming is the default: a re-run fills only the cells that have not been run.
* Results are written and flushed per cell, so a run killed mid-way keeps its completed cells.
* Do not edit the scoring code during a run. A batch holding rows from two judging versions is
  refused by the publish gate.
* A run writes into the batch directory it evaluates. `eval.jsonl`, `responses.jsonl` and
  `trace.jsonl` land next to the pack's `cases.jsonl`, including for a run against a frozen pack.
  Give an offline smoke test `--fresh` so that it builds a new batch instead of writing into the
  latest one.

### Tokens and caching

* **Generation cache.** `--gen llm` caches every model response in `cases/_llm_cache/`, keyed by
  sha256 of the model and the prompt (`haenv/llm.py`). Rebuilding a pack from an unchanged job is
  answered from the cache; `batch.json:gen_usage` then reads `absent:cache_only` with `n_calls: 0`
  and the number of hits. `--regen` forces fresh calls.
* **Answer budget.** For models with a measured basis, `max_tokens` in `config.yaml` must be at
  least the floor that `llm.recommended_budgets()` derives from their output lengths
  (`max(p99 × TAIL_FACTOR, p99 + ANSWER_RESERVE_TOKENS)`, rounded up to 4,000; the observed cap
  replaces p99 when the readings were cut off by it). A run below the floor is refused before
  any call is made. The floor reduces the chance of a budget-limited answer, which would score
  as a lower bound and may require another paid call; it cannot guarantee that no answer is
  truncated.
* **Resume and recompute.** A resumed run skips every cell that already has a verdict and
  re-sends only retryable failures. Stored responses make a judging change a recompute
  (section 2).
* **Accounting.** `batch.json:eval_usage` totals measured input and output tokens per model.
  If the provider omits usage, a cell carries `missing:no_usage`; if no response was persisted,
  it carries `missing:no_response`. Neither is filled with invented token counts. Offline
  reference solvers are listed as `absent:stub` and cost nothing.
  Generation usage likewise records an explicit status when there was no call, only cache hits,
  or no usage returned. In the leaderboard batches of `ddx-timeline` and `ddx-workup`, every
  persisted real-model response has measured usage.
  A batch whose usage is partly missing reports a measured subtotal, a lower bound on its spend.

Token scale in real runs:

| Pack | Models × cases | Input per model per case | Output per model per case |
|---|---|---|---|
| `ddx-workup` (budgeted tests) | 10 × 145 | 14,579 | 11,831 |
| `ddx-timeline` (questions at several time points) | 10 × 145 | 85,860 | 42,587 |

Output length depends mostly on the model, input length on the pack. To recompute from a batch,
first inspect `eval_usage.status` and each model's `missing` counts; the following averages cover
only measured rows and are undefined if none were measured:

```bash
uv run python -c "import json,sys; u=json.load(open(sys.argv[1]))['eval_usage']; by=u.get('by_solver',{}); m=[v for v in by.values() if v.get('n_measured')]; n=sum(v['n_measured'] for v in m); print('status:', u['status'], 'unmeasured:', {k:v['missing'] for k,v in by.items() if v.get('missing') and any(not s.startswith('absent:') for s in v['missing'])}); n or sys.exit('no measured rows'); print(len(m), 'models', n, 'rows', sum(v['in_total'] for v in m)//n, 'in/row', sum(v['out'] for v in m)//n, 'out/row')" results/joint_dx/ddx-workup/<batch>/batch.json
```

## 4. Self-checks

```bash
uv run python tools/verify_selftest.py        # item-verifier negative and positive controls
uv run python tools/leak_probe_selftest.py    # kernel leak-gate controls
```

Each feeds deliberately broken inputs and requires every one of them to be caught, plus clean
inputs that must not be flagged. Both must exit 0.

They make no model calls and do not run at installation or import. Runtime depends on the
interpreter and machine; the offline end-to-end example and the browser and packaging CI
checks are separate workloads.

These controls cover their listed inputs, not complete emission or semantic-leak coverage;
the [data card's known gaps](DATA_CARD.md#known-gaps) describes the remaining defects.

What each check shows, and what to look for:

| Check | Command | Expected |
|---|---|---|
| The item verifier catches deliberately broken streams and events | `uv run python tools/verify_selftest.py` | `68 checks, 0 missed.` (45 negative controls, 23 positive controls) |
| The kernel leak gate fires on every field it scans | `uv run python tools/leak_probe_selftest.py` | `30 checks, 0 missed.` |
| The emission gate withholds a case whose declaration does not hold and names the reason | the Quick Start in the repository README | 3 of the 4 bundled cases emitted; `EWX-04` refused with `event_density_mismatch` |
| `verifier_core/` imports nothing from the clinical layer | `grep -rnE "^\s*(from\|import) haenv" verifier_core/` | no output |
| A judging change makes stored scores unpublishable | after the Quick Start, add one line of code to any file in `haenv/judges/`, then run `uv run haenv report inputs/example-ew.job.yaml` and `uv run python tools/restamp_batch.py <batch-dir> --dry` | `report` warns that all 72 stored rows carry the old judging stamp; the publish gate refuses the batch (`NotPublishable`). Revert the line and it passes |
| A scoring fix is a recompute | `uv run python tools/recompute_judges.py <batch-dir> --full` | rows with a stored model response are re-scored under the current code and the changed fields are listed. An offline batch has no model responses and reports `0/72 rows recomputed` |
| A score difference has a noise floor | `uv run python tools/reliability_passk.py inputs/example-ew.job.yaml --k 3 --offline` | offline stubs are deterministic, so identical samples are dropped and the tool exits 1 with `only 1 comparable samples left (<3) -- no noise floor below k=3`. With `--models <name>` it reports the per-dimension floor and `pass^k` (this calls a model). It reads its source batch from, and creates its sample directories under, `results/` in the repository, while the `haenv run` calls it launches honour `HAENV_OUTPUT_ROOT`: run it with `HAENV_OUTPUT_ROOT` unset, or it finds no samples |

`verify_selftest.py` and `leak_probe_selftest.py` also run in CI on every pull request.

> Do not pipe these through `tail` or similar: a pipe can turn a crash into exit code 0;
> redirect to a file and read the file.

## 5. What cannot be reproduced from this repository

| | Why |
|---|---|
| Model responses behind the published board | The responses and judge votes are run artifacts. A board you run yourself (§3) reproduces the procedure and gives new model answers, so its scores differ from the published board by sampling noise. The published board has no repeat-noise reading of its composite; the per-dimension repeat noise is in the [data card](DATA_CARD.md#current-evaluation) |
| The real-EMR calibration round | Its derivatives are not in the repository. See [`ETHICS.md`](ETHICS.md) §2 |

## 6. Before you read a number

* [`ETHICS.md`](ETHICS.md): the data is synthetic, and exactly where real EMR entered.
* Noise floor. Before reading any difference between two scores, measure `pass^k` with
  k ≥ 3. The noise level changes sharply with the number of cells. The published board ran once
  per cell (k = 1): its tiers cover case sampling only.

**Source and wheel provenance.** Fingerprints resolve code and resources in either installation
layout using the same canonical file labels. With identical scoring/world files and registered
plugins, a source checkout and a non-editable wheel produce identical fingerprints.

*SYNTHETIC data, evaluation only, not medical advice.*
