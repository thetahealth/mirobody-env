# Reproduction guide

For anyone who has this repository and wants to run the numbers. Commands run from the
repository root; `uv` builds the environment from `pyproject.toml`, so no venv needs to be set
up first.

## 0. One minute to see that it is alive (free, no model calls)

```bash
uv run python -c 'import haenv_kernel,haenv.report; print("kernel ok")'   # (1) does the kernel mount? (seconds)
uv run haenv run benchmarks/inputs/example-ew.job.yaml --offline --fresh # (2) end to end
```

(1) imports the installed simulation kernel and supported reporting interface. An import
failure surfaces before a run; an editable install resolves the namespaces from `src/`.

(2) `--offline` uses formula-driven generation and deterministic baseline stubs: reproducible and
free. Exit code 0 means it passed. Artifacts land in
`artifacts/results/<task_type>/<job_id>/<timestamp>/` and reports in `artifacts/reports/`
under the default output policy.

## 1. Prerequisites

| What | Notes |
|---|---|
| Python ≥ 3.10 and [`uv`](https://docs.astral.sh/uv/) | `uv` manages dependencies and the environment. Runtime dependencies are declared in `pyproject.toml` |
| L0 kernel | The `haenv_kernel` simulation package, part of this distribution. The executing package is imported normally from `src/haenv_kernel/` or the installed distribution. `haenv.kernel_path()` locates source for inspection; it does not switch the executing code. |
| API key (model runs only) | Read as `KEY=VALUE` from `config.yaml:env_file` and injected into the subprocess environment; never logged, never written to an artifact. Point it at your own file with `HAENV_ENV_FILE=/path/to/env` |
| An OpenAI-compatible relay (optional) | Models declared with `backend: relay` need the gateway's url, set as `backends.relay.url` in a `config.local.yaml` next to `config.yaml` (merged over it, ignored by git; see the comment above `backends:` in `config.yaml`). Without it those models are skipped, and naming one with `--models` fails |

### Code, resources and output

Source execution, read-only resources and writable artifacts have separate locators. The
checkout default is `<checkout>/artifacts/`; the wheel default is `<cwd>/artifacts/`.
`HAENV_OUTPUT_ROOT` explicitly selects the writable root. `HAENV_DATA_ROOT` selects resources
and retains its historical combined output redirect for compatibility; prefer the output
variable for new configuration.

Fixed jobs and source data are under `benchmarks/`, with policies and references in `registry/`.
New runtime files go under the output root's `results/`, `reports/`, `cases/` and `derived/`.
Bundled benchmark jobs loaded without an explicit job root use `output_root()`; external job
files preserve their legacy root convention. A saved job or manifest path remains historical
identity, not permission to rewrite the source record.

The default kernel is the `haenv_kernel` namespace imported from the executing source or
installation, independently of resource overrides. An explicit comparison path must exist;
it never selects another path when missing and does not switch the executing implementation.

## 2. What can be reproduced for free

Saved reports and receipts can be replayed without new model calls. Explicit legacy
recomputation is a new identified measurement: retain the original artifacts before running
a command that writes new evaluation rows. It does not replace historical criterion identity.
The following legacy restamp workflow uses stored responses and deterministic judges; it is
not the current semantic-vote or management-Q pipeline:

```bash
# Re-judge a batch under the current judging code and prove it passes the publish gate
uv run python tools/evaluation/restamp_batch.py artifacts/results/joint_dx/ddx-timeline/<batch>

# Re-render the report (reads existing JSONL only; no regeneration, no model calls)
uv run haenv report benchmarks/inputs/ddx-timeline.job.yaml --batch <batch>
```

`restamp_batch` drops the stubs, recomputes from the stored responses, re-runs the stubs offline,
and then proves the batch passes the publish gate. A failed gate exits non-zero.

### Run an audited task offline

This exercises lab-change attribution with deterministic reference solvers, including
generation, the pack audit and scoring:

```bash
uv run --with scikit-learn --with joblib python tools/generation/make_pack.py --pack p4 --n 50 --out artifacts/lab-smoke
HAENV_OUTPUT_ROOT=artifacts/lab-smoke/pack uv run haenv run artifacts/lab-smoke/pack/p4-followup.job.yaml --offline
```

Keep the output root on the audited `pack` directory: otherwise `run` builds a separate
batch without the audit marker. The audit dependencies are needed only for the first command.
This reproduces the task and scoring procedure with new synthetic cases and offline answers.
Recomputing the published model scores still requires the original stored answers and judge
votes; the shipped board JSON files contain readouts, not those run artifacts.

### The judging version is part of the citation

A result records the identity of each criterion and of the complete criterion set.
Unchanged criterion columns remain comparable when another criterion changes; combined
scores also require matching aggregation rules and task weights. Legacy results retain
their `judging_sha16` and freeze revision; see [`DATA_CARD.md`](DATA_CARD.md).

### Recomputing the 1.3.1 overall board

Given the archived thirteen-model answer tree and its per-answer Q JSONL file, recompute
the current board without calling a model:

```bash
uv run python tools/evaluation/board_score.py --tree . \
  --work /path/to/answers --q /path/to/q.jsonl \
  --aggregation k3 --boot 2000 --out /path/to/readout-k3
```

Use the 1.3.1 stored answers and scores, including the refreshed diagnosis answers. The shipped
readouts are [`k3.json`](boards/1.3.1/k3.json) and [`pk1.json`](boards/1.3.1/pk1.json);
`--aggregation pk1` selects the preregistered first-answer score. Compared with k3, pk1 swaps
minimax-m3 and deepseek-v4-flash at positions 8 and 9; the remaining order is unchanged.
The answer-root layout and missing-Q handling are the same as for 1.3.0 below. The readouts
alone do not contain the stored answers or judge votes needed to recompute the board.

### Recomputing the 1.3.0 overall board

Given an archived 1.3.0 answer tree and its per-answer Q JSONL file, recompute the
main board without calling a model:

```bash
uv run python tools/evaluation/board_score.py --tree . \
  --work /path/to/answers --q /path/to/q.jsonl \
  --aggregation k3 --boot 2000 --out /path/to/readout-k3
```

Use `--aggregation pk1` for the original first-answer result. Each answer root
contains provider/pack directories with the stored `-pk1`, `-pk2` and `-pk3`
batches. Multiple roots can be supplied as a comma-separated `--work` value.
The script reads stored decision and gate results; it does not regenerate
questions, rerun agents or obtain new rubric judgments. It reports missing Q
explicitly when it prevents a score. The main result averages each question's
three answer scores and gives the four tasks equal weight.

The evaluated question sets ship as frozen pack files. Each batch records its generation commit and world settings, and regenerating at those commits reproduces all four sets byte for byte (verified 2026-10-17: ① diagnosis and ② triage at `c3cecbee`, the ③ medication candidate arm at `fcf3b389` with its pairing at `c3cecbee`, ④ lab-change attribution at `05f97181`). `Q` averages two judgments by one LLM judge: of 7,246 Q records, 7,215 average two judgments, 20 contain one, and 11 have no management plan and score zero.

### Recomputing the released re-ask results

Re-score the released diagnosis answers and recompute the pooled results offline:

```bash
python tools/evaluation/reask_pooled.py --check
```

Expected summary: `61 facts matched; 535 answers re-scored; 0 D mismatches`.
The [released data and conditions](../benchmarks/results/1.3.1/reask/README.md) include
the stored answers, inclusion list and diagnosis gold. A new `haenv reask` run selects its
own units; see the [re-asking guide](reask.md) for the input contract and arm definitions.

### Checking HTTP replay

Replay the stored leaderboard tool calls through `haenv serve fhir`, compare decoded outputs
with the function-call tools and the stored traces, and validate the returned FHIR R4B resources:

```bash
python tools/release/replay_http.py --work leaderboard-pool/ --workers 32 --out artifacts/replay-http-results.json
```

Expected summary: `questions=400`, `responses=15600`, `calls=256609`, `identical=256609`,
`differing=0`, `stored_differing=0`, `pending_purchases_replayed=234`, `validation_failures=0`;
`missing=[]`, `worker_errors=[]`. Purchases and refusals are included. The pool contains the
stored leaderboard runs and is supplied separately from the source tree. Install the `serve`
extra for FHIR validation (`pip install 'haenv[serve]'`).

An offline rehearsal uses the bundled stored-session subset, with no model calls or external pool:

```bash
python tools/release/replay_http.py --offline
```

The subset checks one question, one response and 31 calls; it does not reproduce the full-pool counts.

### Historical candidate board (2026-10-06)

This subsection retains the commands and paths of the unreleased 2026-10-06 candidate checkout. Its
board archives were packaged but never published; the commands apply to a local copy of them.

`tools/evaluation/board_readout.py` reads the stored batches of the four packs and makes no model call:

```bash
python tools/evaluation/board_readout.py --out board/ \
    --pack m2=<batch prefix> --pack pack2=<batch prefix> --pack p3=<batch prefix> --pack p4=<batch prefix>
```

A batch prefix `<dir>/<stamp>` reads the rounds `<stamp>-pk1` to `<stamp>-pk3`. The batches must pass the
publish gate (restamp them as above when the judging code has moved).

The following commands describe the candidate's archive layout, not the 1.3.0 overall-score format.
A candidate archive set holds a common bundle (README, `MANIFEST.tsv`, `rebuild.sh`, the board's job files,
the readout), one bundle per pack and a `SHA256SUMS` file. Put them in one directory, check them and unpack,
then rebuild the board with the script of the bundle, which runs the two commands below with the bundle's
batch names and compares `board.json` with the bundled one:

```bash
sha256sum -c <archive>.SHA256SUMS && for f in <archive>-*.tar.gz; do tar xzf "$f"; done
<archive>/rebuild.sh <candidate checkout> board-rebuilt      # then: sha256sum -c MANIFEST.tsv inside the bundle
```

A pack whose items were run in two batches (the candidate board: the first 50 items, then the other 50 on a
byte copy of the batch) is given as `--pack <pack>=<prefix A>+<prefix B>`, with `--items <prefix>=<job file>`
naming the items each batch owns, and the pack ① random-menu stub rows run on the same 100 items:

```bash
python tools/evaluation/board_readout.py --out <out> --m2-stub-rows <stub rows of the 100 items> \
  --pack m2=<m2 A>+<m2 B> --pack pack2=<pack2 A>+<pack2 B> --pack p3=<p3 A>+<p3 B> --pack p4=<p4 A>+<p4 B> \
  --items <m2 A>=<job of its 50 items> --items <m2 B>=<job of its 50 items> \
  --items <pack2 A>=<job of its 50 items> --items <pack2 B>=<job of its 50 items> \
  --items <p3 A>=<job of its 50 items> --items <p3 B>=<job of its 50 items> \
  --items <p4 A>=<job of its 50 items> --items <p4 B>=<job of its 50 items>
```

The readout the command writes equals the bundled one in every value (only its `inputs` paths differ). The
demo's `board.json` is rebuilt from it byte for byte with the same batches and items:

```bash
python web/demo/export_board.py --readout <out>/readout.json --results <bundle>/results \
  --batch <pack>=<job>/<A>+<job>/<B> (for each pack) --items <job>/<A>=<job of its 50 items> (for each batch) \
  --out web/demo/board.json
```

Any other board is read with one batch prefix per pack: `--pack <pack>=<batch prefix>` for each pack.

### Regenerating the board's packs from its seed

An official board draws its packs with a private seed; the jobs and batches record only the seed's sha256. The
seed is published once the board retires; the 1.3.0 board uses the public seed. Given a retired board's seed, the
bundle's `verify_seed.py` rebuilds the four packs from it with the code of the board's checkout and compares
them with the bundle:

```bash
cd <archive> && python verify_seed.py --haenv <board checkout> --seed-file <seed file>
```

It checks that the sha256 of the seed equals the hash the jobs record, builds each pack at N = 50 and N = 100
with `tools/generation/make_pack.py` (the seed reaches the child processes through the environment only and is never printed
or written), and compares the published pack jobs byte for byte and, at N = 100, every case and every payload of
the rebuilt batch with the bundle's. It exits 0 when everything is identical and needs scikit-learn and joblib for
the audit, like `make_pack.py`.

The parts' item sets must be disjoint and hold the same models and rounds; their rows are read as one pack.
The readout holds:

- **Pack scores** from the production scorers. Packs ② to ④ are chance-corrected balanced accuracies. Pack ①
  is mapped onto the same scale with the random-menu stub measured on the same items:
  `cc = (score − stub) / (1 − stub)`.
- **The overall board**: the equal-weight mean of the four pack scores, a case × round bootstrap
  (independent per-pack draws combined draw by draw), each model's rank interval (2.5th to 97.5th
  percentile of its rank) and every model pair's paired difference. A pair is separable when its
  difference exceeds 2 paired standard errors; models whose rank intervals overlap are not
  separable.
- **A sensitivity analysis.** The SD-equalized total divides each pack's chance-corrected score by
  that pack's between-model standard deviation before averaging, so a pack that separates the models more
  weighs less. On the candidate board this makes the effective pack weights 12% (①), 39% (②), 34% (③) and
  15% (④). The ten ranks are the same under both weightings. The readout writes both totals; the
  published board shows the equal-weight total.
- **Unresolved semantic fields in pack ①.** The semantic judge needs two votes, a third on disagreement,
  and at least two agreeing labels; a field without that is unresolved, and no further votes are bought when
  the board is read. The pack ① scorer uses the code value `m2_tests_precision` for F1–F3 items and 0 for F0
  items. On the candidate board 302 of the 3,000 pack ① cells have an unresolved semantic field; in 271 of them
  it is the tests precision (218 cells F1–F3, 53 cells F0). Leaving the F0 zeros out moves a model's pack ①
  score by 0.000 to 0.039 and swaps only the pack ① order of gpt-6-luna and gemini-3.7-flash. The scores of
  packs ② to ④ read no semantic field, so a field left unresolved there does not enter a score.
- **How the board was extended.** The board packs were drawn once at 100 items per pack. Pack ① was run on
  all 100 items. For packs ② to ④ the first 50 items (the nested 50-item pack) were run first, and the other
  50 items were run on a byte copy of the batch. The board reads the 100 items of a pack as one pack: the rows
  of the two parts are pooled by item, and pack ①'s random-menu stub is run on the same 100 items.
  `tools/evaluation/board_readout.py` reads the two parts as one pack (command above) and writes every model pair's
  separability (`all_pairs`, the matrix the demo page shows); `web/demo/export_board.py` builds `board.json` from
  that readout, taking the two parts as `--batch <pack>=<partA>+<partB>` with the same `--items` option.

## 3. The step that costs money

```bash
uv run haenv run benchmarks/inputs/ddx-timeline.job.yaml --models <model> --limit 1   # one cell first
uv run haenv run benchmarks/inputs/ddx-timeline.job.yaml                              # full run; resumes the latest batch
uv run haenv run benchmarks/inputs/ddx-timeline.job.yaml --fresh                      # starts a new batch
```

* Resuming is the default: a re-run fills only the cells that have not been run.
* Results are written and flushed per cell, so a run killed mid-way keeps its completed cells.
* Do not edit the scoring code during a run. A batch holding rows from two judging versions is
  refused by the publish gate.
* A run writes into the batch directory it evaluates. `eval.jsonl`, `responses.jsonl` and
  `trace.jsonl` land next to the pack's `cases.jsonl`, including for a run against a frozen pack.
  Give an offline smoke test `--fresh` so that it builds a new batch instead of writing into the
  latest one.

### Model routes and prices

The default models are reached directly: Gemini models through Google's API, Qwen models through
DashScope, and every other model through OpenRouter, pinned to one upstream with fallbacks off:
the vendor's own endpoint where the account's zero-data-retention setting allows it, otherwise a
named host serving the same weights (OpenAI models on Azure, Claude on Google Vertex, MiniMax M3 on
Novita). No third-party relay sits between the pipeline and a model.

A billed run counts every request against `--judge-budget-usd`. On `openrouter`, `google` and
`dashscope` the price comes from the provider; on a backend of your own it is the `price`
declared for each model, applied to the token counts the endpoint returns in `usage` (0 for a
free local endpoint). A model there without a `price` is refused before any request is sent.
The built-in `google` and `dashscope` prices were checked against the vendors' pages on a dated day:
a week later each run warns once per backend that they may be stale. Compare them with the pages and
set `paid.tariff_checked_on: <YYYY-MM-DD>` in `config.yaml` to record your own check and silence it.

### Tokens and caching

* **Generation cache.** `--gen llm` caches every model response in `artifacts/cases/_llm_cache/`, keyed by
  sha256 of the model and the prompt (`src/haenv/runtime/llm.py`). Rebuilding a pack from an unchanged job is
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

| Pack | Models × cases | Input per answered cell | Output per answered cell |
|---|---|---|---|
| `ddx-workup` (budgeted tests) | 10 × 145 | 14,579 | 11,831 |
| `ddx-timeline` (questions at several time points) | 10 × 145 | 85,860 | 42,587 |

Each figure is the mean over every cell with measured usage, pooled across models (1,405 cells for
`ddx-workup`, 1,323 for `ddx-timeline`), so a model that answered more cells weighs more. Output
length depends mostly on the model, input length on the pack. To recompute from a batch, first
inspect `eval_usage.status` and each model's `missing` counts; the following averages cover only
measured rows and are undefined if none were measured:

```bash
uv run python -c "import json,sys; u=json.load(open(sys.argv[1]))['eval_usage']; by=u.get('by_solver',{}); m=[v for v in by.values() if v.get('n_measured')]; n=sum(v['n_measured'] for v in m); print('status:', u['status'], 'unmeasured:', {k:v['missing'] for k,v in by.items() if v.get('missing') and any(not s.startswith('absent:') for s in v['missing'])}); n or sys.exit('no measured rows'); print(len(m), 'models', n, 'rows', sum(v['in_total'] for v in m)//n, 'in/row', sum(v['out'] for v in m)//n, 'out/row')" artifacts/results/joint_dx/ddx-workup/<batch>/batch.json
```

## 4. Self-checks

```bash
uv run python tools/testing/verify_selftest.py        # item-verifier negative and positive controls
uv run python tools/testing/leak_probe_selftest.py    # kernel leak-gate controls
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
| The item verifier catches deliberately broken streams and events | `uv run python tools/testing/verify_selftest.py` | `68 checks, 0 missed.` (45 negative controls, 23 positive controls) |
| The kernel leak gate fires on every field it scans | `uv run python tools/testing/leak_probe_selftest.py` | `30 checks, 0 missed.` |
| The emission gate withholds a case whose declaration does not hold and names the reason | the Quick Start in the repository README | 3 of the 4 bundled cases emitted; `EWX-04` refused with `event_density_mismatch` |
| `src/verifier_core/` imports nothing from the clinical layer | `grep -rnE "^\s*(from\|import) haenv" src/verifier_core/` | no output |
| A judging change makes stored scores unpublishable | after the Quick Start, add one line of code to any file in `src/haenv/judges/`, then run `uv run haenv report benchmarks/inputs/example-ew.job.yaml` and `uv run python tools/evaluation/restamp_batch.py <batch-dir> --dry` | `report` warns that all 72 stored rows carry the old judging stamp; the publish gate refuses the batch (`NotPublishable`). Revert the line and it passes |
| A scoring fix is a recompute | `uv run python tools/evaluation/recompute_judges.py <batch-dir> --full` | rows with a stored model response are re-scored under the current code and the changed fields are listed. An offline batch has no model responses and reports `0/72 rows recomputed` |
| A score difference has a noise floor | `uv run python tools/evaluation/reliability_passk.py benchmarks/inputs/example-ew.job.yaml --k 3 --offline` | offline stubs are deterministic, so identical samples are dropped and the tool exits 1 with `only 1 comparable samples left (<3) -- no noise floor below k=3`. With `--models <name>` it reports the per-dimension floor and `pass^k` (this calls a model). Historical rerun helpers require their original compatible artifact tree; a new default directory does not supply missing samples. Check the selected output root before attempting a repeat |

`verify_selftest.py` and `leak_probe_selftest.py` also run in CI on every pull request.

Disk census guards read local evaluation artifacts, which are absent on a clean clone.
Maintainers may reuse historical artifacts through the explicitly registered `results/`
symlinks; those artifacts and their measured usage belong to the original runs. Adding a
generation batch does not supply model responses or measured evaluation usage. A private
scratch surface may combine those historical artifacts with current generation batches,
with each origin recorded separately. Passing that local census does not make its data
available in a clone; missing objects remain unmeasured there.

> Do not pipe these through `tail` or similar: a pipe can turn a crash into exit code 0;
> redirect to a file and read the file.

## 5. What cannot be reproduced from this repository

| | Why |
|---|---|
| Model responses behind the published board | The responses and judge votes are run artifacts. A board you run yourself (§3) reproduces the procedure and gives new model answers, so its scores differ from the published board by sampling noise. The published board has no repeat-noise reading of its composite; the per-dimension repeat noise is in the [data card](DATA_CARD.md#1x-evaluation-historical) |
| The real-EMR calibration round | Its derivatives are not in the repository. See [`ETHICS.md`](ETHICS.md) §2 |

## 6. Before you read a number

* [`ETHICS.md`](ETHICS.md): the data is synthetic, and exactly where real EMR entered.
* Noise floor. Before reading any difference between two scores, measure `pass^k` with
  k ≥ 3. The noise level changes sharply with the number of cells. The original 1.x board ran once
  per cell (k = 1): its tiers cover case sampling only. The 1.3.0 main board averages three
  responses per question and resamples question clusters.

**Source and wheel provenance.** Fingerprints resolve code and resources in either installation
layout using the same canonical file labels. With identical scoring/world files and registered
plugins, a source checkout and a non-editable wheel produce identical fingerprints.

*SYNTHETIC data, evaluation only, not medical advice.*
