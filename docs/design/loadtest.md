# Rehearsing a paid run offline (`haenv loadtest`)

A rehearsal runs the real run commands — `tools/evaluation/reliability_passk.py`, `haenv run`, the judges,
the shared ledger, receipts, retries, resume — against a fake provider on localhost. The fake
provider answers as OpenRouter, Google and DashScope with recorded answers and measured service
times, so a whole board runs for free in a fraction of its real time, and the speed report says
where the time goes before any money is spent.

## Quick start

```bash
# 1. Measure a provider profile and a replay library from finished batches (free, local).
uv run haenv loadtest profile \
    --batches <batch dir>... \
    --service-batches <service batch dir>... \
    --generation-samples generation_samples.json \
    --out board.profile.yaml --library board.library.jsonl

# 2. Rehearse the board at 1x, then at 2x, then with a kill -9 at 40 % of the answers.
uv run haenv loadtest run board.spec.yaml
uv run haenv loadtest run board.spec.yaml --scale 2
uv run haenv loadtest run board.spec.yaml --kill-at 0.4

# 3. Re-read a finished rehearsal.
uv run haenv loadtest report runs/<stamp>-x1
```

The tree must be committed (paid judging, rehearsed too, seals the code it runs). Nothing leaves
the machine: the run gets `transport.rehearsal_url` (every HTTPS request goes to the fake
provider), a key file of dummy keys, its own output root and its own ledger.

## The rehearsal spec

```yaml
profile: board.profile.yaml        # from `haenv loadtest profile`
library: board.library.jsonl
time_scale: 0.05                   # every service time x 0.05 (a 9 h board in about 30 min)
k: 3                               # repeats per job; --scale multiplies k and workers
workers: 24                        # lanes per job, as in the board command
budget_usd: 300                    # the rehearsal ledger's limit
models: [gemini-3.7-flash, gpt-6-sol]   # default: config.default_models
config:                            # merged into the run's configuration (an overlay)
  backend_limits: {openrouter: 48, google: 16, dashscope: 16}
probes:                            # where the report's timings come from
  slot: ["haenv.runtime.slots:request_slot"]
  ledger: ["haenv.scoring.semantic.budget:BudgetLedger.reserve", "haenv.scoring.semantic.budget:BudgetLedger.settle"]
  send: ["haenv.runtime.solvers:OpenAICompatSolver._send_wire", "haenv.runtime.solvers:GoogleSolver._send_wire"]
  cell: ["haenv.scoring.rows:_row_single@1", "haenv.scoring.rows:_row_gated@1"]
acceptance: {overhead_share: 0.05, ledger_p99_s: 0.05, judge_lag_share: 0.10}
max_wall_s: 7200                   # optional: cut the run after this long; the report marks it incomplete
burst_s: 300                       # optional: seconds counted as the start-up burst (ledger latency reported apart)
today: 2026-10-05                  # optional: the day tariff checks see (a profile older than a tariff's validity still rehearses)
jobs:
  - {job: benchmarks/inputs/m2-pack1.job.yaml, batch: <batch dir>}
```

Each job's built batch is copied into the rehearsal's output root and sampled `k` times, exactly
as the board does. Probes name functions as `module:Qualified.name`; a call is timed, and a call
that returns a context manager (a slot) is timed until it is entered. Calls made under the judge
transport are tagged `judge`.

## The profile

`board.profile.yaml` is generated and should not be edited by hand; its header holds the command
that rebuilt it. Per model, as named on the wire:

| field | meaning | measured from |
|---|---|---|
| `service_s.<kind>` | service-time samples (s) per answer kind (`judge`, a job id, `*`) | uncontended runs (`--service-batches`: receipt lock → receipt), OpenRouter `generation_time`, judge receipts |
| `errors.rate_limited` | per-attempt chance of HTTP 429 before generation | failed attempts in `responses.jsonl` / `gated_rounds.jsonl` |
| `errors.extra_data` | chance of a second JSON object after the answer | the same |
| `errors.disconnect` | chance the connection closes after the service time | the same |
| `served_by` | the upstream named in OpenRouter responses | solver receipts |

`max_concurrency: {backend: N}` makes the fake answer 429 to requests beyond N in flight.
`catalog` holds the OpenRouter model and endpoint records and the Google model limits the client
verifies before a paid run. `--service-like WIRE=OTHER*F` gives a model with no uncontended
samples another model's times multiplied by F. Client-side latencies of a loaded run are never
used as service times: they include the wait for a paid slot.

The library holds recorded answers, grouped by answer kind and the top-level keys of their JSON.
A solver request gets an answer from the group whose keys its prompt names most completely; ties
(the query rounds and the final answer of a gated protocol) are drawn in their recorded
proportions. A judge request gets an instance of its JSON schema.

## Reading the speed report

```
wall 1730.2 s (time scale 0.05) · cells 6000/6000 · last answer 1690 s · last vote 1702 s · judge lag 12 s (0.7%)
solver cells (6000): slot wait 0.41 · provider 0.55 · local overhead 0.04
ledger p99 3.2 ms · reserve n=… p50=… p99=… max=…
```

| line | meaning | acceptance |
|---|---|---|
| `wall`, `cells` | driver wall time; answered cells against `jobs × models × k × items` | all cells answered |
| `judge lag` | last judge vote − last solver answer, as a share of the answering time | ≤ 10 % |
| `slot wait` | share of solver cell time spent waiting for a paid slot | (diagnostic) |
| `provider` | share spent inside the provider call | (diagnostic) |
| `local overhead` | the rest: building, grading, writing, accounting; also given at real time (`overhead_share_real`) | ≤ 5 % at real time |
| `backend load` | mean requests in flight per backend over the run, against its configured limit; far below the limit while cells wait for slots means the wait is not capacity | (diagnostic) |
| `ledger p99` | latency of ledger operations, all processes, in all and, apart, in the first `burst_s` seconds and after them (`ledger_p99_steady` is checked against the same limit); the report also gives the p99 of the calling thread's CPU time inside them, so a gap between the two is waiting (write lock, interpreter lock), not ledger work | < 50 ms |
| `no_crash` | every run process exits 0, or 6 when semantic judging left cells unresolved (the report's `judging` line gives their states); no rehearsed model's cell ends in an error | — |
| `no_budget_trip` | no budget pause or refusal at a limit the run never approaches | — |
| `resume` (with `--kill-at`) | cells answered before the kill and missing after (`lost`), receipts written after the kill for cells already answered (`rebought`), generations delivered before the kill that no receipt records | all 0 |

Shares are measured at the rehearsal's time scale. Provider and slot time shrink with
`time_scale` while local work does not, so the report also gives the overhead share at real
time: local overhead ÷ (local overhead + (slot wait + provider) ÷ `time_scale`).

## Running the fake provider by hand

```bash
uv run haenv loadtest serve --profile board.profile.yaml --library board.library.jsonl \
    --time-scale 0.05 --port 18731 --log requests.jsonl
uv run haenv run x.job.yaml --set transport.rehearsal_url=http://127.0.0.1:18731 ...
```

Every request is one line in the request log: arrival and end time, backend, model, answer kind,
outcome (`ok`, `extra_data`, `rate_limited`, `disconnect`, `client_gone`), service time and
generation id.
