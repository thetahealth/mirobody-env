# Concurrency caps

Repeated gates, provider budget scopes and graceful stop are described in
[Gates and the shared ledger](gate-and-ledger.md).

A model's in-flight cells are bounded by four ceilings. Several models on one backend
share its cell cap; every process on one budget ledger shares its paid request pools.

| Ceiling | Configuration | Scope |
|---|---|---|
| Backend cells | `backend_limits.<backend>` | one process, all models on that backend |
| Model cells | `eval.model_caps.<model>`; absent means no separate cap | one process |
| Pool threads | `eval.elastic_max` or `eval.model_workers.<model>`; `workers` for fixed pools | one process |
| Paid solver lanes | `paid.backend_request_limits.<backend>`, otherwise `paid.global_request_limit`, minus `paid.judge_reserved_lanes` | every process sharing a ledger |

`workers=1` is serial. Otherwise each model's effective cap is the minimum of these
ceilings, with fewer threads when fewer cells remain. Backends without their own paid
pool share the global pool. Live ledger capacity (`haenv ops`) takes precedence over
configured paid capacity; the startup table reads the actual ledger pools.

Set `proxy: direct` on `models.<name>` to bypass environment proxies for that model's
solver, judge and probe requests, even when other models use the same host through the
proxy. The direct opener retains connection pacing and connect-phase retries. An omitted
model setting inherits `backends.<backend>.proxy`, which defaults to `env`; an explicit
model setting wins. These routing fields also work in deployment configuration, overlays
and `--set`. Before raising concurrency, probe the model with `proxy: direct` in a separate
output directory to confirm that the endpoint is reachable and accepts the requests from
your location. `probe.json`, `probe.md` and request receipts record the selected route;
startup logs list the direct and env model groups. Balance queries use backend defaults,
or the selected model's route with `haenv spend --balances --model <name>`.

Concurrency-only validation lives in `src/haenv/app/config.py`, extending the existing schema;
`settings.py` is a whole-file dependency of pack WORLD stamps.

Measure and keep one deployment profile outside the Git checkout:

```bash
haenv probe concurrency --ledger budget.json --limit-usd 5 --out "$HOME/.config/haenv"
export HAENV_DEPLOY_CONFIG="$HOME/.config/haenv/concurrency.yaml"
haenv run job.yaml --judge-budget-ledger budget.json
```

**The probe is paid.** It ramps tiny requests through each model's actual route and stops
on excessive rejection rate or latency growth. `probe.md` explains the stopping step;
`probe.json` preserves the measurements. `concurrency.yaml` supplies all four ceilings,
including judge reserves, so neither backend defaults nor pool defaults silently bind.
Use the same absolute `HAENV_DEPLOY_CONFIG` file for all runs on that deployment.
Tiny prompts do not measure the token-per-minute limit of long prompts. Transport failures
and timeouts can indicate a network-path ceiling; repeat through a different path to check.

## Provider quota or network path

| Signal | Provider quota | Network path |
|---|---|---|
| Rejections | HTTP 429, rate-limit bodies, provider 5xx | resets, closed connections, transport timeouts |
| Onset | often a different concurrency step per model | often a similar step across models sharing a path |
| `mean_in_flight` | follows requested concurrency until refusals begin | can plateau below requested concurrency |
| Direct-route control | failures persist through another path | failures disappear with `proxy: direct` or another proxy |

Probe the same model into separate output directories with only the route changed. A
ceiling that moves with the path belongs to that path. Tiny probes measure concurrency
and request rate; long board prompts may reach a token-per-minute limit sooner. A model
cap is per process, so repeated runs can jointly demand more capacity than one model's cap.

## Deployment and experiments

Merge order is `config.yaml → config.local.yaml` (legacy endpoints/keys)
`→ HAENV_DEPLOY_CONFIG → HAENV_CONFIG_OVERLAY → --set`. Move concurrency values out of
legacy local files and experiment overlays into the deployment profile. Legacy local
concurrency copies are rejected. An overlay or `--set` that
sets `backend_limits`, `eval.model_caps`, `eval.elastic_max`, `eval.model_workers`, or
`paid.backend_request_limits` / `paid.global_request_limit` is rejected. A justified
experiment may **only lower** concurrency and must give a nonempty top-level
`concurrency_override_reason`; raising concurrency belongs in the deployment profile.
An absent model cap is bounded by its backend/pool when checking such a reduction.
Adding a backend paid pool belongs in the deployment layer, since it changes global
sharing. An experiment reduction still limits its own cells when live capacity is higher.
Layer paths and SHA-256 hashes appear in `config_runs.jsonl`; deployment caps enter no
judging fingerprint.

At startup the **生效并发表** prints backend and model rows: the four ceilings, effective
cap, actual threads, binding ceiling(s), and live process demand versus paid solver lanes.
Identical processes read as `N×threads=demand`; differing process profiles are summed.
A backend row aggregates its models; a global pool aggregates all backends using it.
`通道将成为瓶颈：N 个单元争 M 条通道` warns that their combined demand exceeds that pool.
Only live registered solver processes count; exited or killed processes drop out through
kernel locks. Older runners that do not register are not counted; restart them when migrating.

`concurrency.json` records the startup table and every cell's slot-wait/wall-clock ratio,
with backend totals. Waiting includes the paid lane queue and the reserve-ahead gate,
but excludes provider service time and scheduler admission queues. A ratio above
`paid.starvation_threshold` (default `0.3`) for `paid.starvation_units` consecutive completed
cells on one backend (default `5`) emits **one WARNING per backend per run**, also recorded
in the live receipt. A shorter cell below the threshold resets the streak. Inspect these
ratios when calls appear slow despite available model threads.

For paid runs and reliability repetitions, **omit `--judge-budget-usd`** to reuse the
ledger's pinned dollar cap, which is printed before launch. Supply it only to create a
new ledger or assert the same cap; a differing explicit value is refused. Change a live
budget through the ledger's operator command, then resume without a copied dollar cap.
