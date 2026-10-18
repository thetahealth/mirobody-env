# Equivalent judge routes

Management judging can distribute one sealed run over several routes to the same model
snapshot. All routes share the saved cell queue, output directory and budget ledger.
The management scorer, prompts, reasoning effort, sampling and output cap come from the
sealed policy. Routes never enter the `management_rubric` criterion fingerprint.

Configure the pool in the normal configuration layers accepted by `src/haenv/app/settings.py`
(`config.local.yaml`, `HAENV_CONFIG_OVERLAY`, or `--set`). Without `judge_routes`, the
existing management executor is used unchanged. The supported management command is
`uv run python tools/evaluation/management_rejudge.py`; board rejudging uses the same outer entry.

```yaml
judge_identity:
  canonical_model: example-judge
  snapshot: '2026-09-22'
judge_routes:
  gateway:
    backend: relay
    model: gateway-alias
    canonical_model: example-judge
    snapshot: '2026-09-22'
    base_url_env: EXAMPLE_BASE_URL
    env_file: /path/to/gateway.env
    key_env_pool: EXAMPLE_KEY_
    proxy: none
    max_tokens_field: max_completion_tokens
    concurrency: 6
    price:
      input_per_million_usd: 1.0
      output_per_million_usd: 4.0
      source: 'operator-verified gateway tariff'
      tariff_checked_on: '2026-10-15'
  router:
    backend: openrouter
    model: vendor/example-judge
    canonical_model: example-judge
    snapshot: '2026-09-22'
    env_file: /path/to/router.env
    key_env: EXAMPLE_ROUTER_KEY
    proxy: http://proxy.example:15001
    concurrency: 4
paid:
  global_request_limit: 10
  tariff_expiry: refuse
```

Prices in this example are synthetic. Set the gateway's actual tariff before using it.
An explicit `url` is the complete chat-completions endpoint. `base_url_env` instead
reads a base URL and appends `/v1/chat/completions` (or `/chat/completions` when the base
already ends in `/v1`). Backend defaults may supply the URL and credential variable names.
Route `env_file` values override the common `env_file` and process environment. A pool
collects nonempty variables with the prefix and a numeric suffix; a single key is optional.
Keys rotate between accounted attempts, never through an unrecorded retry.

`proxy` accepts `env`, `none` (`direct` is also accepted), or an HTTP(S) URL. Explicit
URLs cannot contain credentials. Proxy selection is per request; concurrent routes do not
change process proxy variables. Catalog reads use the same proxy and credentials as sends.

Every route must declare the same `canonical_model` and `snapshot`. `judge_identity` is
an optional explicit anchor for those declarations; when absent, their common declaration
is used. The canonical name must match the sealed policy model after its vendor prefix is
removed. A dated sealed model must also match the declared date. Legacy sealed aliases
carry no date: their snapshot declaration is recorded in `route-execution.json` and cannot
change on resume. The declaration must come from verified provider snapshot evidence.

Snapshot evidence depends on the backend. OpenRouter preflight reads the free model
and endpoint catalogs through the route's proxy. The exact requested model's
`canonical_slug` must match the declared canonical name and snapshot. Either catalog
may supply the slug; if both supply one, both must agree. Missing, undated or conflicting
catalog evidence rejects the entire route before any paid dispatch; other routes continue.
Each response must then name the same canonical model. An undated response alias is
accepted only with that verified catalog evidence; an explicitly contradictory response
date is rejected. Every attempt records `snapshot_evidence: catalog`, the canonical slug
and its catalog paths, including in the saved journal used for receipt replay.

Other backends check the response's `model` (Google: `modelVersion`) against the declared
canonical model and date, and record `snapshot_evidence: response`. Only the vendor prefix
and the two date spellings `YYYY-MM-DD` / `YYYYMMDD` are normalized. A missing model,
a different model, or a different response date makes that cell unscored with
`SnapshotMismatch`, and disables that route for the run. Its response and charge remain
in the receipts. A different upstream alias never establishes equivalence on its own.
Result reconstruction rechecks the recorded evidence without querying a live catalog;
already completed cells are never automatically rejudged.

The existing bounded judge scheduler runs at most the sum of route concurrency limits,
`paid.global_request_limit`, and 64 cells concurrently. An explicit `--concurrency` lowers
that ceiling. Each request also holds a ledger-wide global paid lane, including when the
configuration declares separate backend pools. An available route takes the next request.
401/403 or exhausted credentials disable a route; other failures back off exponentially
from `failure_backoff_s` (default 2 seconds) up to `max_backoff_s` (default 60 seconds).
Another route may acquire the failed request with a distinct receipt ID. A cell is never
scheduled twice. Completed results and cached attempts are reused on resume; changing
routes does not change the sealed request or already measured score. If all preflights
fail, no paid calls start. A budget exhaustion stops new work and drains in-flight work.

OpenRouter reserves against the maximum verified model/endpoint catalog prices and
settles from `usage.cost`. Relay routes require a configured tariff, source and
`tariff_checked_on`. The check has the native tariff's seven-day validity: on day seven,
`paid.tariff_expiry: refuse` blocks new sends; `warn` logs and continues. Future dates
are refused. Relay settlement multiplies prompt, completion and reasoning usage by the
configured USD-per-million rates. Completion counts include reasoning by default, so
reasoning is never charged twice. A gateway that reports completion excluding reasoning
sets `completion_includes_reasoning: false`; an optional `reasoning_per_million_usd`
overrides its output rate. The output cap must bound completion including reasoning.
Optional `cached_input_per_million_usd` and `cache_creation_per_million_usd` price the
corresponding counts in `prompt_tokens_details`; absent rates use the input rate. Counts
are inclusive of the prompt, so they are subtracted before the uncached input is priced.
`prompt_tiers` can declare increasing `prompt_tokens_above` thresholds, each with its
own input/output rates and optional cache/reasoning rates. The exact prompt count selects
the settlement tier; the reservation uses the maximum rate over every tier.

Missing or invalid cost/usage keeps the reservation's upper bound in the same ledger and
warns. A valid answer can still be scored; an absent answer cannot. Unknown charges never
turn into a fabricated zero and never stop the run by themselves. Every raw sample and
cell result lists actual route attempts, returned snapshot, request IDs and failure class.
Unmodified provider responses and billing evidence remain in `receipts/`.

`haenv routes check` validates configuration, endpoint availability in the environment,
credential presence and declared tariff dates entirely offline; it prints only key masks.
It reports live catalog pricing as unchecked. It sends nothing unless `--paid` is explicit:

```bash
HAENV_CONFIG_OVERLAY=/path/to/routes.yaml uv run haenv routes check
HAENV_CONFIG_OVERLAY=/path/to/routes.yaml uv run haenv routes check --paid \
  --ledger /path/to/check-budget.json --limit-usd 0.20 --out /path/to/check
```

The paid form makes one tiny completion per route, checks the returned snapshot and
keeps durable receipts. It does not establish grading agreement: compare a fixed saved
cohort with separate route-only overlays and fresh run IDs to measure that.

```bash
uv run haenv routes compare --left /path/to/gateway-run --right /path/to/router-run \
  --key-file /path/to/selected-keys.json --out /path/to/comparison.json
```

Comparison reads individual completed results, checks that saved inputs and judge prompt
bytes match, and reports exact agreement and mean absolute difference for both the capped
rubric score and quality Q, along with verdict
label agreement. Missing/unscored cells are listed explicitly. It never sends requests or
changes either source run, and applies no invented acceptance threshold.
