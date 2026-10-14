# Proxy routes, price preflight and recovery

Set `proxy` to `env` to use the environment's HTTP proxy settings, or `direct` to bypass
proxies. A model's setting overrides its backend's default; an unset route uses `env`.
Place deployment settings in the untracked `config.local.yaml`, or use an overlay selected
by `HAENV_CONFIG_OVERLAY`:

```yaml
backends:
  openrouter:
    proxy: env
models:
  sol:
    proxy: direct
```

Use the key of a model already present in your configuration. A proxy setting changes the
route recorded in request receipts without changing the accounted answer settings. Solver
requests, semantic judge requests and their price and billing lookups use the same selector.

Free model-catalog reads retry transient failures with bounded backoff. Runs whose budget
ledgers share a directory also share an atomically written, locked `price-catalogs` cache.
`paid.catalog_cache_ttl_s` sets its lifetime in seconds (default 3600; 0 disables reuse).
An expired or invalid snapshot triggers a new read; a failed read prevents paid dispatch.

An online run requires `--judge-budget-ledger`. A new ledger also requires an approved
`--judge-budget-usd`; an existing ledger supplies its own limit when that flag is omitted.
An explicit limit must match the existing ledger. Change a ledger limit through the operator
command, which records the change:

```sh
haenv ops budget set 10 --ledger budget.json --reason "approved budget change"
```

Resuming a cell after an infrastructure failure uses a distinct request identity, with at
most `paid.max_resends` additional rounds per original request (default 2). Earlier unknown
charges remain counted at their bounds. A saved successful answer replays without a new
send; a model's format or answer failure keeps the existing terminal policy. Exhausting the
resend limit preserves the failed cell instead of issuing more requests.

A batch records cell starts in `cells.jsonl`. Resume prioritizes interrupted cells and writes
`resume.json`, including recovered cells, attempts, remaining interrupted cells and retained
failures. Completed result rows remain authoritative. A second scheduler cannot take over a
batch while its current scheduler holds the recovery lock.

Runtime identity checks still apply. A paid batch created under older runtime code can be
refused on upgrade; use a fresh batch when its recorded identity does not permit the change.
The scoring rules, world data and frozen question packs keep their existing fingerprints.

The existing graceful drain command blocks new paid requests and waits for held request
slots. It does not terminate in-flight requests:

```sh
haenv ops drain --ledger budget.json --reason "maintenance"
```
