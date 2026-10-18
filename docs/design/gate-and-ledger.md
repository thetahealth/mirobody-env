# Gates, diagnostics and the shared ledger

`haenv gate` prepares a separate work directory from frozen packs or built jobs,
removes baseline solvers and runs repeated samples concurrently through the reliability
runner. Use the same command and output directory to resume. Changed questions or panel
parameters require a new work directory. Semantic judging is off unless `--semantic` is
specified. `--offline` uses reference stubs and never calls a provider.

```bash
haenv gate packs --packs p2,p4 --models model-a,model-b --n 50 --k 3 --ledger budget.json --out gate-work
haenv gate --from-work gate-work/work --out diagnostics --baseline baseline-gate.json
```

The second command is free: it reads saved eval rows, without opening a paid ledger or
writing into the source batches. Both modes write `gate.md` and `gate.json`.
G1 reports the strongest raw accuracy against 0.90. G2 requires at most half the questions
to have every model/repeat cell correct. G3 requires the strongest two main scores to have
a question-cluster paired bootstrap 95% interval excluding zero (2,000 draws). Only G2
and G3 determine PASS. Missing grid cells make the panel INCOMPLETE. ABORT is wrong;
inapplicable raw atoms stay unknown. Baseline differences are descriptive, and comparisons
report whether the model sets and repeat counts match.

A pack can expose a `ReadoutPack`, a subclass with optional `dimensions`, `base_key`,
`raw_correct` and `main_score` fields, through a `haenv.pack_readouts` entry point. The
callbacks read saved cases and production atoms. They never generate a case or change a
score. Saturation reports category-wise all-right fractions, model accuracy and history
versus ordinary cases. `make_pack` uses the same declaration to report base count, category
counts and the largest category share in `diversity.json`; concentration warnings never
block publication or modify published bytes.

```yaml
gate:
  models: [model-a, model-b]
  pack_models:
    p2: [model-a, model-b, model-c]
diversity:
  max_category_share: 0.5
budget:
  counted_backends: [openrouter]
```

These execution schemas extend the frozen settings schema in `config.py`; the frozen
Pack skeleton also stays unchanged. A missing/null budget scope counts all backends.
Scope is pinned when a ledger is created; an existing ledger changes only through
`haenv ops budget --counted-backends openrouter --ledger budget.json --reason R`.
Use `all` to restore the default. Every provider is still recorded. `haenv spend` shows
backend columns, total charges and budget-counted charges. Legacy unattributed entries
remain counted conservatively.

`haenv ops stop --ledger budget.json --wait 60` broadcasts drain, then waits for active
schedulers and settlement, including the interval after a request releases its lane.
Timeout reports remaining requests and exits nonzero without killing them. Sent requests
settle from actual receipts. Unsent queued holds are released and can reserve again;
hard-killed sent requests retain the existing conservative bound-reconciliation policy.
`ops undrain --reason R` permits resumption.

Schedulers persist each cell's start before executing it. A resumed run requeues starts
without eval results and replays saved responses through the existing accounting path.
`resume.json` lists interrupted, recovered, pending and failed cells with attempt counts.
Model-caused format, token-budget and no-answer failures remain final. Infrastructure
and authentication failures may resume under a distinct request identity, with at most
`paid.max_resends` new rounds (default 2). Each old unknown charge remains counted at its
bound; successful answer receipts are always replayed. Exhausting the resend limit records
an ABORT and its reason. An active batch lock prevents concurrent takeover.

C2ST measurements add the sample SD of the paired classifiers' permutation-null maximum.
This is a sampling-scale estimate under the null, not a fitted-model population-AUC CI.
`haenv realism new.json --baseline previous.json --thresholds thresholds.json --out comparison.json`
accepts the existing threshold mapping. A new crossing within 1.96 SD is report-only;
a crossing beyond that margin is a regression (also beyond combined arm error where
both scales are available). Existing crossings, missing baselines and unmeasured error
remain distinct. Point estimates and frozen threshold files are unchanged.


Pack and gate reports identify a public or private seed from the recorded batch seed
hash. Missing identity is reported as unknown. `board.official: true` emits a WARNING when
the pack uses the public sample seed; it does not block publication or change a gate verdict.
A successful `make_pack --publish-only` retains only the released batch under `pack/results`;
the audited pool and its provenance remain available outside that directory.

Solver price preflights share a catalog within the budget ledger's directory. The first
process fetches it under a kernel lock; others read the same complete snapshot for
`paid.catalog_cache_ttl_s` seconds (default 3600). Missing, invalid or expired snapshots
are fetched through the usual retries; a failed refresh refuses paid startup. Exact-model
price validation and tariff-expiry behavior remain unchanged.

The default judge stage and `semantic_rejudge` use the same catalog scope. Applications
calling the semantic APIs directly can wrap execution in
`haenv.runtime.solver_accounting.catalog_scope(ledger, cfg)` to share that ledger directory's cache.
The transport serves cached catalog GETs; the judge's exact-model and capability checks
continue to read the complete provider metadata.

```yaml
board:
  official: true
paid:
  max_resends: 2
  catalog_cache_ttl_s: 3600
backends:
  provider:
    proxy: direct
models:
  model-a:
    backend: provider
    model: model-a
  model-b:
    backend: provider
    model: model-b
    proxy: env
```

A model's proxy setting overrides its backend default. Backend routing fields are read
by transport before backend registration, so they do not enter the solver field whitelist.
