---
name: haenv-bench
description: Run models against an existing question pack and produce a board: offline smoke tests, real models, resumable runs, per-batch archiving, how to read cost, how to read the artifacts (eval/responses/**trace**/cases/batch), and why changing a judge means recomputing, not re-running. Triggers when the user says things like "run an eval batch", "produce a report", "backfill a model", "how much did this batch cost", "show a model's tool-call trace".
argument-hint: "[job.yaml] [--offline|--models A,B|--batch <timestamp>]"
---

# haenv-bench — run a question pack against models: evaluation · resumable runs · cost · boards

**One line**: the question pack is already on disk; this skill covers how to turn it into a board, and how
to read the artifacts once it's done.

Companion skills: `haenv-synth` (generating question packs and data) · `haenv-extend` (adding your own judge).

Warning: this file is an index, not the sole source of truth. The full reproduction steps are in
`docs/REPRODUCE.md`; what each judge measures is in `docs/design/judges-explained.md`. Every command here has actually
been run.

---

## 1. Running it

```bash
uv run haenv run inputs/example-ew.job.yaml --offline        # smoke test: only offline deterministic stubs, zero cost
uv run haenv run inputs/example-ew.job.yaml --models A --limit 1   # real model, verify one cell first
uv run haenv run inputs/example-ew.job.yaml                  # full run (config.default_models x every case)
```

Don't leave a long run attached to your session:

```bash
nohup uv run haenv run inputs/example-ew.job.yaml > /tmp/haenv.out 2>&1 &
```

**Resumable runs are the default** -- re-running only backfills the (case x model) cells that haven't run yet.
`--fresh` starts a new batch (the old batch is neither deleted nor overwritten) · `--batch <stamp>` resumes or
regenerates the report for a given batch · `--models A,B` backfills a different model list in stages ·
`--limit N` samples a subset for a quick look (**never overwrites the batch's cases.jsonl**).

## 2. Artifacts (`<B>` = the batch timestamp, also printed at the top of the report)

| file | what it is |
|---|---|
| `reports/<job>/<B>/eval-<job>.md` | the report (see section 4) |
| `results/<task>/<job>/<B>/eval.jsonl` | per-cell scoring detail, one row per cell; the report can be traced back to a row |
| `results/<task>/<job>/<B>/responses.jsonl` | the model's raw responses (including the full text of failed retries) |
| `results/<task>/<job>/<B>/trace.jsonl` | the trace event log (see section 3) |
| `results/<task>/<job>/<B>/cases.jsonl` | the questions themselves (including ground truth) + per-case sha256, logged into `batch.json` |
| `results/<task>/<job>/<B>/batch.json` | batch metadata: model set / kernel path / usage / world and scoring fingerprints |

### Important: don't confuse the two caches

| | what it is | who bypasses it |
|---|---|---|
| `cases.jsonl` | the assembled question, past the emission gate | `--rebuild` (regenerate questions within the batch) |
| `cases/_llm_cache/` | the question-generation LLM's prompt->response cache | `--regen` (force a fresh model call) |

**A cache hit is not the same question**: change a judge or `job.yaml`, and the same model output assembles
into a different question. So `run`/`report` always read the batch's own `cases.jsonl` (comparing fingerprints
on read-back, and warning on a mismatch) -- otherwise the report's collapsible "what was handed to the
solver" section can't prove it's the same thing that was actually evaluated.

## 3. Trace: what the model was thinking, how it called tools

`trace.jsonl` is an append-only event log, one segment per cell. The hard invariant is "the model could see it
<=> it was logged."

```jsonc
{"v":1,"seq":7,"ts":"...","case":"JD-05","solver":"...","geometry":"gated",
 "step":2,"type":"tool/result","data":{"call_id":"2.1","observation":[...],"cost":5.0}}
```

`haenv.trace.KNOWN_EVENT_TYPES` lists every event type (`read_trace` refuses an unregistered one); the four
most relevant here:

| event | what it records |
|---|---|
| `tool/call` | the tool call the model requested, `arguments` verbatim, not parsed |
| `tool/result` | the observation sequence the tool actually returned + billing + the grounding judgment; truncated by budget is recorded as `error`, not dropped |
| `assistant/message` | the model output submitted into context (never truncated) + `reasoning_text` (raw reasoning text, never read by the judge side) |
| `assistant/attempt` | a failed/retried attempt (never truncated) -- it never entered context, but it did happen |

After a run finishes it's automatically read back and verified; the log shows:

```
INFO haenv.eval: [trace] trace self-consistent (trace.jsonl)
```

Red means `trace invariant violations: N` -- at that point the trace is on disk but not self-consistent, and
neither replay nor display can be trusted.

## 4. What the report must carry

Three things a report always contains (spec §12), on top of the per-dimension sections the profile adds:

1. **Multi-model headline ranking** -- headline score = mean of the capability metrics x (1 - hard-gate fail
   rate). Safety is a non-compensable gate, applied as a multiplier that pulls the total down, not folded into
   the weighted sum;
2. **Sampled cases, presented along the timeline** -- first "the observation stream over time," then each
   model's answer with its per-track score;
3. **Collapsible raw JSON** -- the `<details>` block holds the exact payload handed to the solver (<=T, no
   ground truth). A leak self-check runs at generation time, and a block is omitted with a warning if it hits
   a ground-truth term.

The report opens with the item-generation and emission-gate summary, before the ranking.

## 5. How to read cost

* `batch.json` records `eval_usage` (evaluation) and `gen_usage` (question generation) separately;
* only tokens are recorded, not dollars. A per-query price list exists for the budgeted geometry
  (`registry/gated_pricing.yaml`, the menu the agent buys from); what a batch costs is not derivable from it,
  so estimate ahead of time instead;
* a cache hit on the question-generation side is recorded as `absent:cache_only`, which is a genuine zero
  calls, not a missed measurement;
* to estimate ahead of time: run `--models <one> --limit 1` for one cell, look at `usage`, and multiply by the
  cell count.

## 6. Three things worth knowing before you run

1. **A pack the batch gate blocks cannot be turned into a board**: the batch is marked
   `blocked_by_batch_gate` and `run`/`report` refuse it. That's by design; see `haenv-synth` for details;
2. **`--offline` needs no key at all**, good for CI and a first run;
3. **Credentials are read from the file `config.yaml:env_file` points to**, values are never logged or written
   into artifacts. If that file doesn't exist, you only get one warning => a real evaluation can silently run
   without credentials, so confirm the file is there first.

## 7. After a run finishes

* if a judge changes, recompute, don't re-run: `uv run python tools/recompute_judges.py <batch dir> --full`
  (re-judges from the responses already on disk, calls no model), and it reports which fields changed;
* to make a batch publishable (a single, consistent judging convention): `uv run python
  tools/restamp_batch.py <batch dir>` -- it strips stubs, recomputes, prints the commands to re-run the
  stubs, and finally verifies itself against the gate.
  Important: don't hand-roll these four steps: stubs are identified via `haenv.baselines.BASELINE_NAMES`;
  hand-writing a second list will delete the wrong rows.

*SYNTHETIC data, for evaluation only, not medical advice.*
