# Ask a failed diagnosis response again

`haenv reask` resumes a stored query conversation immediately before its final answer.
It reproduces the experiment's mechanism per unit. The pooled 1.3.1 re-ask results are
recomputed from the shipped package in [`benchmarks/results/1.3.1/reask/`](../benchmarks/results/1.3.1/reask/README.md)
(stored answers, inclusion list and diagnosis gold):
`tools/evaluation/reask_pooled.py --check` re-scores the 535 stored answers and checks the
61 reported readings, intervals included. A new `haenv reask` run selects its own units; it
does not apply the original per-model sampling, per-question inclusion caps or prefix-check
exclusions. Units with one missed result have no half request; the pooled half-evidence
analysis keeps their original D=0 rather than a new resample outcome.

## Select responses and arms

It selects failed, answered diagnosis responses with an evidence list and asks the original
model once per arm: `resample` adds nothing, `decisive` adds results carrying the missed
evidence, `placebo` adds the same number of results carrying no evidence-list item, and
`half` adds a seeded subset of `floor(k/2)` of the decisive results, only when `k >= 2`.
Failed responses with complete evidence coverage receive resample only.

```bash
haenv reask saved-batch/ --models glm-5.3-flash --questions QUESTION_ID \
  --sources saved-result-corpus/ --dry-run
haenv reask saved-batch/ --models glm-5.3-flash --questions QUESTION_ID \
  --sources saved-result-corpus/ --out reasked/ --model-cap glm-5.3-flash=32 \
  --judge-budget-ledger reask-ledger --judge-budget-usd 1
```

## Source artifacts and replay checks

The source needs `query_trace.jsonl`, `payloads.jsonl`, and final-request fingerprints in
`responses.jsonl` or accounting receipts. Purchase observations in `trace.jsonl` rebuild the
stored tool declarations. Unrecorded decisive purchases use the environment's own purchase
path and require `cases.jsonl`. `--sources` adds recorded results only after matching the
question's solver/verifier payload identity, prediction time T and archived case fingerprints
where available. Same-question world mismatches and conflicting returns for the same call
stop preparation. Use the same source corpus, `--seed` and `--half-seed` to reproduce a selection. A fingerprint
mismatch or insufficient placebo pool stops preparation before model calls. Placebo excludes
all attempted purchases, including refused ones, and can include purchased monitoring streams.

Every injected arm uses the same empty assistant tool-call wrapper and original result objects;
no evidence labels reach the model. Tools are closed for the final answer; the GLM Flash route
uses `auto` for compatibility, with any returned tool call scored as unanswered and never executed.
The dry run prints test names and carried evidence keys without loading a provider.

Before any model call, the current solver's canonical model, upstream, sampling, output
limit, reasoning effort and response format must match the source accounting manifest or
final-request receipts. Equivalent transport routes use the runtime's canonical answer
view. Missing source settings also stop execution. `--allow-settings-override` explicitly
permits missing or different settings and labels the output accordingly; it cannot claim
verified settings replay.

## Output and resuming

Output includes `injection-rows.jsonl` (`model` is the actual model id; `source_model` is the
original solver name; `answer_settings` records the actual settings; `settings_replay` is
`verified`, `mismatch` or `unavailable`; also `question`, `repeat`, `kind`, `arm`, `status`,
`D`, `n_added`; arm codes `A/B/C/H`), the exact request/selection manifest, answers, scheduler
receipts and normal cost accounting. `D` uses the shipped diagnosis scorer and stored gold;
transport and scoring errors have `D: null`. A scoring/conversion error saves the raw answer
and error details for that cell and allows the batch to continue. Such cells are complete
outcomes; transport failures can retry. Completed arms resume without another call; changed plans
require a new output directory. Repeats come from stored metadata, a batch's `-pkN`/`-repN`
suffix, or otherwise default to the first repeat.

## Scheduling and Python API

The command defaults to 64 workers per model, raises legacy cell caps to that concurrency,
and remains bounded by configured backend limits and shared paid request lanes.
`--model-cap MODEL=N` takes precedence within these ceilings and the worker count.
Python callers can use `haenv.reask.prepare(...)`, `dry_rows(units)` and
`run(units, output, ledger_path=..., budget_usd=..., model_caps=...)` with the same semantics.

See the [reproduction guide](REPRODUCE.md#recomputing-the-released-re-ask-results) for the offline check.

*SYNTHETIC data, evaluation only, not medical advice.*
