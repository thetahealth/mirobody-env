# The LLM judge plugin — reference implementation and its boundaries

**Code**: `haenv/judges/llm.py` (in the package, off by default); `examples/llm_judge_demo/` is a thin
shell that demonstrates the entry-point plugin route
**Judge name**: `llmj_rival_discriminator` · **key prefix**: `llmj_disc_`

> The quantities in this document are offline and recomputable (the number of judgeable objects,
> the request count after cache dedup, key coverage); none of them is a scoring result. See §6.

---

## 1. Which gap this fills

All other judges are deterministic code; this plugin is the only judge that calls a model.

It plugs into the standard judge surface (`register_judge` / `load_judge_plugins` in
`haenv/judges/`, `Job.plugins` in `haenv/job.py`).

---

## 2. What it judges

For every candidate where the model claims to have ruled out a registered look-alike, it
asks the judge model with four pieces of text:

```
gold diagnosis × look-alike (registry/rivals.yaml) × that look-alike's discriminating point
(discriminator, free text)
                                  vs.
          the model's own exclusion reasoning (differential[*].ruled_out_by, free text)
```

The verdict is a closed three-way choice: `DISCRIMINATIVE` / `GENERIC` / `IRRELEVANT`.
The first line must equal one of these words exactly, with no substring matching (something like
"not DISCRIMINATIVE" also contains that word); a reply that cannot be parsed is recorded as
"unparseable", never guessed at.

## 3. Why a rule-based judge can't do this

A rule-based judge cannot evaluate free-text exclusion reasoning. `rivals.yaml`'s `discriminator`
field is free text ("serum/24h urine metanephrines; this value is normal during a panic attack"), not
a set of evidence ids, and it points to a test that should have been ordered, which the case text
does not contain. Text-similarity matching would be circular: it would encode the answer into the
check itself.

The rule-based judges ask three questions, and none of them reads what the reasoning says:

| Rule-based judge | What it asks | What it can't ask |
|---|---|---|
| `rival_ruled_out` (`tracks.rival_status`) | is `ruled_out_by` non-empty | what it says |
| `sce_*` (`tracks._pseudo_exclusion`) | is the cited evidence an empty set · is the same evidence cited for multiple exclusions | same as above |
| `disc_recall` (`judges.judge_discriminative_tool`) | did the model order that discriminating test | the exclusion reasoning itself |

`sce_*` catches "all four exclusions cite the same single weight reading"; but at the rule-based
layer, an empty phrase like "doesn't fit this disease's presentation" is field-for-field identical
to a genuinely discriminating argument.

Read this dimension side by side with `disc_recall`, never merged with it: ordering the
discriminating test is not the same as using it to rule out the look-alike, and the gap between the
two is the shape of confirmation bias.

## 4. Cost model (recomputable offline)

* **Pricing unit** = one `(row × claimed-excluded registered look-alike)` pair = one request.
  Requests are only made for the look-alikes the model itself claims to have excluded, so the upper
  bound is `Σ rival_n_declared` and the actual count is far below it.
* **Disk cache** reuses the one in `haenv/llm.py` (key = `sha256(model ‖ argv ‖ prompt)`, stored
  under `cases/_llm_cache/`). The prompt contains only those four pieces of text, no `case_id` and
  no disease course, so the same exclusion sentence shares one cache entry across cases, models
  under test and reruns.
* **The raw text stays in the cache file** (`{"prompt": …, "response": …}`); the row itself keeps
  only a 120-character summary. Changing the judge's convention means recomputing, not
  re-purchasing model output.

To estimate the cost on a batch already on disk, run the demo package's read-only scan script (zero
model calls):

```bash
uv run python examples/llm_judge_demo/scan_claims.py <results-batch-dir>
```

It reports the rows scanned, the rows with a registered look-alike, the rows that claim an
exclusion, the number of judgeable objects and the number of requests after cache dedup. On a
diagnosis batch of several models × tens of cases the result is on the order of a few dozen
requests; the cache saves roughly a third, because different models write the same exclusion
sentence.

## 5. The offline path: say clearly when it can't judge, never silently score 0

The switch is a single environment variable, off by default:

```bash
HAENV_LLM_JUDGE=terra uv run haenv run inputs/<pack>.job.yaml   # on (names the judge model, costs money)
# (unset / set to off, 0, false, no)                       # off — `--offline` and CI take this path
```

When off, the judge still mounts and still emits rows, producing:

```json
{"llmj_disc_status": "no_dispatch", "llmj_disc_rate": null, "llmj_disc_n_claims": 3}
```

`n_claims` is the positive control for this statement: there were 3 judgeable objects and none
of them were judged. Without it, `rate=null` would read as "there was nothing to judge in this case."

Six states, each with its own name, none of them folded into 0 (`STATUS` is a closed set):

| `llmj_disc_status` | Meaning | `rate` |
|---|---|---|
| `ok` | at least one verdict was obtained | a number (0.0 is reachable) |
| `no_rival` | this case has no registered look-alike ⇒ the judge doesn't apply | `null` |
| `no_claim` | look-alikes are registered, but the model claimed to exclude none of them | `null` |
| `no_dispatch` | the switch is off / the model key is misconfigured ⇒ cannot be judged | `null` |
| `dispatch_failed` | a dispatcher exists but the call failed (no key / timeout / empty response) | `null` |
| `unparsed` | the model answered, but no verdict word could be parsed | `null` |

`llmj_disc_via` distinguishes "turned off" (`off:HAENV_LLM_JUDGE`) from "misconfigured"
(`unavailable:ValueError`): both mean it can't be judged, but the former is configuration and the
latter is a fault.

The denominator is `n_claims` (the number of claimed exclusions), not the number of registered
look-alikes; otherwise "excluded only one, and excluded it correctly" would beat "excluded three,
got two right." `n_rivals` is reported alongside it as context.

## 6. Known gaps

* **Not yet run against a real model.** All quantities above are offline.
* **The offline end-to-end demo cannot produce `ok`.** The offline stubs in `haenv/baselines.py`
  never write `ruled_out_by`, so every row of the demo job's offline run is `no_claim` with
  `n_claims=0` and `n_rivals=2`, and the extraction path does not run in the demo. The scan in §4
  (against real model responses on disk) and a fake dispatcher that walks through all six states
  cover that path instead.
* **Mounted only on `single` / `gated`, not on `slices`.** In per-slice answering, the same
  exclusion reasoning repeats across slices; judging per slice would pay for the same sentence N
  times and double-count the same look-alike in `rate`'s denominator.
  `dx_rival` takes `LAST` on slices; this judge does not follow that convention, so the cell is
  left blank with a stated reason (a blank cell with a reason is not a hole in the cross-product).
* **`rivals_of` uses one of two lookup methods.** `judges._rivals_of` goes through
  `_gold(vp,"spec_id")`, while `judge_dx_rival` internally goes through `_ddx(vp).get("spec_id")`.
  This plugin uses only the former, matching `dx_rival`'s `when` precondition.

## 7. Negative control: it only shows up when turned on

The demo package's two job files, with and without the plugin, differ only in `job_id`, `report`
and the `plugins:` section; the case text is identical byte for byte.

```bash
uv pip install --no-deps -e examples/llm_judge_demo                            # once
uv run haenv run examples/llm_judge_demo/demo.job.yaml           --offline --fresh
uv run haenv run examples/llm_judge_demo/demo-noplugin.job.yaml  --offline --fresh
```

The difference between the two `eval.jsonl` files' key sets is exactly the 8 `llmj_disc_*` keys,
with no difference the other way. The run with the plugin logs one external judge,
`['llmj_rival_discriminator']`.

"It ran" and "it's registered but never runs" look identical in the artifact, so this control is
part of the reading.

## 8. The two halves of mounting

```
register_judge(...)     ← returned by the judges() factory, called by load_judge_plugins
mount_table.mount(...)  ← _ensure_mounted(), called inside the factory before it returns
```

`load_judge_plugins` calls `register_judge`, but not `mount`. Without the mount every geometry is
`NONE`: the judge is registered but never runs, which looks the same in the artifact as not
registered at all.
`kinds` and `when` are aligned exactly with `dx_rival` (the three ddx question classes + this case
having a registered look-alike); a misalignment would make the two dimensions' denominators
disagree when read side by side.

The entry-point group name is `haenv.judges.llm_demo`. It does not use the default group
`haenv.judges`, nor does it share the other reference plugin's `haenv.judges.demo`: the group name
goes into `job_sha256`, so it must map one-to-one to a single package.

---

*SYNTHETIC data, evaluation use only, not medical advice.*
