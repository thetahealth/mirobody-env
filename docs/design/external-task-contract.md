# External Task-Plugin Integration Contract

> This is for anyone who wants to add a new case type on top of haenv.
> A reference implementation (an "event effect attribution" case, `examples/attrib_demo`)
> demonstrates this end to end, and it changes no haenv code — everything goes through the
> registration surface.

## 0 · One-line summary

A new case type = six things, written in a package outside this repo, mounted by one `plugins:`
line in `job.yaml`.
Missing any one of them doesn't produce an error — some reading silently turns into `None`, which
in the report looks like "this dimension doesn't apply".

---

## 1 · The six things

| # | Registration point | What it decides | What happens if missing |
|---|---|---|---|
| 1 | `gold_kinds.register_kind_rule` | whether this case is your case type | the kind falls to a built-in rule ⇒ your judge can never reach it in `applicable` |
| 2 | `judges.register_judge` | how it's judged | the judge isn't in the live registry |
| 3 | `mount_table.mount` | which geometry it's judged on | every geometry is `NONE` ⇒ it's registered but never runs — on the artifact, indistinguishable from not being registered at all |
| 4 | `external_gold.register_gold_block(gold_fn=…)` | how the gold gets into `vp.adjudication` | the judge is called as usual, then can't find anything to judge ⇒ every key is `None` |
| 5 | the same call's `probe_fn=…` | how the prompt gets into `prediction_context` | the question never gets asked ⇒ the model answers a different question |
| 6 | `external_gold.register_framing` + `register_probe` | which phrasing is used to ask | falls back to the built-in phrasing ⇒ the answer field you need never appears |

4 and 5 are in the same registration call deliberately. Visibility on the two sides is opposite
(`gold_fn` is verifier-only · `probe_fn` goes to the solver); bundling them into one call forces a
visibility decision for each item. Two separate functions would allow "gold registered, prompt
forgotten" — or worse, handing the solver the entire answer-bearing object.

`probe_fn` returns a dict or a string, written to `prediction_context.<name>`; either function may
return `None` for a case, and then writes no key. A block whose `gold_fn` always returns `None`
places a solver-only field (an instruction, attachment pointers) at the top level of
`prediction_context`, where the system under test reads it.

## 1a · Generation side: gates, latent classes, and the world stamp

| Registration point | What it decides | What happens if missing |
|---|---|---|
| `register_gold_block(..., latent_classes={key: class})` | the provenance class (`patient_fact` / `gold` / `knob`) of each latent key your block carries | **required** whenever `latent_keys` is non-empty: the call is rejected, and nothing is registered |
| `external_gold.register_gate(name, fn, source=, why=)` | emission checks only your case type knows (answer derivable from the prompt, no leak of the diagnosis, attachments intact) | your cases are emitted without them |

A gate is `fn(raw, cs, sp, T) -> list[dict]`, called after the built-in gate chain with the same
arguments the built-in gates see. Each hit is `{kind, severity, detail}`; `kind` must start with
`<name>_`, `severity` is `gate` (blocks emission), `warn` or `info`. A gate that raises aborts the build.

Everything registered here enters `world_sha`, together with a semantic fingerprint of the files in
the package that defines each registered function (comments and docstrings excluded). With nothing
registered, the stamp is byte-identical to a build without this surface. Files your code reads from
outside its own package are not in that fingerprint: record their sha256 in the job's latents.

The fingerprint covers the whole package, so editing a judge that lives in the same package also
moves `world_sha`. That over-reports on purpose: fingerprinting only the defining module would miss
the modules and tables it reads, and a world that changes under an unchanged stamp is the failure this
stamp exists to prevent. If the over-reporting matters to you, put the generation side and the
judging side in two packages.

## 1b · The task card

A task is the shape of its gold plus the answer contract its judges read. The geometry and the
wording of the question are conditions inside a task; they do not make a new one.

The development repository records each task in a hand-written card with these fields:
`task_id` · `title` · `status` (`active` / `demo`) · `claim` · `gold.kinds` · `gold.blocks` ·
`answer_contract.probes` / `output` / `scored_by` · `geometries` · `task_types` · `source` · `owner`.
For a plugin task, `gold.blocks` holds the name you passed to `register_gold_block`, `gold.kinds`
the kinds your cases derive to (your kind rule's output, or a built-in kind if you add none),
`scored_by` the judges that carry the claim, and `source` your package name.

The task-card catalog and its maintainer audit command are not included in the public release.
They are not required to implement a task plugin. For a public plugin, describe its gold,
answer contract and supported geometries in the package README, and verify them with the paired
offline examples. Case assignment follows the gold contract, not just the job's task-type label.

## 2 · Minimal skeleton

```python
# the __init__.py of your package (here: mypkg)
LATENT_KEY = "mytask_payload"          # used in job.yaml's latent: block to carry a reference

def _gold(spec):                       # → adjudication.mytask   (verifier only)
    ref = (spec.get(EG.SLOT) or {}).get(LATENT_KEY)
    return None if not ref else {"gold": ...}      # return None if not found, not an empty dict

def _probe(spec):                      # → prediction_context.mytask (visible to the solver)
    ...                                # every field emitted here must answer "what can the model infer from this"

def judges():                          # entry point points here
    EG.register_framing("MYTASK_PROMPT", TEMPLATE, source=..., why=...)
    EG.register_probe({"probe_id": "mytask.direct", "framing_ref": "MYTASK_PROMPT",
                       "framing_sha256": "<computed>"}, source=...)
    EG.register_gold_block("mytask", _gold, probe_fn=_probe,
                           latent_keys=(LATENT_KEY,), latent_classes={LATENT_KEY: "gold"},
                           source=..., why=...)
    EG.register_gate("mytask", _gates, source=..., why=...)   # optional, see §1a
    GK.register_kind_rule(GK.KindRule("mytask", _derive_kind, ...),
                          source=..., before="not_ddx")      # see §3
    MT.mount("mytask", {"single": "out"}, why_not={...})     # write a reason for every other geometry
    return [Judge(name="mytask", fn=_judge, kinds=("mytask",), category="deterministic")]
```

```toml
# pyproject.toml — the group name must be unique: it goes into job_sha256, and is how "which package is mounted on this board" gets traced
[project.entry-points."haenv.judges.mytask"]
mytask = "mypkg:judges"
```

```yaml
# job.yaml
plugins: [haenv.judges.mytask]
probe_id: mytask.direct
cases:
  - case_id: X-01
    raw: {...}                    # the harness's required shell, unrelated to your case (see §5, limitation 1)
    latent:
      mytask_payload: {...}       # carries a reference, not the whole case
```

---

## 3 · Five known pitfalls

### ① `register_kind_rule`'s documented default is wrong here

By default a rule is appended at the end and doesn't jump ahead of the built-in rules. With the
default, your judge never runs: `_rule_not_ddx` is first in line and consumes every case without ddx
gold ⇒ your kind never gets a turn.

The failure does not report "judge not mounted": the artifact shows `gold_kind: forecast` plus an
`AttributeError` from the core judge trying to read `out.forecast` off your answer — it looks like a
malformed answer, but the kind was inferred wrong.

⇒ Use an explicit `before="not_ddx"`, and write your rule as a narrow, default-deny rule (matches
only when your adjudication block is present), paired with a negative control: it must not match
on a plain case's `vp`.

### ② External content can only go into two free-form dicts

`LatentPremise` and `SolverPayload` are both kernel dataclasses with fixed fields; any extra
top-level key is silently dropped at construction time. `LatentPremise` lives in `core/latent.py`
(generation segment) and `SolverPayload` in `core/schema.py` (judging segment), so changing either
means regenerated packs or recomputed scores.

⇒ The only landing spots are `meta` (premise side, `m = p.meta` in `generate`) and
`prediction_context` (solver side).
Symptom: a unit test shows `blocks_for(spec)` returning your block, but it's absent from the
artifact — the dataclass boundary sits in between.

### ③ The answer object is a dataclass, not a dict

The `out` a judge receives is a kernel `SolverOutput`. To get the raw text, use
`getattr(out, "_raw_text", "")` (as `evaluate.py` does). Writing
`(out or {}).get(...)` gets you `'SolverOutput' object has no attribute 'get'`.

### ④ External probes must not be placed in this repo's `probes/`

`load_probes` is default-deny and validates the entire directory in one pass: if any single
yaml's `framing_ref` isn't registered, it raises and invalidates the whole directory.
Your template is only registered when your plugin is mounted on the job ⇒ drop one file in, and
every job that doesn't mount your plugin stops running.

⇒ Ship it via `register_probe`, bundled with your package.

### ⑤ Use the kernel's extraction path — don't write a second one

Write judges against the kernel's `solver._extract_json`; do not hand-roll a second extractor. Two
extractors that handle `<think>` tags differently disagree on which text is the answer, and that
disagreement doesn't raise an error — it silently lets bad rows through.

---

## 4 · Checks you must bring yourself

The registration surface doesn't exempt you from any gate, and these are checks only you can
provide:

| | Check | Why |
|---|---|---|
| **leakage** | pair every field the prompt side emits with a negative control: inject the answer field, the gate must fail it | only you know your prompt's allow-list |
| **answerability** | gold must fall in the candidate set / value domain; fewer than 2 candidates fails | otherwise the case is unanswerable or a free point |
| **degenerate ceiling** | run at least three data-blind strategies (always abstain / always pick the first / always pick the most recent), check per bucket, not just the total score | a shortcut can score far above random on the positive-case bucket while a negative-heavy aggregate hides it |
| **flip one bit, must be caught** | changing one bit of the gold must change the score | otherwise the judge might always return the same value |
| **two-sided control** | both "empty table, no effect" and "registered and actually takes effect" must be verified | verifying only one side lets the other pass silently |

---

## 5 · Limitations (the only such section in this document)

1. The `raw:` fields are the harness's required shell, unrelated to your case — it still needs to
   build its own world. This is the cost of plugging in an external task type.
2. `--offline` doesn't render the prompt: the stub returns a fixed string, and `probe_id` /
   `framing_sha256` / `prompt_sha256` are all `None`. ⇒ "the model was actually asked your question
   and answered it" can only be verified with a real model — an offline run only verifies the
   rendering layer.
3. Nothing caps the prompt size. The reference implementation's item sequence runs to thousands of
   points ⇒ the rendered prompt comes out to about 256k characters (≈64k tokens). Downsampling is a
   case-generation policy decision; the registration surface doesn't make it for you.
4. This contract only covers the reference implementation for the `single` geometry. How it wires
   up on slice / multi-round / gated cases is unverified.
5. **Your judges are not in `judging_sha16`.** That fingerprint (`anchor.judging_fingerprint`) covers the
   files in `tools/make_freeze.py`'s `JUDGING` list, and no plugin package is on it — this includes the
   in-repo `examples/`. The publish gate keys on it, so rows scored before
   you changed a judge are not flagged as stale. Your package fingerprint does enter `world_sha` (§1a)
   for batches built after the change; recompute older batches yourself.
6. **The task card lives in this repo's `tasks/`, not in your package.** A plugin cannot ship its own
   card yet, and the card carries neither a version stamp nor a cost figure.

*SYNTHETIC data, evaluation use only, not medical advice.*
