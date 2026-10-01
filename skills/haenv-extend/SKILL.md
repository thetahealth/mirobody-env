---
name: haenv-extend
description: >-
  Wire your own thing in from a package outside the repo, without changing one line of haenv -- two sides: the judge side (judges/observed subjects/gold-kind rules) and the world side (indicator streams/devices/events/drug effects/dirty-data injectors). Entry-point groups + job.yaml's `plugins: {judges, world}` + fail-closed registration entry points, including the one seam that will bite you: "registered but not mounted = never runs, and the output can't tell." Triggers when the user says things like "add my own judge", "add my own indicator stream/device/drug", "wire in an external benchmark", "how do I write a plugin", "register an observed subject", "custom dirty data".
---

# haenv-extend — add your own judge: a package outside the repo, not one line of haenv changed

**One line**: both judges and the world can be wired in from a package outside the repo, through
`job.yaml`'s `plugins:` field, with not a single character of haenv itself needing to change.

Companion skills: `haenv-synth` (generating questions) · `haenv-bench` (running a board).

Warning: this file is an index. There are two runnable minimal demos -- read the matching one first:
`examples/plugin_demo/` (judge side) · `examples/world_plugin_demo/` (world side).

---

## 0-before · pick a side first: the two sides freeze different things

| what you want to add | which side | what changing it does |
|---|---|---|
| a judge / observed subject / gold-kind rule | judge side, `plugins.judges` | changes the judge fingerprint -- the score of a cell already paid for changes |
| an indicator stream / device / physiology kernel / event / drug effect / dirty-data injector | world side, `plugins.world` | changes `world_sha` -- the question changes, old and new batches aren't comparable, but the money already spent isn't wasted |

```yaml
plugins:
  judges: [my_pkg.judges]
  world:  [my_pkg.world]
```

**A bare list, `plugins: [a, b]`, is still read as the judge side** -- existing job.yaml files don't need to
change at all.

---

## 0. Judge side: a four-stage dispatch chain, and missing any one stage means it isn't wired up

```
gold kind rule    ->  gold_kinds.register_kind_rule(...)
judge             ->  judges.register_judge(...)
mount (geometry x judge) -> mount_table.mount(...)
load point        ->  job.yaml's `plugins:` field
```

Important: missing either `register_judge` or `mount` means it isn't wired up.
Register without mounting, and `applicable()` reads `NONE` on every geometry --
the judge is registered and never runs, indistinguishable in the output from not being registered at all.

Warning: `load_judge_plugins` only calls `register_judge` on your behalf, not `mount`.
Mounting has to be done by the plugin itself -- the one seam that will bite you if it isn't written down,
which is why the demo's `judges()` factory mounts first, then returns the judges.

---

## 1. The minimal skeleton

**1. A package outside the repo** declares an entry-point group:

```toml
# the pyproject.toml of your plugin package
[project.entry-points."haenv.judges.myorg"]
my_judge = "my_plugin:judges"
```

**2. A factory function** returns the judges, mounting before it returns:

```python
def judges():
    _ensure_mounted()                  # mount_table.mount(...) goes here
    return [Judge(name=NAME, fn=_run, kinds=("*",), keys=KEYS)]
```

**3. job.yaml declares which groups to load**:

```yaml
plugins:
  - haenv.judges.myorg
```

`haenv/job.py:load_plugins` calls `judges.load_judge_plugins(<group name>)` for each one,
printing how many judges got mounted; it warns if a group was declared but not one got mounted.

**4. A negative control**: following `examples/plugin_demo/demo-noplugin.job.yaml`, make a job that's
byte-identical except missing those two `plugins:` lines -- run both, and the difference is your judge's
entire contribution.
Important: without this control, "the judge is wired up" and "the judge is wired up but always vacuous"
are indistinguishable.

---

## 2. Four registration entry points, all fail-closed

| entry point | adds what | what it rejects |
|---|---|---|
| `judges.register_judge` | one judge | a name collision is rejected by default (not silently overwritten) · a built-in judge may not be replaced at runtime · `after=` pointing at an unknown name raises, it does not fall back to appending |
| `mount_table.mount` | a judge x geometry mount | every geometry not listed must have a `why_not`, missing one raises |
| `mount_table.register_subject` | the fifth and later observed subject | `source` / `why` are required; the four built-in ones may not be overridden |
| `gold_kinds.register_kind_rule` | a gold-kind derivation rule | order is semantics |

> Important: `register_kind_rule` has a structural pitfall: it defaults to appending at the end, and the
> built-in last rule, `_rule_join_gold`, always returns a truthy value => a rule appended after it can
> structurally never be reached. The only way to actually take effect is passing `before=`, and that changes
> existing questions' kind => changes their mounting => changes their readings.
> The demo package deliberately avoids this, using `kinds=("*",)` instead.

---

## 3. Two naming disciplines (violating them raises no error, they just make the readings wrong)

1. **A judge's name must not collide with a built-in one** -- `register_judge` rejects it, and that's a
   good thing;
2. Important: **the keys it produces must always carry your own prefix**. A key name that collides with a
   built-in judge gets silently overwritten by `run_judges`'s `out.update()`, and "overwritten" and "never
   mounted" look identical in the output.

---

## 4. Five runnable examples, pick by what you're trying to do

| example | what it demonstrates |
|---|---|
| `examples/plugin_demo` | the minimal end-to-end case: one judge + mount + negative-control job. Start here |
| `examples/llm_judge_demo` | a judge that calls an LLM internally (its own entry-point group, so it can be mounted alone) |
| `examples/external_subject_demo` | registering a fifth observed subject |
| `examples/world_plugin_demo` | the world side: an indicator stream, an event pool, a drug effect and a post-injector |
| `examples/attrib_demo` | a whole new case type: its gold, prompt, probe, emission gates and latent keys, zero lines of haenv changed. Contract: `docs/design/external-task-contract.md` |

Each carries a `demo-noplugin.job.yaml` negative control: the same job with the `plugins:` line removed, so
"the plugin ran" and "the plugin was registered but never ran" cannot be confused.

Installed the same way as in this repo: `uv pip install -e examples/plugin_demo`.

---

## 5. How to prove your judge is actually running

| what you want to know | what to run |
|---|---|
| did the package load | run a batch: the job's first lines print `[job] N plugin judge(s): ... (built-in N · total N)` |
| is it mounted | `judge-inventory` is generated from the mount table; a judge mounted nowhere does not appear there |
| what's its contribution | compare `eval.jsonl` against the `demo-noplugin` negative-control job |

Warning: "registered" and "ran" look the same in a single output. The failure this section exists for is a
judge that is mounted, whose self-test is green, and which never runs on the production path: the control
shows green, so it looks covered. The negative-control job is the check that catches it, and it is the one
most often skipped.

---

## 5-after · World side: four registration points

```
register_stream(name, spec, source=..., kind="stream|kernel|indicator")
register_event_pool(name, spec, source=..., kind="benign_event|life_event|symptom|lookalike")
register_effect(drug, spec, source=...)
register_post_injector(name, fn, params=..., source=...)
```

Minimal skeleton (a complete, runnable one is in `examples/world_plugin_demo/`):

```toml
[project.entry-points."haenv.world.myorg"]
world = "my_pkg:world"
```

```python
def world() -> None:                       # the factory returns nothing, it calls the registration points internally
    from haenv import world_plugins as wp
    wp.register_stream("my_hr", {"unit": "bpm", "normal": 68.0}, source="haenv.world.myorg")
```

### Important: three constraints that raise on the spot

| constraint | why |
|---|---|
| **must not write `weight` / `weight_ref`** | they decide `outcome_label` via `label_rule` -- code outside the repo being able to write them would mean the gold standard is decided from outside the repo (§C1 / §2a). Checked on both the key and the spec's values |
| **a key collision needs an explicit `replace=True`** | a silent overwrite would make "the plugin took effect" and "the plugin never got wired up" look the same in the output |
| **`source=` is not optional** | the output must be able to answer "who added this" |

### The plugin is not exempt from the gates, and it does enter the world fingerprint

A registration point produces a "proposed change to the world": `GEN14` / `GEN15` / the footprint gate run
*after* the change. `world_plugins.manifest_sha()` feeds into `anchor.world_fingerprint()` --
not wiring it in would mean "the world changed but `world_sha` didn't move."
**With no world plugin mounted, the world fingerprint is byte-identical to not having the plugin installed
at all**, so existing batches are unaffected.

Warning: the judge side cannot see what you registered (`gates.py` / `gated.py` / `llm_rubric.py`
deliberately don't read the override layer -- the judge fingerprint must never become a function of "which
plugins are installed"). Consequence: a stream you add gets the strictest default on GEN15 (noise is treated
as 0 => not exempt from SNR; `coupling_judgeable` unregistered => judged by default). This fail-closed
behavior is intentional.

---

## 6. Boundaries: what cannot be changed from outside the repo

* **the scoring kernel** -- it lives in this repository, at `core/` (`config.yaml:kernel_path`), and its
  files belong to the frozen judging and generation segments: changing one moves a code fingerprint, so
  every batch scored or generated under the old one is stamped with that older version. `docs/REPRODUCE.md`
  says which files those are and what a change costs;
* **the built-in judges and their mounts** (`haenv.judges._CORE` lists them) -- both `register_judge` and
  `mount` refuse to overwrite a built-in entry. You can add, not replace;
* **the gold standard** -- given deterministically from the hidden control variables. A judge reads the
  gold standard, it never produces one; a world-side plugin is the same -- this is exactly what the
  `LABEL_BEARING` gate blocks.

*SYNTHETIC data, for evaluation only, not medical advice.*
