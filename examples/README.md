# Examples: five plugin packages

Each directory is a small, separately installable package that extends haenv without changing a
line of it. Every extension is registered explicitly in a job file (`plugins:`), never by scanning
the environment, so "which judges ran on this board" is recorded in the job's fingerprint.

Every demo that produces a board ships with a negative control: a job file identical apart from
the `plugins:` lines. "The plugin ran" and "the plugin was registered but never ran" look the same in
a single output; the control is what tells them apart, so read the two together.

Run everything from the repository root. Installing an example is a one-off step that makes its entry
point discoverable; nothing here calls a model or costs money.

| Directory | What it adds | Install, then run | What to look for |
|---|---|---|---|
| `plugin_demo` | a judge | `uv pip install --no-deps -e examples/plugin_demo`<br>`uv run haenv run examples/plugin_demo/demo.job.yaml --offline --fresh` | every row of `eval.jsonl` carries the four `pdemo_*` keys; the `demo-noplugin` batch carries none |
| `llm_judge_demo` | a judge that may call an LLM for the one cell rules cannot decide | `uv pip install --no-deps -e examples/llm_judge_demo`<br>`uv run haenv run examples/llm_judge_demo/demo.job.yaml --offline --fresh` | the eight `llmj_disc_*` keys are present; offline, `llmj_disc_rate` is `null`, never `0.0`; the control carries none |
| `external_subject_demo` | a fifth observed subject, `episode` | `uv pip install --no-deps -e examples/external_subject_demo`<br>`uv run haenv run examples/external_subject_demo/demo.job.yaml --offline --fresh` | multi-round rows carry the three `xdemo_*` keys; the control carries none |
| `world_plugin_demo` | a world-side change (streams, events, drug effects, dirty data) | `uv pip install --no-deps -e examples/world_plugin_demo`<br>`uv run haenv build examples/world_plugin_demo/demo.job.yaml --gen deterministic --fresh` | the demo batch and the `demo-noplugin` batch record different `world_sha` in `batch.json` |
| `attrib_demo` | a whole external task type: its gold, its prompt, its probe and its judge | `uv pip install --no-deps -e examples/attrib_demo`<br>`uv run haenv run examples/attrib_demo/demo.job.yaml --offline --fresh` | rows carry the four `attrib_*` keys, `attrib_gold_kind` is `event` / `no_event`; the control is refused with the unregistered key named |

The negative control of each demo is the `demo-noplugin.job.yaml` beside it, run with the same
command. Its output lands under `results/` at the repository root, not under `examples/`.

How to write your own: [`docs/design/llm-judge-plugin.md`](../docs/design/llm-judge-plugin.md) for
judges, [`docs/design/external-task-contract.md`](../docs/design/external-task-contract.md) for task
types.

*SYNTHETIC data, for evaluation only, not medical advice.*
