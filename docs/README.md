# Documentation

| You want to know | Read |
|---|---|
| Freeze/source resources, validity evidence and shared references | [`anchor/README.md`](anchor/README.md) |
| The current requirements and supported behavior | [`spec/spec.md`](spec/spec.md) |
| How those requirements are verified | [`spec/testing.md`](spec/testing.md) |
| What the data is, how it is generated, and where its gaps are | [`DATA_CARD.md`](DATA_CARD.md) |
| How to reproduce the numbers — what is free, what costs money | [`REPRODUCE.md`](REPRODUCE.md) |
| Where the data comes from, and why it must not be used clinically | [`ETHICS.md`](ETHICS.md) |
| How the generator composes a trajectory (skeleton + residual) | [`design/skeleton-residual-diffusion.md`](design/skeleton-residual-diffusion.md) |
| The architecture end to end, with diagrams: kernel contract, isolation, gatekeeping, scoring | [`design/architecture.html`](design/architecture.html) |
| What each judge measures, in plain language | [`design/judges-explained.md`](design/judges-explained.md) |
| The full list of judges, with their mounting and status | [`design/judge-inventory.md`](design/judge-inventory.md) |
| What the model sees and returns, and how it is scored | [`design/llm-io-and-scoring.md`](design/llm-io-and-scoring.md) |
| The tracks (task types) and how their scoring differs | [`design/tracks.md`](design/tracks.md) |
| Geometries: single / gated / slices / multi | [`design/multi-geometry-episode-subject.md`](design/multi-geometry-episode-subject.md) |
| How distractors are conditioned so they cannot leak the answer | [`design/distractor-conditioning-spec.md`](design/distractor-conditioning-spec.md) |
| One declaration per lab indicator (`registry/indicators.yaml`) | [`design/indicator-dossier.md`](design/indicator-dossier.md) |
| Adding an LLM-based judge as a plugin | [`design/llm-judge-plugin.md`](design/llm-judge-plugin.md) |
| Plugging in an external task type (gold, prompt, probe) | [`design/external-task-contract.md`](design/external-task-contract.md) |
| Driving the pipeline from a conversation (three skills) | [`../skills/`](../skills) |
| Terms used throughout the code and documents | [`design/glossary.md`](design/glossary.md) |

Published results retain their original generation and judging sources. The reproduction guide
identifies their inputs; changing the current code does not relabel stored results.


## Repository layout

The repository installs the `haenv` distribution and command. The source namespaces have
separate responsibilities; their directory names do not identify duplicate projects.

| Directory | Responsibility |
|---|---|
| `src/haenv/` | Main application and its owned capabilities. |
| `src/haenv_kernel/` | Simulation, latent controls, case construction, noise and baseline verification. |
| `src/verifier_core/` | Domain-independent score audits, version consistency and gate/ceiling checks. |
| `plugins/rare/` | Optional rare-disease coding project with its own package metadata and commands. |
| `benchmarks/` | Fixed job inputs, task cards, narrative source cases, question framings and saved replay material. |
| `registry/` | Clinical policies, provenance anchors, formal validity resources and shared references. |
| `artifacts/` | Default writable root for results, reports, caches and extracted jobs. |
| `examples/`, `plugins/` | Extension examples and optional plugin projects or checkout shims. |
| `tools/`, `tests/` | Commands grouped by purpose, and regression/contract verification. |
| `web/`, `docs/` | Browser demonstration and current documentation. |

Within `src/haenv/`, `app/` owns orchestration, `contracts/` the shared pack/view interfaces,
`world/` generation and clinical rendering, and `interaction/` solver-visible views and tools.
`scoring/` owns decision matching, semantic and management judgements; `runtime/` owns requests,
budgets and recovery. `provenance/` identifies source/criteria, `reporting/` reads and presents
saved measurements, and `resources/` resolves versioned registry content. Task-specific code
remains in the M2/P2/P3/P4 adapters rather than generic contracts.

Use the installed `haenv` command or `python -m haenv`. `haenv.cli` remains its supported entry;
its implementation is in `haenv.app.cli`. An editable install resolves `src/` automatically.
A checkout root on `PYTHONPATH` alone does not supply these namespaces. The optional rare plugin
is installed from `plugins/rare/`; ordinary core use does not require it.

Source, read-only resources and writable output have distinct locators. `haenv.source_root()`
identifies loaded code; `haenv.data_root()` identifies resources; `haenv.output_root()` identifies
writable artifacts. Resource or output overrides do not select another implementation or change
a saved result's historical identity. Default output is `<checkout>/artifacts/` in a source
checkout and `<cwd>/artifacts/` for a wheel. `HAENV_OUTPUT_ROOT` wins explicitly; the combined
`HAENV_DATA_ROOT` output redirect remains for compatibility. Fixed benchmark inputs remain
read-only and separate from `artifacts/{results,reports,cases,derived}`. Use the [reproduction guide](REPRODUCE.md) for commands
and artifact requirements, and the current [specification](spec/spec.md) for their contract.

*SYNTHETIC data, for evaluation only, not medical advice.*
