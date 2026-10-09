# Documentation

| You want to know | Read |
|---|---|
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
| Combining synthesis capabilities and verification across tasks | [Multi-task synthesis](MULTI_TASK_SYNTHESIS.md) · [中文](MULTI_TASK_SYNTHESIS.zh-CN.md) |
| Driving the pipeline from a conversation (three skills) | [`../skills/`](../skills) |
| Terms used throughout the code and documents | [`design/glossary.md`](design/glossary.md) |

`anchor/freeze-2026-09-02.json` is the current freeze snapshot: the fingerprints of the generation
and judging code, and of the frozen packs, that every published score is tied to.

*SYNTHETIC data, for evaluation only, not medical advice.*
