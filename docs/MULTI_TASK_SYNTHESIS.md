# Multi-task synthesis and verification

[简体中文](MULTI_TASK_SYNTHESIS.zh-CN.md) · [README](../README.md)

A new synthesis task defines its gold standard and answer contract, then combines the patient
timelines, noise, distractors and event-density controls it needs from existing tasks. Verification
must cover the resulting record: shared checks apply where their assumptions hold, and new checks
connect the task's gold to the evidence actually available to the agent.

## Example: rare-disease phenotype coding

The [rare-disease extension](https://github.com/thetahealth/mirobody-env/tree/rare-code-bench)
lives on the `rare-code-bench` branch; it is not bundled in the main-branch `haenv` package.
It asks an agent to code Human Phenotype Ontology (HPO) terms, including whether a finding is
present or absent and whether it concerns the patient or a relative, and to identify an Orphanet
diagnosis and, where applicable, a causal gene and variant.

| Component | What the example reuses or adds |
|---|---|
| Patient record | Reuses patient-profile generation, index time and course length; rare findings are overlaid on the existing metabolic patient world. |
| Event density | Reuses the default density settings for measurements, symptoms and life events. This example does not demonstrate a density sweep. |
| Interpretation difficulty | Adds negated findings, findings attributed to relatives, and narrative variants. These complement the framework's existing noise and distractor controls. |
| Verification | Reuses shared event filtering and sex-consistency rules; adds checks that gold terms remain in the visible record, variants can be read back from VCF files, and attachment text does not leak the answer. |
| Evaluation | Adds coding judges through the plugin interface alongside existing task judges. |

In that branch, `haenv_rare/haenv_rare/gen_job.py` shows the reused profile, timeline and density
functions; `haenv_rare/haenv_rare/gen.py` contains phenotype sampling and verification gates.

## Why the checks must cover the combination

During early development, shared event filtering removed symptom sentences containing a blocked
word, while their HPO terms remained in the gold. The rare-disease recoverability check caught
the missing evidence. Shared sex-consistency checks also caught menstrual findings assigned to
male patients; attachment read-back checks caught intended variants missing from generated VCFs.
The fixes belonged in generation, before evaluating a model against those cases.

A shared check can also have assumptions that do not transfer. The batch-level
`footprint_discriminates_real_symptom` check assumes symptoms belong to the symptom-topic registry;
the HPO-sampled findings do not. The rare workflow explicitly overrides this batch gate and records
the override in `batch.json`. Its per-case checks still apply. The job generator documents this
exception; it is not a claim that every shared audit passed.

## Using the existing skills

- [haenv-synth](../skills/haenv-synth/SKILL.md): configure generation, build and verify cases, and inspect refusals.
- [haenv-extend](../skills/haenv-extend/SKILL.md): add world components and judges through plugins.
- [External task contract](design/external-task-contract.md): connect a task's gold, solver-visible evidence and checks.

These skills support the workflow together. Adding another task requires selecting compatible
components and validating their combination; it does not automatically make every existing check
applicable.
