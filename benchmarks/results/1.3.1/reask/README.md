# Stored re-ask outcomes

These synthetic diagnosis outcomes let readers recompute the correction rates and
95% intervals offline, without credentials or model calls. Install the analysis
extra and run from the repository root:

```bash
uv sync --extra analysis
uv run python tools/evaluation/reask_pooled.py --check --out reask-analysis.json
```

`inclusion.jsonl` has one row per model/question/repeat/kind/arm. The 142 targets
were originally wrong with incomplete decisive-evidence coverage. A resamples the
final answer with nothing added; B adds all missed decisive results; C adds the
same number of non-decisive results (placebo). The 30 controls were originally
wrong despite complete coverage and receive A only. The count `n_added` measures
tool-result messages, so a panel containing several findings counts as one result.

On the 79 targets missing at least two results, H adds a seeded subset of
`floor(k/2)` of B's results. For the other 63 targets (`k=1`), the all-target half
bar HALL reuses A, since `floor(1/2)=0`. H−A, B−H and H−C contrasts use only the
79 paired H units. HALL's four bars use the same 142 targets.

`answers.jsonl` contains the 535 final answer texts and their parsed differentials;
`gold.json` supplies only the diagnosis metadata required by the released judges.
`--check` reparses each final text, checks the saved differential, runs the current
unified/comorbidity diagnosis judge, and requires every stored D to match, except the
answers listed in `matcher-drift-1.4.json`. The 1.4 diagnosis judge states its answer
equivalence explicitly: an acute kidney injury attributed to volume depletion, or chronic
kidney disease with its progression, counts when one candidate names both the disease and
the required qualifier; a disease or qualifier left open (to watch for, under assessment,
to be ruled out, or queried) is not named. It also accepts registered judging names of
the gold that are not kernel aliases: sourced synonyms (`threads.yaml: judging_synonyms`),
the reviewed English name of each Chinese gold name (`judging_aliases_en.yaml`) and its
spelling variants.
Under it, 20 stored answers that 1.3.1 scored wrong are correct.
The table lists each one with its stored and current D and the diagnosis concept or
judging name that changed it; `--check` fails on any other difference and on a listed answer that no longer
differs. `--write-drift` regenerates the table. The released facts use the stored D. Error
rows are excluded; an unanswered or empty final answer counts as wrong. The first
non-error row per unit/arm is retained when preparing the inclusion list.

The analysis uses 2,000 bootstrap draws with NumPy's `default_rng(20261014)`.
Questions are sampled jointly across models; every repeat of a question stays in
its cluster. Each draw computes per-model rates, then weights models by their
fixed number of included units for the pooled estimate. Intervals are the 2.5th
and 97.5th percentiles. `facts.json` records all 61 released INJ readings, including
paired contrasts and the half-only subset; `--check` compares each one exactly.
The output JSON also contains per-model and pooled rates and intervals for all
arms and contrasts, including the controls.

For the source tree's separate HTTP replay rehearsal, install the `serve` extra
and run `uv run python tools/release/replay_http.py --offline`. It replays one
stored leaderboard session (31 calls) over loopback HTTP, checks both the direct
function outputs and stored results, and validates every returned FHIR resource.
The complete leaderboard replay requires the separately supplied run pool.

Data: CC BY 4.0, like the other generated benchmark data. SYNTHETIC, evaluation
only, not medical advice.
