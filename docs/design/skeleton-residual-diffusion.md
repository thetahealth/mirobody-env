# Skeleton + residual diffusion — the synthetic-patient generation layer

---

## 1. The problem: a deterministic renderer is a zero-variance conditional map

haenv separates "condition" from "rendering" — the gold label lives in the latent control variables
`z = (outcome, reversal_week, driver, index_time_T, event_density, ddx_condition)`, and the code
renders an observation from `z`. If the rendering step is deterministic, every channel is a fixed
function of `z`:

| Channel | Deterministic rendering | Location |
|---|---|---|
| Primary signal (weight) | Piecewise trend (decline → plateau → rebound) with a small fluctuation whose amplitude `amp ≤ 0.12` is back-computed from the physiological budget | `haenv/build.py` `_weight_series` |
| Clinical labs (10 items) | A univariate ramp lagging weight: `normal + (abnormal−normal)·min(1,(d−onset)/28)` | `haenv/build.py` `CLINICAL_SPEC` |
| Daily auxiliary streams | A waveform around `base`, with `base` looked up from BMI/age/comorbidity | `haenv/events.py` `MetricSpec` |
| Noise | Six kinds of episodic artifact (device-swap offset, transient spikes, contextual confounds, unit errors, MNAR gaps, adherence cliffs), each acting on a window `[d0,d1]` | kernel `haenv_kernel/noise.py` |

The six artifact kinds are events, not a process: without a stationary measurement/physiological
noise process, the gaps between them are smooth deterministic curves. Daily weight is then too
smooth (real day-to-day fluid fluctuation is ±0.3–0.8 kg), cross-channel correlation is artificial
(labs driven by weight are over-correlated, independently generated streams under-correlated), and
each `z` yields exactly one trajectory, so distribution-edge samples cannot be produced.

## 2. Principle: fluctuation belongs to the observation layer

The clean trajectory is the latent ground truth; measurement fluctuation is noise added in the
observation layer (the kernel's weekly slope check would reject it on the clean trajectory).
Diffusion is the strongest way to implement that layer, not a precondition for it.

## 3. Decomposition

```
x_obs(t)  =  s(t; z, θ)  +  r(t; c)  +  a(t)
             ─────────      ────────     ────
             skeleton       residual     episodic
                            field        artifact
```

| Layer | What it is | Determined by | Verified by | Where it lives |
|---|---|---|---|---|
| `s` skeleton | The clean disease course: piecewise trend + labs constrained by the relationship graph | Latent control variables `z` + constitution profile `c`, deterministic | the generator's premise-conflict check + the calibration track's literature relationship set | `build.py`; this design does not change it |
| `r` residual field | Stationary physiological/measurement fluctuation: fluid drift, the AR structure of resting heart rate, the weekly rhythm of step count, scale quantization error and routine missed weigh-ins | Learned from real wearable data, conditioned only on `c`, independent of `z` | New judges, each paired with a deliberately injected defect | haenv side: `haenv/events.py`, `haenv/build.py`, `haenv/physio/` (see D4) |
| `a` artifact | The six kinds of episodic artifact | `noise.inject` | Existing | Unchanged |

The gold label reads only `s`, never `x_obs`. This is the load-bearing wall of the whole design.

## 4. Five hard constraints

### D1 · Gold labels are unchanged (the repo's gold-isolation rule)
`s` is retained as a verifier-only field, as a complete clean series; `label_rule` and the offline
solver read only that. Before and after residual injection, the gold-label fields and the offline
solver's conclusion must be byte-for-byte identical.

The kernel's `_add_clean_ref` (`haenv_kernel/noise.py`) keeps a 28-day sampled sparse clinic-scale reference
(`weight_ref`) as a prompt-side field for the solver to cross-check against — that is separate from
the complete clean series `s` itself, which is carried as a verifier-only out-of-band field (an
`_`-prefixed key at the top of `manifest`) and popped out before anything reaches `injected`: it never
enters `injected`, is never attached to `raw`, and never enters the report.

### D2 · Diffusion does not produce gold labels (the rule the calibration track applies to its relationship graph: a validator, never the label)
The residual layer is a surface-level generator. A wrong residual only makes a case uglier, never
makes the answer wrong. Any design that lets the model "read the residual to infer the outcome" is
out.

### D3 · Residual is independent of `z`, and this must be machine-checkable
In real data, auxiliary streams and outcome are correlated (adherence drops, weight rebounds); a
model learned from that data will carry that correlation into its samples — which trips the leakage
gate directly. So:

- At training time, `outcome` enters the condition and is marginalized out at sampling time; or the
  residual is decorrelated from `outcome` adversarially.
- Machine check: fix profile `c`, sweep over the values of `z`; the distribution test on the residual
  samples must be non-significant. This must be an executable check, not only documentation.

### D4 · Where the residual layer lives
The ground-truth isolation this needs — splitting the clean trajectory from the noisy observation
before any residual is added — lives on the haenv side: the ground-truth trajectory is deep-copied
before `events._apply_physio_if_enabled` adds noise, the slope/anchor gates read that ground truth
(`build.py`'s `premise_conflicts(_truth, p)`, `gates.check_anchors_honored(_truth, cs)`), and a
separate check reads the observation (`gates.check_observed_in_domain(raw, p)`) — all in
`haenv/physio/` and `registry/`, with no kernel change. The parametric residual (P1, §5) for both the
daily auxiliary streams and the primary weight signal lives in `haenv/events.py` and `haenv/build.py`.
A full multi-channel residual generator registered as its own noise kind in the kernel's `noise.py`
is a separate, larger change: it moves the generation segment and requires regenerating the question
packs.

### D5 · Negative control and counter-metric
Every new residual verifier must come with a matching deliberately injected defect in
`tools/verify_selftest.py`. Every change must report, together: the gold label is still judgeable ·
the emission rate hasn't collapsed · the self-checks are green.

## 5. Three phases, each independently stoppable

| Phase | What it does | Training cost | Is stopping here still worth something? |
|---|---|---|---|
| **P0 · Residual resampling** | Trains no model. Slices real wearable data and resamples the residual directly into the observation layer. Not implemented; it is arm B of the §8 experiment | Zero | Yes — it's the lower bound for P1/P2, and arm B of the §8 experiment |
| **P1 · Parametric residual** | AR(1) + weekly cycle + MNAR missed readings, fit per channel. Implemented for both the daily auxiliary streams and the primary weight signal (`_weight_series` uses an AR(1) process seeded per case) | Minutes, CI-feasible | Yes — low-dimensional, interpretable, checkable, free to reproduce |
| **P2 · Conditional diffusion** | A jointly modeled multi-channel residual, conditioned on `c`. Continuous streams use a score-based family; discrete events use an Add-Thin-style TPP diffusion; the hard range/cross-`T`-drift checks from `verify` become a differentiable penalty added to the reverse step (classifier guidance) | GPU, needs external data | — |

P2 is the actual "diffusion"; P0/P1 are the lower bound it must first beat. Without P0/P1 readings,
P2 cannot show that it added anything — a negative result needs a positive control to be readable.

## 6. Data: the one real blocker

60 hand-written raw cases cannot train any generative model.

Cross-channel covariance is identifiable on the daily auxiliary streams, where every channel pair
co-occurs. It is not identifiable on lab channels: most lab metric pairs never co-occur, and
different sampling grids share no timestamps. External data for P2 is therefore needed mainly for the
lab side, and also because the covariance structure among generated streams is artificial, so
learning from it would mean learning the artificial structure.

| Candidate source | What it has | Obstacle |
|---|---|---|
| All of Us (Fitbit subset) | Daily steps / HR / sleep + EHR weight + prescriptions — closest domain match | Requires an application, timeline unknown |
| UK Biobank accelerometer | Large-scale daily activity | No daily weight, no medication titration |
| MIMIC-IV | Real correlation between lab pairs | Inpatient data, no daily behavioral stream, wrong domain; can only support relationship calibration for the `s` layer |
| A 671-case real outpatient EMR sample | Lab baselines and reference ranges | No daily sampling: enough to calibrate `s`, not enough to learn `r` |

What makes P0/P1 feasible: they only learn residual structure, not disease course. Resting heart
rate's autocorrelation, step count's weekly rhythm, a scale's quantization error and missed-weigh-in
pattern — none of these depend on the disease or carry outcome information — so any large wearable
dataset can be used, and D3 is automatically satisfied. Only P2's jointly modeled cross-channel
residual needs in-domain data.

## 7. Relationship to the calibration track: the order cannot be reversed

The calibration track fixes the mean structure (the literature relationship between the ten clinical
indicators); this design fixes the covariance/noise structure. The two are complementary, but strictly
ordered:

1. The relationship graph is the residual layer's verifier — without it, once the residual is added
   there's nothing to judge "is this still self-consistent?" ⇒ the calibration track goes first.
2. They share one precondition, the lab-channel decision (the longitudinal `CLINICAL_SPEC` stream
   versus the textual lab panel). It governs only a residual attached to the lab channel — the
   daily-sampled auxiliary streams and the primary weight signal are unaffected by it, which is why
   P1's parametric residual applies to both without waiting for that decision.
3. The §8 comparison experiment follows the same rule as any paid evaluation: it costs model money,
   and must come after the code freeze.

## 8. The falsifiable experiment to run first (before P0)

Freeze the measurement script first, then touch the data. Then:

- Same batch of cases, arm A the existing rendering, arm B real residual resampling (P0, no model
  training);
- Run 3 models each at `pass^k` (k≥3) to establish a noise floor first — a difference is not read
  without one;
- Compare: aggregate ranking · hard-gate rejection rate · emission rate (counter-metric).

| Result | Conclusion |
|---|---|
| Difference stays within the noise floor | Realism has no value for evaluation ⇒ drop this line of work |
| Ranking changes | The current rendering is handing out free points or unfairly penalizing ⇒ this is itself a publishable finding, and it motivates P2 |
| Emission rate collapses | The residual is crowding out the physiological budget (see §9) ⇒ resolve budget allocation before continuing |

## 9. Limitations and known risks

*(every limitation in this document is collected in this section)*

1. Data acquisition is off the critical path and long-tailed. All of Us's application timeline is
   unknown; P2's schedule cannot be set until the application resolves.
2. The residual competes for the physiological budget. `_weight_series`'s `amp` is back-computed
   from `mw/7·0.9 − trend` — the residual amplitude has to come out of the same budget, or the generate-verify loop
   won't converge and the case won't emit: a trend of 1.33 kg/wk plus a constant 0.25 fluctuation is
   enough to cross the kernel's `max_weekly_delta=1.5` and get rejected after repeated
   non-convergence.
3. The kernel judges the weekly slope point-by-point, which assumes weekly sampling. A daily
   residual will necessarily exceed that point-by-point tolerance. Two ways out: change the kernel to
   a windowed convention (a change to `haenv_kernel/synth.py`, in the generation segment, which requires
   regenerating the question packs), or add the residual only at the observation layer while leaving
   the clean skeleton untouched. This design takes the latter — the cost is that the clean skeleton
   stays too smooth; only what the solver sees becomes realistic.
4. D3's independence test is underpowered at small sample sizes. At small n, "not significant" does
   not mean "independent" — this check's failure threshold needs to be calibrated to the actual batch
   size, not copied from a conventional significance level.
5. Attaching a residual to the `CLINICAL_SPEC` lab channel is blocked on the same channel decision as
   the calibration track — that decision doesn't affect the daily-sampled streams, where P1 applies.
6. P2's guidance direction runs against the literature mainstream. Work like TarDiff steers toward
   downstream task signal; the goal here is to remove outcome information. Existing implementations
   can't be reused directly — D3's adversarial decorrelation has to be written from scratch, and as
   part of the measurement layer it must be held to the same standard of verification as any result.
