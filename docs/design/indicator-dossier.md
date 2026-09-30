# Indicator Profiles — One Declaration per Indicator, Machine-Checkable Consistency Between Attributes

`registry/indicators.yaml` holds one profile per lab indicator: unit, precision, reference and
physiological ranges, which diseases it diagnoses, and a per-cohort distribution. Every attribute has
a source, and the invariants below check the attributes against each other, so a value is never
guessed from an indicator's name or declared twice.

---

## 1. Structure

```yaml
HbA1c:
  unit: "%"
  ndigits: 1
  reference_range:      {low: 4.0, high: 6.0, source: ...}    # normal or not
  physiological_range:  {low: 3.0, high: 20.0, source: ...}   # physiologically possible or not
  diagnostic_for:                                             # which diseases are diagnosed on this value
    T2D: {ge: 6.5, source: "ADA Standards of Care"}
  cohorts:                                                    # a distribution per cohort, not a constant
    T2D:     {median: 7.4, p_abnormal: 0.901, source: "real EMR cohort, 671 patients"}
    obesity: {median: 5.6, p_abnormal: null,  source: "estimated from NHANES", review: pending}
```

### 1.1 `diagnostic_for`

The rule it encodes: what matters is whether the marker is used to diagnose the disease, not the
cohort's population mean.

| Disease | Diagnosed by | Is the marker the basis | ⇒ Baseline |
|---|---|---|---|
| T2D · dyslipidemia | diagnosis is made directly on that value | yes | must fall on the diagnostic side, otherwise the definition contradicts itself |
| MASLD · obesity | imaging/biopsy · BMI | no | set by the cohort's distribution; a normal value is legitimate |

### 1.2 `p_abnormal`: dispersion from published quantities

A distribution needs a median and a spread; the spread is rarely reported directly. Under a
log-normal assumption, the median `m` and the fraction exceeding the reference upper bound
`p` jointly determine σ:

```
σ = ln(ref.high / m) / Φ⁻¹(1 − p)
```

and `p_abnormal` (the abnormal rate) is what the literature and EMR cohort calibrations report.

`p_abnormal: null` means "this cohort has not been measured", not zero. Without it, the case is
not emitted (fail-closed).

---

## 2. Invariants

| | Invariant | Defect it catches |
|---|---|---|
| **INV-1** | for a disease in `diagnostic_for`, `cohorts[disease].median` must fall on the diagnostic side | a dyslipidemia TG baseline below its threshold |
| **INV-2** | if `p_abnormal` is given, the emitted cases must produce that fraction (measured on emitted cases) | a constant ALT baseline that is always normal for a cohort with a high abnormal rate |
| **INV-3** | `physiological_range` ⊇ `reference_range` | a reference bound lying outside the physiologically possible range |
| **INV-4** | every value has a `source:`; `review: pending` propagates all the way to the batch | a fabricated citation (an explicit `pending` is preferred) |
| **INV-5** | `unit`/`ndigits` are declared exactly once; code must not derive a second copy | precision overridden downstream |
| **INV-6** | every indicator that reaches the prompt must have a profile entry (fail-closed) | an attribute guessed from a substring of the name |
| **INV-7** | a fitted distribution's p5/p95 must fall inside `payload_range` | parameters that are individually sourced but jointly inconsistent (§6.3) |

### INV-3 applies to the high side only

Plain containment, `phys.lo ≤ ref.low`, gives a false positive on `ALT`: its `ref.low = 0` is a
notational convention meaning "anything below the upper bound counts as normal," not "0 is a
reachable value." A physiological lower bound above 0 is correct.

INV-3 therefore requires containment on the `high` side, and exempts the `low` side when
`ref.low == 0`, with the reason recorded.

---

## 3. Both lab channels read the profile

Lab values reach the prompt as `CLINICAL_SPEC` time-series streams (`render_clinical`) and as the
`findings_render` text panel (`ROUTINE_PANEL`); both take their constants from this profile.

---

## 4. Distribution sampling

### 4.1 Two parameterizations, both from published quantities

```
(median, p_abnormal) ⇒ σ = ln(ref.high/median) / Φ⁻¹(1−p)
(mean,   sd)         ⇒ σ² = ln(1+(sd/mean)²) · median = mean/exp(σ²/2)
```

A cohort whose numbers do not determine a distribution falls back to the median and reports which
number is missing. In the prompt, "a distribution was built" and "the numbers did not line up, so it
is still a constant" look the same; `distribution_status()` is the place that tells them apart.

### 4.2 ULN definitions must match

Ma X et al. (PMC6961232; 11 studies, 4,084 patients) report that 25% of NAFLD patients have normal ALT, which looks like `p_abnormal = 0.75`.
That fraction is computed against each study's own ULN, usually lower than this repo's 40 (AASLD:
30 for men / 19 for women); pairing it with this repo's ULN is a definition mismatch, the same shape
of error as applying the mg/dL Friedewald formula to mmol/L values.
The profile therefore uses the same cohort's `mean ± sd` (41 ± 38 IU/L, independent of ULN):
median 30.07 · σ 0.7874 ⇒ abnormal rate under this repo's ULN 35.9%.

### 4.3 Umbrella diagnoses are conditioned by definition

A patient diagnosed with dyslipidemia must have at least one abnormal lipid value. Sampling each
lipid independently does not guarantee this: about half of such patients would have no abnormal
lipid at all. Umbrella conditioning samples under the constraint, so the rate of "diagnosed, yet no
abnormal lipid" is 0.

### 4.4 Paired indicators: De Ritis

Sampling AST and ALT independently makes the De Ritis ratio implausible for a large share of cases.
The `known_defect` note on AST's profile entry records this ("don't sample them independently").

`pair_constraints` applies rejection sampling (the same mechanism as the umbrella case), so the ratio
stays in a plausible band, with most early-NAFLD cases below 1.

Counter-metric: a pairwise constraint truncates the joint distribution, so the marginal moves —
ALT's abnormal rate under the constraint is lower than the registered 35.9%. The shift is measured
and reported rather than assumed to be zero.

---

## 5. Coverage

### 5.1 Scope: one declaration per quantity, not one big file

`physio_streams.yaml` (noise) and `gated_pricing.yaml` (pricing) are each a single source of truth
for their own question; folding them into the profile would only relocate them.

"A quantity may only have one way to look it up" calls for one authoritative declaration per
quantity, not one big file. The quantities duplicated across tables are `ndigits` and `unit`, and
those are declared in the profile.

### 5.2 Kinds

Profile entries cover every indicator that reaches the prompt, split by `kind` (clinical,
`monitoring`, `panel`). A daily-sampled monitoring stream belongs to no disease cohort, so
`cohorts: {}` on `monitoring` is empty and valid; inventing a cohort would create a dimension that
is never exercised.

### 5.3 `ndigits` and `CLINICAL_SPEC`

`ndigits` has exactly one declaration, read by `events._declared_ndigits` from the profile.
`_NON_METRIC_NDIGITS` remains as an empty table so the name stays discoverable.

`CLINICAL_SPEC` is a 2-tuple, `(per_kg, step)`; `base` and `ndigits` both come from the profile.

### 5.4 INV-5 is structural

Because `CLINICAL_SPEC` carries no `base`/`ndigits` copy, the tuple cannot diverge from the profile —
a stronger guarantee than reconciling two copies at runtime. A check asserts the tuple length is
exactly 2.

### 5.5 Scope of the real-EMR abnormal-rate check

`CLINICAL_SPEC[k][0]` is `per_kg`, not a baseline, so a baseline cannot be read from that slot.
The real-EMR abnormal-rate check applies only to cohorts whose profile `source` traces back to the
EMR calibration (a diabetic cohort). Applying those rates to `HbA1c/obesity` would treat every
patient as diabetic. Other cohorts are checked against their own `p_abnormal`.

---

## 6. Reference bounds, direction, and INV-7

### 6.1 Bounds and their sources

| Indicator | Bound | Source |
|---|---|---|
| `fasting_glucose` | normal < 5.6 mmol/L | ADA · IFG is 5.6–6.9. Not the same number as `diagnostic_for.T2D.ge = 7.0` |
| `CGM_TIR` | target ≥ 70% | International Consensus on Time in Range (Diabetes Care 2019;42:1593), target range 3.9–10.0 mmol/L |
| `FIB4` | < 1.30 rules out advanced fibrosis | NPV ≥90%; >2.67 rules it in. An age-stratified cutoff exists; this table has no stratification |
| `systolic/diastolic_bp` | 130 / 80 | 2017 ACC/AHA. ESC/ESH uses 140/90; with that choice a diastolic baseline of 88 would count as normal |

### 6.2 Direction

TIR is the fraction of time glucose falls in the target range, so higher is better. Judged with the
default lab-test direction, patients who meet the target would be flagged abnormal and patients who
miss it flagged normal. The profile declares `direction: lower_abnormal` for such indicators;
`abnormal_side` and `sigma_of` both use it to pick the bound and the quantile.

### 6.3 INV-7: sourced parameters can still be jointly inconsistent

An abnormal rate calibrated from EMR data counts the abnormal flag on each lab report, i.e. the
reference range of the issuing lab,
not this repo's `findings.yaml` bound. This is the same family as the ALT ULN mismatch (§4.2) and
applies to every such rate.

Combined with a stricter bound, such a rate can produce an implausible distribution, for example:

```
fasting_glucose/T2D  median 8.2 · p_abnormal 0.642 · bound 5.6
⇒ σ = 1.048 ⇒ p5 = 1.46 · p95 = 45.99      while the prompt's value domain is [3.0, 25.0]
```

INV-7 requires a solved distribution's p5/p95 to fall inside `payload_range`; otherwise the
distribution is not built (the value falls back to the median) and the reason is stated. The tails
are not clamped: clamping would make "the parameters are self-consistent" and "the parameters
conflict but were clamped" look the same in the prompt.

With these bounds, `abnormal_side` is defined for every clinical indicator, so INV-1's scan surface
is complete.

---

## 7. Stream membership: the stream manifest

The dossier answers what is known about one indicator. A second question needs its own single
source: which tables must know that a stream exists. A derived wearable stream appears in many
tables, each fail-closed on its own, and a missing entry in one of them (such as gated pricing)
would surface only downstream, in a scored evaluation, for every solver at once.

`registry/streams.yaml` declares every stream once. Derived from it:

| Table | How |
|---|---|
| `events.METRICS`, `events.AUX_WHITELIST` | at import; unit and `ndigits` read from the dossier |
| kernel `synth.AUX_SIGNALS` | registered by haenv when it mounts the kernel; the kernel keeps only the signals its own generators emit |
| physiology-layer exclusions | `physio: exclude`; a `physio: render` stream without a parameter block in `physio_streams.yaml` fails at load |
| `gates.DEVICE_SIGNALS` (GEN7) | `proves_device` |
| `wearable.DERIVED_BINDINGS`, `GRID_PARENT` | `derived_from`, `grid_parent` |
| gated pricing | written to `registry/gated_pricing_streams.yaml` by `tools/gen_stream_tables.py`, because scoring reads it; the judging fingerprint then moves only when a price does |

Parameters stay with the subsystem that owns them: noise blocks in `physio_streams.yaml`,
calibration in `haenv/wearable.py`, clinical facts in the dossier. §5.1 still holds for
quantities; the manifest declares membership.

A check requires the dossier range of every rendered stream to equal the range the pipeline
enforces. Nothing on the generation path reads those ranges for monitoring streams, so the check
does not affect any case.

---

## 8. Limitations

- the profile does not decide who reviews these values; it guarantees that every value has a
  source and every contradiction is flagged;
- INV-2 can only be judged from emitted cases, which is more expensive than the other invariants;
- the kernel's `DISEASE_SIGNAL_DOMAIN` lives in `core/latent.py` (generation segment); the profile
  reconciles against its `range`/`max_weekly_delta` rather than taking them over, because changing
  them in the kernel changes the question packs;
- `fasting_glucose` has no age- or cohort-specific bound, `FIB4` has no age stratification, and the
  blood-pressure bound follows one guideline (§6.1).

*SYNTHETIC data, evaluation use only, not medical advice.*
