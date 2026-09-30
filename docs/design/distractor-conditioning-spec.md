# Conditioning Distractor Events on the Patient

> **Scope**: which benign and life distractor events are selected for a case — `BENIGN_EVENTS` /
> `LIFE_EVENTS` / `_event_ok` / `event_prior` / `plan_benign_evs` in `haenv/events.py`, with the
> event pools in `registry/benign_events.yaml`. Injection timing, density tiers and daily metric
> streams are out of scope.
>
> SYNTHETIC, evaluation-only, not medical advice.

## 1 Purpose and the one constraint

Conditioning lets distractors depend on attributes of the patient (for example the season of the
index date, or whether the patient wears a wearable), so they look like they belong to that
patient. **A conditioning axis must be answer-neutral**: if distractor selection correlates with
the diagnosis, the set of distractors becomes a proxy for the answer.

Conditioning is opt-in per case: `latent.distractor_cond_axes` in a job file lists the axes to turn
on (default empty, which leaves generation unchanged). An unregistered or non-openable axis name
raises when the case is loaded.

## 2 Axes

`events.CONDITION_AXIS_SPECS` declares every axis as data
`(name, value domain, value source, neutrality evidence, status, note)`:

| Axis | Value source | Status |
|---|---|---|
| `season` | `events.season_of`: drawn from `case_id` | `opt_in` |
| `activity` | `events.activity_of`: wearable in `Facts.devices` (drawn from `case_id`) | `opt_in` |
| `stage` | `index_time_T` / `course_end_day` | `deferred`: `_event_ok` receives only `Facts`, which has no `T` |
| `drug` | `Facts.drug` | `opt_in_restricted`: the events it could shift are medication side effects, a driver proxy |
| `sex` | `ddx._sex_of` (prompt symptoms + `CONDITION_SEX_SKEW[spec_id]`) | `forbidden`: a function of the diagnosis |
| `disease`, `comorbidity` | `Facts` fields, or `latent.ddx_*` | `forbidden`: constant/empty in `Facts`; the `latent` fields are the gold label |

Only `opt_in` axes can be turned on. Age conditioning is the existing `age_w` prior.

## 3 Per-entry declarations

An entry in `registry/benign_events.yaml` may carry a `cond` block per axis:

```yaml
- text: 前臂晒了一下午后发红、碰到就刺痛
  topic: sunburn
  age_w: [1.4, 1.0, 0.6]
  cond:
    season:
      deny: [winter]                               # admission: cannot occur for this value
      w: {summer: 2.0, spring: 1.2, autumn: 0.8}   # multiplier after admission
    activity:
      w: {active: 1.3, sedentary: 0.8}
```

Entries unrelated to an axis are listed under `condition_neutral_items.<axis>`.

- **Admission** (`deny`) is applied in `_event_ok` via `events.cond_denied`; the rejection reason is
  `<axis>_implausible_for_profile`. Every `deny` removes the entry from that patient's pool, so it
  is reserved for events that cannot happen (sunburn in winter). The dense density tier draws up to
  9 benign and 4 life events, so the tightest patient's pools must keep headroom above those counts.
- **Multipliers** (`w`) are applied in `event_prior`:

      w(item, patient) = age_w[band] × Π_axes  m_axis(item, value)     # default m = 1.0

  Multiplication, not addition: an additive combination can drive a weight to or below zero, which
  `weighted_order` treats as "impossible", and each multiplier is a relative rate that can be
  reviewed independently. The composed multiplier is bounded by `COND_MULTIPLIER_RANGE = (1/8, 8)`.

`events.check_condition_specs` enforces at load time: complete coverage (each entry and openable axis
has a declaration or a neutral listing, never both), `w > 0` with keys equal to the domain minus
`deny` (zero is written as `deny`), registered axis names and values, a `deny` that leaves at least
one value, and the multiplier bound over every combination of axis values.

## 4 Leakage guards

The within-case check `verify.check_conclusion_invariance` and the solver-text scan cannot see
leakage through conditioning, which is a cross-case channel (distractor distribution ↔ label).
Three further layers apply:

- **L1, constructive.** An axis's value function must not read the spec, the diagnosis, `join_gold`
  or any `ddx_*` field. `events.check_axis_neutrality` scans each openable value function's code for
  the tokens in `AXIS_LEAK_TOKENS`. This is the basis for the `forbidden` axes: `sex`'s value source
  reads `spec_id`.
- **L2, text.** Distractor text must not contain answer aliases: `plan_benign_evs` scans against the
  case's own aliases plus the registry's `leak_only` words (`overlay.leak_aliases_for`).
- **L3, statistical.** The label-independence audit (best single-threshold F1 over
  `gates.case_features`, minus the all-positive baseline, against a label-permutation null). At
  pack-sized samples a spike shows a real channel, but its absence proves nothing, so admission of
  an axis rests on L1.

Independently of the axis, an entry sharing a domain with a diagnosis's differentiating clue
(photosensitivity ↔ SLE, menstrual cycle ↔ PCOS, heat intolerance ↔ Graves, thirst/polyuria ↔ the
glucose-metabolism thread) must not be amplified on a condition related to that diagnosis.

## 5 Limitations

1. The weights in `registry/benign_events.yaml` are judgments about relative incidence, not
   calibrated against incidence data. They never enter the gold label or the prompt's numbers.
2. L2 scans each case against its own aliases plus `leak_only` words, not the union of all
   diagnoses' aliases.
3. `gates.case_features` has no feature derived from a benign event's topic, so L3 does not see
   topic-level leakage; a single-feature threshold also misses signals carried jointly by several
   topics.
4. Pool-level statistics (distinct pools, identical pairs) reflect only admission; multipliers are
   visible only through `event_prior`.
5. `stage` is deferred and `drug` restricted (§2).
