"""Deterministic per-case demographic sampling for ddx questions.

Age band, devices, primary disease, drug and background comorbidities are drawn from the
counter-based RNG keyed by `case_id` only, never by spec, diagnosis or `join_gold`, so
demographics cannot predict the answer by construction. Sex is the one field set by the
caller from the question's symptoms. Age is sampled uniformly, not by disease epidemiology.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import math
from haenv.resources.regpath import registry_cached as _registry_cached

import haenv.world.generation.rng as rng

AGE_BANDS: tuple[str, ...] = ("25-29", "30-34", "35-39", "40-44", "45-49",
                              "50-54", "55-59", "60-64", "65-69", "70-74")

# A scale is always present (weight is the primary signal). Disease<->device constraints live in
# `registry/disease_device_requirements.yaml`; an empty `DISEASE_SIGNAL_DOMAIN ∩ CLINICAL_BY_DEVICE`
# would leave a device idle.
SCALES: tuple[str, ...] = ("smart_scale",)
OPTIONAL_DEVICES: tuple[str, ...] = ("wearable",)


@_registry_cached("disease_device_requirements.yaml")
def _pairing() -> dict:
    from haenv.resources.regpath import load_registry
    return load_registry("disease_device_requirements.yaml")


@_registry_cached("drug_indications.yaml")
def _indications() -> dict:
    from haenv.resources.regpath import load_registry
    return load_registry("drug_indications.yaml")


#: Indication tiers that may be sampled (`mismatch` may not).
ALLOWED_INDICATIONS: frozenset[str] = frozenset({"approved", "off_label"})


def drugs_for(disease: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Usable `(drug, dose ladder)` pairs for this disease, in registry order. Falls back to the
    global `DRUGS` when the disease is not in the table.
    """
    pairs = ((_indications().get("pairs") or {}).get(disease) or {})
    out = tuple((d, tuple(float(x) for x in (spec or {}).get("dose_steps") or ()))
                for d, spec in pairs.items()
                if indication_of(disease, d) in ALLOWED_INDICATIONS
                and (spec or {}).get("dose_steps"))
    return out or DRUGS


def _pick_drug_for(disease: str, case_id: str):
    return rng.pick(drugs_for(disease), case_id, "drug")


def indication_of(disease: str, drug: str) -> str:
    """Indication tier for `(disease, drug)`, or `unregistered`.

    The judging side reads this table itself (`gates._judging_registry`), because `regpath` merges
    plugin overlays and a judge must not depend on installed plugins.
    """
    spec = ((_indications().get("pairs") or {}).get(disease) or {}).get(drug)
    return str((spec or {}).get("status") or "unregistered")


def disease_pool() -> tuple[tuple[str, int], ...]:
    d = (_pairing().get("diseases") or {})
    return tuple((k, int((v or {}).get("weight", 1))) for k, v in d.items())

# Background comorbidities must be physiologically modeled and must not be a gold diagnosis;
# see `registry/background_comorbidity.yaml`.

DRUGS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("metformin", (500.0, 1000.0)),          # oral, once daily
    ("semaglutide", (0.25, 0.5, 1.0)),       # weekly
    ("tirzepatide", (2.5, 5.0)),             # weekly
    ("liraglutide", (0.6, 1.2, 1.8)),        # daily injection
    ("dulaglutide", (0.75, 1.5)),            # weekly
)


#: adult obesity: BMI >= 28.0 kg/m2 (WS/T 428-2013)
OBESE_BMI = 28.0
#: Synthetic height sampling bounds: historical GB/T 10000-1988 tables 1–2, P5/P95.
#: Not clinical limits or current population percentiles; see the inv-small ADR.
HEIGHT_RANGE_M = {"M": (1.583, 1.775), "F": (1.484, 1.659)}
#: Synthetic BMI sampling range, not a clinical upper threshold.
OBESE_BMI_DRAW = (28.0, 32.0)


#: Sampling policy for unqualified current-period complaints: use bands ending by 54.
#: WHO (2024-10-16) gives 45–55 as typical menopause ages, not a universal cutoff;
#: this synthetic restriction does not diagnose menopause from age.
MENSES_WORDS = ("月经", "例假", "经期", "来事")
MENSES_MAX_AGE = 54


def menstruates(symptom_texts) -> bool:
    return any(any(w in str(t) for w in MENSES_WORDS) and "绝经" not in str(t) for t in symptom_texts or ())


def pack_height_m(case_id: str, sex: str) -> float:
    if str(sex) not in HEIGHT_RANGE_M:
        raise ValueError(f"unknown sex for height sampling: {sex}")
    lo, hi = HEIGHT_RANGE_M[str(sex)]
    return lo + (hi - lo) * rng.unit(case_id, "height")


def obese_weight_scale(case_id: str, sex: str, conditions, start_kg: float) -> float:
    """Factor on the whole weight course of a patient whose conditions include obesity, so the
    starting weight reaches BMI 28 at the patient's height; 1.0 when it already does or when
    obesity is not a condition. A rescaled patient lands on a BMI drawn from `OBESE_BMI_DRAW`."""
    if "obesity" not in {str(c) for c in conditions or ()}:
        return 1.0
    if not math.isfinite(start_kg) or start_kg <= 0:
        raise ValueError("obese patient requires a finite, positive starting weight")
    h = pack_height_m(case_id, sex)
    if start_kg >= OBESE_BMI * h * h:
        return 1.0
    lo, hi = OBESE_BMI_DRAW
    return (lo + (hi - lo) * rng.unit(case_id, "obese_bmi")) * h * h / start_kg


def doses_per_week(drug: str) -> int:
    """Doses per week for this drug, from `gates.DRUG_DOSES_PER_WEEK` (also read by GEN20b)."""
    from haenv.world.validation.gate_tables import DRUG_DOSES_PER_WEEK
    return DRUG_DOSES_PER_WEEK[drug]


#: a patient's own words that she is not fat; such a patient is never drawn with obesity
LEAN_WORDS = ("不胖", "一直瘦", "瘦瘦的", "偏瘦", "一直很瘦", "没有肥胖")


def is_lean(symptom_texts) -> bool:
    return any(any(w in str(t) for w in LEAN_WORDS) for t in symptom_texts or ())


def sample_profile(case_id: str, sex: str, avoid: tuple = (), lean: bool = False) -> dict:
    """Sample one demographic profile keyed by `case_id`; never takes the diagnosis as input.

    `avoid` lists primary diseases already used by other variants of the same condition; they are
    avoided when the pool allows, otherwise sampling proceeds normally. `lean`: the patient says she
    is not fat (`is_lean`), so obesity is neither the primary disease (redrawn without it) nor a
    comorbidity; every other draw is unchanged.
    """
    age = rng.pick(AGE_BANDS, case_id, "age")
    scale = rng.pick(SCALES, case_id, "scale")

    # Primary disease by registry weight (weights expanded into repeated entries, since `rng` has
    # no weighted primitive). Every kernel disease domain includes `weight`.
    _pool = disease_pool()
    _bag = [name for name, w in _pool for _ in range(max(1, w))]
    disease = rng.pick(_bag, case_id, "disease")
    if avoid and disease in set(avoid):
        _left = [x for x in _bag if x not in set(avoid)]
        if _left:                       # only avoid when the pool is not exhausted; otherwise sample normally
            disease = rng.pick(_left, case_id, "disease_avoid")
    if lean and disease == "obesity":
        disease = rng.pick([x for x in _bag if x != "obesity" and x not in set(avoid)] or
                           [x for x in _bag if x != "obesity"], case_id, "disease_lean")

    _spec = (_pairing().get("diseases") or {}).get(disease) or {}
    _uni = (_pairing().get("universal") or {})
    _req = list(_uni.get("required") or [scale]) + list(_spec.get("requires_devices") or [])
    _optpool = list(_uni.get("optional") or []) + list(_spec.get("optional_devices") or [])
    opt = rng.subset(_optpool, 0, len(_optpool), case_id, "devices")
    devices = list(dict.fromkeys(_req + opt))

    drug, steps = _pick_drug_for(disease, case_id)

    comorbidities = [c for c in _sample_comorbidities(disease, case_id) if not (lean and c == "obesity")]

    return {
        "age_range": age,
        "sex": sex,
        "devices": devices,
        "drug": drug,
        "dose_steps": list(steps),
        "comorbidities": comorbidities,
        "disease": disease,
    }


@_registry_cached("background_comorbidity.yaml")
def _comorbid_registry() -> dict:
    from haenv.resources.regpath import load_registry
    return load_registry("background_comorbidity.yaml") or {}


def comorbid_pool() -> tuple[tuple[str, float], ...]:
    el = (_comorbid_registry().get("eligible") or {})
    return tuple((k, float((v or {}).get("rate", 0.0))) for k, v in el.items())


def comorbid_excluded() -> dict:
    return dict(_comorbid_registry().get("excluded") or {})


def _sample_comorbidities(disease: str, case_id: str) -> list[str]:
    """Background comorbidities for this case: independent coin flips in registry order (never
    sorted), capped at `max_per_case`, never the primary disease.
    """
    cap = int(_comorbid_registry().get("max_per_case") or 2)
    out: list[str] = []
    for name, rate in comorbid_pool():
        if name == disease:
            continue
        if len(out) >= cap:
            break
        if rng.unit(case_id, "comorbid", name) < rate:
            out.append(name)
    return out


def _band_bounds(band: str) -> tuple[int, int]:
    lo, hi = band.split("-", 1)
    return int(lo), int(hi)


def _prevalence(cond: str, band: str, registry=None) -> float | None:
    registry = _comorbid_registry() if registry is None else registry
    rows = ((registry.get("age_prevalence") or {}).get(cond) or {}).get("rows")
    if not rows:
        return None
    lo, hi = _band_bounds(band)
    mid = (lo + hi) / 2.0
    for a, b, p in rows:
        if a <= mid <= b:
            return float(p)
    return None


def age_weights(conditions, min_age=None, max_age=None) -> dict[str, float]:
    """P(age band | conditions) up to a constant: uniform prior over `AGE_BANDS` times the
    registry prevalence of each condition in that band (`age_prevalence`); bands outside the
    spec's `min_age` / `max_age` get 0. A condition without a row leaves the weights as they are."""
    out = {}
    registry = _comorbid_registry()
    for band in AGE_BANDS:
        lo, hi = _band_bounds(band)
        if (min_age is not None and lo < int(min_age)) or (max_age is not None and hi > int(max_age)):
            out[band] = 0.0
            continue
        w = 1.0
        for c in dict.fromkeys(str(x) for x in conditions or () if x):
            p = _prevalence(c, band, registry)
            if p is not None:
                w *= p
        out[band] = w
    return out


def pack_age_band(case_id: str, conditions, current: str | None = None, min_age=None, max_age=None) -> str:
    """The age band of a plugin pack's patient: drawn from `age_weights` keyed on `case_id`
    (deterministic). With no usable weight (every band excluded) the current band stays."""
    w = age_weights(conditions, min_age, max_age)
    tot = sum(w.values())
    if tot <= 0:
        return current or AGE_BANDS[0]
    u = rng.unit(case_id, "age_given_conditions") * tot
    acc = 0.0
    for band in AGE_BANDS:
        acc += w[band]
        if u < acc and w[band] > 0:
            return band
    return max(w, key=w.get)


def pregnancy_capable(sex: str, age_range: str) -> bool:
    """Sex F and an age band starting at or below `pregnancy_capable.age_max`."""
    t = _comorbid_registry().get("pregnancy_capable") or {}
    try:
        lo = int(str(age_range).split("-", 1)[0])
    except ValueError:
        return False
    return str(sex) == "F" and lo <= int(t.get("age_max", 44))


def avoid_for(drug: str, sex: str, age_range: str) -> bool:
    """True when `drug` is on the avoid list for a woman who can become pregnant (unless indicated:
    the caller decides whether the indication overrides)."""
    t = _comorbid_registry().get("pregnancy_capable") or {}
    return pregnancy_capable(sex, age_range) and str(drug).split("_")[0] in set(t.get("avoid") or ())


def profile_fields_are_answer_neutral(specs: dict) -> list[str]:
    bad: list[str] = []
    for cid, spec in (specs or {}).items():
        a = sample_profile(cid, "F")
        b = sample_profile(cid, "F")
        if a != b:
            bad.append(f"{cid}: sampling the same case_id twice gave different results (non-deterministic)")
    return bad
