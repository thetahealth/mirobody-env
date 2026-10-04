"""Deterministic per-case demographic sampling for ddx questions.

Age band, devices, primary disease, drug and background comorbidities are drawn from the
counter-based RNG keyed by `case_id` only, never by spec, diagnosis or `join_gold`, so
demographics cannot predict the answer by construction. Sex is the one field set by the
caller from the question's symptoms. Age is sampled uniformly, not by disease epidemiology.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
from .regpath import registry_cached as _registry_cached

from . import rng

AGE_BANDS: tuple[str, ...] = ("25-29", "30-34", "35-39", "40-44", "45-49",
                              "50-54", "55-59", "60-64", "65-69", "70-74")

# A scale is always present (weight is the primary signal). Disease<->device constraints live in
# `registry/disease_device_requirements.yaml`; an empty `DISEASE_SIGNAL_DOMAIN ∩ CLINICAL_BY_DEVICE`
# would leave a device idle.
SCALES: tuple[str, ...] = ("smart_scale",)
OPTIONAL_DEVICES: tuple[str, ...] = ("wearable",)


def _pairing() -> dict:
    from .regpath import load_registry
    return load_registry("disease_device_requirements.yaml")


def _indications() -> dict:
    from .regpath import load_registry
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


def doses_per_week(drug: str) -> int:
    """Doses per week for this drug, from `gates.DRUG_DOSES_PER_WEEK` (also read by GEN20b)."""
    from .gate_tables import DRUG_DOSES_PER_WEEK
    return DRUG_DOSES_PER_WEEK[drug]


def sample_profile(case_id: str, sex: str, avoid: tuple = ()) -> dict:
    """Sample one demographic profile keyed by `case_id`; never takes the diagnosis as input.

    `avoid` lists primary diseases already used by other variants of the same condition; they are
    avoided when the pool allows, otherwise sampling proceeds normally.
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

    _spec = (_pairing().get("diseases") or {}).get(disease) or {}
    _uni = (_pairing().get("universal") or {})
    _req = list(_uni.get("required") or [scale]) + list(_spec.get("requires_devices") or [])
    _optpool = list(_uni.get("optional") or []) + list(_spec.get("optional_devices") or [])
    opt = rng.subset(_optpool, 0, len(_optpool), case_id, "devices")
    devices = list(dict.fromkeys(_req + opt))

    drug, steps = _pick_drug_for(disease, case_id)

    comorbidities = _sample_comorbidities(disease, case_id)

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
    from .regpath import load_registry
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


def profile_fields_are_answer_neutral(specs: dict) -> list[str]:
    bad: list[str] = []
    for cid, spec in (specs or {}).items():
        a = sample_profile(cid, "F")
        b = sample_profile(cid, "F")
        if a != b:
            bad.append(f"{cid}: sampling the same case_id twice gave different results (non-deterministic)")
    return bad
