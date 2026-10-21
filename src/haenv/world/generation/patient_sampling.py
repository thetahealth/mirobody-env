"""Patient sampling shared by the pack generators, outside the clinical core.

The diagnosis template supplies only diagnostic facts and hidden truth. Every
patient has a fresh identity and profile; independent-type complaints are drawn
from the sourced event pool with its existing age/context priors. No model calls.
"""
from __future__ import annotations

import copy
import hashlib
import functools
from types import SimpleNamespace


@functools.lru_cache(maxsize=1)
def _specs():
    from haenv.world.overlay import condition_registry
    return condition_registry(include_draft=True)


def benign_symptoms(case: dict) -> dict:
    """Replace an independent diagnosis's template complaints, preserving count/timing.

    Signal-bearing complaints of unified/comorbidity specs remain diagnostic
    facts. Their visible wording is already sampled by events.enter_symptoms.
    """
    if case["latent"].get("ddx_join_gold") != "independent":
        return case
    import haenv.world.observations.events as E
    from haenv.app.job import driver_of
    c = copy.deepcopy(case)
    f = E.facts_of(SimpleNamespace(**c))
    cond = E.normalize_cond_axes(f.cond_axes)
    banned = E.event_proxy_tags(f, driver_of(c["case_id"], c["latent"]))
    side_effects = E.side_effect_topics(E.shown_drugs(f.case_id, f.disease, f.comorbidities, f.drug,
                                                    sex=f.sex, age_range=c["raw"]["age_range"]))
    lay = E.symptom_lay(f.sex)
    pool = [e for e in E.event_pools()[0] if E._event_ok(e, f, banned, cond=cond)[0]
            and e["topic"] not in side_effects
            and e["text"] in lay]
    old = c["raw"].get("symptoms") or []
    order = E.weighted_order(pool, f, "patient-complaints", cond)
    if len(order) < len(old):
        raise ValueError(f"{c['case_id']}: benign phrase pool cannot supply {len(old)} complaints")
    c["raw"]["symptoms"] = [{"day": int(s["day"]), "text": pool[k]["text"],
                               "context": pool[k]["context"]} for s, k in zip(old, order)]
    return c


def fresh_patient(case: dict, pack: str, seed: str) -> dict:
    """Draw a patient independently of other patients of the same spec.

    All probabilities are read by demographics from disease/device requirements,
    drug indications and background_comorbidity. No cross-variant `avoid` list.
    The relative weight course is a diagnostic fact and remains unchanged.
    """
    import haenv.world.generation.demographics as D
    import haenv.world.clinical.ddx as ddx
    import haenv.world.generation.rng as rng
    c = copy.deepcopy(case)
    cid = "PT-" + hashlib.sha256(f"{pack}|{seed}|{case['case_id']}".encode()).hexdigest()[:20]
    c["case_id"] = cid
    spec_id = c["latent"]["ddx_spec_id"]
    spec = _specs()[spec_id]
    sex = ddx._sex_of(spec, case_id=cid, spec_id=spec_id)
    symptoms = c["raw"].get("symptoms") or []
    profile = D.sample_profile(cid, sex, lean=D.is_lean(s["text"] for s in symptoms))
    profile = ddx.clinical_profile(cid, spec_id, profile)
    c["raw"].update(profile)
    scale = 0.8 + 0.4 * rng.unit(cid, "patient-weight")
    if D.is_lean(s["text"] for s in symptoms):
        # Respect a diagnostic statement that the patient is lean. This scales
        # the whole course, never a single endpoint or its relative change.
        h = D.pack_height_m(cid, sex)
        top = max(float(case["raw"][k]) for k in ("start_weight", "nadir_weight"))
        top = max(top, float(c["latent"].get("regain_end_kg", top)))
        scale = min(scale, 27.0 * h * h / top)
    for key in ("start_weight", "nadir_weight"):
        c["raw"][key] = round(float(case["raw"][key]) * scale, 1)
    if "regain_end_kg" in c["latent"]:
        c["latent"]["regain_end_kg"] = round(float(c["latent"]["regain_end_kg"]) * scale, 1)
    c["latent"]["event_density"]["dosing_per_week"] = D.doses_per_week(profile["drug"])
    return benign_symptoms(c)
