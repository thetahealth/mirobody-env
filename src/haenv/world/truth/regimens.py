"""Question-owned medicines and sourced background courses, generation only.

Declarations name existing recipe/registry dependencies. They are saved on the
case so readers never reconstruct them from the current generation settings.
SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from copy import deepcopy


def declarations(p, overlay, legacy=(), *, case_id=None, T=None, course_end=None):
    from haenv.contracts.external_gold import SLOT
    from haenv.resources.regpath import load_registry
    ext, out = (p.meta or {}).get(SLOT) or {}, []
    primary = (p.patient_basics.get("regimen") or {}).get("drug")

    def add(drug, reason, source):
        row = {"drug": drug, "reason": reason, "source": source}
        if drug and row not in out:
            out.append(row)

    if plan := ext.get("p3"):
        for medicine in overlay:
            add(medicine["drug"], "target or registered comorbidity premise", "p3_meds.yaml:drugs/comorbidities")
        if case_id is not None and T is not None and course_end is not None:
            from haenv.p3.plan import background_on, h01
            from haenv.p3.realize import _tier
            from haenv.world.clinical import med_course as mc
            from haenv_kernel.latent import drug_pkpd
            confounders = load_registry("p3_meds.yaml").get("confounders", {}).get("other_drug", {})
            reg = p.patient_basics.get("regimen") or {}
            # Preserve only the old diary's selected, active confounder. The
            # exact salt/rate/dose test is the existing p3 realization rule;
            # merely appearing in the confounder registry is not decisive.
            pk = drug_pkpd(primary) or {}
            kwargs = {"min_titration_days": pk["min_titration_days"]} if "min_titration_days" in pk else {}
            wm = {"primary": {"drug": primary, "dose_timeline": mc.dose_course(
                case_id, list(reg.get("dose_steps") or ()), T, course_end, **kwargs)},
                "concurrent": list(legacy)}
            if background_on():
                for drug, dose, medicine in mc.active_medicines(wm, T):
                    if drug == plan["drug"] or medicine.get("overlay"):
                        continue
                    row = confounders.get(drug)
                    if row and h01("p3od", case_id, drug) < _tier(row["rate"], dose):
                        add(drug, "registered other-drug confounder", "p3_meds.yaml:confounders.other_drug")
                        break
    if plan := ext.get("p2"):
        from haenv.p2.render import WORLD_DRUG, _dm_agent
        from haenv.p2 import rules
        tier = rules.table()["tiers"][plan["tier"]]
        if plan.get("beta_blocker"):
            add(WORLD_DRUG[plan.get("bb_kind") or "bb"][0], "plan beta blocker", "p2.render:WORLD_DRUG/beta_blocker")
        dm = _dm_agent(plan)
        if dm in ("insulin", "su"):
            for medicine in overlay:
                if medicine["drug"] == WORLD_DRUG[dm][0]:
                    add(medicine["drug"], "plan glucose-lowering agent", "p2.render:DM_AGENT/_dm_agent")
        if tier.get("context") == "volume":
            add("furosemide", "dated diuretic reduction read by volume rule", "p2_history_triage.yaml:clinical_context")
        if tier["rule"] == "R0a_head_anticoag":
            from haenv.world.truth.soc import drug_classes
            for medicine in overlay:
                if "oac" in drug_classes(medicine["drug"]):
                    add(medicine["drug"], "head-injury anticoagulant premise", "p2_history_triage.yaml:R0a_head_anticoag")
    if plan := ext.get("p4"):
        cfg = load_registry("p4_diversity.yaml")
        an = plan.get("analyte")
        factors = {plan.get("cause"), (plan.get("factor") or {}).get(an)}
        factors.update((plan.get("decoys") or {}).get(an, ()))
        # The diverse renderer writes both active and inactive factor courses.
        # Their timing is read by the gold even when another factor is causal.
        if plan.get("difficulty_r2"):
            related = cfg.get("analytes", {}).get(an, {})
            for kind in ("pre", "method", "true_causes"):
                factors.update(related.get(kind, ()))
        have = {p.patient_basics.get("disease"), *p.patient_basics.get("comorbidities", ())}
        for factor in sorted(f for f in factors if f):
            if row := cfg.get("medication_world", {}).get(factor):
                indication = (cfg.get("rules", {}).get(factor) or {}).get("indication")
                if indication and indication not in have:
                    continue
                add(row["drug"], "registered item factor: " + factor, "p4_diversity.yaml:medication_world." + factor)
        if not plan.get("difficulty_r2") and an == "fasting_glucose":
            add(primary, "primary glucose course read by item", "labworld.yaml:effects.glp1_on_fasting_glucose")
        if not plan.get("difficulty_r2") and an == "creatinine":
            add("enalapril", "ACEi course read by item", "labworld.yaml:effects.acei_on_creatinine")
    if ext.get("m2") or ext.get("m2_bridge"):
        from haenv.m2.premises import requirements
        from haenv.world.clinical.med_course import CONCURRENT
        for component, check in requirements(p.meta.get("ddx") or {}):
            if check["kind"] == "medication_exposure":
                add(CONCURRENT[check["indication"]][0][0], "diagnosis exposure premise: " + component, check["source"])
        # The F2 question explicitly asks about its named primary drug; both the
        # visible scene and gold's dose-state/target relationship read that drug.
        if (ext.get("m2") or {}).get("frame") == "F2":
            add(primary, "named F2 medication direction and dose state", "m2_pre_review.yaml")
    return out


def finalize_dependencies(raw):
    """Save dependencies of records realized after background-course assignment."""
    lp = raw.latent_premise or {}
    if "decisive_medications" not in lp:
        return
    g = (raw.adjudication or {}).get("meds_p3") or {}
    if not g:
        return
    from haenv.p3.policy import specification
    from haenv.resources.regpath import load_registry
    out = lp["decisive_medications"]

    def add(drug, reason, source):
        row = {"drug": drug, "reason": reason, "source": source}
        if row not in out:
            out.append(row)

    administration = g.get("administration_record") or {}
    supplement = administration.get("supplement")
    if supplement in specification()["levothyroxine_absorption"]["supplements"]:
        add(supplement, "recorded LT4 absorption interaction", "p3_clinical_sources.yaml:levothyroxine_absorption")
    confounders = load_registry("p3_meds.yaml").get("confounders", {}).get("other_drug", {})
    for entry in g.get("diary", ()):
        drug = entry.get("source_drug")
        if entry.get("cause") == "other_drug" and drug in confounders:
            add(drug, "registered other-drug confounder", "p3_meds.yaml:confounders.other_drug")


def current_background_courses(medicines, day, decisive=()):
    """Condition a current-use draw on taking therapy at the observation visit.

    NHANES class rates describe current use. A legacy practice-course hold that
    is still ongoing cannot turn a selected current user into an untreated draw.
    Keep completed historical holds, future records and question-owned courses.
    Reuse the last verified positive dose; never invent an initiation or dose.
    """
    from haenv.world.clinical.med_course import dose_at
    for medicine in medicines:
        if not medicine.get("soc") or not medicine.get("drug") or medicine["drug"] in decisive:
            continue
        points = medicine.get("dose_timeline") or []
        if dose_at(points, day) > 0:
            continue
        positive = [p for p in points if p["ts"] <= day and p["value"] > 0]
        if not positive:
            continue
        last = max(positive, key=lambda p: p["ts"])
        medicine["dose_timeline"] = [p for p in points if not
            (last["ts"] < p["ts"] <= day and p["value"] == 0)]
        medicine["current_use_conditioned"] = True


def finalize_m2_record_facts(raw):
    """Refresh existing M2 rubric facts after background notes are recorded."""
    adj = raw.adjudication or {}
    snapshot = adj.get("open_management")
    if ("decisive_medications" not in (raw.latent_premise or {})
            or not adj.get("m2_workup") or not snapshot
            or snapshot.get("visible_facts") == raw.prediction_context):
        return
    from haenv.scoring.management.plan import finalize
    finalize(raw, str(raw.case_id), "m2", derive_workup=False)


def prepare(p, cid, T, ce):
    """Replace covered background assignments once, before any laboratory render."""
    if "background_regimen" in p.meta:
        return p.meta["background_regimen"]
    from haenv.contracts.external_gold import SLOT
    if not set((p.meta or {}).get(SLOT) or {}) & {"m2", "m2_bridge", "p2", "p3", "p4"}:
        return None
    from haenv.world.truth import soc
    from haenv.world.clinical import med_course as mc
    from haenv.world.generation.build_clinical import _overlay_kw, _meds_history
    pb = p.patient_basics
    overlay = _overlay_kw(p)
    old = deepcopy(pb.get("regimen") or {})
    disease, original = str(pb.get("disease") or ""), str(old.get("drug") or "")
    history = _meds_history(p, cid, T, ce)
    legacy = mc.concurrent_medicines(cid, disease, pb.get("comorbidities") or (), original, ce,
        sex=pb.get("sex"), age_range=pb.get("age_range"), overlay=overlay, history=history, legacy_only=True)
    decisive = declarations(p, overlay, legacy, case_id=cid, T=T, course_end=ce)
    drugs = {r["drug"] for r in decisive}
    replace_primary = original not in drugs
    kw = soc.premise_kwargs(p, cid)
    context = {**kw["constraints"], "decisive_medications": sorted(drugs), "background_regimen": True}
    concurrent = mc.concurrent_medicines(cid, disease, pb.get("comorbidities") or (),
        "" if replace_primary else original, ce, sex=pb.get("sex"), age_range=pb.get("age_range"),
        overlay=overlay, history=history, constraints=context)
    current_background_courses(concurrent, T, drugs)
    primary = None
    if replace_primary:
        # Preserve registry ordering (metformin first in the T2D ladder). An
        # untreated/uncovered draw stays untreated: never force an extra class
        # or invent a rate for an indication absent from the registry.
        primary = next((m for m in concurrent if m.get("soc") and m.get("drug") and m.get("indication") == disease), None)
        if primary is not None:
            concurrent.remove(primary)
            points = primary["dose_timeline"]
            # A stable historical prescription may have no record inside the
            # observation window. Primary trajectories require such a record;
            # carry the existing dose forward at the window origin, without
            # changing exposure or inventing a prescription/dose change.
            if not any(0 <= q["ts"] <= T for q in points):
                dose = mc.dose_at(points, 0)
                if dose > 0:
                    primary["dose_timeline"] = sorted(
                        [*points, {"ts": 0, "value": dose}], key=lambda q: q["ts"])
            drug = primary["drug"]
            pb["regimen"] = {"drug": drug, "dose_steps": list(soc.dose_ladder(drug) or ())}
        else:
            pb["regimen"] = {"drug": "", "dose_steps": []}
            primary = {"drug": "", "dose_timeline": [], "indication": disease, "soc": True,
                       "source": "standard_of_care.yaml:conditions",
                       "reason": "no registered therapy drawn for primary indication"}
        from haenv.world.validation.gate_tables import DRUG_DOSES_PER_WEEK
        p.event_density.pop("dosing_per_week", None)
        if primary["drug"]:
            p.event_density["dosing_per_week"] = DRUG_DOSES_PER_WEEK[primary["drug"]]
    record = {"decisive_medications": decisive, "concurrent": concurrent,
              "primary": primary, "legacy_primary": old}
    p.meta["background_regimen"] = record
    return record
