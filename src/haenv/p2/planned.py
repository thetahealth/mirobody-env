"""Generation-only planning of pack 2 history and medication premises. P2_WORLD."""
from __future__ import annotations

from haenv.world.history.history_dependency import rate, uniform


def planned_conditions(plan):
    """Diagnoses knowable before OBSERVE, using the existing history decisions.

    The caller supplies context_cid for native history draws. No rendered text,
    realized measurements or new random draw participates in this plan.
    """
    from haenv.p2 import rules as R
    from haenv.p2.render import needs
    from haenv.world.validation.difficulty_r2 import enabled
    spec = R.table()["tiers"][plan["tier"]]
    conditions = needs(plan["tier"], plan["stratum"], bool(plan.get("beta_blocker")),
                       plan.get("comorbid") or (), plan.get("dm_agent"))["add_conditions"]
    conditions = list(dict.fromkeys([*conditions, *plan.get("known_conditions", ())]))
    from haenv.p2.reading import enabled as reading_enabled, core as reading_core
    if reading_enabled(plan) and reading_core(plan['context_cid'], plan) and spec['rule'] == 'R11_hypo_level1':
        conditions.append('atrial_fibrillation')
    if enabled(plan) and spec.get("context"):
        family = spec["context"]
        if family == "volume" or family == "rhythm" and spec["acts"]:
            conditions.append("heart_failure" if family == "volume" else "atrial_fibrillation")
        return list(dict.fromkeys(conditions))
    cid = plan["context_cid"]
    role = uniform("history-role", cid) < rate("p2", plan)
    rule = spec["rule"]
    hf = rule in ("R8_hf_gain", "R7a_htn_severe_hf") or enabled(plan) and "heart_failure" in conditions
    if enabled(plan):
        family = "heart_failure" if hf else "atrial_fibrillation"
        if not hf and rule in ("R2_news2_medium", "R10_news2_low", "R10b_infection_medium_risk", "R13_news2_zero_symptom", "R14_nothing"):
            choices = R.table()["clinical_context"]["native_history_families"]["weights"]
            draw = uniform("p2-native-history-family", cid) * sum(choices.values())
            family = plan.get("history_family")
            if not family:
                for candidate, weight in choices.items():
                    draw -= weight
                    if draw < 0:
                        family = candidate
                        break
        if hf or role and rule != "R9_af_alerts":
            conditions.append(family)
    elif role and (hf or rule == "R14_nothing"):
        conditions.append("heart_failure" if hf else "atrial_fibrillation")
    return list(dict.fromkeys(conditions))



def prepare_assignment(raw, cid):
    """Capture assigned courses before observation; queries use the stored plan."""
    from haenv.world.truth import coupling_enabled, soc
    if not coupling_enabled():
        return
    from haenv.p2.gate import plan_of
    from haenv.world.validation.difficulty_r2 import enabled
    plan = plan_of(raw)
    if plan is None:
        return
    pb = raw.latent_premise.get("patient_basics") or {}
    raw.latent_premise["p2_planned_conditions"] = planned_conditions({
        **plan, "context_cid": cid,
        "known_conditions": [pb.get("disease"), *pb.get("comorbidities", ())],
    })
    if rate("p2", plan) > 0 or enabled(plan):
        wm = raw.latent_premise.get("world_medication") or {}
        plan["soc_oac"] = [m for m in wm.get("concurrent", ()) if "oac" in soc.drug_classes(m.get("drug", ""))]
