"""P2-only rule/table extension; common hospital-library inputs retain their bytes."""
from copy import deepcopy
from haenv.i18n.text import L
from haenv.p2 import rules as R
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached

_base_table, _base_readings, _base_assess = R.table, R.readings, R.assess


@registry_cached("pack2_triage_scenarios.yaml", "p2_history_triage.yaml")
def table():
    base, extra = deepcopy(_base_table()), load_registry("p2_history_triage.yaml")
    for key, value in extra.items():
        if key == "rules":
            for rule in value:
                rule = dict(rule)
                rule["printed"] = L(rule["printed"])
                before = rule.pop("before")
                base[key].insert(next(i for i, r in enumerate(base[key]) if r["id"] == before), rule)
        elif key == "tiers":
            base[key] = {**value, **{k: v for k, v in base[key].items() if k not in value}}
        elif key in ("sources", "thresholds", "strata"):
            base[key].update(value)
        else:
            base[key] = value
    return base


def readings(block):
    from haenv.p2.context import facts
    return {**_base_readings(block), **facts(block)}


def assess(block):
    return _assess(block, visible=False)


def assess_visible(block):
    return _assess(block, visible=True)


def _assess(block, *, visible):
    result = _base_assess(block)
    r = result["readings"]
    if visible:
        from haenv.p2.context import facts
        r["relative_hypotension"] = facts(block, home_baseline=True)["relative_hypotension"]
    if r["r0"]:
        return result
    if r["head_anticoag"]:
        rid = "R0a_head_anticoag"
    elif r["chemo_unwell"]:
        rid = "R0b_chemo_unwell"
    elif r["relative_hypotension"] and result["rule"] not in (
            "R1_news2_high", "R2_news2_medium", "R3a_single3_vital"):
        rid = "R3d_relative_hypotension"
    elif (result["rule"] == "R10_news2_low" and r["stable_rate"] and r["news2"] == 1
          and any(lo <= block["home_check_today"]["heart_rate_bpm"] <= hi
                  for lo, hi, points in table()["news2"]["hr"] if points == 1 and hi < 90)):
        rid = "R10a_stable_rate"
    elif result["rule"] == "R10_news2_low" and r["infection_medium_risk"]:
        rid = "R10b_infection_medium_risk"
    elif result["rule"] == "R14_nothing" and moderate_postdischarge_gain(block, r):
        if result["set"] == ["watchful_waiting"]:
            result["set"].append("routine_followup")
        return result
    else:
        return result
    accepted = list(R.rule_of(rid)["set"])
    single = table()["acceptance"]["single_level2"]
    if r["level2_nights"] == 1 and R.floor_of(accepted) in single["when_floor"]:
        accepted = list(single["set"])
    return dict(rule=rid, set=accepted, readings=r)


def moderate_postdischarge_gain(block, readings):
    from haenv.i18n.text import is_text
    from haenv.p2.narrative import records
    gain = readings["weight_gain_3d"]
    limit = table()["thresholds"]["hf_gain_kg_gt"]
    return (readings["heart_failure"] and gain is not None and limit / 2 < gain <= limit and
            any(is_text(r.get("medication_name"), "drug.furosemide")
                and r.get("discharge_day") is not None and r.get("dose_change_day") == r["discharge_day"]
                and r.get("previous_dose_mg", 0) > r.get("current_dose_mg", 0)
                and r["discharge_day"] < block["today_day"] for r in records(block)))


def af_clauses() -> list:
    """P2's printed AF exemption (`p2_history_triage.yaml` `af_exemption`), one phrase per clause:
    the documented-AF exemption from R9 needs no dizziness or mild new symptom ticked today, as
    `rules.assess` reads it."""
    return [L(k) for k in table()["af_exemption"]["protocol"]]


def af_protocol():
    from haenv.i18n.text import J
    return J("lines", af_clauses())


def install():
    R.table, R.readings, R.assess = table, readings, assess

