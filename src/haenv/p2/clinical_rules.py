"""P2-only rule/table extension; common hospital-library inputs retain their bytes."""
from copy import deepcopy
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
    result = _base_assess(block)
    r = result["readings"]
    if r["r0"]:
        return result
    if r["head_anticoag"]:
        rid = "R0a_head_anticoag"
    elif r["chemo_unwell"]:
        rid = "R0b_chemo_unwell"
    elif (result["rule"] == "R10_news2_low" and r["stable_rate"] and r["news2"] == 1
          and any(lo <= block["home_check_today"]["heart_rate_bpm"] <= hi
                  for lo, hi, points in table()["news2"]["hr"] if points == 1 and hi < 90)):
        rid = "R10a_stable_rate"
    else:
        return result
    accepted = list(R.rule_of(rid)["set"])
    single = table()["acceptance"]["single_level2"]
    if r["level2_nights"] == 1 and R.floor_of(accepted) in single["when_floor"]:
        accepted = list(single["set"])
    return dict(rule=rid, set=accepted, readings=r)


def install():
    R.table, R.readings, R.assess = table, readings, assess

