"""Derive conditional initial-stage requirements from the generated record."""
from __future__ import annotations

import json


def install():
    from haenv.app.pack_hooks import register_pack_extension
    register_pack_extension("m2:workup", register)


def register(pack):
    if pack.name != "m2":
        return
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    def with_workup(next_call, raw, case_id):
        result = next_call(raw, case_id)
        finalize(raw, case_id)
        return result
    register_hook(Stage.OBSERVE, "m2:workup", with_workup)


def finalize(raw, case_id):
    from haenv.m2 import core as C
    from haenv.m2.workup import pre_review_rules
    from haenv.m2.workup import workup_tests
    plan = raw.adjudication.get("m2_workup")
    if plan is None:
        return
    ddx = raw.adjudication["ddx"]
    sid = str(ddx["spec_id"])
    comps = C.components(sid)
    T = int(raw.prediction_context["prediction_time_T"])
    optional, facts = [], {}

    def point(fid):
        rows = C._on_demand(raw, fid, T) or []
        return max(rows, key=lambda r: int(r.get("ts", T))) if rows else {}

    if "HD-UNI-01" in comps:
        ttg, iga = point("tTG_IgA"), point("TotalIgA")
        # Equivocal or weakly positive reports retain mandatory EMA. The
        # numeric assay currently models a positive rather than equivocal band.
        positive = (isinstance(ttg.get("value"), (int, float))
                    and ttg.get("flag") in ("H", "HIGH", "POSITIVE")
                    and ttg["value"] > ttg.get("ref_high", float("inf"))
                    and not ttg.get("weakly_positive") and not ttg.get("equivocal"))
        adequate = (isinstance(iga.get("value"), (int, float))
                    and iga.get("ref_low") is not None and iga["value"] >= iga["ref_low"])
        facts["celiac"] = {"tTG_IgA": ttg, "TotalIgA": iga,
                            "positive_not_equivocal": positive, "adequate_IgA": adequate}
        if positive and adequate:
            optional.append(pre_review_rules()["conditional_optional"]["HD-UNI-01"]["test"])
    if "JD-PCOS" in comps:
        testosterone = point("Testosterone")
        records = raw.prediction_context.get("event_history_record") or []
        menstrual = [r for r in records if int(r.get("day", T + 1)) <= T
                     and any(s in json.dumps(r, ensure_ascii=False) for s in ("月经稀发", "周期推迟40", "周期超过35"))]
        androgen = (isinstance(testosterone.get("value"), (int, float))
                    and testosterone.get("ref_high") is not None
                    and testosterone["value"] > testosterone["ref_high"])
        adult = C._age_mid(raw) is not None and C._age_mid(raw) >= 18
        facts["pcos"] = {"ovulatory_records": menstrual, "Testosterone": testosterone,
                         "hyperandrogenism": androgen, "adult": adult}
        if menstrual and androgen and adult:
            optional.append(pre_review_rules()["conditional_optional"]["JD-PCOS"]["test"])
    fibrosis = pre_review_rules()["fibrosis_evidence"]
    if "HD-UNI-13" in comps:
        fib = C.fib4_point(raw, T) or []
        age = C._age_mid(raw)
        low = fibrosis["lower_age_over_65"] if age is not None and age > 65 else fibrosis["lower"]
        indeterminate = any(isinstance(r.get("value"), (int, float))
                            and low <= r["value"] <= fibrosis["upper"] for r in fib)
        facts["fibrosis"] = {"FIB4": fib, "indeterminate": indeterminate,
                             "key_evidence": [fibrosis["finding"]] if indeterminate else [],
                             "source": fibrosis["source"]}
    required = [] if ddx.get("insufficient") else workup_tests(sid, optional)
    menu = C.workup_menu()
    plan.update(required=required, optional=optional, conditional_facts=facts,
                minimum_cost=C.workup_cost(required, menu), budget=C.workup_budget(required, menu),
                version=pre_review_rules()["version"])
    ddx["tests"] = required
    from haenv.m2.diversity import finalize as diversify
    diversify(raw, case_id)
