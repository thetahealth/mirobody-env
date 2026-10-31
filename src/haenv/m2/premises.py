"""One emission check for the observable premises of diagnostic gold.

The registry declares requirements by condition, never by generated item id.
Medication courses are constructed in the world before observations exist.
"""
from __future__ import annotations

from dataclasses import replace
from haenv.m2.evidence import registry


def requirements(ddx):
    from haenv.m2.core import components
    if ddx.get("insufficient") or ddx.get("join_gold") == "independent":
        return []
    table = registry().get("premise_checks") or {}
    return [(comp, check) for comp in components(str(ddx.get("spec_id") or ""))
            for check in table.get(comp, ())]


def prepare_case(cs):
    if not (cs.latent.get("m2") or cs.latent.get("m2_bridge")):
        return cs
    from haenv.world.clinical.ddx import clinical_profile
    from haenv.m2.core import components
    raw = dict(cs.raw)
    for comp in components(str(cs.latent.get("ddx_spec_id") or "")):
        raw = clinical_profile(cs.case_id, comp, raw)
    return replace(cs, raw=raw)


def medication_overlay(p):
    """Use an existing indicated regimen for a declared exposure scenario.

    Start at day zero, before the symptom schedule. This is a selected synthetic
    exposure design, not an estimate of drug initiation frequency or causality.
    """
    from haenv.world.clinical.med_course import CONCURRENT
    ext = (p.meta or {}).get("_external") or {}
    if not (ext.get("m2") or ext.get("m2_bridge")):
        return []
    out = []
    for _comp, check in requirements((p.meta or {}).get("ddx") or {}):
        if check["kind"] == "medication_exposure":
            drug, ladder = CONCURRENT[check["indication"]][0]
            if drug != p.patient_basics["regimen"]["drug"]:
                out.append({"drug": drug, "indication": check["indication"],
                            "dose_timeline": [{"ts": 0, "value": ladder[0]}]})
    return out


def emission(raw, cs, sp, T):
    from haenv.m2.core import is_m2_case, _on_demand
    from haenv.world.clinical.med_course import DISPLAY, dose_at
    from haenv.world.observations.events_text import phrase_ids
    if not is_m2_case(raw):
        return []
    ddx = raw.adjudication.get("ddx") or {}
    hits = []
    for comp, check in requirements(ddx):
        if not check.get("source"):
            raise ValueError(f"{comp}: premise check has no source")
        ok = False
        if check["kind"] == "medication_exposure":
            wm = raw.latent_premise.get("world_medication") or {}
            meds = [wm.get("primary") or {}, *(wm.get("concurrent") or [])]
            records = sp.prediction_context.get("medication_record") or []
            current = sp.prediction_context.get("current_medications") or []
            symptoms = [int(s["day"]) for s in cs.raw.get("symptoms") or [] if int(s["day"]) <= T]
            onset = min(symptoms) if symptoms else T
            for m in meds:
                drug = str(m.get("drug") or "")
                if check["drug_class"] not in drug or drug not in DISPLAY:
                    continue
                exposed = dose_at(m.get("dose_timeline") or [], onset) > 0
                # A hidden exposure, an undated name or a prescription AFTER onset
                # cannot establish the causal premise. The visible dated entry is
                # checked against the same world course, not against gold wording.
                dated = any(r.get("drug") == DISPLAY[drug][0] and int(r.get("day", T+1)) <= onset
                            for r in records)
                visible_now = any(DISPLAY[drug][0] in phrase for r in current for phrase in phrase_ids(r))
                ok |= exposed and dated and (visible_now or any(
                    int(r.get("day", T+1)) <= T and r.get("drug") == DISPLAY[drug][0] for r in records))
        elif check["kind"] == "glycaemic_diagnosis":
            from haenv.m2.glycaemia import has_fasting_statement
            try:
                fpg = _on_demand(raw, "FBG", T) or []
                hba1c = _on_demand(raw, "HbA1c", T) or []
                margin = registry()["glycaemic_course"]["hba1c_margin_pct"]
                ok = (any(p.get("ts") == T and isinstance(p.get("value"), (int, float))
                          and p["value"] >= check["fpg_min_mmol_l"] and has_fasting_statement(p) for p in fpg)
                      and any(p.get("ts") == T and isinstance(p.get("value"), (int, float))
                              and p["value"] >= check["hba1c_min_pct"] + margin for p in hba1c))
                if ok and check.get("post_onset_fpg"):
                    # A visible fasting glucose below the diabetic threshold after the
                    # line's onset contradicts the established diagnosis on day T.
                    from haenv.m2.core import components
                    from haenv.m2.glycaemia import _onset, visible_readings
                    lines = set(components(str(ddx.get("spec_id") or ""))) & {comp}
                    onset = _onset(raw, lines, T)
                    ok = all(p["value"] >= check["fpg_min_mmol_l"]
                             for p in visible_readings(raw, "FBG", T) if onset <= p["ts"] < T)
            except ValueError:
                ok = False
        elif check["kind"] == "renal_course":
            try:
                reports = _on_demand(raw, check["finding"], T) or []
                for report in reports:
                    details = report.get("details") or {}
                    rows = details.get("uacr_history") or []
                    measured = [r for r in rows if isinstance(r.get("value"), (int, float))
                                and int(r.get("ts", T+1)) <= T]
                    ok |= (len(measured) >= check["minimum_observations"]
                           and max(r["ts"] for r in measured)-min(r["ts"] for r in measured) >= check["span_days"]
                           and all(r["value"] >= check["albuminuria_mg_g"] for r in measured)
                           and min(r["value"] for r in measured[-2:]) >= check["ratio"]*measured[0]["value"]
                           and details.get("same_method") is True
                           and details.get("acute_factors_reviewed") is True
                           and bool(report.get("value")))
            except ValueError:
                ok = False
        else:
            raise ValueError(f"unsupported premise check {check['kind']}")
        if not ok:
            hits.append({"kind": "m2_premises_missing", "severity": "gate",
                         "detail": f"{comp}: solver-reachable {check['kind']} premise missing"})
    return hits
