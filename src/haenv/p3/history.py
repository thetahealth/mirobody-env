"""Documented dose-specific statin myopathy, with content-matched inactive records."""
from __future__ import annotations
from copy import deepcopy

from haenv.i18n.text import L, T as template, is_text
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.p3.gate import BLOCK
from haenv.p3.gate import plan_of


def limiting_inputs(rows):
    """Only the original clinical premises read by limiting_record.

    Directory labels and a redundant symptom summary are not decision inputs.
    This projection lets C1 compare a source reconstruction with the hidden
    record without requiring the reconstruction to recreate unused metadata.
    """
    from haenv.p3.textread import canonical
    keys = {"day", "drug_id", "prescriptions", "labs", "other_causes", "exposure_class",
            "reaction", "verification", "record_state", "dose_mg", "ck_uln", "secondary_cause",
            "dechallenge", "rechallenge"}
    lab_keys = {"day", "ck_uln", "creatinine_umol_L", "potassium_mmol_L",
                "external_assistance", "unable_to_self_treat"}
    def canon(obj):
        if isinstance(obj, str):
            return canonical(obj)
        if isinstance(obj, list):
            return [canon(v) for v in obj]
        if isinstance(obj, dict):
            return {k: canon(v) for k, v in obj.items()}
        return obj
    out = []
    for row in rows or ():
        if "secondary_cause" in row and "dose_mg" in row and "ck_uln" in row:
            # Legacy statin archives are one dated prescription/lab event.
            # The EMR reader adds convenient arrays, but limiting_record's
            # legacy branch reads these original flat clinical fields. Compare
            # that same representation on both sides, including drug identity.
            from haenv.p3.policy import identify_drug
            flat = {k: row[k] for k in ("day", "dose_mg", "ck_uln", "secondary_cause",
                                       "dechallenge", "rechallenge") if k in row}
            flat["drug_id"] = row.get("drug_id") or identify_drug(row.get("drug"))
            out.append(canon(flat))
            continue
        if row.get("drug_id") not in ("atorvastatin", "enalapril", "gliclazide") and "exposure_class" not in row and not row.get("ck_uln"):
            continue  # this record cannot establish a limit in limiting_record
        projected = {k: v for k, v in row.items() if k in keys and v is not None}
        if "labs" in projected:
            projected["labs"] = [{k: v for k, v in lab.items() if k in lab_keys} for lab in projected["labs"]]
            if row.get("drug_id") == "gliclazide":
                for lab in projected["labs"]:
                    for key in ("external_assistance", "unable_to_self_treat"):
                        lab[key] = bool(lab.get(key))
        if not row.get("prescriptions") and "drug" in row:
            projected["drug"] = row["drug"]
        out.append(canon(projected))
    return out


def acei_contraindicated(rows, drug):
    """Verified active ACEI angioedema excludes enalapril at any BP response."""
    return drug == "enalapril" and any(
        is_text(r.get("reaction"), "p3.hist.reaction.angioedema")
        and r.get("exposure_class") == "ACEI"
        and is_text(r.get("verification"), "p3.hist.verified")
        and is_text(r.get("record_state"), "p3.hist.state.active") for r in rows or ())


def limiting_record(rows, drug, dose):
    from haenv.world.validation.difficulty_r2 import config
    limits = config("p3")["dose_limit"]
    if acei_contraindicated(rows, drug):
        return True
    for r in rows or ():
        seq = r.get("prescriptions") or []
        labs = r.get("labs") or []
        if r.get("drug_id") != drug or len(seq) < 3 or len(labs) < 3:
            continue
        before, higher, later = seq[:3]
        a, b, c = labs[:3]
        raised = float(higher["dose_mg"]) > float(before["dose_mg"]) == float(dose)
        returned = float(later["dose_mg"]) == float(before["dose_mg"])
        ordered = before["day"] < higher["day"] <= b["day"] < later["day"] <= c["day"]
        if not (raised and returned and ordered):
            continue
        if drug == "enalapril":
            renal = float(b["creatinine_umol_L"]) / float(a["creatinine_umol_L"]) >= limits["enalapril"]["creatinine_ratio"]
            potassium = float(b["potassium_mmol_L"]) >= limits["enalapril"]["potassium_mmol_L"]
            restored = float(c["potassium_mmol_L"]) < limits["recovery"]["potassium_mmol_L"] and float(c["creatinine_umol_L"]) < limits["recovery"]["creatinine_ratio"] * float(a["creatinine_umol_L"])
            if (renal or potassium) and restored and is_text(r.get("other_causes"), "p3.hist.cause.none_renal"):
                return True
        elif drug == "gliclazide" and b.get("external_assistance") is True and b.get("unable_to_self_treat") is True:
            return True
        elif drug == "atorvastatin" and float(b["ck_uln"]) > limits["atorvastatin"]["ck_uln"] and float(c["ck_uln"]) < limits["recovery"]["ck_uln"] and is_text(r.get("other_causes"), "p3.hist.cause.none_statin"):
            return True
    if drug != "atorvastatin":
        return False
    return any(is_text(r.get("drug"), "drug.atorvastatin") and float(r.get("dose_mg", 0)) > float(dose)
               and float(r.get("ck_uln", 0)) > 10 and is_text(r.get("secondary_cause"), "p3.hist.secondary.none")
               and is_text(r.get("dechallenge"), "p3.hist.dechallenge.ck_normal")
               and is_text(r.get("rechallenge"), "p3.hist.rechallenge.lower_tolerated")
               for r in rows or ())


def prepare(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        return prepare_r2(raw, cid, plan)
    if "history_record" in plan:
        return
    T = int(raw.prediction_context["prediction_time_T"])
    active = plan["drug"] == "atorvastatin" and plan["class"] == "switch" and rate("p3", plan) > 0
    phr = table()["phrasing"]
    source = phr["encounter_sources"][int(uniform("history-source", cid) * len(phr["encounter_sources"]))]
    symptom = phr["muscle_symptoms"][int(uniform("history-word", cid) * len(phr["muscle_symptoms"]))]
    cklo, ckhi = table()["packs"]["p3"]["ck_active" if active else "ck_control"]["value"]
    ck = round(cklo + uniform("history-ck", cid) * (ckhi - cklo), 1)
    control = phr["statin_control"][int(uniform("history-control", cid) * len(phr["statin_control"]))]
    row = {"day": old_day("p3", cid, T), "source": source, "drug": L("drug.atorvastatin"), "dose_mg": 20,
           "symptom": symptom, "ck_uln": ck, "secondary_cause": L("p3.hist.secondary.none" if active else "p3.hist.see_course"),
           "dechallenge": L("p3.hist.dechallenge.ck_normal" if active else "p3.hist.see_course"),
           "rechallenge": L("p3.hist.rechallenge.lower_tolerated" if active else "p3.hist.rechallenge.same_tolerated"),
           "course": L("p3.hist.course.statin_recovered") if active else template(control.key, ck=ck)}
    plan["history_record"] = row
    plan["history_active"] = active
    if active:
        plan["position"] = "min"
        plan["n_inc"] = 0


def finalize(raw, cid, plan):
    g = raw.adjudication[BLOCK]
    if "class" not in g:
        return
    if plan.get("allergy_control"):
        g["adverse_event_record"].append(deepcopy(plan["allergy_control"]))
    active = bool(plan["history_active"])
    lower = min(int(x["day"]) for x in g["visits"])
    raw.prediction_context[BLOCK]["adverse_event_record"] = deepcopy(g["adverse_event_record"])
    g["review_from_day"] = lower
    raw.prediction_context[BLOCK]["review_from_day"] = lower
    g["history_dependency"] = {"active": active, "recent_lower": lower,
                               "records": [{"section": "adverse_event_record", "day": g["adverse_event_record"][0]["day"]}],
                               "source": table()["packs"]["p3"]["source"]}
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import register
    if enabled(plan):
        register(g, "adverse_event_record", g["adverse_event_record"][0]["day"], active, recent_lower=lower)
        align_home_pressure_record(raw, g)


def align_home_pressure_record(raw, g):
    """The measured home-pressure axis requires a cuff and its dated aggregate series."""
    if g.get("axis") != "home_sbp":
        return
    devices = list((raw.user_profile or {}).get("device_inventory") or ())
    if "bp_cuff" not in devices:
        devices.append("bp_cuff")
    raw.user_profile["device_inventory"] = devices
    from haenv.p3.render import home_pressure_rows
    raw.prediction_context.setdefault(BLOCK, {})["home_pressure_record"] = home_pressure_rows(g)


def answer(raw):
    from haenv.p3.gold import g3
    g = raw["adjudication"][BLOCK]
    return g3(g)["class"]


def twin(raw, day=None):
    out = {**raw, "adjudication": {**raw["adjudication"], BLOCK: deepcopy(raw["adjudication"][BLOCK])}}
    g = out["adjudication"][BLOCK]
    g["adverse_event_record"] = [x for x in g.get("adverse_event_record", ()) if day is not None and int(x["day"]) != day]
    return out


POLICY = HistoryDependency(BLOCK, plan_of, prepare, finalize, answer, twin)


def prepare_r2(raw, cid, plan):
    if "history_record" in plan:
        return
    from haenv.world.labworld.meds import drug
    name = plan["drug"]
    eligible = name in ("atorvastatin", "enalapril", "gliclazide")
    active = (eligible and plan["class"] == "switch" and plan.get("subtype", "off") == "off"
              and uniform("history-role", cid) < rate("p3", plan))
    family = name
    spec = drug(family)
    rung = int(uniform("history-tolerated-rung", cid) * (len(spec["ladder"]) - 1))
    low, high = spec["ladder"][rung:rung + 2]
    T = int(raw.prediction_context["prediction_time_T"])
    day = old_day("p3", cid, T)
    sources = table()["phrasing"]["encounter_sources"]
    row = {"day": day, "source": sources[int(uniform("history-source", cid) * len(sources))],
           "drug_id": family, "drug": spec["display"],
           "prescriptions": [{"day": day - 28, "dose_mg": low}, {"day": day - 14, "dose_mg": high},
                             {"day": day + 1, "dose_mg": low}],
           "labs": []}
    if family == "atorvastatin":
        # A striking CK result is also possible after strenuous exercise. Its
        # cause, rather than an independently lower control value, determines
        # whether the old episode limits the statin dose.
        ck = round(12 + 6 * uniform("history-ck", cid), 1)
        row["labs"] = [{"day": day - 28, "ck_uln": 0.7}, {"day": day, "ck_uln": ck}, {"day": day + 28, "ck_uln": 0.8}]
        from haenv.world.observations.findings_render import _value
        from haenv.resources.registry import load_findings
        findings = load_findings()
        from haenv.world.validation.difficulty_r2 import config
        baseline = row["labs"][0]
        for fid in config("p3")["archived_safety_panel"]["findings"]:
            baseline[fid.lower() + "_u_L"] = _value(findings[fid], "normal", None,
                cid, fid, baseline["day"], declared=False)
        row["symptom"] = table()["phrasing"]["muscle_symptoms"][int(uniform("history-word", cid) * 3)]
        row["other_causes"] = L("p3.hist.cause.none_statin" if active else "p3.hist.cause.hiking")
    elif family in ("enalapril", "losartan"):
        from haenv.p3.realize import _renal
        renal = _renal(cid, [day - 28, day, day + 28], 0)
        c0, k0 = renal[0][1:]
        hyper = uniform("history-event", cid) < 0.5
        row["labs"] = [{"day": day - 28, "potassium_mmol_L": k0, "creatinine_umol_L": c0},
            {"day": day, "potassium_mmol_L": round(6.2 + 0.6 * uniform("history-k", cid), 1) if active and hyper else renal[1][2],
             "creatinine_umol_L": round(c0 * (1.4 + 0.2 * uniform("history-rise", cid))) if active and not hyper else renal[1][1]},
            {"day": day + 28, "potassium_mmol_L": renal[2][2], "creatinine_umol_L": renal[2][1]}]
        row["other_causes"] = L("p3.hist.cause.none_renal")
    elif family == "gliclazide":
        row["labs"] = [{"day": day - 28, "glucose_mmol_L": 6.1, "external_assistance": False},
            {"day": day, "glucose_mmol_L": 2.3 if active else 4.3, "external_assistance": active, "unable_to_self_treat": active},
            {"day": day + 28, "glucose_mmol_L": 6.0, "external_assistance": False}]
        row["symptom"] = L("p3.hist.symptom.sweat_tremor" if uniform("history-word", cid) < 0.5 else "p3.hist.symptom.dizzy_sweat")
        row["event_record"] = L("p3.hist.event.assisted" if active else "p3.hist.event.self_treated")
    elif family == "amlodipine":
        row["course"] = L("p3.hist.course.amlodipine_tolerated")
    else:
        row["labs"] = [{"day": day - 28, "tsh_mIU_L": 12.1},
                       {"day": day, "tsh_mIU_L": 3.6}, {"day": day + 28, "tsh_mIU_L": 2.5}]
        row["symptom"] = L("p3.hist.symptom.cold_fatigue_resolved")
    plan["history_record"], plan["history_active"] = row, active
    if active:
        plan.update(position="min" if rung == 0 else "mid", tolerated_rung=rung,
                    n_inc=min(int(plan.get("n_inc", 0)), rung), subtype="off")
    prepare_allergy(raw, cid, plan)


def prepare_allergy(raw, cid, plan):
    """Rare patient-specific records; applicability is in their clinical facts."""
    from haenv.world.validation.difficulty_r2 import config
    cfg = config("p3")["allergy_record"]
    share = cfg["acei_exposed_record_share"] if plan["drug"] == "enalapril" else cfg["patient_record_share"]
    if uniform("p3-patient-allergy-record", cid) >= share["value"]:
        return
    eligible = plan["drug"] == "enalapril" and plan["class"] == "switch" and plan.get("subtype") == "off"
    # Absolute contraindications are single-factor safety anchors. A response
    # intersection instead uses the dose-specific intolerance history; give
    # it the same content-matched inactive allergy controls as other items.
    active = eligible and not plan.get("interaction_family") and uniform("p3-allergy-confirmation", cid) < cfg["confirmed_applicable_share"]["value"]
    T = int(raw.prediction_context["prediction_time_T"])
    original = plan["history_record"]
    verified, pending = L("p3.hist.verified"), L("p3.hist.unverified")
    revoked, valid = L("p3.hist.state.revoked"), L("p3.hist.state.active")
    variants = [("ACEI", verified, revoked), (L("p3.hist.exposure.penicillin"), verified, valid), ("ACEI", pending, valid)]
    exposure, verification, state = ("ACEI", verified, valid) if active else variants[int(uniform("p3-allergy-content", cid) * len(variants))]
    record = {"day": old_day("p3", cid, T), "source": original["source"],
              "exposure_class": exposure, "reaction": L("p3.hist.reaction.angioedema"), "verification": verification,
              "record_state": state, "symptom": L("p3.hist.symptom.angioedema"),
              "record_received": L("p3.hist.received.outside_record")}
    if active:
        # This replaces the limiting episode before it is generated, so there
        # is only one decisive record, never two independent overrides.
        plan["history_record"] = record
        plan["history_active"] = True
        plan["history_mechanism"] = "contraindication"
    else:
        plan["allergy_control"] = record
