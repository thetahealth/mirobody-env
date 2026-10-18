"""Documented dose-specific statin myopathy, with content-matched inactive records."""
from __future__ import annotations
from copy import deepcopy
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.p3.gate import BLOCK
from haenv.p3.gate import plan_of


def limiting_record(rows, drug, dose):
    from haenv.world.validation.difficulty_r2 import config
    limits = config("p3")["dose_limit"]
    for r in rows or ():
        if (drug == "enalapril" and r.get("reaction") == "血管性水肿"
                and r.get("exposure_class") == "ACEI" and r.get("verification") == "已核实"
                and r.get("record_state") == "有效"):
            return True
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
            if (renal or potassium) and restored and r.get("other_causes") == "未见脱水、NSAID、补钾或其他并用药变动":
                return True
        elif drug == "gliclazide" and b.get("external_assistance") is True and b.get("unable_to_self_treat") is True:
            return True
        elif drug == "atorvastatin" and float(b["ck_uln"]) > limits["atorvastatin"]["ck_uln"] and float(c["ck_uln"]) < limits["recovery"]["ck_uln"] and r.get("other_causes") == "未见运动、甲减或其他诱因":
            return True
    if drug != "atorvastatin":
        return False
    return any(r.get("drug") == "阿托伐他汀" and float(r.get("dose_mg", 0)) > float(dose)
               and float(r.get("ck_uln", 0)) > 10 and r.get("secondary_cause") == "未发现"
               and r.get("dechallenge") == "症状及 CK 恢复正常"
               and r.get("rechallenge") == "该剂量未再使用，低剂量长期耐受"
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
    row = {"day": old_day("p3", cid, T), "source": source, "drug": "阿托伐他汀", "dose_mg": 20,
           "symptom": symptom, "ck_uln": ck, "secondary_cause": "未发现" if active else "见经过",
           "dechallenge": "症状及 CK 恢复正常" if active else "见经过",
           "rechallenge": "该剂量未再使用，低剂量长期耐受" if active else "原剂量再用耐受",
           "course": "停用后恢复；随后阿托伐他汀 10 mg 每日长期耐受，无肾损伤。" if active else
              phr["statin_control"][int(uniform("history-control", cid) * len(phr["statin_control"]))].format(ck=ck)}
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
        row["other_causes"] = "未见运动、甲减或其他诱因" if active else "近期长途徒步，休息后肌肉症状消失"
    elif family == "enalapril":
        from haenv.p3.realize import _renal
        renal = _renal(cid, [day - 28, day, day + 28], 0)
        c0, k0 = renal[0][1:]
        hyper = uniform("history-event", cid) < 0.5
        row["labs"] = [{"day": day - 28, "potassium_mmol_L": k0, "creatinine_umol_L": c0},
            {"day": day, "potassium_mmol_L": round(6.2 + 0.6 * uniform("history-k", cid), 1) if active and hyper else renal[1][2],
             "creatinine_umol_L": round(c0 * (1.4 + 0.2 * uniform("history-rise", cid))) if active and not hyper else renal[1][1]},
            {"day": day + 28, "potassium_mmol_L": renal[2][2], "creatinine_umol_L": renal[2][1]}]
        row["other_causes"] = "未见脱水、NSAID、补钾或其他并用药变动"
    elif family == "gliclazide":
        row["labs"] = [{"day": day - 28, "glucose_mmol_L": 6.1, "external_assistance": False},
            {"day": day, "glucose_mmol_L": 2.3 if active else 4.3, "external_assistance": active, "unable_to_self_treat": active},
            {"day": day + 28, "glucose_mmol_L": 6.0, "external_assistance": False}]
        row["symptom"] = "出汗、手抖" if uniform("history-word", cid) < 0.5 else "突然头晕、冒汗"
        row["event_record"] = "当时意识与反应受影响，无法自行进食；家属喂糖，急救人员给予葡萄糖" if active else "当时自行进食，无意识变化或外人救助"
    else:
        row["labs"] = [{"day": day - 28, "tsh_mIU_L": 12.1},
                       {"day": day, "tsh_mIU_L": 3.6}, {"day": day + 28, "tsh_mIU_L": 2.5}]
        row["symptom"] = "当时怕冷、乏力，后续复查记录症状缓解"
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
    active = eligible and uniform("p3-allergy-confirmation", cid) < cfg["confirmed_applicable_share"]["value"]
    T = int(raw.prediction_context["prediction_time_T"])
    original = plan["history_record"]
    variants = [("ACEI", "已核实", "已撤销"), ("青霉素类", "已核实", "有效"), ("ACEI", "尚待核实", "有效")]
    exposure, verification, state = ("ACEI", "已核实", "有效") if active else variants[int(uniform("p3-allergy-content", cid) * len(variants))]
    record = {"day": old_day("p3", cid, T), "source": original["source"],
              "exposure_class": exposure, "reaction": "血管性水肿", "verification": verification,
              "record_state": state, "symptom": "嘴唇和舌头肿胀、声音嘶哑",
              "record_received": "本次携外院原始记录核对"}
    if active:
        # This replaces the limiting episode before it is generated, so there
        # is only one decisive record, never two independent overrides.
        plan["history_record"] = record
        plan["history_active"] = True
        plan["history_mechanism"] = "contraindication"
    else:
        plan["allergy_control"] = record
