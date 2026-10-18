"""P3 source records, projected from existing facts without adding clinical events.

The reader below reconstructs decision inputs from the visible notes, results and
prescriptions. Neither that reader nor the renderer uses a gold class or event role.
"""
from __future__ import annotations

from copy import deepcopy
import re

LABS = {
    "ck_uln": ("肌酸激酶", "倍参考上限"),
    "potassium_mmol_L": ("血钾", "mmol/L"),
    "creatinine_umol_L": ("血肌酐", "μmol/L"),
    "glucose_mmol_L": ("血糖", "mmol/L"),
    "tsh_mIU_L": ("TSH", "mIU/L"),
    "alt_u_L": ("ALT", "U/L"), "ast_u_L": ("AST", "U/L"),
}


def _field(text, label):
    m = re.search(rf"(?:^|\n){re.escape(label)}：([^\n]*)", text)
    return m[1] if m else None


def medication_rows(rows):
    """Use the same prescribing fields for every medicine in the dated record."""
    from haenv.world.clinical.med_course import DISPLAY
    from haenv.world.clinical.med_course import display
    names = {v[0]: k for k, v in DISPLAY.items()}
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("prescribing history requires dated records")
        row = dict(r)
        row["drug"] = row["drug"].split("（", 1)[0]
        if "prescription" not in row:
            key, event = names[row["drug"]], row["event"]
            if event.startswith(("暂停", "停用")):
                row["prescription"] = "停用"
            else:
                m = re.search(r"→\s*([0-9.]+)", event) if "→" in event else re.search(r"([0-9.]+)", event)
                if m is None:
                    raise ValueError("recorded medication dose is missing")
                row["prescription"] = display(key, float(m[1]))
        row.setdefault("event", "处方")
        out.append(row)
    return sorted(out, key=lambda r: (r["day"], r.get("drug", "")))


def source_block(legacy, current_medications, patient_quote):
    """One outpatient-note stream; old chemistry and prescribing leave its prose."""
    from haenv.world.labworld.meds import drug
    from haenv.p3.score import parse_drug
    b = deepcopy(legacy)
    key = parse_drug(b["drug"])
    medication_name = b["drug"].split("（", 1)[0]
    unit = drug(key)["unit"].split("/")[0]
    notes, labs, prescriptions, home = [], [], [], []
    for r in b["visits"]:
        day = int(r["day"])
        has_rx = r["prescription"] not in ("本次待定", "未用药")
        text = f"主诉：慢病复诊。\n诊断：{b['condition']}。\n处方：" + ("见当日处方记录" if has_rx else r["prescription"])
        if not has_rx:
            text += f"\n记录：{r['change']}"
        if r["reading"] not in ("未测", "未到复查时间"):
            dest = home if drug(key).get("self_measured") else labs
            dest.append({"day": day, "test": b["control_axis"], "result": r["reading"]})
        else:
            text += f"\n检查：{r['reading']}"
        if r.get("labs"):
            labs.append({"day": day, "test": "肾功能与电解质", "result": r["labs"]})
        if day == min(int(v["day"]) for v in b["visits"]):
            text += f"\n医嘱：{b['clinic_target']}"
            if b.get("known_comorbidity"):
                text += f"\n既往史：{b['known_comorbidity']}"
        notes.append({"day": day, "source": "门诊病程记录", "text": text})
        if r["prescription"] not in ("本次待定", "未用药"):
            prescriptions.append({"day": day, "drug": b["drug"], "prescription": r["prescription"],
                                  "event": r["change"]})
    for r in b.get("encounter_record") or ():
        # Structured symptom/onset indexing is ordinary encounter metadata;
        # the clinical history remains in the note body, never in a directory label.
        notes.append({**r, "text": f"主诉：{r['symptom']}。\n现病史：{r['history']}"})
    for r in b.get("adverse_event_record") or ():
        day = int(r["day"])
        if r.get("exposure_class"):
            text = (f"主诉：{r['symptom']}。\n用药史：{r['exposure_class']}\n临床记录：{r['reaction']}"
                    f"\n资料核对：{r['verification']}\n登记状态：{r['record_state']}\n就诊记录：{r['record_received']}")
        else:
            text = f"主诉：{r.get('symptom', '复诊')}。\n用药：{r['drug'].split('（', 1)[0]}"
            for field, label in (("other_causes", "问诊"), ("event_record", "经过"),
                                 ("secondary_cause", "检查记录"), ("dechallenge", "处理"),
                                 ("rechallenge", "随访"), ("course", "现病史")):
                if r.get(field):
                    text += f"\n{label}：{r[field]}"
            for p in r.get("prescriptions") or ():
                prescriptions.append({"day": int(p["day"]), "drug": r["drug"],
                                      "prescription": f"{r['drug'].split('（', 1)[0]} {p['dose_mg']:g} {unit} 每日"})
            old_labs = r.get("labs") or [{"day": day, **{k: r[k] for k in LABS if k in r}}]
            for p in old_labs:
                for field, (name, lab_unit) in LABS.items():
                    if field in p:
                        labs.append({"day": int(p["day"]), "test": name, "result": f"{p[field]} {lab_unit}"})
            if "dose_mg" in r:
                prescriptions.append({"day": day, "drug": r["drug"],
                                      "prescription": f"{r['drug'].split('（', 1)[0]} {r['dose_mg']:g} mg 每日"})
        notes.append({"day": day, "source": r["source"], "text": text})
    prescriptions.extend(deepcopy(b.get("other_medications") or []))
    out = {"today_date": b.get("today_date", f"病程第 {b['today']} 天"),
           "encounter_type": "慢病用药复诊", "patient_quote": patient_quote,
           "current_medications": deepcopy(current_medications),
           "encounter_record": sorted(notes, key=lambda r: r["day"]),
           "lab_results": sorted(labs, key=lambda r: (r["day"], r["test"])),
           "prescription_record": medication_rows(prescriptions),
           "dispensing_record": [{**r, "drug": medication_name} for r in b["refill_record"]],
           "medication_review": [{"day": int(b["today"]), "drug": medication_name, "text": b["pill_count_today"]}],
           "symptom_diary": b["symptom_diary"], "text_source": b.get("text_source", "template")}
    if b.get("documented_course_only"):
        out["medication_review"][0]["text"] += "；症状日记未填写的经过为未记载。"
    if home:
        out["home_measurements"] = home
    if "protocol_library" in b:
        out.update(protocol_library=b["protocol_library"], rules_delivery=b["rules_delivery"])
    else:
        # Printed delivery is retained as a source document for offline controls.
        out["clinic_documents"] = b["clinic_protocol"]
    return out


def clinical_view(b):
    """Parse only source contents into the existing G3 reader's input schema."""
    if "lab_results" not in b:
        return b
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    from haenv.p3.score import parse_drug
    notes = b.get("encounter_record") or []
    visits = [r for r in notes if _field(r.get("text", ""), "主诉") == "慢病复诊。"]
    orders = [(r, _field(r["text"], "医嘱")) for r in visits if _field(r["text"], "医嘱")]
    if len(orders) != 1:
        raise ValueError("source records do not identify a single review course")
    order, target = orders[0]
    candidates = [k for k in ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
                  if axis(drug(k)["axis"])["display"] in target]
    if len(candidates) != 1:
        raise ValueError("review target is missing or ambiguous")
    key = candidates[0]
    name, ax = drug(key)["display"], axis(drug(key)["axis"])["display"]
    T = max(int(r["day"]) for r in visits)
    rows = []
    for r in visits:
        day = int(r["day"])
        results = [p["result"] for p in b["lab_results"] + (b.get("home_measurements") or [])
                   if int(p["day"]) == day and p["test"] == ax]
        prescription = _field(r["text"], "处方")
        change = _field(r["text"], "记录")
        if prescription == "见当日处方记录":
            matches = [p for p in b.get("prescription_record") or []
                       if int(p["day"]) == day and parse_drug(p.get("drug")) == key]
            if len(matches) != 1:
                raise ValueError("dated prescription missing or ambiguous")
            prescription, change = matches[0]["prescription"], matches[0]["event"]
        row = {"day": day, "reading": results[0] if len(results) == 1 else _field(r["text"], "检查"),
               "prescription": prescription, "change": change}
        renal = [p["result"] for p in b["lab_results"] if int(p["day"]) == day and p["test"] == "肾功能与电解质"]
        if renal:
            row["labs"] = renal[0]
        rows.append(row)
    reviews = [r for r in b.get("medication_review") or [] if parse_drug(r.get("drug")) == key]
    if len(reviews) != 1:
        raise ValueError("medication reconciliation missing")
    out = {"drug": name, "control_axis": ax, "clinic_target": target, "today": T,
           "visits": sorted(rows, key=lambda r: r["day"]), "symptom_diary": b.get("symptom_diary") or [],
           "refill_record": [{k: v for k, v in p.items() if k != "drug"} for p in b.get("dispensing_record") or []
                             if parse_drug(p.get("drug")) == key],
           "pill_count_today": reviews[0]["text"],
           "encounter_record": [r for r in notes if "symptom" in r],
           "other_medications": [r for r in b.get("prescription_record") or [] if parse_drug(r.get("drug")) != key],
           "adverse_event_record": []}
    if "未填写的经过为未记载" in reviews[0]["text"]:
        out["documented_course_only"] = True
    if _field(order["text"], "既往史"):
        out["known_comorbidity"] = _field(order["text"], "既往史")
    if b.get("protocol_library"):
        out["protocol_library"] = b["protocol_library"]
    elif "clinic_documents" in b:
        out["clinic_protocol"] = b["clinic_documents"]
    lower = min(int(r["day"]) for r in visits)
    for note in notes:
        text, day = note.get("text", ""), int(note["day"])
        if day >= lower:
            continue
        exposure = _field(text, "用药史")
        if exposure:
            out["adverse_event_record"].append({"day": day, "exposure_class": exposure,
                "reaction": _field(text, "临床记录"), "verification": _field(text, "资料核对"),
                "record_state": _field(text, "登记状态")})
            continue
        old_drug = _field(text, "用药")
        old_key = parse_drug(old_drug)
        if old_key is None:
            continue
        old = {"day": day, "drug": old_drug, "drug_id": old_key,
               "other_causes": _field(text, "问诊")}
        rx = sorted([r for r in b.get("prescription_record") or [] if parse_drug(r.get("drug")) == old_key
                     and "prescription" in r and int(r["day"]) < lower], key=lambda r: int(r["day"]))
        prior, later = [r for r in rx if r["day"] <= day], [r for r in rx if r["day"] > day]
        selected = prior[-2:] + later[:1]
        old["prescriptions"] = [{"day": int(r["day"]),
            "dose_mg": float(re.search(r"([0-9.]+) (?:mg|μg)", r["prescription"])[1])} for r in selected]
        panels = {}
        for result in b.get("lab_results") or []:
            if not day - 28 <= int(result["day"]) <= day + 28:
                continue
            for field, (lab_name, _) in LABS.items():
                if result["test"] == lab_name:
                    panels.setdefault(int(result["day"]), {"day": int(result["day"])})[field] = float(result["result"].split()[0])
        old["labs"] = [panels[d] for d in sorted(panels)]
        if old_key == "gliclazide":
            event = _field(text, "经过")
            for panel in old["labs"]:
                panel["external_assistance"] = panel["day"] == day and event == "当时意识与反应受影响，无法自行进食；家属喂糖，急救人员给予葡萄糖"
                panel["unable_to_self_treat"] = panel["external_assistance"]
        if _field(text, "处理"):
            old.update(secondary_cause=_field(text, "检查记录"), dechallenge=_field(text, "处理"),
                       rechallenge=_field(text, "随访"))
            old["dose_mg"] = old["prescriptions"][-1]["dose_mg"]
            old["ck_uln"] = next(p["ck_uln"] for p in old["labs"] if p["day"] == day)
        out["adverse_event_record"].append(old)
    return out


def finalize(raw):
    """Project only P3 after its existing history finalizer, before snapshot collection."""
    import haenv.world.clinical.med_course as MC
    from haenv.p3.gate import BLOCK
    from haenv.p3.render import render_block
    g = (raw.adjudication or {}).get(BLOCK) or {}
    if "class" not in g:
        return
    wm = (raw.latent_premise or {}).get("world_medication") or {}
    medications = sorted(MC.display(k, dose) for k, dose, _ in MC.active_medicines(wm, int(g["T"])))
    utterances = sorted((e for e in raw.evidence_ledger or [] if e.get("source_type") == "patient_reported_symptom"),
                        key=lambda e: int(e.get("source_timestamp", 0)))
    if not utterances:
        raise ValueError("P3 needs a recorded patient utterance")
    quote = utterances[-1]["symptom"]
    g.update(record_format="emr", current_medications=medications, patient_quote=quote)
    raw.prediction_context[BLOCK] = source_block(render_block(g), medications, quote)
    for manifest in (g.get("history_dependency") or {}, g.get("look_alike") or {}):
        for ref in manifest.get("records") or []:
            if ref["section"] == "adverse_event_record":
                ref["section"] = "encounter_record"


def observe(next_call, raw, case_id):
    result = next_call(raw, case_id)
    finalize(raw)
    return result


def install():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.OBSERVE, "p3:emr", observe)
