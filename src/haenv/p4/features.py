"""Features of a pack-4 item read off what the solver sees (for the generation-time audits).

`surface(sp, H)` -- the PREREG L4-1 surface set: |dobs|/RCV, sign, analyte, H, interval, draw
hours, age, sex, known conditions, number of history draws. `factor_flags(sp)` -- presence of
each factor on the two compared draws or in the window between them, read from the rendered
report, diet record and medication record and from the visible weight stream. Nothing here reads
the gold block, so a feature that separates the classes is a property of the rendered item.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math
import re
import statistics

FACTORS = ("glp1_dose_change", "weight_change", "acei_start_or_stop", "postprandial_labelled_fasting",
           "delayed_processing", "meat_meal", "hemolysis_note", "poc_meter", "jaffe_enzymatic_switch",
           "lab_switch_zero_bias")
from haenv.p4.diversity import table as _diversity_table
FACTORS = tuple(dict.fromkeys((*FACTORS, *_diversity_table()["rules"])))
ANALYTES = ("fasting_glucose", "creatinine", "potassium", "glucose", "tsh", "alt", "ck", "triglycerides")
#: primary drug by the first three characters of its displayed name (scenario variable)
DRUG_INDEX = {"司美格": 1, "利拉鲁": 2, "度拉糖": 3, "替尔泊": 4, "二甲双": 5, "氨氯地": 6, "阿托伐": 7}


def _blk(sp) -> dict:
    pc = sp.get("prediction_context") if isinstance(sp, dict) else getattr(sp, "prediction_context", {})
    block = (pc or {}).get("followup_p4") or {}
    from haenv.p4.records import clinical_view
    return clinical_view(block)


def _ld(sp) -> dict:
    return (sp.get("longitudinal_data") if isinstance(sp, dict) else getattr(sp, "longitudinal_data", {})) or {}


def _up(sp) -> dict:
    return (sp.get("user_profile") if isinstance(sp, dict) else getattr(sp, "user_profile", {})) or {}


def compared(sp) -> tuple[dict, dict]:
    b = _blk(sp)
    rows = {int(r["day"]): r for r in b.get("report") or ()}
    return rows[int(b["previous_draw_day"])], rows[int(b["today"])]


def _wmed(ld, day, span=15):
    w = [float(q["value"]) for q in ld.get("weight") or () if day - span <= int(q["ts"]) <= day
         and isinstance(q.get("value"), (int, float))]
    return statistics.median(w) if w else None


def factor_flags(sp, window_only: bool = False) -> dict[str, int]:
    """Presence of each factor in what the item shows. Medication events count anywhere in the
    span of the displayed report (a reader who only looks for factors sees them wherever they
    sit, PREREG R7); with `window_only`, only events inside (t0, T] count (the timing half of
    the knowledge the item tests). Diet, specimen, method and lab flags read the two compared
    draws in both modes."""
    from haenv.world.labworld import tables
    b = _blk(sp)
    from haenv.p4.diversity_reader import applies
    if applies(b):
        cfg = _diversity_table()
        flags = dict.fromkeys(FACTORS, 0)
        for key in cfg["analytes"][b["analyte"]]["pre"] + cfg["analytes"][b["analyte"]]["method"] + cfg["analytes"][b["analyte"]]["true_causes"]:
            rule = cfg["rules"][key]
            rows = b.get(rule["section"]) or []
            flags[key] = int(bool(rows) if rule["section"] in ("report", "diet_record") else
                             any(r.get("exposure") == rule.get("exposure") or r.get("comparison") == rule.get("exposure") or
                                 r.get("drug") == rule.get("exposure") for r in rows))
        return flags
    t0, T = int(b["previous_draw_day"]), int(b["today"])
    r0, r1 = compared(sp)
    R = tables()["render"]
    meats = set(R["meals_meat"]) | set(R.get("breakfast_meat") or ())
    glp1_names = {tables()["render"]["drugs"][d] for d in R["glp1_class"]}
    med = b.get("medication_record") or ()
    diets = {int(d["draw_day"]): d for d in b.get("diet_record") or ()}
    f = dict.fromkeys(FACTORS, 0)
    lo = t0 if window_only else min(int(r["day"]) for r in b.get("report") or ()) - 1
    f["glp1_dose_change"] = int(any(m["drug"] in glp1_names and not m["event"].startswith("起始")
                                    and lo < int(m["day"]) <= T for m in med))
    f["acei_start_or_stop"] = int(any("ACEI" in m["drug"] and m["event"] != "" and not m["event"].startswith("长期")
                                      and lo < int(m["day"]) <= T for m in med))
    ld = _ld(sp)
    w0, w1 = _wmed(ld, t0), _wmed(ld, T)
    f["weight_change"] = int(bool(w0 and w1 and abs(math.log(w1 / w0)) >= 0.04))
    for d in (t0, T):
        ent = (diets.get(d) or {}).get("entries") or ()
        if any(e.get("meal") == "早餐" and e.get("time") for e in ent):
            f["postprandial_labelled_fasting"] = 1
        if any(e.get("food") in meats for e in ent):
            f["meat_meal"] = 1
    for r in (r0, r1):
        sp_ = r.get("specimen") or {}
        proc = str(sp_.get("processing") or "")
        if "离心" in proc:
            m = re.search(r"(\d+(?:\.\d+)?)\s*小时", proc)
            if m and float(m.group(1)) >= 2.0:
                f["delayed_processing"] = 1
        if "轻度" in str(sp_.get("hemolysis_index") or ""):
            f["hemolysis_note"] = 1
        if "床旁" in str(r.get("method")) or "床旁" in str(r.get("note") or ""):
            f["poc_meter"] = 1
        if "Jaffe" in str(r.get("method")) or "Jaffe" in str(r.get("note") or ""):
            f["jaffe_enzymatic_switch"] = 1
    if (r0.get("lab") != r1.get("lab") and r0.get("method") == r1.get("method")) or (
            r0.get("method") != r1.get("method") and not f["jaffe_enzymatic_switch"] and not f["poc_meter"]):
        f["lab_switch_zero_bias"] = 1
    return f


def surface(sp, H: int) -> dict:
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    b = _blk(sp)
    an = b["analyte"]
    r0, r1 = compared(sp)
    v0, v1 = float(r0["value"]), float(r1["value"])
    dobs = math.log(v1 / v0)
    up = _up(sp)
    age = str(up.get("age_range") or "40-44")
    kc = list(up.get("known_conditions") or ())
    import json as _json
    med = b.get("medication_record") or ()
    drugs = sorted({m["drug"] for m in med if "ACEI" not in m["drug"]})
    jf = lambda r: "jaffe" if "Jaffe" in str(r.get("method")) else None     # noqa: E731
    return {"ratio": abs(dobs) / rcv95(cv_total(an, jf(r0)), cv_total(an, jf(r1))), "sign": 1 if dobs > 0 else -1,
            # text length and structure of what the solver reads (pack-1 review r3: length leaked)
            "block_chars": len(_json.dumps((sp.get("prediction_context") if isinstance(sp, dict) else sp.prediction_context)["followup_p4"], ensure_ascii=False)),
            "payload_chars": len(_json.dumps(sp, ensure_ascii=False, default=str)) if isinstance(sp, dict) else 0,
            "n_med_rows": len(med), "n_notes": sum(1 for r in b.get("report") or () if r.get("note")),
            "drug": (DRUG_INDEX.get(drugs[0][:3], 9) if drugs else 0), "T": int(b["today"]),
            "analyte": ANALYTES.index(an), "H": int(H),
            "interval": int(b["today"]) - int(b["previous_draw_day"]),
            "hour1": int(r1["time"][:2]), "hour0": int(r0["time"][:2]),
            "age": int(age.split("-")[0]) if age[:2].isdigit() else 40,
            "sex": int(up.get("sex") == "F"), "n_known": len(kc), "T2D": int("T2D" in kc),
            "HTN": int("hypertension" in kc), "n_hist": len(b.get("report") or ()) - 2,
            "comorbid": int(bool(b.get("known_comorbidity"))),
            "dobs": dobs, "v0": v0, "v1": v1}


#: |dobs|/RCV at or below which a visible change is read as inside the RCV, and the meal window (hours before
#: the draw) in which cooked meat is read as acting (`registry/labworld.yaml` render notes, NICE NG203 1.1.4)
READ_NOISE_MAX, READ_CHANGE_MIN = 0.7, 1.3
READ_MEAL_WINDOW_H = (1.0, 24.0)


def visible_class(sp) -> tuple[str, list]:
    """What a reader who has only the solver-visible block and the documented rules concludes: `(class, acting
    candidates)`. The change is read against the RCV of the two printed methods; beyond it, the acting
    candidates the block shows are a cooked-meat meal inside the window before a compared draw, a Jaffe draw
    against an enzymatic one, and an ACEI start or stop between the two draws, each with its direction
    (a meal and Jaffe raise the draw they sit on, a start raises, a stop lowers); the one that agrees with the
    sign of the change names the class. `unresolved` (no candidate agrees) and `ambiguous` (several do) mean
    the block does not decide the gold; `gap` is a ratio between the two bands."""
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    b = _blk(sp)
    from haenv.p4.diversity_reader import applies
    from haenv.p4.diversity_reader import read_class
    if applies(b):
        return read_class(b)
    an = b["analyte"]
    t0, T = int(b["previous_draw_day"]), int(b["today"])
    r0, r1 = compared(sp)
    jf = lambda r: "jaffe" if "Jaffe" in str(r.get("method")) else None     # noqa: E731
    d = math.log(float(r1["value"]) / float(r0["value"]))
    ratio = abs(d) / rcv95(cv_total(an, jf(r0)), cv_total(an, jf(r1)))
    clinical = bool(b.get("interpretation_rule"))
    if not clinical and ratio <= READ_NOISE_MAX:
        return "analytic_biological_noise", []
    if not clinical and ratio < READ_CHANGE_MIN:
        return "gap", []
    from haenv.world.labworld import tables
    R = tables()["render"]
    meats = set(R["meals_meat"]) | set(R.get("breakfast_meat") or ())
    cands: list[tuple[str, int, str]] = []
    for dd in b.get("diet_record") or ():
        if int(dd["draw_day"]) not in (t0, T):
            continue
        draw_min = int(dd["draw_time"][:2]) * 60 + int(dd["draw_time"][3:5])
        for e in dd.get("entries") or ():
            if e.get("food") in meats and e.get("time"):
                gap = (int(dd["draw_day"]) - int(e["day"])) * 24 + (draw_min - int(e["time"][:2]) * 60 - int(e["time"][3:5])) / 60
                if READ_MEAL_WINDOW_H[0] <= gap <= READ_MEAL_WINDOW_H[1]:
                    cands.append(("preanalytical", 1 if int(dd["draw_day"]) == T else -1, "meat_meal"))
    if jf(r0) != jf(r1):
        # The clinical assay comparison is sample dependent and can have
        # either sign. The legacy positive-Jaffe model remains unchanged.
        direction = (1 if d > 0 else -1) if clinical else (1 if jf(r1) else -1)
        cands.append(("method_difference", direction, "jaffe_enzymatic_switch"))
    for m in b.get("medication_record") or ():
        if "ACEI" in m["drug"] and t0 < int(m["day"]) <= T:
            if str(m["event"]).startswith("开始"):
                cands.append(("true_change", 1, "acei_start_or_stop"))
            elif str(m["event"]).startswith("停用"):
                cands.append(("true_change", -1, "acei_start_or_stop"))
    agree = [c for c in cands if c[1] == (1 if d > 0 else -1)]
    if not agree:
        return ("analytic_biological_noise" if clinical and not cands else "unresolved"), cands
    return (agree[0][0] if len(agree) == 1 else "ambiguous"), cands
