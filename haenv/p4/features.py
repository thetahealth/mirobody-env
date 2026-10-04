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
import statistics

FACTORS = ("glp1_dose_change", "weight_change", "acei_start_or_stop", "postprandial_labelled_fasting",
           "delayed_processing", "meat_meal", "hemolysis_note", "poc_meter", "jaffe_enzymatic_switch",
           "lab_switch_zero_bias")
ANALYTES = ("fasting_glucose", "creatinine")
#: primary drug by the first three characters of its displayed name (scenario variable)
DRUG_INDEX = {"司美格": 1, "利拉鲁": 2, "度拉糖": 3, "替尔泊": 4, "二甲双": 5, "氨氯地": 6, "阿托伐": 7}


def _blk(sp) -> dict:
    pc = sp.get("prediction_context") if isinstance(sp, dict) else getattr(sp, "prediction_context", {})
    return (pc or {}).get("followup_p4") or {}


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
    from ..labworld import tables
    b = _blk(sp)
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
        if "小时离心" in proc:
            try:
                h = float(proc.split("采血后 ")[1].split(" 小时")[0])
            except (IndexError, ValueError):
                h = 0.0
            if h >= 2.0:
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
    from ..labworld import cv_total, rcv95
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
    return {"ratio": abs(dobs) / rcv95(cv_total(an)), "sign": 1 if dobs > 0 else -1,
            # text length and structure of what the solver reads (pack-1 review r3: length leaked)
            "block_chars": len(_json.dumps(b, ensure_ascii=False)),
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
