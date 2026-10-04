"""Pack 4 production renderer: the follow-up report the solver reads.

Writes `prediction_context.followup_p4` from the realized item (`realize.realize`). It carries
every feature the gold depends on and nothing that names the gold:

* `report` -- every draw of the followed analyte up to today: day, draw time, printed value,
  fasting label, specimen handling (hours to centrifugation, haemolysis index), lab, method and
  a free-text note when the lab printed one;
* `diet_record` -- the meals before each of the two compared draws: the evening meal before, and
  on a non-fasting draw the breakfast (and the lunch of an afternoon draw) at meal times;
* `medication_record` -- the primary drug's visible dose line (named, with its class), each
  change with its reason, and the ACEI record with its indication;
* `previous_draw_day` / `today` -- which two draws the question compares.

A draw between `fasting_hours` is a fasting draw; any other draw is recorded as non-fasting. A
pre-analytic or method perturbation is visible only through these fields (a cooked-meat breakfast
or lunch 1-4 h before a non-fasting draw, three hours to centrifugation, a Jaffe method), and a
decoy is rendered the same way as a cause. The leak gate (`gold.leak_hits`) scans the
rendered block and the solver payload.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from .plan import h01

TEXT_SOURCE = "template"


def _R():
    from ..labworld import tables
    return tables()["render"]


def _minus(clock: str, hours: float) -> str:
    h, m = int(clock[:2]), int(clock[3:5])
    t = (h * 60 + m - int(round(float(hours) * 60))) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"


def _min(clock: str) -> int:
    return int(clock[:2]) * 60 + int(clock[3:5])


def _hhmm(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def is_fasting(clock: str) -> bool:
    lo, hi = (int(x) for x in _R()["fasting_hours"])
    return lo <= int(clock[:2]) < hi


def meat_slot(clock: str, case_id: str, day: int) -> tuple[str, str] | None:
    """`(meal, HH:MM)` of a cooked-meat meal 1-4 h before a non-fasting draw at `clock`: the lunch
    when the lunch window allows it, else the breakfast; None for a fasting draw."""
    if is_fasting(clock):
        return None
    R, c = _R(), _min(clock)
    for meal, win in (("午餐", R["lunch_window"]), ("早餐", R["breakfast_window"])):
        a, b = max(_min(win[0]), c - 240), min(_min(win[1]), c - 60)
        if a <= b:
            return meal, _hhmm(a + int(h01("p4meat-t", case_id, day) * (b - a + 1)))
    return None


def meat_dinner_time(clock: str, case_id: str, day: int) -> str | None:
    """HH:MM of a cooked-meat dinner (18:00-20:59 the evening before) at least
    `meat_dinner_min_hours` before a draw at `clock`; None when no dinner time is that early."""
    last = min(20 * 60 + 59, _min(clock) + 24 * 60 - int(round(60 * float(_R()["meat_dinner_min_hours"]))))
    if last < 18 * 60:
        return None
    return _hhmm(18 * 60 + int(h01("p4meat-dinner", case_id, day) * (last - 18 * 60 + 1)))


def _pick(seq, *key):
    return seq[int(h01(*key) * len(seq))]


def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}" if nd else f"{int(round(v))}"


def _by_factor(rec: dict, name: str) -> dict | None:
    return next((p for p in rec.get("perturbations") or () if p.get("factor") == name), None)


def report_row(item: dict, rec: dict, case_id: str) -> dict:
    from ..labworld import analyte as A
    from ..labworld.perturb import factor as F
    R = _R()
    a = A(item["analyte"])
    nd = int(a["ndigits"])
    d = int(rec["day"])
    lab, method, note = R["default_lab"], a["default_method"], None
    dp = _by_factor(rec, "delayed_processing")
    lo, hi = (float(x) for x in R["specimen_normal_h"])
    hours = float(dp["params"]["hours"]) if dp else round(lo + (hi - lo) * h01("p4spec", case_id, d), 1)
    poc = _by_factor(rec, "poc_meter")
    jf = _by_factor(rec, "jaffe_enzymatic_switch")
    if poc and (poc.get("role") == "cause" or (poc.get("params") or {}).get("method_throughout")):
        lab, method = "门诊床旁检测点", F("poc_meter")["display"]
    elif poc:
        note = "同日空腹指尖血糖用床旁血糖仪测定"
    if jf and (jf.get("role") in ("cause", "minor") or (jf.get("params") or {}).get("method_throughout")):
        method = F("jaffe_enzymatic_switch")["jaffe_method"]
    elif jf:
        note = "同次生化套餐中肌酐改用 " + F("jaffe_enzymatic_switch")["jaffe_method"] + " 检测"
    ls = _by_factor(rec, "lab_switch_zero_bias")
    if ls and (ls.get("params") or {}).get("reagent_change"):
        method = method + "，换用另一厂家试剂"
    elif ls:
        lab = R["alt_lab"]
    hemo = "轻度（1+）" if _by_factor(rec, "hemolysis_note") else "无"
    if method.startswith("床旁"):
        specimen = {"sample": "指尖毛细血管全血", "processing": "即时检测", "hemolysis_index": "—"}
    else:
        specimen = {"sample": "静脉血（促凝血清管）",
                    "processing": f"采血后 {hours:.1f} 小时离心" + ("，期间室温放置" if hours >= 2.0 else ""),
                    "hemolysis_index": hemo}
    row = {"day": d, "time": rec["time"][:5], "value": _fmt(float(rec["printed"]), nd), "unit": a["unit"],
           "fasting": R["fasting_label"] if is_fasting(rec["time"][:5]) else R["nonfasting_label"],
           "specimen": specimen, "lab": lab, "method": method}
    if note:
        row["note"] = note
    return row


def diet_rows(item: dict, rec: dict, case_id: str) -> dict:
    R = _R()
    d = int(rec["day"])
    clock = rec["time"][:5]
    dinner_food = _pick(R["dinner_light"], "p4dinner", case_id, d)
    dinner_time = f"{18 + int(h01('p4dinner-h', case_id, d) * 3):02d}:{int(h01('p4dinner-m', case_id, d) * 60):02d}"
    meat = _by_factor(rec, "meat_meal")
    if meat and meat.get("params", {}).get("meal") == "dinner":
        dinner_food = _pick(R["meals_meat"], "p4meat", case_id, d)
        dinner_time = meat_dinner_time(clock, case_id, d) or dinner_time
    entries = [{"day": d - 1, "time": dinner_time, "meal": "前一日晚餐", "food": dinner_food}]
    if is_fasting(clock):
        entries.append({"day": d, "time": None, "meal": "早餐", "food": "采血后进食"})
        return {"draw_day": d, "draw_time": clock, "entries": entries}
    slot = meat_slot(clock, case_id, d) if meat and meat.get("params", {}).get("meal") != "dinner" else None
    b0, b1 = (_min(x) for x in R["breakfast_window"])
    if slot and slot[0] == "早餐":
        entries.append({"day": d, "time": slot[1], "meal": "早餐", "food": _pick(R["breakfast_meat"], "p4meat", case_id, d)})
    else:
        t = b0 + int(h01("p4bf-t", case_id, d) * (b1 - b0 + 1))
        entries.append({"day": d, "time": _hhmm(t), "meal": "早餐", "food": _pick(R["meals_light"], "p4bf", case_id, d)})
    l0, l1 = (_min(x) for x in R["lunch_window"])
    c = _min(clock)
    if slot and slot[0] == "午餐":
        entries.append({"day": d, "time": slot[1], "meal": "午餐", "food": _pick(R["meals_meat"], "p4meat", case_id, d)})
    elif c - 30 >= l0:
        t = l0 + int(h01("p4lunch-t", case_id, d) * (min(l1, c - 30) - l0 + 1))
        entries.append({"day": d, "time": _hhmm(t), "meal": "午餐", "food": _pick(R["lunch_light"], "p4lunch", case_id, d)})
    return {"draw_day": d, "draw_time": clock, "entries": entries}


def _reasons(drug: str) -> dict:
    from ..labworld import tables
    t = tables()["render"]
    rs = t.get("dose_reasons") or {}
    key = "glp1" if drug in (t.get("glp1_class") or ()) else drug
    return rs.get(key) or rs["default"]


def medication_rows(item: dict, raw, T: int) -> list[dict]:
    from ..labworld import tables
    names = tables()["render"].get("drugs") or {}
    drug = item.get("drug") or ""
    why = _reasons(drug)
    nm0 = names.get(drug, drug)
    out = []
    prev, lowered = None, False
    for q in sorted((raw.longitudinal_data or {}).get("dose_timeline") or [], key=lambda x: int(x["ts"])):
        if int(q["ts"]) > T:
            break
        v = q.get("value")
        if prev is None:
            out.append({"day": int(q["ts"]), "drug": nm0, "event": f"起始 {v}"})
        elif v != prev and not v:
            out.append({"day": int(q["ts"]), "drug": nm0, "event": f"暂停（原剂量 {prev}）", "reason": why["hold"]})
            lowered = True
        elif v != prev and not prev:
            out.append({"day": int(q["ts"]), "drug": nm0, "event": f"恢复 {v}", "reason": why["back"]})
        elif v != prev:
            r = why["down"] if v < prev else (why["back"] if lowered else why["up"])
            lowered = lowered or v < prev
            out.append({"day": int(q["ts"]), "drug": nm0, "event": f"剂量 {prev} → {v}", "reason": r})
        prev = v
    eff = tables()["effects"]["acei_on_creatinine"]
    ac = item.get("acei") or {}
    nm = f"{eff['drug']}（ACEI，口服）"
    ind = eff["indications"].get(ac.get("indication") or "", "")
    if ac.get("initial_on"):
        out.append({"day": 0, "drug": nm, "event": f"长期服用 {eff['dose']}（病程开始前已在用）", "reason": ind})
    for ev in list(ac.get("events") or ()) + list(ac.get("decoy_events") or ()):
        out.append({"day": int(ev["day"]), "drug": nm, "event": (f"开始 {eff['dose']}" if ev["action"] == "start" else "停用"),
                    "reason": ind if ev["action"] == "start" else eff["stop_reason"]})
    return sorted(out, key=lambda r: (r["day"], r["drug"]))


def render_block(item: dict, raw, case_id: str) -> dict:
    from ..labworld import analyte as A
    a = A(item["analyte"])
    T = int(item["t1"])
    recs = sorted(item["records"], key=lambda r: int(r["day"]))
    by = {int(r["day"]): r for r in recs}
    return {
        "analyte": item["analyte"], "name": a["display"], "unit": a["unit"],
        "reference_range": a["reference"],
        "previous_draw_day": int(item["t0"]), "today": T,
        "report": [report_row(item, r, case_id) for r in recs],
        "diet_record": [diet_rows(item, by[int(item["t0"])], case_id), diet_rows(item, by[T], case_id)],
        "medication_record": medication_rows(item, raw, T),
        **({"known_comorbidity": f"{cm['display']} {cm['stage']} 期（肾内科随诊）"} if (cm := item.get("comorbidity")) else {}),
        "text_source": TEXT_SOURCE,
    }
