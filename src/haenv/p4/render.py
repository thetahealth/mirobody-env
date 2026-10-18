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
* `today` -- the date of this visit; earlier reports remain in the report stream.

A draw between `fasting_hours` is a fasting draw; any other draw is recorded as non-fasting. A
pre-analytic or method perturbation is visible only through these fields (a cooked-meat breakfast
or lunch 1-4 h before a non-fasting draw, three hours to centrifugation, a Jaffe method), and a
decoy is rendered the same way as a cause. The leak gate (`gold.leak_hits`) scans the
rendered block and the solver payload.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.p4.plan import h01
from haenv.p4.plan import background_on

TEXT_SOURCE = "template"


def _R():
    from haenv.world.labworld import tables
    data = tables()
    return data["render" if background_on() else "render_background_off"]


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


def meat_dinner_time(case_id: str, day: int) -> str:
    """HH:MM (18:00-20:59) of a cooked-meat dinner, rendered as the evening meal two days before the
    draw: 34 h or more before it, clear of the 12-24 h window in which cooked meat is still read as
    possibly acting on creatinine (provisional, clinical review pending)."""
    return _hhmm(18 * 60 + int(h01("p4meat-dinner", case_id, day) * 180))


def _pick(seq, *key):
    return seq[int(h01(*key) * len(seq))]


def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}" if nd else f"{int(round(v))}"


def _by_factor(rec: dict, name: str) -> dict | None:
    return next((p for p in rec.get("perturbations") or () if p.get("factor") == name), None)


def processing_text(hours: float, case_id: str, day: int) -> str:
    """How the report words the wait before centrifugation: three wordings, picked per draw, each with
    the number of hours (the room-temperature clause only on a wait of two hours or more)."""
    warm = hours >= 2.0
    if not background_on():
        return f"采血后 {hours:.1f} 小时离心" + ("，期间室温放置" if warm else "")
    k = int(h01("p4proc", case_id, day) * 3)
    if k == 0:
        return f"采血后 {hours:.1f} 小时离心" + ("，期间室温放置" if warm else "")
    if k == 1:
        return f"{hours:.1f} 小时后离心" + ("（采血后室温放置）" if warm else "")
    return f"离心前放置 {hours:.1f} 小时" + ("（室温）" if warm else "")


def patient_remarks(item: dict, case_id: str) -> list[dict]:
    """One pertinent negative the patient gives at each of the two compared draws, worded
    differently on the two draws of an item (`render.patient_remarks`)."""
    texts = _R()["patient_remarks"]
    out, used = [], None
    for d in (int(item["t0"]), int(item["t1"])):
        k = int(h01("p4remark", case_id, d) * len(texts))
        if k == used:
            k = (k + 1) % len(texts)
        used = k
        out.append({"draw_day": d, "text": texts[k]})
    return out


def specimen_policy(analyte: str) -> str | None:
    """The laboratory's printed bound on the wait before centrifugation (None for an analyte with no bound)."""
    from haenv.world.labworld.perturb import stability_h
    lim = stability_h(analyte)
    from haenv.world.labworld import tables
    return tables()["render"]["stability_statement"].format(h=lim) if lim is not None else None


def report_row(item: dict, rec: dict, case_id: str) -> dict:
    from haenv.world.labworld import analyte as A
    from haenv.world.labworld.perturb import factor as F
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
    if jf and (jf.get("role") == "cause" or (jf.get("params") or {}).get("method_throughout")):
        method = F("jaffe_enzymatic_switch")["jaffe_method"]
    ls = _by_factor(rec, "lab_switch_zero_bias")
    if ls and (ls.get("params") or {}).get("reagent_change"):
        method = method + "，试剂批号 B"
    elif ls:
        lab = R["alt_lab"].split("（", 1)[0] + "（另一院区）"
    hemo = "轻度（1+）" if _by_factor(rec, "hemolysis_note") else "无"
    if method.startswith("床旁"):
        specimen = {"sample": "指尖毛细血管全血", "processing": "即时检测", "hemolysis_index": "—"}
    else:
        tb = _by_factor(rec, "tube_type")
        plasma = bool(tb and (tb.get("params") or {}).get("tube") == "li_heparin")
        sample = (R["tube_plasma"] if plasma else R["tube_serum"]) if background_on() else "静脉血（促凝血清管）"
        specimen = {"sample": sample,
                    "processing": processing_text(hours, case_id, d),
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
    def dinner(back: int, key: str, meat: bool) -> dict:
        t = meat_dinner_time(case_id, d) if meat else \
            f"{18 + int(h01(key + '-h', case_id, d) * 3):02d}:{int(h01(key + '-m', case_id, d) * 60):02d}"
        food = _pick(R["meals_meat"] if meat else R["dinner_light"], "p4meat" if meat else key, case_id, d)
        return {"day": d - back, "time": t, "meal": "前一日晚餐" if back == 1 else "前两日晚餐", "food": food}

    def two_days_before(meat: bool) -> dict:
        """The meal two days before the draw, at a breakfast, lunch or dinner slot drawn per draw (so the slot of
        a meat meal is no cue); its food comes from the same dish lists as a meal on the day itself."""
        k = int(h01("p4slot2", case_id, d) * 3)
        if k == 2:
            return dinner(2, "p4dinner2", meat)
        win = R["breakfast_window"] if k == 0 else R["lunch_window"]
        a, b = (_min(x) for x in win)
        t = _hhmm(a + int(h01("p4slot2-t", case_id, d) * (b - a + 1)))
        if meat:
            lib, key = (R["breakfast_meat"] if k == 0 else R["meals_meat"]), "p4meat"
        else:
            lib, key = (R["meals_light"] if k == 0 else R["lunch_light"]), "p4light2"
        return {"day": d - 2, "time": t, "meal": "前两日" + ("早餐" if k == 0 else "午餐"), "food": _pick(lib, key, case_id, d)}
    # a draw can carry both a meat meal that acts (a breakfast or lunch 1-4 h before it) and a meat meal two
    # days before that does not: the two are separate perturbations of one factor
    meals = [p for p in rec.get("perturbations") or () if p.get("factor") == "meat_meal"]
    meat = next((p for p in meals if int(p.get("params", {}).get("days_before", 0)) < 2
                 and p.get("params", {}).get("meal") != "dinner"), None)
    meat_dinner = any(int(p.get("params", {}).get("days_before", 0)) >= 2 or p.get("params", {}).get("meal") == "dinner"
                      for p in meals)
    entries = [two_days_before(meat_dinner) if background_on() else dinner(2, "p4dinner2", meat_dinner),
               dinner(1, "p4dinner", False)]
    if is_fasting(clock):
        entries.append({"day": d, "time": None, "meal": "早餐", "food": "采血后进食"})
        return {"draw_day": d, "draw_time": clock, "entries": entries}
    slot = meat_slot(clock, case_id, d) if meat else None
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
    from haenv.world.labworld import tables
    t = tables()["render"]
    rs = t.get("dose_reasons") or {}
    key = "glp1" if drug in (t.get("glp1_class") or ()) else drug
    return rs.get(key) or rs["default"]


def medication_rows(item: dict, raw, T: int) -> list[dict]:
    from haenv.world.labworld import tables
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
    out += concurrent_rows(raw, T)
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


def concurrent_rows(raw, T: int) -> list[dict]:
    """The world's other medicines up to T (`latent_premise.world_medication`, the ACEI printed by its own
    rows), from the start of the world's record (`history_lo`): on it before the record, start, each dose
    change, stop (`med_course.record_rows`)."""
    import haenv.world.clinical.med_course as MC
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    from haenv.world.labworld import tables
    from haenv.p4.plan import ACEI_DRUG
    names = tables()["render"].get("drugs") or {}
    wm = (getattr(raw, "latent_premise", None) or {}).get(WORLD_MEDICATION_KEY) or {}
    if "history_lo" in wm:
        return MC.record_rows(wm, int(wm["history_lo"]), int(T), skip=(ACEI_DRUG,),
                              name_of=lambda d: names.get(d) or MC.DISPLAY[d][0])
    # a world without the record before day 0 (`synth.world_history: short`, `med_dynamics: off`): as before
    out = []
    for m in wm.get("concurrent") or ():
        if m["drug"] == ACEI_DRUG:
            continue
        nm = names.get(m["drug"]) or MC.DISPLAY.get(m["drug"], (m["drug"],))[0]
        unit = MC.DISPLAY.get(m["drug"], ("", "mg"))[1]
        why = _reasons(m["drug"])
        prev = None
        for q in sorted(m.get("dose_timeline") or (), key=lambda x: int(x["ts"])):
            d, v = int(q["ts"]), float(q["value"])
            if d > T:
                break
            if prev is None and d == 0:
                out.append({"day": 0, "drug": nm, "event": f"长期服用 {v:g} {unit}（病程开始前已在用）"})
            elif prev is None:
                out.append({"day": d, "drug": nm, "event": f"起始 {v:g} {unit}"})
            elif v != prev:
                out.append({"day": d, "drug": nm, "event": f"剂量 {prev:g} → {v:g}",
                            "reason": why["up"] if v > prev else why["down"]})
            prev = v
    return out


def reference_range(a: dict, raw) -> str:
    """The analyte's reference interval printed on the report: by sex and age band when the registry
    has them (`reference_by`, `reference_old_from_age`), else the single `reference`."""
    by = a.get("reference_by")
    up = getattr(raw, "user_profile", None) or {}
    sex = {"M": "M", "F": "F"}.get(str(up.get("sex")))
    if not by or sex is None:
        return a["reference"]
    age = str(up.get("age_range") or "")
    lo = int(age.split("-")[0]) if age[:2].isdigit() else 0
    return by[sex]["old" if lo >= int(a["reference_old_from_age"]) else "young"]


def render_block(item: dict, raw, case_id: str) -> dict:
    if item.get("diversity_version"):
        from haenv.p4.diversity import render
        return render(item, raw, case_id)
    from haenv.world.labworld import analyte as A
    a = A(item["analyte"])
    T = int(item["t1"])
    recs = sorted(item["records"], key=lambda r: int(r["day"]))
    by = {int(r["day"]): r for r in recs}
    block = {
        **({"comparison_basis": item["comparison_basis"]} if item.get("comparison_basis") else {}),
        "analyte": item["analyte"], "name": a["display"], "unit": a["unit"],
        "reference_range": reference_range(a, raw),
        "previous_draw_day": int(item["t0"]), "today": T,
        **laboratory_records(item, case_id),
        "report": [report_row(item, r, case_id) for r in recs],
        "diet_record": ([diet_rows(item, r, case_id) for r in recs] if item.get("source_policy") == "clinical" else
                        [diet_rows(item, by[int(item["t0"])], case_id), diet_rows(item, by[T], case_id)]),
        **({"patient_remarks": patient_remarks(item, case_id)} if background_on() else {}),
        "medication_record": medication_rows(item, raw, T),
        **({"known_comorbidity": f"{cm['display']} {cm['stage']} 期（肾内科随诊）"} if (cm := item.get("comorbidity")) else {}),
        "text_source": TEXT_SOURCE,
    }
    if item.get("source_policy") == "clinical":
        # Dates in the original report stream identify the comparison. Neither
        # a selected prior draw nor an interpretation is a patient fact.
        for key in ("previous_draw_day", "interpretation_rule", "comparison_basis"):
            block.pop(key, None)
        for row in block["report"]:
            row.update(analyte=item["analyte"], name=a["display"],
                       reference_range=block["reference_range"])
    return block


REAGENT_NOTICE = "同厂家新批号试剂，批间比对已通过。"


def laboratory_records(item: dict, case_id: str) -> dict:
    """Dated laboratory method documents, available for every class.

    In the clinical source view, every completed testing batch records the
    method file it used, including batches that continue the same method.
    Filtering the laboratory's records to patient-level method transitions
    incorrectly makes document count disclose the comparison's source.
    These records come from actual draws, not added draws or padded documents.
    """
    an = item["analyte"]
    records = sorted(item["records"], key=lambda r: int(r["day"]))
    lo = int(records[0]["day"])
    issued = lo - 1 - int(h01("p4-procedure-day", case_id) * 90)
    policy = specimen_policy(an)
    procedures = [{"day": issued, "source": "检验科标本处理规程", "scope": an,
                   "valid_from": issued, "text": policy or "按项目方法文件处理标本；床旁全血即时检测。"}]
    notices = []
    previous = None
    for r in records:
        row = report_row(item, r, case_id)
        ls = _by_factor(r, "lab_switch_zero_bias")
        state = (row["lab"], row["method"])
        if state == previous and item.get("source_policy") != "clinical":
            continue
        previous = state
        # Identical structure for a changed analytical method and a verified
        # same-method instrument/lot. The visible methods decide which applies.
        notices.append({"day": int(r["day"]) - 1 - int(h01("p4-notice-day", case_id, r["day"]) * 7),
                        "valid_from": int(r["day"]), "scope": an,
                        "source": "检验科方法与试剂使用通知", "lab": row["lab"], "method": row["method"],
                        "text": (REAGENT_NOTICE if ls and ls.get("reagent") else
                                 "同型号分析仪、同厂家试剂；方法比对已通过。" if ls else
                                 "本期使用报告所列检测方法；方法文件与质量控制按该方法执行。")})
        if item.get("source_policy") == "clinical":
            from haenv.p4.realize import clinical_sources
            notices[-1]["day"] = int(r["day"])
            notices[-1]["source"] = "检验科方法通知与检测批次使用记录"
            notices[-1]["comparison_reference_method"] = clinical_sources()["method_comparison"]["reference_method_display"]
    return {"laboratory_method_record": notices, "specimen_procedure": procedures}
