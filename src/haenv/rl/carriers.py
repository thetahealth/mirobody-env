"""The carriers of families A, B and G: the record rows that hold the red line (redline arm) or its safe
counterpart (control arm) at the same place.

* A (old contraindication): an allergy entry, a past-history entry (with an ECG result for QT), or an outside
  antenatal / gynaecology letter; the item drug is a new prescription about to start.
* B (interaction from another prescriber): an outside letter and the other prescriber's dispensing rows (a
  source column names the pharmacy and prescriber); control = a non-interacting alternative.
* G (unit or label arithmetic): a home glucose log from a replacement meter in mmol/L, outside creatinine in
  µmol/L, a weight recorded in pounds, or a pharmacy label (concentration, frequency, per-flare maximum).
* F (a follow-up the assistant itself committed to): earlier conversations with this assistant (the same text in
  both arms), one at the drug's start where the assistant commits to a dated safety check; redline = no result
  of that check since the start; control = the check's rows on the committed schedule, normal values.

`build(rule, variant, ctx)` returns the rows and the facts slots; every draw is keyed on (case, rule), never on
the arm, so both arms differ only in the rows the patch names. Texts are i18n templates
(`registry/i18n/*/templates/rl.yaml: rl.carrier.<rule id>.*`); numbers are registry parameters (ours).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import datetime

from haenv.rl import plan as P
from haenv.rl import zh as Z

NEW_FAMILIES = ("A", "B", "G", "F")


def _t(key: str, **slots) -> str:
    from haenv.i18n.text import text_of
    return text_of(key, Z.lang(), **slots)


def _supply(text: str, days: int) -> str:
    return f"{text}，{days} 天量" if Z.zh() else f"{text}, {days}-day supply"


def _disp(rule) -> str:
    return Z.rf(rule, "display", rule["trigger"]["display"])


def _k(rid: str, name: str) -> str:
    return f"rl.carrier.{rid}.{name}"


def _iso(day, ctx) -> str:
    return P.iso(day, ctx["T"], ctx["cid"])


def _year(day, ctx) -> int:
    return P.date_of(day, ctx["T"], ctx["cid"]).year


def _month(day, ctx) -> str:
    return _iso(day, ctx)[:7]


def _empty() -> dict:
    return {"allergies": None, "past_carrier": None, "visits": [], "patch_visits": [], "dispensing": [],
            "patch_disp_keys": set(), "target_fills": None, "results": [], "patch_analytes": [], "from_date": None,
            "glucose_log": None, "glucose_note": None, "patch_glucose_from": None, "messages": [], "draw_end": None,
            "chem_override": None, "trigger": None, "slots": {}, "status": None, "extra_problems": [],
            "conversation": None}


def build(rule: dict, variant: str, ctx: dict) -> dict:
    kind = rule["trigger"]["kind"]
    cx = _empty()
    cx["extra_problems"] = list(rule.get("extra_problems") or ())
    red = variant == "redline"
    f = {"history_contraindication": _history, "pregnancy_trimester": _pregnancy,
         "interaction_current": _interaction, "converted_low_reading": _glucose,
         "converted_creatinine_egfr": _creatinine, "label_dose": _label,
         "converted_weight_dose": _weight, "committed_check_missing": _followup}[kind]
    f(rule, red, ctx, cx)
    return cx


# ------------------------------------------------------------------------------------------------ A

def _history(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    ev = rule.get("event") or {}
    arm = "redline" if red else "control"
    secs = rule["trigger"]["sections"]
    if "before" in ev:
        day = T - P.randint(*ev["before"], cid, rid, "event_day")
    else:                                                       # an age at diagnosis (years back)
        day = T - int(365.25 * (ctx["age"] - P.randint(*ev["age"], cid, rid, "event_age")))
        if day > T - 3 * 365:
            raise _unrealizable("the past diagnosis would be too recent")
    slots = {"date": _iso(day, ctx), "month": _month(day, ctx), "year": _year(day, ctx),
             "year2": _year(day + 365, ctx), "year3": _year(day + 3 * 365, ctx)}
    value = None
    if rid.endswith("qt-prolongation"):
        rng = rule["redline"]["qtc"] if red else rule["resolution"]["qtc"]
        q = P.randint(*rng, cid, rid, "qtc")
        q2 = q + P.randint(3, 9, cid, rid, "qtc2")
        slots.update(qtc=q, qtc2=q2, hr=P.randint(58, 74, cid, rid, "ecg_hr"))
        cx["results"].append({"date": slots["date"], "analyte": _disp(rule),
                              "result": _t(_k(rid, f"ecg.{arm}"), **slots)})
        cx["patch_analytes"] = [_disp(rule)]
        value = str(q)
    if "allergies" in secs:
        cx["allergies"] = [_t(_k(rid, "allergy"), **slots)] if red else [ctx["allergy"]]
        cx["patch_allergies"] = True
        value = value or _t(_k(rid, "value"))
    if "past_history" in secs:
        sex = ctx["sex"]
        key = f"past.{arm}" if red else (f"past.control_{sex}" if _has(_k(rid, f"past.control_{sex}")) else
                                          "past.control")
        cx["past_carrier"] = _t(_k(rid, key), **slots)
        value = value or _t(_k(rid, "value"))
    if _has(_k(rid, "visit.redline")):
        v = {"date": slots["date"], "clinician": _t(_k(rid, "visit.who")), "type": _t(_k(rid, "visit.type")),
             "summary": _t(_k(rid, f"visit.{arm}"), **slots)}
        cx["visits"].append(v)
        cx["patch_visits"].append(v)
    when = str(slots[ev.get("date_slot", "date")])
    cx["trigger"] = {"date": when, "value": value}
    cx["slots"] = {"date": when, "value": value}


def _pregnancy(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    rl = rule["redline"]
    weeks = P.randint(*rl["weeks_at_today"], cid, rid, "ga_weeks")
    lmp_red = T - 7 * weeks - P.randint(0, 6, cid, rid, "ga_days")
    visit = lmp_red + 7 * P.randint(*rl["booking_weeks"], cid, rid, "booking")
    if visit >= ctx["presc_day"] - 3:
        raise _unrealizable("the antenatal visit would follow the prescription")
    if red:
        lmp = lmp_red
        ga = (visit - lmp) // 7
        slots = {"lmp": _iso(lmp, ctx), "edd": _iso(lmp + 280, ctx), "ga": ga, "date": _iso(visit, ctx),
                 "next": _iso(visit + 28, ctx)}
    else:
        lmp = visit - P.randint(*rule["resolution"]["lmp_before_visit"], cid, rid, "lmp_ctl")
        slots = {"lmp": _iso(lmp, ctx), "date": _iso(visit, ctx), "year": _year(visit - 900, ctx)}
    arm = "redline" if red else "control"
    v = {"date": _iso(visit, ctx), "clinician": _t(_k(rid, "visit.who")), "type": _t(_k(rid, f"visit.type")),
         "summary": _t(_k(rid, f"visit.{arm}"), **slots)}
    cx["visits"].append(v)
    cx["patch_visits"].append(v)
    cx["trigger"] = {"date": _iso(visit, ctx), "value": _iso(lmp_red, ctx)}
    cx["slots"] = {"date": _iso(visit, ctx), "value": (T - lmp_red) // 7, "lmp": _iso(lmp_red, ctx)}


# ------------------------------------------------------------------------------------------------ B

def _interaction(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    c = rule["course"]
    arm = "redline" if red else "control"
    start = T - P.randint(*c["before"], cid, rid, "course_start")
    src = _t(_k(rid, "source"))
    sup0 = int(c["supply_control"]) if (not red and "supply_control" in c) else int(c["supply"])
    rows = [(start, _t(_k(rid, f"disp.{arm}")), "__other_0__", sup0)]
    slots = {"date": _iso(start, ctx)}
    for name in ("step", "refill"):                             # a dose step or a refill of the same course
        if name in c:
            d = start + P.randint(*c[name], cid, rid, f"course_{name}")
            if d >= T:
                raise _unrealizable(f"{name} after today")
            rows.append((d, _t(_k(rid, f"disp2.{arm}")), "__other_1__", int(c.get(f"{name}_supply", c["supply"]))))
            slots[name] = _iso(d, ctx)
    for d, txt, key, supply in rows:
        cx["dispensing"].append((d, _supply(txt, supply), key, src))
        cx["patch_disp_keys"].add(key)
    if c.get("same"):                                           # a medicine of the same visit, the same in both arms
        cx["dispensing"].append((start, _supply(_t(_k(rid, 'disp_same')), int(c['same'])), "__other_same__",
                                 src))
    if red and not any(d <= T < d + sup for d, _, _, sup in rows):
        raise _unrealizable("the interacting course ended before today")
    vday = start - int(c.get("visit_offset", 0))
    v = {"date": _iso(vday, ctx), "clinician": _t(_k(rid, "visit.who")), "type": _t(_k(rid, "visit.type")),
         "summary": _t(_k(rid, f"visit.{arm}"), **slots)}
    cx["visits"].append(v)
    cx["patch_visits"].append(v)
    ing = rule["trigger"]["interacting"][0]["ingredient"]
    if Z.zh():
        ing = Z.interacting_names(rule)[ing][0]
    cx["trigger"] = {"date": _iso(start, ctx), "value": ing}
    cx["slots"] = {"date": _iso(start, ctx), "value": ing}


# ------------------------------------------------------------------------------------------------ G

def _glucose(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    t = rule["trigger"]
    n_new = P.randint(*t["days_new"], cid, rid, "days_new")
    n_old = P.randint(*t["days_usual"], cid, rid, "days_usual")
    switch = T - n_new
    rows = []
    for i in range(n_old, 0, -1):
        d = switch - i
        v = P.uniform(*t["usual_mg_dl"], cid, rid, "usual_mgdl", d)
        rows.append({"date": _iso(d, ctx), "value": str(int(round(v))), "unit": "mg/dL"})
    k0 = len(rows)
    rl = rule["redline"]
    n_low = P.randint(*rl["n_low"], cid, rid, "n_low")
    start = P.uniform(*rl["start_mmol"], cid, rid, "start_mmol")
    thr_mmol = float(t["threshold"]) / float(t["mg_dl_per_mmol"])
    vals = []
    for j in range(n_new):
        d = switch + j
        if red:
            last3 = j >= n_new - 3
            if j >= n_new - n_low:
                v = P.uniform(*rl["low_mmol"], cid, rid, "low_mmol", j)
            elif last3:
                v = P.uniform(thr_mmol + 0.05, thr_mmol + 0.5, cid, rid, "edge_mmol", j)
            else:
                frac = j / max(1, n_new - 3)
                v = start + (thr_mmol + 0.6 - start) * frac + 0.3 * (P.draw(cid, rid, "noise_mmol", j) - 0.5)
        else:
            v = P.uniform(*rule["resolution"]["mmol"], cid, rid, "safe_mmol", j)
        v = round(v, 1)
        vals.append(v)
        rows.append({"date": _iso(d, ctx), "value": f"{v:.1f}", "unit": "mmol/L"})
    if red and sum(v * float(t["mg_dl_per_mmol"]) < float(t["threshold"]) for v in vals[-int(t["last_n"]):]) != n_low:
        raise _unrealizable("low readings off by rounding")
    cx["glucose_log"] = rows
    cx["patch_glucose_from"] = k0
    cx["glucose_note"] = _t(_k(rid, "note"), date=_iso(switch, ctx))
    cx["messages"].append({"date": _iso(switch, ctx), "from": "患者" if Z.zh() else "patient",
                           "text": _t(_k(rid, "message"))})
    lows = [(i, v) for i, v in enumerate(vals) if v * float(t["mg_dl_per_mmol"]) < float(t["threshold"])]
    i, v = lows[-1] if lows else (len(vals) - 1, vals[-1])
    cx["trigger"] = {"date": _iso(switch + i, ctx), "value": f"{v:.1f}"}
    cx["slots"] = {"date": _iso(switch + i, ctx), "value": f"{v:.1f}",
                   "converted": int(round(v * float(t["mg_dl_per_mmol"]))), "n_low": n_low}


def _creatinine(rule, red, ctx, cx):
    from haenv.rl.realize import ckd_epi_2021_inverse
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    t = rule["trigger"]
    st = rule["stay"]
    dis = T - P.randint(*st["discharge_before"], cid, rid, "discharge")
    adm = dis - P.randint(*st["length"], cid, rid, "stay_length")
    mid = adm + P.randint(2, 3, cid, rid, "mid_draw")
    src = rule["redline"] if red else rule["resolution"]
    eg = P.uniform(*src["egfr"], cid, rid, "egfr_dis")
    scr = ckd_epi_2021_inverse(eg, ctx["age"], ctx["sex"])
    per = float(t["umol_per_mg_dl"])
    u_dis = int(round(scr * per))
    u_adm = int(round(u_dis * P.uniform(*src["rise"], cid, rid, "rise")))
    u_mid = int(round((u_adm + u_dis) / 2 + 4 * (P.draw(cid, rid, "mid_noise") - 0.5)))
    hi = 90 if ctx["sex"] == "F" else 110
    ref = "45–90" if ctx["sex"] == "F" else "60–110"
    for d, u in ((adm, u_adm), (mid, u_mid), (dis, u_dis)):
        cx["results"].append({"date": _iso(d, ctx), "analyte": _disp(rule), "value": str(u), "unit": t["unit"],
                              "ref": ref, "flag": "H" if u > hi else ""})
    cx["patch_analytes"] = [_disp(rule)]
    cx["from_date"] = _iso(adm, ctx)
    cx["draw_end"] = adm - P.randint(30, 60, cid, rid, "draw_end_before_stay")
    slots = {"adm": _iso(adm, ctx), "dis": _iso(dis, ctx),
             "city": Z.ph(P.pick(("Toronto", "Lisbon", "Dublin", "Madrid", "Sydney"), cid, rid, "city"))}
    cx["visits"].append({"date": _iso(dis, ctx), "clinician": _t(_k(rid, "visit.who"), **slots),
                         "type": _t(_k(rid, "visit.type")), "summary": _t(_k(rid, "visit"), **slots)})
    from haenv.rl.realize import ckd_epi_2021
    cx["trigger"] = {"date": _iso(dis, ctx), "value": str(u_dis)}
    cx["slots"] = {"date": _iso(dis, ctx), "value": str(u_dis), "converted": f"{u_dis / per:.2f}",
                   "egfr": int(round(ckd_epi_2021(u_dis / per, ctx["age"], ctx["sex"])))}


def _label(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    kind = rule["trigger"]["label"]
    arm = "redline" if red else "control"
    pharmacy = ctx["pharmacy"]
    fills = []
    if kind == "frequency":                                     # a discharge label from the hospital pharmacy
        st = rule["stay"]
        dis = T - P.randint(*st["discharge_before"], cid, rid, "discharge")
        adm = dis - P.randint(*st["length"], cid, rid, "stay_length")
        cad = int(rule["fills"]["cadence"])
        last = dis - P.randint(3, 20, cid, rid, "last_home_fill")
        d = last
        while d >= max(T - 300, ctx["target_start"]):
            fills.append((d, _supply(_t(_k(rid, "label.usual"), **ctx["weekdays"]), cad), "__target__", pharmacy))
            d -= cad
        lab = _t(_k(rid, f"label.{arm}"), **ctx["weekdays"])
        fills.append((dis, lab, "__label__", _t(_k(rid, "source"))))
        slots = {"adm": _iso(adm, ctx), "dis": _iso(dis, ctx)}
        cx["visits"].append({"date": _iso(dis, ctx), "clinician": _t(_k(rid, "visit.who")),
                             "type": _t(_k(rid, "visit.type")), "summary": _t(_k(rid, "visit"), **slots)})
        when = dis
        conv = int(rule["trigger"]["prescribed_weekly_mg"]) * 7
    else:
        fp = rule["fills"]
        cad = P.randint(*fp["cadence"], cid, rid, "label_cadence")
        last = T - P.randint(*fp["last_fill"], cid, rid, "label_last")
        d = last - cad
        while d >= max(T - 300, ctx["target_start"]):
            fills.append((d, _t(_k(rid, "label.usual")), "__target__", pharmacy))
            d -= cad
        fills.append((last, _t(_k(rid, f"label.{arm}")), "__label__", pharmacy))
        when = last
        conv = (float(fp["wrong_mg_ml"]) * float(fp["ml"])) if kind == "concentration_volume" else \
            float(fp["wrong_max_mg"])
        conv = f"{conv:g}"
    cx["target_fills"] = fills
    cx["patch_disp_keys"].add("__label__")
    cx["trigger"] = {"date": _iso(when, ctx), "value": _t(_k(rid, "value"))}
    cx["slots"] = {"date": _iso(when, ctx), "value": _t(_k(rid, "value")), "converted": conv}


def _weight(rule, red, ctx, cx):
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    t = rule["trigger"]
    day = T - P.randint(*rule["visit_before"], cid, rid, "preop_day")
    kg = P.uniform(*(rule["redline"]["kg"] if red else rule["resolution"]["kg"]), cid, rid, "preop_kg")
    oz_total = int(round(kg / float(t["kg_per_lb"]) * 16))
    lb, oz = divmod(oz_total, 16)
    conv = (lb + oz / 16.0) * float(t["kg_per_lb"])
    if red != (conv <= float(t["weight_max_kg"])):
        raise _unrealizable("weight off the arm's side after rounding")
    height = P.randint(*((65, 72) if ctx["sex"] == "M" else (59, 66)), cid, rid, "preop_height_in")   # inches
    slots = {"lb": lb, "oz": oz, "sbp": P.randint(124, 142, cid, rid, "preop_sbp"),
             "dbp": P.randint(68, 82, cid, rid, "preop_dbp"), "hr": P.randint(62, 78, cid, rid, "preop_hr"),
             "ft": height // 12, "inch": height % 12, "date": _iso(day, ctx)}
    v = {"date": _iso(day, ctx), "clinician": _t(_k(rid, "visit.who")), "type": _t(_k(rid, "visit.type")),
         "summary": _t(_k(rid, "visit"), **slots)}
    cx["visits"].append(v)
    cx["patch_visits"].append(v)
    cr = rule["creatinine"]
    cx["chem_override"] = lambda i: {"creatinine": P.uniform(*cr, cid, rid, "stable_creatinine", i)}
    shown = f"{lb} 磅 {oz} 盎司" if Z.zh() else f"{lb} lb {oz} oz"
    cx["trigger"] = {"date": _iso(day, ctx), "value": shown}
    cx["slots"] = {"date": _iso(day, ctx), "value": shown, "converted": f"{conv:.1f}"}


# ------------------------------------------------------------------------------------------------ F

def _followup(rule, red, ctx, cx):
    """The check the assistant committed to at the drug's start (day `ctx["start"]`): a baseline draw shortly
    before the start (both arms), the conversation (both arms) and, in the control arm only, the check's rows on
    the committed schedule. The patch span is the check's results after the start."""
    from haenv.rl.realize import baseline_result, panel_rows, panel_specs
    rid, cid, T = rule["id"], ctx["cid"], ctx["T"]
    fu = rule["followup"]
    S = ctx["start"]
    C = S + int(fu["check_after"])
    base = {"sex": ctx["sex"], "age": ctx["age"]}
    chem = fu.get("chem") or {}
    over = (lambda i: {a: P.uniform(*rng, cid, rid, "fu_chem", a, i) for a, rng in chem.items()}) if chem else None
    cx["chem_override"] = over
    panels = [p for p in fu.get("panels") or () if p in ("chem", "lft", "cbc")]
    b = S - P.randint(2, 10, cid, rid, "fu_baseline")
    for name in sorted(set(panels) | {"chem"}):
        cx["results"] += panel_rows(rule, name, b, cid, T, base, 997, over(997) if over and name == "chem" else None)
    for test in fu.get("baseline_tests") or ():
        cx["results"].append(baseline_result(rule, test, cid, _iso(b, ctx)))
    cx["baseline_date"] = _iso(b, ctx)
    cx["draw_end"] = b - P.randint(40, 100, cid, rid, "fu_draw_end")
    if not red:
        days = [max(S + 3, C + P.randint(-2, 2, cid, rid, "fu_first"))]
        days += [S + int(x) + P.randint(0, 2, cid, rid, "fu_then", j) for j, x in enumerate(fu.get("then") or ())]
        lv = fu.get("level")
        for j, d in enumerate(x for x in days if x <= T - 2):
            for name in panels:
                cx["results"] += panel_rows(rule, name, d, cid, T, base, 900 + j,
                                            over(900 + j) if over and name == "chem" else None)
            if lv:
                v = round(P.uniform(*lv["range"], cid, rid, "fu_level", j), int(lv["decimals"]))
                cx["results"].append({"date": _iso(d, ctx), "analyte": Z.rf(rule, "level_display", lv["display"]),
                                      "value": f"{v:.{int(lv['decimals'])}f}", "unit": lv["unit"], "ref": lv["ref"],
                                      "flag": ""})
    names = sorted({Z.ph(sp["display"]) for p in panels for sp in panel_specs(rule, p)}
                   | ({Z.rf(rule, "level_display", fu["level"]["display"])} if fu.get("level") else set()))
    cx["patch_analytes"] = names
    cx["from_date"] = _iso(S + 1, ctx)
    cx["conversation"] = conversation(rule, ctx, S, C)
    cx["trigger"] = {"date": _iso(C, ctx), "value": _iso(C, ctx)}
    cx["slots"] = {"date": _iso(C, ctx), "value": _iso(C, ctx), "start": _iso(S, ctx)}


def conversation(rule, ctx, S: int, C: int) -> list[dict]:
    """2 to 4 dated exchanges with this assistant (keyed; the same in both arms): the commitment at the start day
    and ordinary ones (`redline_questions.yaml: record.conversations`), in about half of the items one naming
    another current medicine (`record.med_conversations`). Ordinary exchanges fall before the committed date."""
    rid, cid = rule["id"], ctx["cid"]
    lib = P.questions_table().get("record") or {}
    sl = {"clinician": str(ctx["clin"]).split("（" if Z.zh() else " (")[0], "date": _iso(C, ctx),
          "date2": _iso(S + 30, ctx), "dose": ctx["dose"], **ctx["weekdays"]}
    out = [{"day": S, "user": _t(_k(rid, "conv.user"), **sl), "assistant": _t(_k(rid, "conv.assistant"), **sl)}]
    n = P.randint(2, 4, cid, rid, "conv_n") - 1
    plain = sorted(lib.get("conversations") or (), key=lambda x: P.draw(cid, rid, "conv_order", x["user"]))
    picks = [{"user": Z.ph(x["user"]), "assistant": Z.ph(x["assistant"])} for x in plain[:n]]
    meds = [m for m in ctx.get("meds") or ()]
    if meds and P.draw(cid, rid, "conv_med") < 0.5:
        m = P.pick(meds, cid, rid, "conv_med_name")
        x = P.pick(lib.get("med_conversations") or (), cid, rid, "conv_med_text")
        picks[P.randint(0, n - 1, cid, rid, "conv_med_pos")] = {"user": Z.ph(x["user"]).format(med=m["name"]),
                                                                "assistant": Z.ph(x["assistant"]).format(med=m["name"])}
    lo = max(ctx["lo_day"] + 10, S - 150)
    used = {S}
    for j, x in enumerate(picks):
        d = P.randint(lo, C - 2, cid, rid, "conv_day", j)
        while d in used:
            d += 1
        used.add(d)
        out.append({"day": d, "user": x["user"], "assistant": x["assistant"]})
    out.sort(key=lambda e: e["day"])
    return [{"date": _iso(e["day"], ctx), "user": e["user"], "assistant": e["assistant"]} for e in out]


# ------------------------------------------------------------------------------------------------ helpers

def _has(key: str) -> bool:
    from haenv.i18n.text import templates
    return templates("en").get(key) is not None


def _unrealizable(msg):
    from haenv.rl.realize import Unrealizable
    return Unrealizable(msg)


def past_history(rule, ctx, carrier: str | None) -> tuple[list[str], int | None]:
    """2 to 4 ordinary past-history entries (keyed; dated years back) with the carrier at a keyed place."""
    rid, cid = rule["id"], ctx["cid"]
    lib = list((P.questions_table().get("record") or {}).get("past_history") or ())
    lib = [x for x in lib if not x.get("sex") or x["sex"] == ctx["sex"]]
    lib.sort(key=lambda x: P.draw(cid, rid, "past_order", x["text"]))
    n = P.randint(2, 4, cid, rid, "n_past")
    uk = rule["jurisdiction"] == "UK"
    out = []
    for x in lib[:n]:
        yr = ctx["today"].year - P.randint(4, 30, cid, rid, "past_year", x["text"])
        if yr < ctx["today"].year - ctx["age"] + 5:
            yr = ctx["today"].year - P.randint(2, 6, cid, rid, "past_year_adult", x["text"])
        out.append(Z.ph(x.get("text_UK", x["text"]) if uk else x["text"]).format(year=yr))
    pos = None
    if carrier is not None:
        pos = P.randint(0, len(out), cid, rid, "past_pos")
        out.insert(pos, carrier)
    return out, pos


def today(ctx) -> datetime.date:
    return ctx["today"]
