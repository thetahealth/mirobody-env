"""Pack 3 production renderer: the chronic-medication clinic record the solver reads.

Writes `prediction_context.meds_p3` from the realized item (`realize.realize`). It carries every
input the gold depends on -- the drug and its ladder, the target, the visit table (reading,
prescription, start / increase / continue), the refill record and today's pill count, the
symptom diary with onset days -- and nothing that names the gold (no cause of a symptom, no
truth, no adherence figure). The leak gate (`gate.leak_hits`) scans the rendered block and the
solver payload.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

TEXT_SOURCE = "template"
RENEWAL_READING = "未到复查时间"
NOTE = ("你的建议由医生审核后执行。日数相对于本数据的第 0 天，负数为第 0 天之前的院内慢病档案记录。")


def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}" if nd else f"{int(round(v))}"


def _dose(v: float, unit: str) -> str:
    return f"{int(v) if float(v).is_integer() else v} {unit.split('/')[0]}"


def _labs(renal: dict, day: int, uacr: dict | None = None, k_nd: int = 1) -> dict:
    """Creatinine and potassium of the visit (A12: an ACEI is monitored on them); with CKD (A24)
    the eGFR beside the creatinine and the urine ACR where it was measured."""
    if day not in renal:
        return {}
    v = renal[day]
    egfr = f"（eGFR {v[2]} mL/min/1.73m²）" if len(v) > 2 else ""
    acr = f"，尿白蛋白/肌酐比 {uacr[day]} mg/g" if uacr and day in uacr else ""
    return {"labs": f"肌酐 {v[0]} μmol/L{egfr}，血钾 {v[1]:.{k_nd}f} mmol/L{acr}"}


def visit_rows(item: dict) -> list[dict]:
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    from haenv.p3.plan import known_of
    d = drug(item["drug"], known_of({"comorbidity": (item.get("comorbidity") or {}).get("code")}))
    ax = axis(item["axis"])
    name = d["display"].split("（")[0]
    line = sorted(item["dose_line"], key=lambda x: int(x[0]))
    on_day = {int(dd): v for dd, v in line}
    renal = {int(r[0]): tuple(r[1:]) for r in item.get("renal") or ()}
    cm = item.get("comorbidity")
    k_nd = int(item.get("k_digits", 1))
    uacr = None
    if cm:
        days = sorted(int(r["day"]) for r in item["visits"])
        uacr = {days[0]: cm["uacr"][0], int(item["T"]): cm["uacr"][1]}
    half = float(rule()["early_max_frac"]) * float(d["tau_rule_days"])
    rows, cur, last = [], None, None
    for r in sorted(item["visits"], key=lambda x: int(x["day"])):
        day = int(r["day"])
        reading = f"{_fmt(float(r['printed']), int(ax['ndigits']))} {ax['unit']}"

        if day == int(item["T"]):
            rows.append({"day": day, "reading": reading, "prescription": "本次待定", "change": "—", **_labs(renal, day, uacr, k_nd)})
            continue
        if cur is None and day not in on_day:
            # A19: a monitoring visit before the start
            rows.append({"day": day, "reading": reading, "prescription": "未用药", "change": "观察", **_labs(renal, day, uacr, k_nd)})
            continue
        if day in on_day:
            last = day
            new = on_day[day]
            change = "起始" if cur is None else f"加量（{_dose(cur, d['unit'])} → {_dose(new, d['unit'])}）"   # A10: the record's own wording
            cur = new
        else:
            # a renewal visit, inside the early window after the last change (`realize._layout`): the
            # patient's own home record is read as at every visit; a laboratory axis is not yet due
            change = "续方"
            if not d.get("self_measured") and day - last <= half:
                reading = RENEWAL_READING
        rows.append({"day": day, "reading": reading, "prescription": f"{name} {_dose(cur, d['unit'])} 每日",
                     "change": change, **_labs(renal, day, uacr, k_nd)})
    iv = item.get("illness_visit")
    if iv:      # a visit for an acute illness between the chronic visits: no reading, the prescription as it stood
        day = int(iv["day"])
        at = [v for dd, v in line if int(dd) <= day][-1]
        rows.append({"day": day, "reading": "未测", "prescription": f"{name} {_dose(at, d['unit'])} 每日",
                     "change": f"因{iv['reason']}就诊，慢病用药未调整"})
        rows.sort(key=lambda r: int(r["day"]))
    return rows


def attribution_rule(d: dict, background: bool = False) -> str:
    """The drug's adverse-effect attribution rule as G3 applies it (`gold.ae_cause`), p3-fixes b:
    `exposure` -- any onset on the drug is dose-related; `timing` -- by the onset windows after the
    last increase (or the start); a drug dosed by its axis (`ae_dose_decision: false`) is not lowered
    for a symptom alone. Onsets inside the gray bands are not emitted, so the rule needs no words for them."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    name = d["display"].split("（")[0]
    head = f"本院不良反应归因：{name}登记的不良反应为{d['ae_display']}。"
    lower = "应减量" if d.get("ae_dose_decision") is not False else "但本药剂量按控制指标调整，控制指标未低于目标下限时不因症状减量"
    pre = "开始用药之前已出现的，不归因于本药。"
    other = ""
    if d["attribution"] == "exposure":
        return head + f"开始用药后出现的，归因于本药剂量，{lower}；" + pre + other
    lo, hi = (int(x) for x in rl["ae_dose_window"])
    return (head + f"起病于最近一次加量（或开始用药）后 {lo}–{hi} 天内的，归因于本药剂量，{lower}；"
            f"起病距最近一次剂量变动已满 {int(rl['ae_unattributed_min'])} 天的，不能归因于本药剂量，"
            "暂不调整剂量，先查其他原因；" + pre + other)


def adherence_rule() -> str:
    """The adherence line G3 reads (`rule.adherence_line` over `adherence_window_days`), p3-fixes d."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    pct = int(round(100 * float(rl["adherence_line"])))
    return (f"本院依从性判定：按取药记录（提前取药的余量顺延计入；加量当天换新规格发药，旧规格余药不再计入）与今天的药盒计数估算最近 {int(rl['adherence_window_days'])} 天"
            f"（开始用药不足 {int(rl['adherence_window_days'])} 天的，自开始用药日起）的服药覆盖率，"
            f"低于 {pct}% 视为依从不足；控制指标未达标而依从不足时，暂不调整剂量，先核实服药情况。")


def clinic_protocol(item: dict) -> list[str]:
    """The clinic's rules the gold reads and no guideline fixes the same way everywhere:
    the reassessment interval of the drug, the adverse-effect attribution rule (p3-fixes b), the
    adherence line (p3-fixes d), and for a known comorbidity the clinic's own policy. All come from
    the registry the gold reads, so the sentence and the rule cannot differ; every item of a drug
    carries the same sentences, whatever its decision."""
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    from haenv.p3.plan import known_of
    d = drug(item["drug"], known_of({"comorbidity": (item.get("comorbidity") or {}).get("code")}))
    tau = int(d["tau_rule_days"])
    half = int(float(rule()["early_max_frac"]) * tau)
    out = [f"本院复评间隔：该药加量（或开始用药）后约 {tau} 天疗效才充分显现；加量（或开始用药）后 {half} 天内复查读数仍未达标，"
           f"属疗效尚未显现，不据此判定加量无效。", attribution_rule(d, bool(item.get("background"))), adherence_rule()]
    if item.get("adverse_event_record"):
        from haenv.world.history.history_dependency import table
        out.append(table()["packs"]["p3"]["protocol"])
        if any(r.get("prescriptions") for r in item["adverse_event_record"]):
            from haenv.world.validation.difficulty_r2 import config
            out.append(config("p3")["protocol"])
    if d.get("clinic_policy"):
        out.append(str(d["clinic_policy"]))
    if item.get("documented_course_only"):
        from haenv.world.validation.difficulty_r2 import config
        out.append("症状状态只登记原始记录明确记载的经过，未登记不表示已缓解或持续。一次症状后来缓解也不能替代对起病时用药、加量日期和其他原因的核对。")
        out.append(config("p3")["allergy_record"]["protocol"])
    return out


PERSIST_DAYS = 90     # KDIGO 2024 PP 1.1.3.1-1.1.3.2: chronicity needs >= 3 months, not a single eGFR
_G = ("G1", "G2", "G3a", "G3b", "G4", "G5")


def _g(v: float) -> str:
    return "G1" if v >= 90 else "G2" if v >= 60 else "G3a" if v >= 45 else "G3b" if v >= 30 else "G4" if v >= 15 else "G5"


def ckd_stage(item: dict) -> str:
    """The KDIGO G stage the record supports today. The stage on file (from the pre-treatment eGFR)
    moves only when today's eGFR and the latest one at least `PERSIST_DAYS` earlier both lie beyond it
    on the same side, and then to the nearer of their two categories: the middle of the three. A
    single reading, or a fall within 3 months of starting the ACEI, keeps the stage on file (KDIGO 2024
    PP 1.1.3.1-1.1.3.2; PP 3.6.4: a creatinine rise of up to 30% after starting an ACEI is expected)."""
    base = str(item["comorbidity"]["stage"])
    eg = sorted((int(r[0]), float(r[3])) for r in item.get("renal") or () if len(r) > 3)
    if not eg:
        return base
    day, today = eg[-1]
    past = [v for d, v in eg if d <= day - PERSIST_DAYS]
    if not past:
        return base
    return sorted((base, _g(today), _g(past[-1])), key=_G.index)[1]


def render_block(item: dict) -> dict:
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    sup = int(item.get("supply", 30))
    d = drug(item["drug"])
    ax = axis(item["axis"])
    up = item.get("upper", ax["upper"])          # this patient's target (A12)
    tgt = f"{ax['display']} < {up} {ax['unit']}"
    if ax.get("lower") is not None:
        tgt = f"{ax['display']} {ax['lower']}–{up} {ax['unit']}"
    diary = sorted(item["diary"], key=lambda e: int(e["onset"]))
    on_day = {int(x): v for x, v in sorted(item["dose_line"], key=lambda x: int(x[0]))[1:]}   # the increases
    cm = item.get("comorbidity")
    known = {}
    if cm:      # A24: the comorbidity the clinic knows, staged by a persistent eGFR (`ckd_stage`), and the nephrology medication
        known = {"known_comorbidity": f"{cm['display']} {ckd_stage(item)} 期，白蛋白尿 A3（肾内科随诊）"
                 + (f"；在用{cm['co_medication']}" if cm.get("co_medication") else "")}
    out = {
        **({"documented_course_only": True} if item.get("documented_course_only") else {}),
        **({"home_pressure_record": home_pressure_rows(item)}
           if item.get("documented_course_only") and item["axis"] == "home_sbp" else {}),
        **({"adverse_event_record": list(item["adverse_event_record"])} if item.get("adverse_event_record") else {}),
        **({"review_from_day": item["review_from_day"]} if "review_from_day" in item else {}),
        "condition": d["condition"], "drug": d["display"], **known,
        "other_medications": list(item.get("other_medications") or ()),
        "dose_ladder": " / ".join(_dose(x, d["unit"]) for x in d["ladder"]) + "（每日）",
        "control_axis": ax["display"], "clinic_target": "本院目标：" + tgt, "clinic_protocol": clinic_protocol(item),
        "visits": visit_rows(item),
        "refill_record": [{"day": int(p), "dispensed": f"{sup} 天量" + (f"（加量至 {_dose(on_day[int(p)], d['unit'])}，新规格）" if int(p) in on_day else "")}
                          for p in item["refills"]],
        "pill_count_today": f"末次取药（第 {int(item['refills'][-1])} 天，{sup} 天量）后，今天药盒剩余 {int(item['pill_count'])} 天量",
        "symptom_diary": [{"symptom": e["symptom"], "onset_day": int(e["onset"]),
                           "severity": e["severity"],
                           **({"status": e["status"]} if course_documented(item, e) else {})}
                          for e in diary],
        "encounter_record": [{"day": int(e["onset"]), "source": "门诊问诊记录",
                              "symptom": e["symptom"], "onset_day": int(e["onset"]),
                              "history": e.get("context") or ("无明显诱因" if e.get("kind") == "drug_symptom" else "—")}
                             for e in diary],
        "today": int(item["T"]), "note": NOTE, "text_source": TEXT_SOURCE,
    }
    from haenv.world.protocols.delivery import library_block
    return library_block(out, "p3")


def home_pressure_rows(item):
    """One observed seven-day aggregate per visit, never a daily sampling stream."""
    return [{"ts": int(r["day"]), "value": float(r["printed"]),
             "aggregation": "7日家庭收缩压均值"} for r in item["visits"]]


def course_documented(item, entry):
    """Recorded course is optional; every symptom and its onset remain visible."""
    if not item.get("documented_course_only"):
        return True
    from haenv.world.validation.difficulty_r2 import config
    from haenv.world.history.history_dependency import uniform
    return uniform("p3-course-documentation", item["course_documentation_key"], entry["onset"], entry["symptom"]) < config("p3")["documented_symptom_course"]["value"]
