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

from haenv.i18n.text import J, L, T

TEXT_SOURCE = "template"
RENEWAL_READING = L("p3.reading.not_due")
NOTE = L("p3.note")
DRUG_KEYS = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")


def drug_name(key: str):
    """The short printed name of a managed drug (its display before the bracketed class)."""
    return L(f"drug.{key}")


def key_of(d: dict) -> str:
    """The managed drug a `meds.drug` table describes."""
    from haenv.p3.plan import identify_drug as parse_drug
    key = parse_drug(d["display"])
    if key is None:
        raise ValueError("unknown managed drug")
    return key


def _fmt(v: float, nd: int) -> str:
    return f"{v:.{nd}f}" if nd else f"{int(round(v))}"


def _dose(v: float, unit: str) -> str:
    return T("p3.dose", value=int(v) if float(v).is_integer() else v, unit=unit.split('/')[0])


def _labs(renal: dict, day: int, uacr: dict | None = None, k_nd: int = 1) -> dict:
    """Creatinine and potassium of the visit (A12: an ACEI is monitored on them); with CKD (A24)
    the eGFR beside the creatinine and the urine ACR where it was measured."""
    if day not in renal:
        return {}
    v = renal[day]
    egfr = T("p3.labs.egfr", egfr=v[2]) if len(v) > 2 else ""
    acr = T("p3.labs.acr", acr=uacr[day]) if uacr and day in uacr else ""
    return {"labs": T("p3.labs.renal", cr=v[0], egfr=egfr, k=f"{v[1]:.{k_nd}f}", acr=acr)}


def visit_rows(item: dict) -> list[dict]:
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    from haenv.p3.plan import known_of
    d = drug(item["drug"], known_of({"comorbidity": (item.get("comorbidity") or {}).get("code")}))
    ax = axis(item["axis"])
    name = drug_name(item["drug"])
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
        reading = T("p3.lab.result", value=_fmt(float(r['printed']), int(ax['ndigits'])), unit=ax['unit'])

        if day == int(item["T"]):
            rows.append({"day": day, "reading": reading, "prescription": L("p3.rx.pending"), "change": "—", **_labs(renal, day, uacr, k_nd)})
            continue
        if cur is None and day not in on_day:
            # A19: a monitoring visit before the start
            rows.append({"day": day, "reading": reading, "prescription": L("p3.rx.none"), "change": L("p3.change.observe"), **_labs(renal, day, uacr, k_nd)})
            continue
        if day in on_day:
            last = day
            new = on_day[day]
            change = L("p3.change.start") if cur is None else T("p3.change.increase", old=_dose(cur, d['unit']), new=_dose(new, d['unit']))   # A10: the record's own wording
            cur = new
        else:
            # a renewal visit, inside the early window after the last change (`realize._layout`): the
            # patient's own home record is read as at every visit; a laboratory axis is not yet due
            change = L("p3.change.renew")
            if not d.get("self_measured") and day - last <= half:
                reading = RENEWAL_READING
        rows.append({"day": day, "reading": reading, "prescription": T("p3.rx.daily", name=name, dose=_dose(cur, d['unit'])),
                     "change": change, **_labs(renal, day, uacr, k_nd)})
    iv = item.get("illness_visit")
    if iv:      # a visit for an acute illness between the chronic visits: no reading, the prescription as it stood
        day = int(iv["day"])
        at = [v for dd, v in line if int(dd) <= day][-1]
        rows.append({"day": day, "reading": L("p3.reading.not_measured"), "prescription": T("p3.rx.daily", name=name, dose=_dose(at, d['unit'])),
                     "change": T("p3.change.illness_visit", reason=iv['reason'])})
        rows.sort(key=lambda r: int(r["day"]))
    return rows


def attribution_rule(d: dict, background: bool = False) -> str:
    """The drug's adverse-effect attribution rule as G3 applies it (`gold.ae_cause`), p3-fixes b:
    `exposure` -- any onset on the drug is dose-related; `timing` -- by the onset windows after the
    last increase (or the start); a drug dosed by its axis (`ae_dose_decision: false`) is not lowered
    for a symptom alone. Onsets inside the gray bands are not emitted, so the rule needs no words for them."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    head = T("p3.rule.ae.head", name=drug_name(key_of(d)), ae=d['ae_display'])
    lower = L("p3.rule.ae.lower" if d.get("ae_dose_decision") is not False else "p3.rule.ae.axis_dosed")
    pre = L("p3.rule.ae.predating")
    if d["attribution"] == "exposure":
        return T("p3.rule.ae.exposure", head=head, lower=lower, pre=pre)
    lo, hi = (int(x) for x in rl["ae_dose_window"])
    return T("p3.rule.ae.timing", head=head, lo=lo, hi=hi, lower=lower,
             umin=int(rl['ae_unattributed_min']), pre=pre)


def adherence_rule() -> str:
    """The adherence line G3 reads (`rule.adherence_line` over `adherence_window_days`), p3-fixes d."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    pct = int(round(100 * float(rl["adherence_line"])))
    return T("p3.rule.adherence", window=int(rl['adherence_window_days']), pct=pct)


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
    out = [T("p3.rule.interval", tau=tau, half=half), attribution_rule(d, bool(item.get("background"))), adherence_rule()]
    if item.get("documented_course_only"):
        out.append(L("p3.rule.safety_course"))
    if item["drug"] == "levothyroxine":
        from haenv.p3.plan import specification
        out.append(T("p3.rule.absorption", hours=specification()["levothyroxine_absorption"]["minimum_separation_hours"]))
    if d.get("when_off_at_top"):
        from haenv.world.labworld.meds import axis
        out.append(T("p3.rule.top_check", name=drug_name(item["drug"]), top=d["ladder"][-1],
                     unit=d["unit"], axis=axis(d["axis"])["display"]))
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
        out.append(L("protocol.p3.documented_course_only"))
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
    tgt = T("p3.target.upper", axis=ax['display'], up=up, unit=ax['unit'])
    if ax.get("lower") is not None:
        tgt = T("p3.target.range", axis=ax['display'], lo=ax['lower'], up=up, unit=ax['unit'])
    diary = sorted(item["diary"], key=lambda e: int(e["onset"]))
    on_day = {int(x): v for x, v in sorted(item["dose_line"], key=lambda x: int(x[0]))[1:]}   # the increases
    cm = item.get("comorbidity")
    known = {}
    if cm:      # A24: the comorbidity the clinic knows, staged by a persistent eGFR (`ckd_stage`), and the nephrology medication
        known = {"known_comorbidity": T("p3.comorbidity.ckd", display=cm['display'], stage=ckd_stage(item),
                                        comed=T("p3.comorbidity.comed", med=cm['co_medication']) if cm.get("co_medication") else "")}
    out = {
        **({"documented_course_only": True} if item.get("documented_course_only") else {}),
        **({"home_pressure_record": home_pressure_rows(item)}
           if item.get("documented_course_only") and item["axis"] == "home_sbp" else {}),
        **({"adverse_event_record": list(item["adverse_event_record"])} if item.get("adverse_event_record") else {}),
        **({"review_from_day": item["review_from_day"]} if "review_from_day" in item else {}),
        "condition": d["condition"], "drug": d["display"], **known,
        "other_medications": list(item.get("other_medications") or ()),
        "dose_ladder": T("p3.ladder", ladder=J("slash", [_dose(x, d["unit"]) for x in d["ladder"]])),
        "control_axis": ax["display"], "clinic_target": T("p3.target.clinic_drug", name=d["display"], target=tgt), "clinic_protocol": clinic_protocol(item),
        # Source instructions cover newly registered medicines until the shared
        # hospital catalog includes their monographs. They never read the gold.
        **({"medication_instructions": clinic_protocol(item)}
           if item["drug"] in ("amlodipine", "losartan", "levothyroxine") else {}),
        "visits": visit_rows(item),
        "refill_record": [{"day": int(p), "dispensed": T("p3.refill.dispensed", sup=sup, extra=(
                               T("p3.refill.new_strength", dose=_dose(on_day[int(p)], d['unit'])) if int(p) in on_day else ""))}
                          for p in item["refills"]],
        "pill_count_today": pill_count(item, sup),
        "symptom_diary": [{"symptom": e["symptom"], "onset_day": int(e["onset"]),
                           "severity": e["severity"],
                           "status": e["status"]}
                          for e in diary],
        "encounter_record": [{"day": int(e["onset"]), "source": L("p3.source.consult"),
                              "symptom": e["symptom"], "onset_day": int(e["onset"]),
                              "history": e.get("context") or (L("p3.context.no_trigger") if e.get("kind") == "drug_symptom" else "—")}
                             for e in diary],
        **({"administration_record": item["administration_record"]} if item.get("administration_record") else {}),
        "today": int(item["T"]), "note": NOTE, "text_source": TEXT_SOURCE,
    }
    from haenv.world.protocols.delivery import library_block
    return library_block(out, "p3")


def home_pressure_rows(item):
    """One observed seven-day aggregate per visit, never a daily sampling stream."""
    return [{"ts": int(r["day"]), "value": float(r["printed"]),
             "aggregation": L("p3.agg.home_sbp_7d")} for r in item["visits"]]


def course_documented(item, entry):
    """Every course G3 can inspect is documented, including resolved events."""
    return True


def pill_count(item, supply):
    from haenv.p3.plan import h01
    from haenv.p3.plan import specification
    keys = specification()["surface"]["pill_count_templates"]
    key = item.get("course_documentation_key") or item.get("record_key") or str(item["dose_line"])
    template_key = keys[int(h01("p3-count-channel", key) * len(keys))]
    return T(template_key, day=int(item["refills"][-1]), sup=supply, left=int(item["pill_count"]))
