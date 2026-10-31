"""Features of a pack-3 item read off what the solver sees (generation-time audits, rule readers).

Nothing here reads the gold block: `inputs(sp)` recovers the decision inputs from the rendered
clinic record (drug, dose position, days since the last increase, control status of today's
printed reading, refill coverage over the last 90 days, the drug's registered symptom and its
onset), `surface(sp, H)` the non-decision scenario variables and the length / structure of the
item, and `rule_reader(sp, knows=...)` the decision of a reader who knows the G3 rules except the
pieces named in `knows`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json

from haenv.i18n.text import is_text
from haenv.p3.gold import CLASSES  # noqa: F401  (re-exported for the audit)
from haenv.p3.textread import canonical, find, number, prescription_dose, slots


def _canon_opt(v):
    return canonical(v) if isinstance(v, str) else v


def _started(change) -> bool:
    return is_text(change, "p3.change.start")


def _increased(change) -> bool:
    return slots(change, "p3.change.increase") is not None


def _pill_left(text) -> int:
    inner = slots(text, "p3.review.documented_only")
    from haenv.p3.policy import pill_count_slots
    count = pill_count_slots(inner["text"] if inner else text)
    if count is not None:
        return int(count["left"])
    # Historical source fixtures predate the dated pill-count template.
    count = slots(text, "p3.pill_count.legacy")
    if count is None:
        raise ValueError("recorded pill count is missing")
    return int(count["left"])


def _new_strength(dispensed) -> bool:
    got = slots(dispensed, "p3.refill.dispensed")
    return got is not None and slots(got["extra"], "p3.refill.new_strength") is not None

from haenv.p3.policy import drug_keys
DRUG_KEYS = drug_keys()
KNOWLEDGE = ("tau", "timing", "attribution", "adherence", "top", "response", "comorbidity")


def _blk(sp) -> dict:
    from haenv.p3.emr import clinical_view
    pc = sp.get("prediction_context") if isinstance(sp, dict) else getattr(sp, "prediction_context", {})
    return clinical_view((pc or {}).get("meds_p3") or {})


def drug_key(b: dict) -> str:
    from haenv.p3.policy import identify_drug as parse_drug
    key = parse_drug(b["drug"])
    if key is None:
        raise ValueError("unknown managed drug")
    return key


def _num(s: str) -> float:
    return number(s)


def inputs(sp) -> dict:
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    b = _blk(sp)
    k = drug_key(b)
    d = drug(k)
    ax = axis(d["axis"])
    L = [float(x) for x in d["ladder"]]
    T = int(b["today"])
    rows = b["visits"]
    past = [r for r in rows if int(r["day"]) < T]
    x = prescription_dose(past[-1]["prescription"])
    pos = "min" if x == L[0] else "top" if x == L[-1] else "mid"
    s = max(int(r["day"]) for r in past if _started(r["change"]) or _increased(r["change"]))
    starts = [int(r["day"]) for r in past if _started(r["change"])]
    incs = [int(r["day"]) for r in past if _increased(r["change"])]
    y = _num(rows[-1]["reading"])
    target = (slots(b["clinic_target"], "p3.target.clinic_drug") or slots(b["clinic_target"], "p3.target.clinic"))["target"]
    bounds = slots(target, "p3.target.upper") or slots(target, "p3.target.range")
    upper = float(bounds["up"])     # this patient's target (A12)
    status = "off" if y > upper else "on"
    if ax.get("over_below") is not None and y < float(ax["over_below"]):
        status = "over"
    pick = [int(p["day"]) for p in b["refill_record"]]
    sup = int(_num(b["refill_record"][-1]["dispensed"]))
    left = _pill_left(b["pill_count_today"])
    from haenv.world.labworld.meds import carry_in
    from haenv.p3.gold import printed_coverage
    reset = [int(p["day"]) for p in b["refill_record"] if _new_strength(p["dispensed"])]
    a_last = min(1.0, max(0.0, (sup + carry_in(pick, sup, reset)[-1] - left) / max(1, T - pick[-1])))
    cov90 = printed_coverage(pick, sup, left, T, min(starts) if starts else min(int(r["day"]) for r in past), reset)[0]
    late = any(q - p > 1.15 * sup for p, q in zip(pick, pick[1:]))
    from haenv.p3.gold import documented_contexts
    doc = {canonical(x) for x in documented_contexts(d)}
    symptoms = {canonical(x) for x in d["ae_symptoms"]}
    histories = {(canonical(e["symptom"]), int(e["onset_day"])): e["history"] for e in b.get("encounter_record") or ()}
    sym = [e for e in b["symptom_diary"] if canonical(e["symptom"]) in symptoms
           and (k == "gliclazide" or not is_text(e.get("status"), "p3.status.resolved"))
           and _canon_opt(histories.get((canonical(e["symptom"]), int(e["onset_day"])), e.get("context"))) not in doc]
    return {"drug": k, "position": pos, "dose": x, "ladder": L, "since_increase": T - s, "T": T,
            "start": min(starts) if starts else None, "increases": incs, "status": status, "reading": y,
            "coverage90": cov90, "late_refill": int(late), "pill_adherence": a_last,
            "drug_symptom": int(bool(sym)), "symptom_onset": (max(int(e["onset_day"]) for e in sym) if sym else None),
            "drug_symptom_any": int(any(canonical(e["symptom"]) in symptoms for e in b["symptom_diary"])),
            "tau": float(d["tau_rule_days"]), "attribution": d["attribution"], "upper": upper,
            "known": known_comorbidities(sp),
            "absorption_interaction": absorption_interaction(b.get("administration_record"))}


def known_comorbidities(sp) -> tuple[str, ...]:
    """The registered comorbidities on the patient's list of known conditions (A24)."""
    from haenv.world.labworld.meds import meds
    up = (sp.get("user_profile") if isinstance(sp, dict) else getattr(sp, "user_profile", {})) or {}
    reg = meds().get("comorbidities") or {}
    return tuple(k for k in up.get("known_conditions") or () if k in reg)


def surface(sp, H: int) -> dict:
    b = _blk(sp)
    pc = sp.get("prediction_context") if isinstance(sp, dict) else getattr(sp, "prediction_context", {})
    visible = (pc or {}).get("meds_p3") or {}
    up = (sp.get("user_profile") if isinstance(sp, dict) else getattr(sp, "user_profile", {})) or {}
    age = str(up.get("age_range") or "40-44")
    ld = (sp.get("longitudinal_data") if isinstance(sp, dict) else {}) or {}
    rows = b["visits"]
    return {"drug": DRUG_KEYS.index(drug_key(b)), "H": int(H),
            "age": int(age.split("-")[0]) if age[:2].isdigit() else 40, "sex": int(up.get("sex") == "F"),
            "n_known": len(up.get("known_conditions") or ()), "T": int(b["today"]),
            "comorbid": int(bool(b.get("known_comorbidity"))),
            "D": int(b["today"]) - min(int(r["day"]) for r in rows), "n_visits": len(rows),
            "n_streams": len(ld),
            "block_chars": len(json.dumps(visible, ensure_ascii=False)),
            "payload_chars": len(json.dumps(sp, ensure_ascii=False, default=str)) if isinstance(sp, dict) else 0,
            "n_refills": len(b["refill_record"]), "n_diary": len(b["symptom_diary"]),
            "n_increase_rows": sum(1 for r in rows if _increased(r["change"]))}


def rule_reader(sp, unknown: tuple[str, ...] = (), generic_tau: float | None = None) -> str:
    """G3 applied to the visible record by a reader who knows every rule except `unknown`:
    `tau` the reassessment interval; `timing` the onset windows of a timing-attributed adverse
    effect; `attribution` which drugs are attributed by exposure (the reader applies the timing
    windows to every drug); `adherence` the 0.80 line; `top` switch-or-add at the top rung.
    `response` the drug's guideline step below the top (A15: titrate or add; without it the
    reader up-titrates every drug below the top). `comorbidity` that a known comorbidity changes
    the step (A24: without it the reader keeps a CKD patient's ACEI on target below the top). `generic_tau` replaces every drug's interval by
    one number and changes nothing else (A12 fixes the r12 reader, which also dropped `attribution`).
    A resolved entry is not an ongoing adverse effect for any reader."""
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    rl = rule()
    f = inputs(sp)
    off, over = f["status"] == "off", f["status"] == "over"
    adh_low = "adherence" not in unknown and f["coverage90"] < float(rl["adherence_line"])
    tau = generic_tau if generic_tau is not None else f["tau"]
    early = "tau" not in unknown and f["since_increase"] <= float(rl["early_max_frac"]) * tau
    exposure = f["attribution"] == "exposure" and "attribution" not in unknown
    ae = None
    if f["drug_symptom"]:
        on = f["symptom_onset"]
        if f["start"] is not None and on < f["start"]:
            ae = None
        elif exposure or "timing" in unknown:
            ae = "dose_related"
        else:
            anchor = max([d for d in [f["start"]] + f["increases"] if d is not None and d <= on], default=None)
            lo, hi = rl["ae_dose_window"]
            ae = "dose_related" if anchor is not None and lo <= on - anchor <= hi else "unattributed"
    from haenv.p3.history import acei_contraindicated
    if acei_contraindicated(_blk(sp).get("adverse_event_record"), f["drug"]):
        return "switch"
    if ae == "unattributed" and not over:
        return "check_adherence_or_adverse_effect"
    if ae == "dose_related" or over:
        return "downtitrate"
    if off and (adh_low or f["absorption_interaction"]):
        return "check_adherence_or_adverse_effect"
    if off and early:
        return "maintain"
    from haenv.p3.history import limiting_record
    if off and limiting_record(_blk(sp).get("adverse_event_record"), f["drug"], f["dose"]):
        return "switch"
    if off and f["position"] == "top" and "top" not in unknown:
        return (drug(f["drug"], f["known"]).get("when_off_at_top") or {}).get("decision", "switch")
    known = f["known"] if "comorbidity" not in unknown else ()
    if off and "response" not in unknown and drug(f["drug"], known)["when_off"] == "add":
        return "switch"
    if off:
        return "uptitrate"
    dk = drug(f["drug"], known)
    if dk.get("on_target") == "titrate" and f["position"] != "top" and f["reading"] >= float(dk.get("titrate_on_target_min") or 0):
        return "uptitrate"
    return "maintain"


def printed_rules(sp) -> dict:
    """The clinic rules printed on the item (`render.clinic_protocol`), parsed from its text alone
    (p3-fixes): the reassessment interval, the adverse-effect attribution rule, the adherence line,
    and the CKD policy with its SBP floor. A rule the item does not print is absent from the dict."""
    b = _blk(sp)
    from haenv.world.protocols.library import clinic_lines
    from haenv.world.labworld.meds import drug
    lines = list(b.get("medication_instructions") or b.get("clinic_protocol") or clinic_lines(b))
    out: dict = {}

    def first(key):
        return next((m for m in (find(x, key) for x in lines) if m is not None), None)
    m = first("p3.rule.interval")
    if m:
        out["tau"], out["early_days"] = int(m["tau"]), int(m["half"])
    timing, exposure = first("p3.rule.ae.timing"), first("p3.rule.ae.exposure")
    if timing or exposure:
        if timing:
            out["attribution"], out["ae_window"] = "timing", (int(timing["lo"]), int(timing["hi"]))
            out["ae_unattributed_min"] = int(timing["umin"])
        else:
            out["attribution"] = "exposure"
        out["ae_lowers"] = not is_text((timing or exposure)["lower"], "p3.rule.ae.axis_dosed")
    m = first("p3.rule.adherence")
    if m:
        out["adherence_line"] = int(m["pct"]) / 100.0
    if first("p3.rule.top_check"):
        out["top_decision"] = "check_adherence_or_adverse_effect"
    ckd = drug("enalapril", ("CKD",))
    policy = ckd.get("clinic_policy")
    hit = next((x for x in lines if policy is not None and is_text(x, policy.key)), None)
    if hit is not None:
        out["titrate_on_target"] = True
        floor = ckd.get("titrate_on_target_min")
        out["floor"] = float(floor) if floor is not None else None
    return out


def printed_rule_reader(sp) -> str:
    """G3 applied by a reader who knows only the rules printed on the item (`printed_rules`) and the
    answer options of the question; the visible record is read as `inputs` reads it (the drug's
    registered symptom words stand in for recognising its adverse effect in the diary). A rule that is
    not printed is read the plain way: any symptom on the drug is lowered for, no adherence check, no
    "not yet arrived" hold, a missed target below the top dose is up-titrated. A derivability
    audit found the gold not derivable from the item for
    gliclazide below the top, adverse-effect attribution and the 0.80 line; this reader is the check."""
    f = inputs(sp)
    P = printed_rules(sp)
    off, over = f["status"] == "off", f["status"] == "over"
    adh_low = P.get("adherence_line") is not None and f["coverage90"] < P["adherence_line"]
    early = "early_days" in P and f["since_increase"] <= P["early_days"]
    ae = None
    if f["drug_symptom"]:
        on = f["symptom_onset"]
        anchor = max([d for d in [f["start"]] + f["increases"] if d is not None and d <= on], default=None)
        if f["start"] is not None and on < f["start"]:
            ae = None
        elif P.get("attribution") == "timing":
            lo, hi = P["ae_window"]
            umin = P.get("ae_unattributed_min")
            if anchor is not None and lo <= on - anchor <= hi:
                ae = "dose_related"
            elif anchor is not None and umin is not None and on - anchor >= umin:
                ae = "unattributed"
        else:
            ae = "dose_related"
    from haenv.p3.history import acei_contraindicated
    if acei_contraindicated(_blk(sp).get("adverse_event_record"), f["drug"]):
        return "switch"
    if ae == "unattributed" and not over:
        return "check_adherence_or_adverse_effect"
    if (ae == "dose_related" and P.get("ae_lowers", True)) or over:
        return "downtitrate"
    if off and (adh_low or f["absorption_interaction"]):
        return "check_adherence_or_adverse_effect"
    if off and early:
        return "maintain"
    if off:
        from haenv.p3.history import limiting_record
        if limiting_record(_blk(sp).get("adverse_event_record"), f["drug"], f["dose"]):
            return "switch"
        return P.get("top_decision", "switch") if f["position"] == "top" else "uptitrate"
    if P.get("titrate_on_target") and f["position"] != "top" and (P.get("floor") is None or f["reading"] >= P["floor"]):
        return "uptitrate"
    return "maintain"


def anomaly_stub(sp) -> str:
    """L3-3: any adherence or symptom signal => check; else by target and dose position."""
    f = inputs(sp)
    if f["late_refill"] or f["pill_adherence"] < 0.85 or f["drug_symptom_any"]:
        return "check_adherence_or_adverse_effect"
    if f["status"] == "off":
        return "switch" if f["position"] == "top" else "uptitrate"
    return "maintain"



from haenv.p3.policy import absorption_interaction
