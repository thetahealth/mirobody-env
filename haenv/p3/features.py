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
import re

from .gold import CLASSES  # noqa: F401  (re-exported for the audit)

DRUG_KEYS = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
KNOWLEDGE = ("tau", "timing", "attribution", "adherence", "top", "response", "comorbidity")


def _blk(sp) -> dict:
    pc = sp.get("prediction_context") if isinstance(sp, dict) else getattr(sp, "prediction_context", {})
    return (pc or {}).get("meds_p3") or {}


def drug_key(b: dict) -> str:
    from ..labworld.meds import drug
    return next(k for k in DRUG_KEYS if drug(k)["display"] == b["drug"])


def _num(s: str) -> float:
    return float(re.match(r"\s*([0-9.]+)", s).group(1))


def inputs(sp) -> dict:
    from ..labworld.meds import axis, drug, rule
    b = _blk(sp)
    k = drug_key(b)
    d = drug(k)
    ax = axis(d["axis"])
    L = [float(x) for x in d["ladder"]]
    T = int(b["today"])
    rows = b["visits"]
    past = [r for r in rows if int(r["day"]) < T]
    x = float(re.search(r"([0-9.]+) (?:mg|μg)", past[-1]["prescription"]).group(1))
    pos = "min" if x == L[0] else "top" if x == L[-1] else "mid"
    s = max(int(r["day"]) for r in past if r["change"] == "起始" or r["change"].startswith("加量"))
    starts = [int(r["day"]) for r in past if r["change"] == "起始"]
    incs = [int(r["day"]) for r in past if r["change"].startswith("加量")]
    y = _num(rows[-1]["reading"])
    upper = float(re.search(r"(?:<|–)\s*([0-9.]+)", b["clinic_target"]).group(1))     # this patient's target (A12)
    status = "off" if y > upper else "on"
    if ax.get("over_below") is not None and y < float(ax["over_below"]):
        status = "over"
    pick = [int(p["day"]) for p in b["refill_record"]]
    sup = int(_num(b["refill_record"][-1]["dispensed"]))
    left = int(re.search(r"剩余 (\d+)", b["pill_count_today"]).group(1))
    a_last = min(1.0, max(0.0, (sup - left) / max(1, T - pick[-1])))
    cov = []
    for dd in range(T - int(rule()["adherence_window_days"]) + 1, T + 1):
        a = a_last if dd >= pick[-1] else next(
            (min(1.0, sup / (q - p)) for p, q in zip(pick, pick[1:]) if p <= dd < q), 1.0)
        cov.append(a)
    late = any(q - p > 1.15 * sup for p, q in zip(pick, pick[1:]))
    sym = [e for e in b["symptom_diary"] if e["symptom"] in d["ae_symptoms"] and e["status"] != "已缓解"]
    return {"drug": k, "position": pos, "dose": x, "ladder": L, "since_increase": T - s, "T": T,
            "start": min(starts) if starts else None, "increases": incs, "status": status, "reading": y,
            "coverage90": sum(cov) / len(cov), "late_refill": int(late), "pill_adherence": a_last,
            "drug_symptom": int(bool(sym)), "symptom_onset": (int(sym[0]["onset_day"]) if sym else None),
            "drug_symptom_any": int(any(e["symptom"] in d["ae_symptoms"] for e in b["symptom_diary"])),
            "tau": float(d["tau_rule_days"]), "attribution": d["attribution"], "upper": upper,
            "known": known_comorbidities(sp)}


def known_comorbidities(sp) -> tuple[str, ...]:
    """The registered comorbidities on the patient's list of known conditions (A24)."""
    from ..labworld.meds import meds
    up = (sp.get("user_profile") if isinstance(sp, dict) else getattr(sp, "user_profile", {})) or {}
    reg = meds().get("comorbidities") or {}
    return tuple(k for k in up.get("known_conditions") or () if k in reg)


def surface(sp, H: int) -> dict:
    b = _blk(sp)
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
            "block_chars": len(json.dumps(b, ensure_ascii=False)),
            "payload_chars": len(json.dumps(sp, ensure_ascii=False, default=str)) if isinstance(sp, dict) else 0,
            "n_refills": len(b["refill_record"]), "n_diary": len(b["symptom_diary"]),
            "n_increase_rows": sum(1 for r in rows if r["change"].startswith("加量"))}


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
    from ..labworld.meds import drug, rule
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
    if (off and adh_low) or ae == "unattributed":
        return "check_adherence_or_adverse_effect"
    if ae == "dose_related" or over:
        return "downtitrate"
    if off and early:
        return "maintain"
    if off and f["position"] == "top" and "top" not in unknown:
        return "switch"
    known = f["known"] if "comorbidity" not in unknown else ()
    if off and "response" not in unknown and drug(f["drug"], known)["when_off"] == "add":
        return "switch"
    if off:
        return "uptitrate"
    if drug(f["drug"], known).get("on_target") == "titrate" and f["position"] != "top":
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
