"""Pack 2 T-time vital-sign snapshot: background in the NEWS2-0 region, the focal indicator at
the grade of the planned disposition, 72 h capillary-glucose and home-BP logs, and the visible
block `prediction_context.triage` built from them. PACK2_WORLD segment.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from ._util import pick as _pick
from ._util import u01
from .events import chief_complaint, distractor_results
from .grading import thresholds

#: Visible field names per focal indicator (what `decisive_findings` should name).
FOCAL_FIELDS = {
    "K": ("point_of_care_now.serum_potassium_mmol_L",),
    "glucose": ("point_of_care_now.capillary_glucose_mmol_L", "capillary_glucose_log_72h"),
    "SBP": ("vitals_now.systolic_bp_mmHg",),
    "NEWS2": ("vitals_now.respiratory_rate_per_min", "vitals_now.heart_rate_bpm",
              "vitals_now.temperature_c", "vitals_now.spo2_pct"),
}



def _times():
    return ("07:00", "11:30", "17:30", "21:30")


def snapshot(case_id: str, T: int, plan: dict, tb: dict | None = None) -> dict:
    """The numbers the item shows. Background from the NEWS2-0 region; the focal indicator at the
    grade of `plan["disposition"]`. Deterministic in (case_id, plan)."""
    tb = tb or thresholds()
    bg, fv = tb["background"], tb["focal_values"]
    cls, focal = plan["disposition"], plan["focal"]
    u = lambda *k: u01("snap", case_id, *k)                       # noqa: E731
    v = {"rr": _pick(*bg["rr"], u("rr"), 0), "spo2": _pick(*bg["spo2"], u("spo2"), 0),
         "sbp": _pick(*bg["sbp"], u("sbp"), 0), "hr": _pick(*bg["hr"], u("hr"), 0),
         "temp": _pick(*bg["temp"], u("temp"), 1), "alert": True,
         "glucose": _pick(*bg["glucose"], u("glu"), 1), "K": _pick(*bg["K"], u("K"), 1)}
    # 72 h capillary glucose log: days T-3 .. T-1 four readings, today two readings before the visit.
    slots = [(T - d, t) for d in (3, 2, 1) for t in _times()] + [(T, "07:00"), (T, "11:30")]
    glog = [{"day": d, "time": t, "mmol_L": _pick(*bg["glucose"], u("glog", i), 1)}
            for i, (d, t) in enumerate(slots)]
    # Home BP log, days T-3 .. T-1 morning and evening, background region.
    bslots = [(T - d, t) for d in (3, 2, 1) for t in ("08:00", "20:00")]
    blog = []
    for i, (d, t) in enumerate(bslots):
        s = _pick(*bg["sbp"], u("blog", i), 0)
        blog.append({"day": d, "time": t, "systolic_mmHg": s, "diastolic_mmHg": _dbp(s, u("blogd", i))})
    w = u("focal")
    if focal == "K":
        v["K"] = _pick(*fv["K"][cls], w, 1)
    elif focal == "SBP":
        v["sbp"] = _pick(*fv["SBP"][cls], w, 0)
    elif focal == "glucose":
        g = fv["glucose"]
        if cls == "ed_now":
            v["glucose"] = _pick(*g["ed_now"], w, 1)
        elif cls in ("within_24h", "routine_followup"):
            n = (_pick(*g["within_24h_lows"], u("nlows"), 0) if cls == "within_24h" else 1)
            nad = _pick(*g[f"{cls}_nadir"], w, 1)
            hi = g[f"{cls}_nadir"][1]
            idx = sorted(range(len(glog) - 1), key=lambda i: u01("lowpos", case_id, i))[:n]
            for j, i in enumerate(sorted(idx)):
                glog[i]["mmol_L"] = nad if j == 0 else _pick(nad, hi, u("lowv", j), 1)
    elif focal == "NEWS2":
        for q, rng_ in fv["NEWS2"][cls].items():
            v[q] = _pick(rng_[0], rng_[1], u("n2", q), 1 if q == "temp" else 0)
    # only the emergency cell may reach a diastolic >= 120
    v["dbp"] = _dbp(v["sbp"], u("dbp")) if cls == "ed_now" else min(_dbp(v["sbp"], u("dbp")), 119)
    v["glucose_log"] = [g_["mmol_L"] for g_ in glog]
    return {"values": v, "glucose_log": glog, "bp_log": blog}


def _dbp(sbp: int, u: float) -> int:
    return int(round(0.58 * sbp + 8 + (u * 7 - 3.5)))


def visible_block(case_id: str, T: int, plan: dict, snap: dict | None = None) -> dict:
    """`prediction_context.triage`: what the solver sees about today's visit."""
    snap = snap or snapshot(case_id, T, plan)
    v = snap["values"]
    return {
        "visit_day": int(T),
        "chief_complaint": chief_complaint(case_id, plan),
        "vitals_now": {"respiratory_rate_per_min": v["rr"], "spo2_pct": v["spo2"],
                       "systolic_bp_mmHg": v["sbp"], "diastolic_bp_mmHg": v["dbp"],
                       "heart_rate_bpm": v["hr"], "temperature_c": v["temp"],
                       "consciousness_acvpu": "A" if v.get("alert", True) else "V",
                       "supplemental_oxygen": False},
        "point_of_care_now": {"capillary_glucose_mmol_L": v["glucose"],
                              "serum_potassium_mmol_L": v["K"]},
        "capillary_glucose_log_72h": snap["glucose_log"],
        "home_bp_log_72h": snap["bp_log"],
        "other_results_today": [{"test": r["test"], "value": r["value"]}
                                for r in distractor_results(case_id, int(plan.get("distractors", 0)))],
        "data_note": "SYNTHETIC data, evaluation only",
    }


def values_from_visible(b: dict) -> dict:
    """Read the G2 inputs back from the visible block (what the item shows)."""
    vn, poc = b["vitals_now"], b["point_of_care_now"]
    return {"rr": vn["respiratory_rate_per_min"], "spo2": vn["spo2_pct"], "sbp": vn["systolic_bp_mmHg"],
            "dbp": vn["diastolic_bp_mmHg"], "hr": vn["heart_rate_bpm"], "temp": vn["temperature_c"],
            "alert": vn.get("consciousness_acvpu", "A") == "A",
            "glucose": poc["capillary_glucose_mmol_L"], "K": poc["serum_potassium_mmol_L"],
            "glucose_log": [g["mmol_L"] for g in b.get("capillary_glucose_log_72h") or []]}

