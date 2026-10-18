"""Rebuilt pack 2 · the rule reader: the acceptable dispositions of an item from its visible block and
the printed rules only (`registry/pack2_triage_scenarios.yaml`). The gold, the emission gate, the
generator's acceptance test and the audit's S-6 reader all call `assess`, so the gold is derivable
from what the solver sees by construction. P2_WORLD segment.

Reads the table through `haenv.yamlcache` / `haenv.regpath` (light: yaml + stdlib).
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json

TABLE = "pack2_triage_scenarios.yaml"
DISPOSITIONS = ("ed_now", "within_24h", "routine_followup", "watchful_waiting")
#: intake-checklist keys whose "yes" fires R0
R0_KEYS = ("chest_pain_or_pressure", "fainted", "new_confusion", "breathless_at_rest",
           "one_sided_weakness_or_slurred_speech")
#: R6 reads this item only; dizziness is its own item and fires no rule
HTN_SYMPTOM = "severe_headache_or_visual_disturbance"
DIZZY = "dizziness"
HYPO_HELP = "needed_help_for_low_glucose_last_3_days"
MILD = "mild_new_symptom"
CHECKLIST = R0_KEYS + (HTN_SYMPTOM, DIZZY, HYPO_HELP, MILD)
#: the item before the split (headache, blurred vision or dizziness in one); a saved block carrying it reads as R6 did then
LEGACY_HTN_SYMPTOM = "severe_headache_blurred_vision_or_dizziness"


def checklist_of(block: dict) -> dict:
    """Normalize the compact, complete symptom screen; absent evidence is never a negative."""
    ck = block.get("intake_checklist") or {}
    if LEGACY_HTN_SYMPTOM in ck and HTN_SYMPTOM not in ck:
        ck = {**{k: v for k, v in ck.items() if k != LEGACY_HTN_SYMPTOM}, HTN_SYMPTOM: ck[LEGACY_HTN_SYMPTOM], DIZZY: False}
    if set(R0_KEYS) <= set(ck):
        return ck  # old saved prototype blocks remain readable
    if ck.get("urgent_screen_complete") is not True or not isinstance(ck.get("urgent_symptoms"), list):
        raise ValueError("incomplete urgent symptom screen")
    if not set(ck["urgent_symptoms"]) <= set(R0_KEYS):
        raise ValueError("unknown urgent symptom")
    if not {HTN_SYMPTOM, DIZZY, HYPO_HELP, MILD} <= set(ck):
        raise ValueError("incomplete other symptom screen")
    return {**ck, **{k: k in ck["urgent_symptoms"] for k in R0_KEYS}}


def table() -> dict:
    """The scenario table, through the shared registry path and YAML cache."""
    from haenv.resources.regpath import registry_dir
    from haenv.resources.yamlcache import load_yaml
    return load_yaml(registry_dir() / TABLE) or {}


def table_sha16() -> str:
    return hashlib.sha256(json.dumps(table(), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def rule_of(rid: str) -> dict:
    return next(r for r in table()["rules"] if r["id"] == rid)


def policy_text() -> str:
    """The printed rules, in order (the question template prints them once for every item)."""
    return "\n".join([*(r["printed"] for r in table()["rules"]), table()["acceptance"]["single_level2"]["printed"]])


def _points(bands, v) -> int:
    for lo, hi, pts in bands:
        if lo <= v <= hi:
            return int(pts)
    raise ValueError(f"NEWS2: {v!r} outside every band")


def news2(check: dict, checklist: dict, scale: int = 1) -> tuple[int, int]:
    """(aggregate, largest single-parameter score) on today's home check; consciousness from the
    checklist's new confusion, including supplemental oxygen when shown."""
    n = table()["news2"]
    parts = [_points(n["rr"], check["respiratory_rate_per_min"]), spo2_points(check, scale),
             _points(n["sbp"], check["systolic_bp_mmHg"]), _points(n["hr"], check["heart_rate_bpm"]),
             _points(n["temp"], check["temperature_c"]),
             int(n["new_confusion_points"]) if checklist.get("new_confusion") else 0]
    return sum(parts) + (2 if check.get("on_oxygen") else 0), max(parts)


def news2_parts(check: dict, checklist: dict, scale: int = 1) -> dict:
    n = table()["news2"]
    return {"rr": _points(n["rr"], check["respiratory_rate_per_min"]), "spo2": spo2_points(check, scale),
            "sbp": _points(n["sbp"], check["systolic_bp_mmHg"]), "hr": _points(n["hr"], check["heart_rate_bpm"]),
            "temp": _points(n["temp"], check["temperature_c"]),
            "acvpu": int(n["new_confusion_points"]) if checklist.get("new_confusion") else 0}


def spo2_points(check, scale=1):
    bands = table()["news2"]["spo2"]
    if scale == 2:
        from haenv.world.validation.difficulty_r2 import config
        bands = config("p2")["scale2"]["on_oxygen" if check.get("on_oxygen") else "room_air"]
    return _points(bands, check["spo2_pct"])


def n2_low_hr() -> int:
    """Upper end of the heart rate's low 3-point NEWS2 band."""
    return int(next(hi for lo, hi, pts in table()["news2"]["hr"] if pts == 3 and lo < 0))


def readings(block: dict) -> dict:
    """The quantities the rules read, off the visible block (`last_day_only` stubs pass a cut block)."""
    th = table()["thresholds"]
    T = int(block["today_day"])
    ck = checklist_of(block)
    check = block["home_check_today"]
    from haenv.p2.history import scale2
    scale = 2 if scale2(block.get("clinical_history_record")) else 1
    s, top = news2(check, ck, scale)
    parts = news2_parts(check, ck, scale)
    #: single-3 parameters read as R3a: respiratory rate in either 3-point band, heart rate >= 131, SBP <= 90;
    #: suspected hypothermia is <35 C (NHS), distinct from the NEWS2 <=35 C band;
    #: a heart rate in the low 3-point band alone is R3c; SpO2 apart (R3b)
    low_hr = parts["hr"] == 3 and check["heart_rate_bpm"] <= n2_low_hr()
    vital3 = (["rr"] if parts["rr"] == 3 else []) + (["hr"] if parts["hr"] == 3 and not low_hr else []) \
        + (["sbp"] if parts["sbp"] == 3 and check["systolic_bp_mmHg"] <= 90 else []) \
        + (["temp"] if check["temperature_c"] < th["hypothermia_temp_lt"] else [])
    today_bp = [r for r in block.get("home_bp_log") or () if int(r["day"]) == T]
    sev = sum(1 for r in today_bp if r["systolic_mmHg"] >= th["htn_sys_ge"] or r["diastolic_mmHg"] >= th["htn_dia_ge"])
    raised = sum(1 for r in today_bp if r["systolic_mmHg"] >= 140 or r["diastolic_mmHg"] >= 90)
    w = {int(r["day"]): float(r["kg"]) for r in block.get("morning_weight") or ()}
    gain = (w[T] - w[T - th["hf_gain_days"]]) if (T in w and T - th["hf_gain_days"] in w) else None
    cgm = [r for r in block.get("cgm_nightly") or () if T - 7 < int(r["night_of_day"]) + 1 <= T]
    l2 = sum(1 for r in cgm if r["min_mmol_L"] < th["hypo_level2_mmol"])
    l1 = sum(1 for r in cgm if r["min_mmol_L"] < th["hypo_level1_mmol"])
    alerts = sum(int(r.get("irregular_rhythm_alerts") or 0) for r in block.get("wearable_daily") or ()
                 if T - th["af_window_days"] < int(r["day"]) <= T)
    known = set((block.get("background") or {}).get("known_conditions") or ())
    if block.get("clinical_history_record"):
        from haenv.p2.history import diagnosed
        known |= diagnosed(block["clinical_history_record"])
    return {"news2": s, "news2_top": top, "single3_vital": vital3, "single3_spo2": parts["spo2"] == 3, "single3_low_hr": low_hr, "r0": [k for k in R0_KEYS if ck.get(k)], "htn_symptom": bool(ck.get(HTN_SYMPTOM)),
            "hypo_help": bool(ck.get(HYPO_HELP)), "mild": bool(ck.get(MILD)), "bp_severe_today": sev,
            "bp_raised_today": raised, "weight_gain_3d": None if gain is None else round(gain, 2),
            "heart_failure": "heart_failure" in known, **({"known_af": "atrial_fibrillation" in known} if block.get("clinical_history_record") or "atrial_fibrillation" in known else {}), "cgm_nights": len(cgm), "level2_nights": l2,
            "level1_nights": l1, "alerts_3d": alerts}


def assess(block: dict) -> dict:
    """{"rule", "set", "readings"}: the first printed rule that fires, and its acceptable set."""
    th = table()["thresholds"]
    r = readings(block)
    if r["r0"]:
        rid = "R0_red_flag"
    elif r["news2"] >= 7:
        rid = "R1_news2_high"
    elif r["news2"] >= 5:
        rid = "R2_news2_medium"
    elif r["single3_vital"]:
        rid = "R3a_single3_vital"
    elif r["single3_spo2"]:
        rid = "R3b_single3_spo2"
    elif r["single3_low_hr"]:
        rid = "R3c_single3_low_hr"
    elif r["news2_top"] >= 3:
        rid = "R3_news2_single3"
    elif r["hypo_help"]:
        rid = "R4_hypo_level3"
    elif r["level2_nights"] >= th["hypo_level2_nights_ge"]:
        rid = "R5_hypo_level2_repeat"
    elif r["bp_severe_today"] >= th["htn_readings_ge"] and r["htn_symptom"]:
        rid = "R6_htn_emergency"
    elif r["bp_severe_today"] >= th["htn_readings_ge"] and r["heart_failure"]:
        rid = "R7a_htn_severe_hf"
    elif r["bp_severe_today"] >= th["htn_readings_ge"]:
        rid = "R7_htn_severe"
    elif r["heart_failure"] and r["weight_gain_3d"] is not None and r["weight_gain_3d"] > th["hf_gain_kg_gt"]:
        rid = "R8_hf_gain"
    elif r["alerts_3d"] >= th["af_alerts_ge"] and not r.get("known_af"):
        rid = "R9_af_alerts"
    elif r["news2"] >= 1:
        rid = "R10_news2_low"
    elif r["cgm_nights"] and r["level1_nights"] >= 1 and r["level2_nights"] == 0:
        rid = "R11_hypo_level1"
    elif r["bp_raised_today"] >= th["htn_readings_ge"]:
        rid = "R12_htn_raised"
    elif r["mild"]:
        rid = "R13_news2_zero_symptom"
    else:
        rid = "R14_nothing"
    accepted = list(rule_of(rid)["set"])
    single = table()["acceptance"]["single_level2"]
    if r["level2_nights"] == 1 and floor_of(accepted) in single["when_floor"]:
        accepted = list(single["set"])
    return {"rule": rid, "set": accepted, "readings": r}


def floor_of(acc: list[str]) -> str:
    """The least urgent acceptable disposition (the composition class, design §5)."""
    return max(acc, key=DISPOSITIONS.index)


def last_day_only(block: dict) -> dict:
    """The block cut to its last day (S-5 stub): series before today removed."""
    T = int(block["today_day"])
    out = dict(block)
    for k, day in (("wearable_daily", "day"), ("home_bp_log", "day"), ("morning_weight", "day")):
        out[k] = [r for r in block.get(k) or () if int(r[day]) == T]
    out["cgm_nightly"] = [r for r in block.get("cgm_nightly") or () if int(r["night_of_day"]) == T - 1]
    return out
