"""Rebuilt pack 2's emission declaration for the shared gate (`haenv/world/validation/shared_audit.py`, SA-1 + SA-8).

Refused: an item that could not be realized; a gold whose rule / set differs from the rule reader
re-run on the shown block, or whose rule is not the planned tier's; a block carrying a disposition
word, a rule id, a tier or stratum name; a message that names an R0 symptom the checklist does not
carry; a base-world patient sentence with a red-flag word (it would contradict a non-urgent gold); a
record condition the world's labs were not drawn for; obesity on the record with weights that cannot be obese.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import re

GATE = "p2"
BLOCK = "triage_p2"
LATENT = "p2"

EN_TOKENS = ("ed_now", "within_24h", "routine_followup", "watchful_waiting", "acceptable", "borderline",
             "inf_n7", "inf_n56", "inf_v3", "inf_s3", "inf_t35", "af_rvr", "htn_lowbp", "brady", "hf_htn_severe", "inf_n14", "inf_n0", "red_normal", "htn_emerg", "htn_severe",
             "htn_raised", "htn_sbp220", "hf_rest", "hf_gain", "hf_stable", "af_unstable", "af_stable", "hypo_l3", "hypo_l2x",
             "hypo_l1", "ben_exercise", "ben_spo2", "ben_sleep", "masked", "atypical")
ZH_TOKENS = ("急诊", "急救", "24小时", "24 小时", "常规复诊", "常规随访", "居家观察", "继续观察", "无需就医", "不用就医",
             "危急值", "红旗", "金标", "可接受")
_RULE_ID = re.compile(r"\bR\d{1,2}\b")
#: copied from `haenv/pack2/world.py:RED_FLAG_WORDS` (= `haenv/scoring/verify.py`), import-light
RED_FLAG_WORDS = ("胸痛", "胸闷压迫", "呼吸困难", "喘不上气", "晕厥", "意识丧失", "抽搐",
                  "偏瘫", "半身无力", "言语不清", "黑便", "呕血", "咯血", "血尿", "剧烈头痛",
                  "视力骤降", "视物模糊加重", "高热不退", "持续呕吐", "腹痛剧烈", "自杀")


def plan_of(raw) -> dict | None:
    from haenv.contracts.external_gold import SLOT
    lp = raw.get("latent_premise") if isinstance(raw, dict) else getattr(raw, "latent_premise", None)
    p = (((lp or {}).get("meta") or {}).get(SLOT) or {}).get(LATENT)
    return p if isinstance(p, dict) and p.get("tier") else None


def _get(raw, k):
    return raw.get(k) if isinstance(raw, dict) else getattr(raw, k, None)


def _unrealizable(g: dict) -> str | None:
    return str(g["unrealizable"]) if g.get("unrealizable") else (None if g.get("set") else "no realized item")


def _recompute(g: dict, raw) -> list[str]:
    from haenv.p2 import rules as R
    blk = (_get(raw, "prediction_context") or {}).get(BLOCK)
    if not isinstance(blk, dict):
        return []
    invalid = measurement_conflicts(blk)
    if invalid:
        return invalid
    a = R.assess(blk)
    out = []
    if a["rule"] != g.get("rule") or a["set"] != g.get("set"):
        out.append(f"rule reader on the shown block = {a['rule']} {a['set']}, gold = {g.get('rule')} {g.get('set')}")
    want = R.table()["tiers"][g["tier"]]["rule"]
    if a["rule"] != want:
        out.append(f"tier {g['tier']} plans {want}, block fires {a['rule']}")
    return out


def leak_hits(block: dict) -> list[str]:
    from haenv.p2 import render as RD
    from haenv.p2 import rules as R
    out = []
    blob = json.dumps(block, ensure_ascii=False)
    if _RULE_ID.search(blob):
        out.append("a rule id in the visible block")
    msg = str(block.get("message") or "")
    if re.search(r"[0-9０-９]", msg):
        out.append(f"digit in the message: {msg!r}")
    try:
        ck = R.checklist_of(block)
    except ValueError as exc:
        return out + [str(exc)]
    for k in R.R0_KEYS + (R.HTN_SYMPTOM, R.DIZZY, R.HYPO_HELP):
        if not ck.get(k) and any(s in msg for s in RD.say_pool(k)):
            out.append(f"message names {k} but the checklist says no")
    return out


def ledger_red_flags(g: dict, raw) -> list[tuple[str, str]]:
    out = []
    for e in _get(raw, "evidence_ledger") or ():
        s = str(e.get("symptom") or "") if isinstance(e, dict) else ""
        for w in RED_FLAG_WORDS:
            if w in s:
                out.append(("ledger_red_flag", f"{e.get('evidence_id')}: {w}"))
    return out


def checklist_vs_ledger(g: dict, raw) -> list[tuple[str, str]]:
    """The intake checklist denies a new symptom while the ledger records one for today or the 3 days before."""
    from haenv.p2.hooks import symptom_recent
    blk = (_get(raw, "prediction_context") or {}).get(BLOCK) or {}
    ck = blk.get("intake_checklist") or {}
    days = symptom_recent(_get(raw, "evidence_ledger"), int(blk["today_day"])) if blk else []
    if ck.get("mild_new_symptom") is False and days:
        return [("checklist_vs_ledger", f"days {days}")]
    return []


def world_in_window(g: dict, raw) -> list[tuple[str, str]]:
    """A world stream the solver view keeps (an evidence-ledger value is read from it) with a point
    in the block's window that differs from the block's reading of that day."""
    from haenv.p2.hooks import measurement_series
    from haenv.p2.render import WINDOW
    pc = _get(raw, "prediction_context") or {}
    T = int(pc.get("prediction_time_T") or 0)
    ld = _get(raw, "longitudinal_data") or {}
    blk = pc.get(BLOCK) or {}
    shown = measurement_series(blk)
    out = []
    for s in sorted(shown):
        for q in ld.get(s) or ():
            d = int(q["ts"])
            if d in shown[s] and shown[s][d] != float(q["value"]):
                out.append(("world_in_window", f"{s}@{d}"))
    return out[:3]


def record_vs_world(g: dict, raw) -> list[tuple[str, str]]:
    """A condition on the shown record that is not a condition of the world (disease + comorbidities)
    the labs were drawn from."""
    blk = (_get(raw, "prediction_context") or {}).get(BLOCK) or {}
    pb = (_get(raw, "latent_premise") or {}).get("patient_basics") or {}
    world = {str(pb.get("disease") or "")} | {str(x) for x in pb.get("comorbidities") or ()}
    shown = list((blk.get("background") or {}).get("known_conditions") or ())
    shown += list((_get(raw, "user_profile") or {}).get("known_conditions") or ())
    return [("record_vs_world", c) for c in dict.fromkeys(map(str, shown)) if c not in world]


#: adult obesity, BMI >= 28.0 kg/m2 (WS/T 428-2013)
OBESE_BMI = 28.0
#: a short adult's height (m): the 5th percentile of GB/T 10000-1988 (men 18-60, women 18-55), shorter than
#: today's adults, so the bound is lenient
SHORT_HEIGHT_M = {"M": 1.583, "F": 1.484}


def record_obesity_vs_weight(g: dict, raw) -> list[tuple[str, str]]:
    """The shown record lists obesity while every visible weight (the block's mornings and the world's
    weighings up to T) is below BMI 28 even at a short adult's height; no height is shown."""
    pc = _get(raw, "prediction_context") or {}
    blk = pc.get(BLOCK) or {}
    prof = _get(raw, "user_profile") or {}
    shown = set((blk.get("background") or {}).get("known_conditions") or ()) | set(prof.get("known_conditions") or ())
    h = SHORT_HEIGHT_M.get(str(prof.get("sex") or ""))
    if "obesity" not in shown or h is None:
        return []
    T = int(pc.get("prediction_time_T") or blk.get("today_day") or 0)
    kg = [float(r["kg"]) for r in blk.get("morning_weight") or ()]
    kg += [float(q["value"]) for q in (_get(raw, "longitudinal_data") or {}).get("weight") or () if int(q["ts"]) <= T]
    need = OBESE_BMI * h * h
    if kg and max(kg) < need:
        return [("record_obesity_vs_weight", f"max visible weight {max(kg):.1f} kg < {need:.1f} kg")]
    return []


def measurement_conflicts(block: dict) -> list[str]:
    """Check complete independent renderings before emission, with mutation controls.

    A different device/context (night SpO2 vs home SpO2, CGM minimum vs fasting
    glucose) is not forced equal. Duplicate views of the same measurement are.
    """
    from haenv.p2.render import WINDOW
    from haenv.p2.render import BP_DAYS
    from haenv.p2 import rules as R
    out = []
    try:
        R.checklist_of(block)
        T = int(block["today_day"])
        c = block["home_check_today"]
        if c.get("on_oxygen") is not False:
            from haenv.p2.history import scale2
            if c.get("on_oxygen") is not True or not scale2(block.get("clinical_history_record")):
                out.append("oxygen status missing or no registered oxygen target")
        for name, key, expected in (
            ("wearable_daily", "day", list(range(T - WINDOW + 1, T + 1))),
            ("morning_weight", "day", list(range(T - WINDOW + 1, T + 1))),
            ("home_bp_log", "day", [d for d in range(T - BP_DAYS + 1, T + 1) for _ in range(2)]),
        ):
            if [int(r[key]) for r in block[name]] != expected:
                out.append(f"{name}: missing, duplicate, out-of-order or future day")
        bp = block["home_bp_log"]
        if [(r["day"], r["time"]) for r in bp] != [(d, t) for d in range(T - BP_DAYS + 1, T + 1) for t in ("07:30", "20:00")]:
            out.append("home_bp_log: duplicate/missing observation time")
        if (bp[-1]["systolic_mmHg"], bp[-1]["diastolic_mmHg"]) != (c["systolic_bp_mmHg"], c["diastolic_bp_mmHg"]):
            out.append("home bp repeat does not equal home_check_today")
        known = set(block["background"]["known_conditions"])
        diabetic = bool(known & {"T2D", "T1D", "type2_diabetes"})
        cg = block["cgm_nightly"]
        if [int(r["night_of_day"]) for r in cg] != (list(range(T - 7, T)) if diabetic else []):
            out.append("cgm_nightly: incomplete or incompatible with diabetes background")
        for r in cg:
            if (r["min_mmol_L"] < R.table()["thresholds"]["hypo_level1_mmol"]) != (r["minutes_below_3_9"] > 0):
                out.append("cgm minimum and duration below threshold disagree")
        R.readings(block)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        out.append(f"incomplete measurements: {exc}")
    return out


def _extra(g: dict, raw) -> list[tuple[str, str]]:
    blk = (_get(raw, "prediction_context") or {}).get(BLOCK) or {}
    return (ledger_red_flags(g, raw) + checklist_vs_ledger(g, raw) + world_in_window(g, raw) + record_vs_world(g, raw)
            + record_obesity_vs_weight(g, raw)
            + [("measurement_consistency", x) for x in measurement_conflicts(blk)])


def emission():
    from haenv.world.validation.shared_audit import Emission
    return Emission(gate=GATE, block=BLOCK, plan_of=plan_of, recompute=_recompute,
                    class_of=lambda g: g.get("floor"),
                    planned_class=lambda p: p.get("floor"),
                    words=EN_TOKENS + ZH_TOKENS, leak=leak_hits, extra=_extra,
                    unrealizable=_unrealizable)
