"""Where the rebuilt pack 2 attaches to the production path, without editing a frozen-segment file
(same arrangement as pack 4, `haenv/p4/hooks.py`). When the `haenv.p2` group loads, `install()` wraps:

* `build.observe_labs` -- after the world's own measurement layer, realize the planned item
  (`render.realize`) on the patient's facts (known conditions, the world's weight before the
  window), write the visible block to `prediction_context.triage_p2`, the gold (rule reader on that
  block) to `adjudication.triage_p2`, add the scenario's conditions to `user_profile.known_conditions`,
  and remove the world's own in-window points of the quantities the block shows (so a world reading
  never sits next to a block reading of the same day);
* `qside.render_for` -- the pack's question template without the diagnosis packs' suffixes.

A case without a `p2` plan goes straight to the originals. Judging code = segment P2_JUDGING
(stamped on rows as `p2_judging_sha16`); world code = P2_WORLD (stamped on the gold as
`p2_world_sha16`, and into `world_sha` through the registered probe). Neither enters
`judging_sha16` or the generation fingerprint.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()

BLOCK = "triage_p2"
#: world streams that would show a second reading of a quantity the block shows (gate.world_in_window)
OVERLAP_STREAMS = ("weight", "resting_hr", "spo2", "body_temp", "steps", "sleep_duration", "sleep_hours", "hrv",
                   "systolic_bp", "diastolic_bp", "fasting_glucose", "CGM_TIR", "skin_temp", "heart_rate")


def p2_judging_fingerprint() -> str:
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P2_JUDGING")


def p2_world_fingerprint() -> str:
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P2_WORLD")


def is_p2_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get(BLOCK), dict)


def _weight_before(raw, day: int):
    pts = [q for q in (raw.longitudinal_data or {}).get("weight") or () if int(q["ts"]) <= day
           and isinstance(q.get("value"), (int, float))]
    return float(max(pts, key=lambda q: int(q["ts"]))["value"]) if pts else None


def measurement_series(block: dict) -> dict:
    """Canonical observations, grouped by world stream meaning (not by similar names)."""
    if not block:
        return {}
    T = int(block["today_day"])
    check = block["home_check_today"]
    wd = block["wearable_daily"]
    out = {"weight": {int(r["day"]): r["kg"] for r in block["morning_weight"]}}
    for stream, field in (("resting_hr", "resting_hr_bpm"), ("hrv", "hrv_ms"), ("steps", "steps"),
                          ("sleep_duration", "sleep_h"), ("sleep_hours", "sleep_h"),
                          ("skin_temp", "skin_temp_delta_c")):
        # Skin-temperature streams in the base world are absolute temperatures,
        # whereas the block carries a delta: they are not the same measurement.
        if stream != "skin_temp":
            out[stream] = {int(r["day"]): r[field] for r in wd if r[field] is not None}
    for stream, field in (("systolic_bp", "systolic_mmHg"), ("diastolic_bp", "diastolic_mmHg")):
        out[stream] = {int(r["day"]): r[field] for r in block["home_bp_log"] if r["time"] == "20:00"}
    for stream, field in (("heart_rate", "heart_rate_bpm"), ("spo2", "spo2_pct"), ("body_temp", "temperature_c")):
        out[stream] = {T: check[field]}
    return out


def prompt_payload(payload) -> dict:
    """Require the opaque solver id on the actual view, including all evidence ids."""
    import re
    data = dict(payload.__dict__)
    sid = str(data.get("case_id") or "")
    if not re.fullmatch(r"C[0-9A-F]{10}", sid):
        raise ValueError("p2 solver payload needs an opaque case id")
    for ev in data.get("evidence_ledger") or ():
        if not str(ev.get("evidence_id") or "").startswith(f"EV-{sid}-"):
            raise ValueError("p2 evidence id is not scoped to the opaque solver id")
    return data


#: `mild_new_symptom` covers today and the 3 days before (the same window as the checklist's
#: `needed_help_for_low_glucose_last_3_days`)
MILD_DAYS = 3


def symptom_recent(ledger, day: int) -> list[int]:
    """Days in [day - MILD_DAYS, day] on which the ledger records a patient-reported symptom."""
    return sorted({int(e.get("source_timestamp", -1)) for e in ledger or ()
                   if e.get("source_type") == "patient_reported_symptom"
                   and day - MILD_DAYS <= int(e.get("source_timestamp", -1)) <= day})


def observe_triage(raw, case_id: str, truth: dict, audit: dict) -> tuple[dict, dict]:
    from haenv.p2 import rules as R
    from haenv.p2.gate import plan_of
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    from haenv.p2.render import WINDOW
    from haenv.p2.render import Unrealizable
    from haenv.p2.render import realize
    from haenv.p2.render import visible_medications
    plan = plan_of(raw)
    if plan is None:
        return truth, audit
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    T = int(raw.prediction_context["prediction_time_T"])
    up = dict(raw.user_profile or {})
    wpts = {int(q["ts"]): float(q["value"]) for q in (raw.longitudinal_data or {}).get("weight") or ()
            if T - WINDOW < int(q["ts"]) <= T and isinstance(q.get("value"), (int, float))}
    patient = {"known_conditions": list(up.get("known_conditions") or ()), "age_range": up.get("age_range"),
               "weight_kg": _weight_before(raw, T - WINDOW), "weight_series": wpts,
               "new_symptom_recent": bool(symptom_recent(getattr(raw, "evidence_ledger", None), T)),
               "medications": visible_medications((raw.latent_premise or {}).get(WORLD_MEDICATION_KEY), T)}
    try:
        item = realize(str(case_id), T, plan, patient)
    except Unrealizable as e:
        adj[BLOCK] = {"plan": dict(plan), "unrealizable": str(e)}
        raw.adjudication = adj
        return truth, {**(audit or {}), "p2": {"unrealizable": str(e)}}
    blk, a = item["block"], item["assess"]
    up["known_conditions"] = list(blk["background"]["known_conditions"])
    raw.user_profile = up
    # One measurement layer: world points of the same context use the block's final
    # observations. The builder subsequently realigns ledger values to these streams.
    series = measurement_series(blk)
    for stream, points in series.items():
        if stream not in (raw.longitudinal_data or {}):
            continue
        # Keep the world's declared sampling cadence. The block's additional
        # simulated daily home measurements do not add world stream timestamps.
        for q in raw.longitudinal_data[stream]:
            if int(q["ts"]) in points:
                q["value"] = points[int(q["ts"])]
    # A day the watch did not record (null field) shows no world reading of that quantity either.
    for stream, field in (("resting_hr", "resting_hr_bpm"), ("steps", "steps"), ("hrv", "hrv_ms"),
                          ("sleep_duration", "sleep_h"), ("sleep_hours", "sleep_h")):
        gone = {int(r["day"]) for r in blk["wearable_daily"] if r[field] is None}
        if gone and stream in (raw.longitudinal_data or {}):
            raw.longitudinal_data[stream] = [q for q in raw.longitudinal_data[stream] if int(q["ts"]) not in gone]
    # the dated record behind the medication list (`med_course.record_rows`): the primary drug and the
    # background medicines over the world's record; the item's own prescriptions stand on the list
    import haenv.world.clinical.med_course as MC
    wm = (raw.latent_premise or {}).get(WORLD_MEDICATION_KEY) or {}
    if "history_lo" in wm:
        blk["background"]["medication_record"] = MC.record_rows(wm, int(wm["history_lo"]), T, primary=True)
    pc = dict(raw.prediction_context or {})
    pc[BLOCK] = blk
    raw.prediction_context = pc
    from haenv.world.observations.events_text import relocate_event_history
    relocate_event_history(raw)
    adj[BLOCK] = {"pack": "p2", "plan": dict(plan), "tier": plan["tier"], "stratum": plan["stratum"],
                  "wording": plan["wording"], "beta_blocker": bool(plan.get("beta_blocker")),
                  "cohort": plan.get("cohort", "ordinary"),
                  "rule": a["rule"], "set": a["set"], "floor": R.floor_of(a["set"]),
                  "borderline": len(a["set"]) > 1, "readings": a["readings"], "attempt": item["attempt"],
                  "rule_source": R.rule_of(a["rule"]).get("source"),
                  "table_sha16": R.table_sha16(), "p2_world_sha16": p2_world_fingerprint()}
    raw.adjudication = adj
    return truth, {**(audit or {}), "p2": {"rule": a["rule"], "attempt": item["attempt"]}}


def install() -> None:
    """Idempotent."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from haenv.runtime.state import run_ctx_of

        def observe_labs(next_call, raw, case_id):
            truth, audit = next_call(raw, case_id)
            return observe_triage(raw, case_id, truth, audit)

        def render_for(next_call, solver, payload):
            import haenv.interaction.qside as QS
            ctx = run_ctx_of(solver)
            if not is_p2_payload(payload) or not ctx.probes:
                return next_call(solver, payload)
            prompt_payload(payload)
            pid = QS.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            return QS.render_with_probe(payload, pid, ctx.probes, allow_retired=ctx.allow_retired), pid

        from haenv.app.pack_hooks import Stage
        from haenv.app.pack_hooks import register_hook
        register_hook(Stage.OBSERVE, "p2:realize", observe_labs)
        register_hook(Stage.PROMPT, "p2:framing", render_for)
        _INSTALLED.append(True)
