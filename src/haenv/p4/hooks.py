"""Where pack 4 attaches to the production path, without editing a frozen-segment file.

`haenv/build.py` (generation) and `haenv/evaluate.py` (judging) stay byte-identical to the
integration branch. Pack 4 wraps two module attributes when its group loads (`plugin.register`);
each wrapper hands a case without a pack-4 plan straight to the original, so every other job
builds and renders exactly as before:

* `build.observe_labs` -- after the world's own lab measurement layer, realize the planned
  follow-up item (`realize.realize`), write the visible block to `prediction_context.followup_p4`
  and the gold with its per-point records to `adjudication.followup_p4`, and add the item's
  records and lab truth to the build audit (`lab_observation.p4`);
* `evaluate.render_for` -- the pack-4 question template without the run-time suffixes of the
  diagnosis packs (noop / quant / premise / gated asks belong to those packs).

The generation-model premise of a pack-4 case drops the weight-task labels `outcome` and `driver`
through the shared skeleton (`plugin.pack().prompt_gold_meta`).

Every patient-reported symptom of a pack-4 case is entered as a patient says it by the production
path itself (`events.enter_symptoms`, `registry/symptom_lay.yaml`), as on every plugin task's case.

The judging code is its own segment, `tools/release/make_freeze.py:P4_JUDGING`, stamped on pack-4 rows as
`p4_judging_sha16`; the world code is `P4_WORLD`, stamped on the gold block as `p4_world_sha16`
and carried into `world_sha` through the registered probe (`plugin.register`). Neither enters
`judging_sha16` or the generation fingerprint.
"""
from __future__ import annotations

import math
import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


def p4_judging_fingerprint() -> str:
    """Semantic fingerprint of `tools/release/make_freeze.py:P4_JUDGING` (same rule as `judging_sha16`)."""
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P4_JUDGING")


def p4_world_fingerprint() -> str:
    """Semantic fingerprint of `tools/release/make_freeze.py:P4_WORLD`."""
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P4_WORLD")


def is_p4_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get("followup_p4"), dict)


def observe_followup(raw, case_id: str, truth: dict, audit: dict) -> tuple[dict, dict]:
    """The pack-4 part of the measurement layer for one case (no-op without a plan)."""
    import haenv.world.generation.build as B
    from haenv.p4.gate import plan_of
    from haenv.p4.realize import Unrealizable
    from haenv.p4.realize import realize
    from haenv.p4.render import render_block
    plan = plan_of(raw)
    if plan is None:
        return truth, audit
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    try:
        item = realize(raw, str(case_id), plan, B._LAB_CTX.get(str(case_id)))
    except Unrealizable as e:
        adj["followup_p4"] = {"plan": dict(plan), "unrealizable": str(e)}
        raw.adjudication = adj
        return truth, {**(audit or {}), "p4": {"unrealizable": str(e)}}
    # the ACEI's indication is on the patient's list of known conditions
    ind = (item.get("acei") or {}).get("indication")
    up = dict(raw.user_profile or {})
    patient = (raw.latent_premise or {}).get("patient_basics") or {}
    for k in list(patient.get("comorbidities") or ()) + [ind] + [c["code"] for c in [item.get("comorbidity")] if c]:
        if k and k not in (up.get("known_conditions") or ()):
            up["known_conditions"] = list(up.get("known_conditions") or ()) + [k]
            raw.user_profile = up
    world_record(raw, item)
    block = render_block(item, raw, str(case_id))
    pc = dict(raw.prediction_context or {})
    pc["followup_p4"] = block
    raw.prediction_context = pc
    from haenv.world.observations.events_text import relocate_event_history
    relocate_event_history(raw)
    from haenv.world.labworld import tables_sha16
    gold = {**item, "plan": dict(plan), "obs_pct": round(100.0 * (math.exp(item["g4"]["dlog_obs"]) - 1.0), 3),
            "p4_world_sha16": p4_world_fingerprint(), "labworld_tables_sha16": tables_sha16()}
    adj["followup_p4"] = gold
    raw.adjudication = adj
    rec = {"analyte": item["analyte"], "class": item["class"], "t0": item["t0"], "t1": item["t1"],
           "lab_truth": [[int(r["day"]), r["truth"]] for r in sorted(item["records"], key=lambda r: int(r["day"]))],
           "records": item["records"], "gap_index": item["gap_index"], "attempt": item["attempt"]}
    return truth, {**(audit or {}), "p4": rec}


def acei_line(item: dict) -> list[dict]:
    """The ACEI's dose line as the item's record prints it (`render.medication_rows`)."""
    from haenv.world.labworld import tables
    dose = float(str(tables()["effects"]["acei_on_creatinine"]["dose"]).split()[0])
    ac = item.get("acei") or {}
    line = [{"ts": 0, "value": dose}] if ac.get("initial_on") else []
    for ev in sorted(list(ac.get("events") or ()) + list(ac.get("decoy_events") or ()), key=lambda e: int(e["day"])):
        line.append({"ts": int(ev["day"]), "value": dose if ev["action"] == "start" else 0.0})
    return line


def world_record(raw, item: dict) -> None:
    """The world's medication record holds the ACEI on the dose line the record prints; an item that
    prints no ACEI leaves it out of the record."""
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    from haenv.p4.plan import ACEI_DRUG
    wm = (raw.latent_premise or {}).get(WORLD_MEDICATION_KEY) or {}
    line = acei_line(item)
    keep = []
    for m in wm.get("concurrent") or ():
        if m.get("overlay") and m["drug"] == ACEI_DRUG:
            if not line:
                continue
            m["dose_timeline"] = line
        keep.append(m)
    if wm:
        wm["concurrent"] = keep


def install() -> None:
    """Idempotent. Registers ordered callbacks at explicit application stages."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from haenv.runtime.state import run_ctx_of

        def observe_labs(next_call, raw, case_id):
            truth, audit = next_call(raw, case_id)
            return observe_followup(raw, case_id, truth, audit)

        def render_for(next_call, solver, payload):
            import haenv.interaction.qside as QS
            ctx = run_ctx_of(solver)
            if not is_p4_payload(payload) or not ctx.probes:
                return next_call(solver, payload)
            pid = QS.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            return QS.render_with_probe(payload, pid, ctx.probes, allow_retired=ctx.allow_retired), pid

        from haenv.app.pack_hooks import Stage
        from haenv.app.pack_hooks import register_hook
        register_hook(Stage.OBSERVE, "p4:realize", observe_labs)
        register_hook(Stage.PROMPT, "p4:framing", render_for)
        _INSTALLED.append(True)
