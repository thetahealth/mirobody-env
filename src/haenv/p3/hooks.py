"""Where pack 3 attaches to the production path, without editing a frozen-segment file.

Pack 3 wraps two module attributes when its group loads (`plugin.register`); each wrapper hands a
case without a pack-3 plan straight to the original (pack 4 wraps the same two; the wrappers chain
and each passes the other's cases through):

* `build.observe_labs` -- after the world's own lab measurement layer, realize the planned
  medication item (`realize.realize`), write the visible clinic record to
  `prediction_context.meds_p3` and the gold with its records to `adjudication.meds_p3`;
* `evaluate.render_for` -- the pack-3 question template without the diagnosis packs' run-time
  suffixes.

The generation-model premise drops the plan through the shared skeleton (`plugin.pack()`), and every
patient sentence is entered as a patient says it by the production path (`events.enter_symptoms`,
`registry/symptom_lay.yaml`).

The judging code is segment `tools/release/make_freeze.py:P3_JUDGING` (stamped on pack-3 rows as
`p3_judging_sha16`); the world code is `P3_WORLD` (stamped on the gold block as `p3_world_sha16`
and carried into `world_sha` through the registered probe). Neither enters `judging_sha16` or the
generation fingerprint.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()
BLOCK = "meds_p3"


def p3_judging_fingerprint() -> str:
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P3_JUDGING")


def p3_world_fingerprint() -> str:
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("P3_WORLD")


def is_p3_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get(BLOCK), dict)


def observe_meds(raw, case_id: str, truth: dict, audit: dict) -> tuple[dict, dict]:
    """The pack-3 part of the measurement layer for one case (no-op without a plan)."""
    from haenv.p3.gate import plan_of
    from haenv.p3.realize import Unrealizable
    from haenv.p3.realize import realize
    from haenv.p3.render import render_block
    plan = plan_of(raw)
    if plan is None:
        return truth, audit
    adj = raw.adjudication if isinstance(raw.adjudication, dict) else {}
    whitelist_ledger(raw, plan["drug"])
    try:
        item = realize(raw, str(case_id), plan)
    except Unrealizable as e:
        adj[BLOCK] = {"plan": dict(plan), "unrealizable": str(e)}
        raw.adjudication = adj
        return truth, {**(audit or {}), "p3": {"unrealizable": str(e)[:500]}}
    item["other_medications"] = world_record(raw, item)
    pc = dict(raw.prediction_context or {})
    pc[BLOCK] = render_block(item)
    raw.prediction_context = pc
    from haenv.world.observations.events_text import relocate_event_history
    relocate_event_history(raw)
    # the managed condition is a known condition of the patient (A5: the visible profile agrees with the block)
    from haenv.world.labworld.meds import drug
    up = dict(raw.user_profile or {})
    kc = list(up.get("known_conditions") or [])
    patient = (raw.latent_premise or {}).get("patient_basics") or {}
    # These were sampled as diagnosed background conditions, not hidden lines.
    for k in list(patient.get("comorbidities") or ()) + [drug(item["drug"])["known_condition"]] + [c["code"] for c in [item.get("comorbidity")] if c]:
        if k not in kc:         # A24: and the comorbidity on the record
            kc = kc + [k]
            up["known_conditions"] = kc
            raw.user_profile = up
    from haenv.world.labworld.meds import TARGETS
    from haenv.world.labworld.meds import MEDS
    adj[BLOCK] = {**item, "plan": dict(plan), "p3_world_sha16": p3_world_fingerprint(),
                  "tables": [MEDS, TARGETS]}
    raw.adjudication = adj
    rec = {"drug": item["drug"], "axis": item["axis"], "class": item["class"], "subtype": item["subtype"],
           "control_truth": [[int(r["day"]), r["truth"]] for r in item["visits"]],
           "layout_attempt": item["layout_attempt"], "level_attempt": item["level_attempt"]}
    return truth, {**(audit or {}), "p3": rec}


def world_record(raw, item: dict) -> list:
    """The world's medication record holds the managed drug on the item's dose line (the world drew it
    for the drug's condition, `med_course.overlay_medicines`); returns the dated record of the other
    medicines up to day T (start, dose changes, stop), as the block lists them (the day-T list of names
    for a world without the record)."""
    import haenv.world.clinical.med_course as MC
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    lp = raw.latent_premise if isinstance(raw.latent_premise, dict) else {}
    wm = lp.get(WORLD_MEDICATION_KEY) or {}
    line = [{"ts": int(d), "value": float(v)} for d, v in sorted(item["dose_line"], key=lambda x: int(x[0]))]
    for m in wm.get("concurrent") or ():
        if m.get("overlay") and m["drug"] == item["drug"]:
            m["dose_timeline"] = line
    T = int(item["T"])
    if "history_lo" in wm:
        # dated rows of the other medicines over the world's record (`med_course.record_rows`): the managed
        # drug and the comorbidity's co-medication are the item's own (overlay) and are printed by the block
        return MC.record_rows(wm, int(wm["history_lo"]), T, skip=(item["drug"],), primary=True)
    # a world without the record (`synth.world_history: short`, `med_dynamics: off`): the day-T list, as before
    co = (item.get("comorbidity") or {}).get("co_medication")
    return [MC.display(d, dose) for d, dose, m in MC.active_medicines(wm, T)
            if d != item["drug"] and not (co and m.get("overlay") and m.get("indication") == item["comorbidity"]["code"])]


def whitelist_ledger(raw, drug_name: str) -> None:
    """A16: the complaints on a pack-3 case's record are drawn from those registered as irrelevant
    to the managed drug: a complaint that matches the drug's `background_deny` words is redrawn, on
    the same day, from the registered complaints that do not. The case's own condition line is
    whitelisted at eligibility (`plan.dx_allowed`)."""
    from haenv.world.observations.events import symptom_lay_every
    from haenv.world.labworld.meds import meds
    from haenv.p3.plan import background_ok
    from haenv.p3.plan import h01
    reg, every = meds()["benign_complaints"], symptom_lay_every()
    sex = str((getattr(raw, "user_profile", None) or {}).get("sex") or "")
    allowed = sorted(t for t in reg if background_ok(drug_name, t, *every.get(t, ()), sex=sex))
    used = {e.get("symptom") for e in raw.evidence_ledger or ()}
    for e in raw.evidence_ledger or ():
        if (e.get("source_type") == "patient_reported_symptom"
                and not background_ok(drug_name, e.get("symptom"), e.get("context"),
                                       *every.get(e.get("symptom"), ()), sex=sex)):
            pick = min((t for t in allowed if t not in used), key=lambda t: h01("p3bg", e.get("evidence_id"), t))
            used.add(pick)
            e["symptom"] = pick
            e.pop("context", None)


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
            return observe_meds(raw, case_id, truth, audit)

        def render_for(next_call, solver, payload):
            import haenv.interaction.qside as QS
            ctx = run_ctx_of(solver)
            if not is_p3_payload(payload) or not ctx.probes:
                return next_call(solver, payload)
            pid = QS.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            return QS.render_with_probe(payload, pid, ctx.probes, allow_retired=ctx.allow_retired), pid

        from haenv.app.pack_hooks import Stage
        from haenv.app.pack_hooks import register_hook
        register_hook(Stage.OBSERVE, "p3:realize", observe_labs)
        register_hook(Stage.PROMPT, "p3:framing", render_for)
        _INSTALLED.append(True)
