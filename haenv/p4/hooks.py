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

The judging code is its own segment, `tools/make_freeze.py:P4_JUDGING`, stamped on pack-4 rows as
`p4_judging_sha16`; the world code is `P4_WORLD`, stamped on the gold block as `p4_world_sha16`
and carried into `world_sha` through the registered probe (`plugin.register`). Neither enters
`judging_sha16` or the generation fingerprint.
"""
from __future__ import annotations

import functools
import math
import threading

_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


def p4_judging_fingerprint() -> str:
    """Semantic fingerprint of `tools/make_freeze.py:P4_JUDGING` (same rule as `judging_sha16`)."""
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("P4_JUDGING")


def p4_world_fingerprint() -> str:
    """Semantic fingerprint of `tools/make_freeze.py:P4_WORLD`."""
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("P4_WORLD")


def is_p4_payload(payload) -> bool:
    pc = getattr(payload, "prediction_context", None) or {}
    return isinstance(pc, dict) and isinstance(pc.get("followup_p4"), dict)


def observe_followup(raw, case_id: str, truth: dict, audit: dict) -> tuple[dict, dict]:
    """The pack-4 part of the measurement layer for one case (no-op without a plan)."""
    from .. import build as B
    from .gate import plan_of
    from .realize import Unrealizable, realize
    from .render import render_block
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
    for k in [ind] + [c["code"] for c in [item.get("comorbidity")] if c]:      # R15: and the comorbidity
        if k and k not in (up.get("known_conditions") or ()):
            up["known_conditions"] = list(up.get("known_conditions") or ()) + [k]
            raw.user_profile = up
    block = render_block(item, raw, str(case_id))
    pc = dict(raw.prediction_context or {})
    pc["followup_p4"] = block
    raw.prediction_context = pc
    from ..labworld import tables_sha16
    gold = {**item, "plan": dict(plan), "obs_pct": round(100.0 * (math.exp(item["g4"]["dlog_obs"]) - 1.0), 3),
            "p4_world_sha16": p4_world_fingerprint(), "labworld_tables_sha16": tables_sha16()}
    adj["followup_p4"] = gold
    raw.adjudication = adj
    rec = {"analyte": item["analyte"], "class": item["class"], "t0": item["t0"], "t1": item["t1"],
           "lab_truth": [[int(r["day"]), r["truth"]] for r in sorted(item["records"], key=lambda r: int(r["day"]))],
           "records": item["records"], "gap_index": item["gap_index"], "attempt": item["attempt"]}
    return truth, {**(audit or {}), "p4": rec}


def install() -> None:
    """Idempotent. Wraps the module attributes the production path looks up at call time."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from .. import build as B
        from .. import evaluate as EV
        from .. import qside as QS
        from ..run_state import run_ctx_of
        o_observe, o_render = B.observe_labs, QS.render_for

        @functools.wraps(o_observe)
        def observe_labs(raw, case_id):
            truth, audit = o_observe(raw, case_id)
            return observe_followup(raw, case_id, truth, audit)

        @functools.wraps(o_render)
        def render_for(solver, payload):
            ctx = run_ctx_of(solver)
            if not is_p4_payload(payload) or not ctx.probes:
                return o_render(solver, payload)
            pid = EV.resolve_probe(solver, getattr(solver, "prompt_mode", "default"))
            return EV.render_with_probe(payload, pid, ctx.probes, allow_retired=ctx.allow_retired), pid

        B.observe_labs = observe_labs
        QS.render_for = render_for
        _INSTALLED.append(True)
