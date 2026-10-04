"""M2 round 2 (spec 1, 2.2, 2.3): after the gated round-1 answer, push the case's five readings and
ask for a revision. One request, menu closed, same solver and the round-1 payload (the readings it
bought included), so the model's context is round 1 plus the push block.

The push block is rendered from the world (`core.render_push`) with the plan from
`adjudication.revision`; it is the same for every model. The cell's round-2 fields go onto the
round-1 row with the prefix `r2_`; the atoms come from `score.r2_atoms`, which reads only what is
persisted on the row, so a judging fix is a recompute.
"""
from __future__ import annotations

import copy
import logging

log = logging.getLogger("haenv.m2.round2")


def round1_payload(raw, T: int, tr):
    """The payload of the last round-1 step: the lean payload plus every reading the model bought,
    re-queried the way `gated.run_gated` revealed it."""
    from haenv_kernel.build import build_instance                    # kernel
    from haenv_kernel.gatekeeper import Gatekeeper                   # kernel
    from ..gated import synth_menu_target, withhold_signals
    sp, _ = build_instance(raw, int(T))
    lean, withheld = withhold_signals(sp)
    gk = Gatekeeper({**(lean.longitudinal_data or {}), **withheld}, int(T))
    revealed: dict = {}
    for c in (getattr(tr, "calls", None) or []):
        if not c.target or c.truncated or not c.revealed or c.target in revealed:
            continue
        res = gk.query(c.billed_kind or c.kind, c.target)
        if res.get("need_synth"):
            pts = synth_menu_target(raw, c.target, int(T))
            if pts:
                res["series"] = pts
        if res.get("series"):
            revealed[c.target] = res["series"]
    payload = copy.deepcopy(lean)
    payload.longitudinal_data = {**(lean.longitudinal_data or {}), **revealed}
    return payload


def alias_sets_of(vp) -> list[list[str]]:
    """Gold-line alias sets exactly as the round-1 dx judges use them."""
    from ..judges._helpers import _names
    from ..judges.differential import _thread_sets
    ddx = (getattr(vp, "adjudication", None) or {}).get("ddx") or {}
    if ddx.get("join_gold") == "comorbidity":
        return _thread_sets(vp)[0]
    if ddx.get("join_gold") == "unified":
        return [_names(vp)]
    return []


def _f3_truth(raw, out1, vp) -> dict:
    """F3 round-1 atoms with the world truth (readings at or before T come from the case)."""
    from .plugin import frame_atoms
    if ((getattr(vp, "adjudication", None) or {}).get("m2_frame") or {}).get("frame") != "F3":
        return {}
    a = frame_atoms(getattr(out1, "_raw", None) or {}, vp, getattr(raw, "longitudinal_data", None) or {})
    return {k: a[k] for k in ("m2_rcv_said", "m2_rcv_gold", "m2_rcv_ok", "m2_rcv_emitted")}


def run_round2(cid, sname, raw, T: int, solver, out1, tr, row: dict, vp) -> dict:
    from ..solve_guard import guarded_solve
    from .core import render_push
    from .score import r2_atoms
    plan = (getattr(vp, "adjudication", None) or {}).get("revision") or {}
    items = render_push(raw, int(T), plan)
    payload = round1_payload(raw, int(T), tr)
    r1 = dict(getattr(out1, "_raw", None) or {})
    prev_ctx = getattr(solver, "gated_context", None)
    solver.gated_context = None                         # menu closed
    solver.m2_revision_context = {"items": items, "round1": r1, "case_id": cid}
    try:
        facts = guarded_solve(solver, payload, int(T), tag=f"{cid}|m2:r2")
    finally:
        solver.m2_revision_context = None
        solver.gated_context = prev_ctx
    base = {"r2_expected": True, "r2_items": items, "m2_r1_raw": r1, "r2_alias_sets": alias_sets_of(vp),
            "r2_warranted": bool((vp.adjudication or {}).get("clinician_action_warranted"))}
    if facts.leak:
        base.update(_f3_truth(raw, out1, vp))
        return {**base, "r2_status": "ABORT(leak)", "r2_leak": list(facts.leak)}
    out2 = facts.out
    if out2 is None or facts.raw_empty:
        return {**base, "r2_status": "ABORT(no_response)"}
    if facts.raw_unparseable:
        return {**base, "r2_status": "ABORT(unparseable)",
                "r2_raw_text": (getattr(out2, "_raw_text", "") or "")[:20000]}
    raw2 = dict(getattr(out2, "_raw", None) or {})
    base.update(_f3_truth(raw, out1, vp))
    rec = {**base, "r2_status": "SCORED", "r2_raw": raw2,
           "r2_raw_text": (getattr(out2, "_raw_text", "") or "")[:20000] or None,
           "r2_usage": getattr(out2, "_usage", None)}
    rec.update(r2_atoms({**row, **rec}, plan))
    return rec
