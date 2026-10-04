"""Where M2 attaches to the production path, without editing a judging-segment file.

`haenv/evaluate.py`, `haenv/gated.py` and `registry/gated_pricing.yaml` stay byte-identical to the
integration branch, so a v1.0.1 cell (and an F0 bridge cell) is judged by the integration judging
code with the integration `judging_sha16`. M2 wraps five functions when its judge group loads
(`plugin.register`); every wrapper hands a case without `prediction_context.m2_frame` straight to
the original, so it adds nothing outside F1-F3:

* `gated.menu_for`           -- the M2 discriminator items on F1-F3 menus (`core.M2_DISCRIMINATORS`)
* `gated.synth_menu_target`  -- the M2 `resolves_to` rows and the sex filter (`core.M2_RESOLVES_TO`)
* `gated.synth_on_demand`    -- base-condition findings (`registry/base_condition_findings.yaml`)
* `gated.run_gated`          -- F3 today's follow-up result on the solver; keeps round 1's answer
                                and trace for round 2
* `evaluate.render_for`      -- the frame ask, the F3 follow-up and the round-2 block after the
                                question (the same suffix `evaluate` would append last)
* `evaluate._row_gated`      -- the M2 round-1 atoms (`score.r1_atoms`, `score.m2_tests_atoms`)
                                and the M2 judging fingerprint (`m2_judging_sha16`) on F1-F3 rows;
                                round 2 only when
                                the optional profile is on (`core.ROUND2`, off by default)

The M2 judging code is its own segment, `tools/make_freeze.py:M2_JUDGING`, stamped on M2 rows as
`m2_judging_sha16`; it never enters `judging_sha16`.
"""
from __future__ import annotations

import functools
import json
import threading

_TL = threading.local()
_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


def m2_suffix(solver, payload) -> str:
    """Frame ask (`prediction_context.m2_frame`), F3 follow-up and the round-2 revision block;
    `""` for every v1.0.1 question and round."""
    from .core import M2_FRAME_ASKS, M2_REVISION_PROMPT
    fr = ((getattr(payload, "prediction_context", None) or {}).get("m2_frame") or {})
    fr = fr if isinstance(fr, dict) else {}
    out = M2_FRAME_ASKS.get(str(fr.get("frame") or ""), "")
    fu = getattr(solver, "m2_followup", None)
    if fu and fr.get("frame") == "followup_interpretation":
        out += "\n  followup_result: " + json.dumps(fu, ensure_ascii=False)
    rc = getattr(solver, "m2_revision_context", None)
    if rc:
        out += M2_REVISION_PROMPT.format(
            items="\n".join(json.dumps(i, ensure_ascii=False) for i in rc.get("items") or []),
            round1=json.dumps(rc.get("round1") or {}, ensure_ascii=False))
    return out


def m2_judging_fingerprint() -> str:
    """Semantic fingerprint of `tools/make_freeze.py:M2_JUDGING` (same rule as `judging_sha16`)."""
    from ..pack_skeleton import segment_fingerprint
    return segment_fingerprint("M2_JUDGING")


def _adjudication(raw) -> dict:
    return getattr(raw, "adjudication", None) or {}


def install() -> None:
    """Idempotent. Wraps the module attributes the production path looks up at call time."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from .. import evaluate as EV
        from .. import gated as G
        from .. import qside as QS
        from . import core

        o_menu, o_smt, o_sod, o_run = G.menu_for, G.synth_menu_target, G.synth_on_demand, G.run_gated
        o_render, o_row = QS.render_for, EV._row_gated

        @functools.wraps(o_menu)
        def menu_for(sp, withheld, case_id=""):
            return core.m2_menu_for(o_menu, sp, withheld, case_id)

        @functools.wraps(o_smt)
        def synth_menu_target(raw, target, T):
            return core.m2_synth_menu_target(o_smt, raw, target, T)

        @functools.wraps(o_sod)
        def synth_on_demand(raw, target, T):
            return core.m2_synth_on_demand(o_sod, raw, target, T)

        @functools.wraps(o_run)
        def run_gated(raw, T, solver, *a, **kw):
            plan = (_adjudication(raw).get("m2_frame") or {}).get("f3_plan")
            if plan:
                solver.m2_followup = core.f3_followup(getattr(raw, "longitudinal_data", None) or {}, plan, int(T))
            try:
                out, tr = o_run(raw, T, solver, *a, **kw)
            finally:
                if plan:
                    solver.m2_followup = None
            if core.is_m2_framed(raw):
                _TL.last = (solver, out, tr)
            return out, tr

        @functools.wraps(o_render)
        def render_for(solver, payload):
            text, pid = o_render(solver, payload)
            return text + m2_suffix(solver, payload), pid

        @functools.wraps(o_row)
        def _row_gated(cid, sname, raw, T, solver, **kw):
            _TL.last = None
            row = o_row(cid, sname, raw, T, solver, **kw)
            if not core.is_m2_framed(raw):
                return row
            last, _TL.last = getattr(_TL, "last", None), None
            row["m2_judging_sha16"] = m2_judging_fingerprint()
            if last is None or last[1] is None or "tool_n_calls" not in row:
                return row                       # an abort path of round 1
            from haenv_kernel.build import build_instance     # kernel
            from .score import r1_atoms
            _, vp = build_instance(raw, int(T))
            row.update(r1_atoms(row, vp))
            from .round2 import _f3_truth         # F3 RCV atoms need the case's readings at or before T
            row.update(_f3_truth(raw, last[1], vp))
            row["m2_r1_raw"] = dict(getattr(last[1], "_raw", None) or {})   # stored: a judging fix is a recompute
            from .score import m2_tests_atoms     # revision r2-2: a gold line's key-signal item is on target
            row.update(m2_tests_atoms(row["m2_r1_raw"], vp, row.get("tests_cap")))
            if core.ROUND2 and (vp.adjudication or {}).get("revision"):
                from .round2 import run_round2
                row.update(run_round2(cid, sname, raw, int(T), last[0], last[1], last[2], row, vp))
            return row

        G.menu_for, G.synth_menu_target, G.synth_on_demand, G.run_gated = (
            menu_for, synth_menu_target, synth_on_demand, run_gated)
        QS.render_for = render_for
        EV._row_gated = _row_gated
        _INSTALLED.append(True)
