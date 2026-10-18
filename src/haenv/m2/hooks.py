"""Where M2 attaches to the production path, without editing a judging-segment file.

M2 wraps production functions when its judge group loads
(`plugin.register`); every wrapper hands a case that is not an M2 case (`core.is_m2_case`: F1-F3, or
an F0 bridge case of the M2 pack) straight to the original; frame asks and atoms stay F1-F3:

* `gated.menu_for`           -- the M2 discriminator items on F1-F3 menus (`core.M2_DISCRIMINATORS`)
* `gated.synth_menu_target`  -- the M2 `resolves_to` rows and the sex filter (`core.M2_RESOLVES_TO`)
* `gated.synth_on_demand`    -- base-condition findings (`registry/base_condition_findings.yaml`)
* `gated.run_gated`          -- F3 today's follow-up result on the solver; the queried world of every
                                M2 case (`core.m2_world`); keeps round 1's answer and trace for round 2
* `events._render_lookalikes` -- the record format of a lookalike entry on F1-F3: the same fields as every
                                other patient-reported entry (F0 and v1.0.1 keep theirs, bytes unchanged)
* `rows.gated_menu`          -- the rebuilt menu of audits and stubs, marked like `run_gated`
* `evaluate.render_for`      -- the frame ask, the F3 follow-up and the round-2 block after the
                                question (the same suffix `evaluate` would append last)
* `evaluate._row_gated`      -- the M2 round-1 atoms (`score.r1_atoms`, `score.m2_tests_atoms`)
                                and the M2 judging fingerprint (`m2_judging_sha16`) on F1-F3 rows;
                                round 2 only when
                                the optional profile is on (`core.ROUND2`, off by default)

The M2 judging code is its own segment, `tools/release/make_freeze.py:M2_JUDGING`, stamped on M2 rows as
`m2_judging_sha16`; it never enters `judging_sha16`.
"""
from __future__ import annotations

import copy
import json
import threading

_TL = threading.local()
_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


def m2_suffix(solver, payload) -> str:
    """Frame ask (`prediction_context.m2_frame`), F3 follow-up and the round-2 revision block;
    `""` for every v1.0.1 question and round."""
    from haenv.m2.core import M2_FRAME_ASKS
    from haenv.m2.core import M2_REVISION_PROMPT
    fr = ((getattr(payload, "prediction_context", None) or {}).get("m2_frame") or {})
    fr = fr if isinstance(fr, dict) else {}
    out = M2_FRAME_ASKS.get(str(fr.get("frame") or ""), "")
    rc = getattr(solver, "m2_revision_context", None)
    if rc:
        out += M2_REVISION_PROMPT.format(
            items="\n".join(json.dumps(i, ensure_ascii=False) for i in rc.get("items") or []),
            round1=json.dumps(rc.get("round1") or {}, ensure_ascii=False))
    return out


def m2_judging_fingerprint() -> str:
    """Semantic fingerprint of `tools/release/make_freeze.py:M2_JUDGING` (same rule as `judging_sha16`)."""
    from haenv.app.packs import segment_fingerprint
    return segment_fingerprint("M2_JUDGING")


def _adjudication(raw) -> dict:
    return getattr(raw, "adjudication", None) or {}


def install() -> None:
    """Idempotent. Registers ordered callbacks at explicit application stages."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from haenv.m2 import core


        def render_lookalikes(next_call, raw, cs, T):
            from haenv.m2.background import lookalike_record_format
            return lookalike_record_format(next_call(raw, cs, T), cs)

        def observe_labs(next_call, raw, case_id):
            truth, audit = next_call(raw, case_id)
            if core.is_m2_case(raw):
                frame = _adjudication(raw).get("m2_frame") or {}
                if frame.get("frame") == "F2":
                    # Identify the decision object from hidden patient truth,
                    # before rendering any current-medication surface.
                    frame["target_drug"] = raw.latent_premise["patient_basics"]["regimen"]["drug"]
                core.sync_current_medications(raw)
                # The block was derived before surface generation. Serialize its
                # initial-stage list on ddx too, including hand-authored M2 jobs.
                workup = _adjudication(raw).get("m2_workup")
                if workup is not None:
                    raw.adjudication["ddx"]["tests"] = list(workup["required"])
            if core.is_m2_framed(raw):
                from haenv.m2.background import sync_known_conditions
                sync_known_conditions(raw)
            return truth, audit

        def gated_menu(next_call, raw, T):
            prev, _TL.m2_case = getattr(_TL, "m2_case", False), core.is_m2_case(raw)
            prev_budget = getattr(_TL, "m2_budget", None)
            try:
                _TL.m2_budget = core.case_budget(raw)
                return next_call(raw, T)
            finally:
                _TL.m2_case = prev
                _TL.m2_budget = prev_budget

        def menu_for(next_call, sp, withheld, case_id=""):
            return core.m2_menu_for(next_call, sp, withheld, case_id, getattr(_TL, "m2_case", False),
                                    getattr(_TL, "m2_budget", None))

        def synth_menu_target(next_call, raw, target, T):
            return core.m2_synth_menu_target(next_call, raw, target, T)

        def synth_on_demand(next_call, raw, target, T):
            return core.m2_synth_on_demand(next_call, raw, target, T)

        def run_gated(next_call, raw, T, solver, *a, **kw):
            plan = (_adjudication(raw).get("m2_frame") or {}).get("f3_plan")
            m2_case = core.is_m2_case(raw)
            if plan:
                ld = getattr(raw, "longitudinal_data", None) or {}
                solver.m2_followup = core.f3_followup(ld, plan, int(T))
            if m2_case:
                ld = getattr(raw, "longitudinal_data", None) or {}
                world = core.m2_world(raw, ld, plan, int(T))
                if world is not ld:
                    raw = copy.copy(raw)                 # the queried world: today's follow-up, FIB-4 from its inputs
                    raw.longitudinal_data = world
            prev_case, prev_budget = getattr(_TL, "m2_case", False), getattr(_TL, "m2_budget", None)
            _TL.m2_case, _TL.m2_budget = m2_case, core.case_budget(raw)
            try:
                out, tr = next_call(raw, T, solver, *a, **kw)
            finally:
                _TL.m2_case, _TL.m2_budget = prev_case, prev_budget
                if plan:
                    solver.m2_followup = None
            if core.is_m2_framed(raw):
                _TL.last = (solver, out, tr)
            return out, tr

        def render_for(next_call, solver, payload):
            text, pid = next_call(solver, payload)
            return text + m2_suffix(solver, payload), pid

        from haenv.app.pack_hooks import Stage
        from haenv.app.pack_hooks import register_hook
        for stage, name, fn in (
            (Stage.LOOKALIKES, "lookalikes", render_lookalikes),
            (Stage.OBSERVE, "observe", observe_labs),
            (Stage.GATED_MENU, "gated_menu", gated_menu),
            (Stage.MENU, "menu", menu_for),
            (Stage.SYNTH_TARGET, "synth_target", synth_menu_target),
            (Stage.SYNTH_FINDING, "synth_finding", synth_on_demand),
            (Stage.RUN_GATED, "run_gated", run_gated),
            (Stage.PROMPT, "framing", render_for),
        ):
            register_hook(stage, "m2:" + name, fn)
        _INSTALLED.append(True)


def gated_row(base_row, cid, sname, raw, T, solver, **kw):
    from haenv.m2 import core
    previous = getattr(_TL, "last", None)
    _TL.last = None
    try:
        row = base_row(cid, sname, raw, T, solver, **kw)
        last = getattr(_TL, "last", None)
    finally:
        _TL.last = previous
    if not core.is_m2_framed(raw):
        return row
    row["m2_judging_sha16"] = m2_judging_fingerprint()
    if last is None or last[1] is None or "tool_n_calls" not in row:
        return row                       # an abort path of round 1
    from haenv_kernel.build import build_instance     # kernel
    from haenv.m2.history import credit
    credit(row, raw, solver)
    from haenv.m2.score import r1_atoms
    _, vp = build_instance(raw, int(T))
    row.update(r1_atoms(row, vp))
    from haenv.m2.round2 import _f3_truth         # F3 RCV atoms need the case's readings at or before T
    row.update(_f3_truth(raw, last[1], vp))
    row["m2_r1_raw"] = dict(getattr(last[1], "_raw", None) or {})   # stored: a judging fix is a recompute
    from haenv.m2.score import m2_tests_atoms     # revision r2-2: a gold line's key-signal item is on target
    row.update(m2_tests_atoms(row["m2_r1_raw"], vp, row.get("tests_cap")))
    if core.ROUND2 and (vp.adjudication or {}).get("revision"):
        from haenv.m2.round2 import run_round2
        row.update(run_round2(cid, sname, raw, int(T), last[0], last[1], last[2], row, vp))
    return row
