"""Innermost M2 purchase hook, reading the case's saved observation plan."""
from __future__ import annotations

import copy

from haenv.world.truth import plan_of


def synth(next_call, orig, raw, target, T):
    plan = plan_of(raw)
    if not plan:
        return next_call(orig, raw, target, T)
    from haenv.m2 import core
    from haenv.interaction import gated
    from haenv.world.clinical.clinical_evidence import _SUPPORTED
    from haenv.world.observations import findings_render as FR
    from haenv.world.truth.plan import constrain, delta, observed
    vocab = core._findings()
    fid = gated._resolve_target(str(target).strip(), vocab)
    if not fid:
        return next_call(orig, raw, target, T)
    ddx = (raw.adjudication or {}).get("ddx") or {}
    comps = core.components(str(ddx.get("spec_id") or ""))
    supported = any(fid in _SUPPORTED.get(c, ()) for c in comps)
    entry = plan["observables"].get(fid) or {}
    # A truth entry wins over the penetrance list: a glycaemic member that does not read its declared
    # direction still follows the case's one glycaemic state.
    if fid in plan["unexpressed"] and not entry and not supported and fid != "ThyroidUptakeUS":
        view = copy.copy(raw)
        view.adjudication = {**raw.adjudication, "ddx": {**ddx, "insufficient": True}}
        return next_call(orig, view, target, T)
    renal_course = plan["observables"].get("Cr", {}).get("kind") == "course"
    if (entry.get("kind") == "approach" or fid in plan["constraints"]) and not (renal_course and fid in ("Cr", "UACR")):
        fspec = vocab[fid]
        sx = core._sex_spec(raw, fid, fspec)
        stream = FR._PANEL_TO_TS.get(fid)
        points = sorted((q for q in (raw.longitudinal_data or {}).get(stream) or []
                         if isinstance(q, dict) and isinstance(q.get("value"), (int, float))
                         and not isinstance(q.get("value"), bool)), key=lambda q: int(q["ts"]))
        if points:
            before = [q for q in points if int(q["ts"]) <= int(T)]
            last = before[-1] if before else points[0]
            # Reuse the named-draw noise, precision mixture and report face exactly.
            view = copy.copy(raw)
            shift = delta(entry, T) - delta(entry, int(last["ts"]))
            view.longitudinal_data = {**raw.longitudinal_data,
                stream: [{**q, "value": float(q["value"]) + shift} if q is last else q for q in points]}
            pts = core._same_day(raw, fid, T, gated._named_on_stream(view, fid, fspec, T))
            from haenv.world.generation.build_clinical import lab_print_ndigits
            nd = lab_print_ndigits(str(raw.case_id), stream)
            spec = sx or fspec
            same_day = any(int(q["ts"]) == int(T) for q in points)
            if not same_day:
                # The stream draw already carries the plan and its background constraint. A new draw on
                # another day may still need it; an existing same-day draw must not be reflected twice.
                pts = [{**p, "value": constrain(plan, fid, p["value"], nd, FR._band(spec)[0])} for p in pts]
            return core._reflag(raw, pts, sx)
        spec = sx or fspec
        if not entry:
            entry = {"kind": "approach", "onset": 0, "half_time": None,
                     "base": FR._value(spec, "normal", None, str(raw.case_id), fid, 0)}
            entry["target"] = entry["base"]
        value = FR._round(observed(str(raw.case_id), fid, entry, int(T)), fid)
        value = constrain(plan, fid, value, FR._ndigits(fid), FR._band(spec)[0])
        lo, hi = FR._band(spec)
        flag = "H" if value > hi else "L" if value < lo else "NORMAL"
        name, sym = gated._report_face(str(raw.case_id), fid, flag)
        return [{"ts": int(T), "finding_id": fid, "value": value, "unit": spec.get("unit", ""),
                 "ref_low": lo, "ref_high": hi, "flag": sym, "name": name or spec["name_cn"]}]
    if renal_course and fid in ("Cr", "UACR", "CKDPersistence", "AKICreatinineChange") and not ddx.get("insufficient"):
        from haenv.world.truth.renal import result
        points = result(raw, fid, int(T))
        if points is not None:
            return points
    return next_call(orig, raw, target, T)


def install_results():
    from haenv.app.pack_hooks import Stage, register_hook
    register_hook(Stage.M2_FINDING, "m2:truth", synth)
