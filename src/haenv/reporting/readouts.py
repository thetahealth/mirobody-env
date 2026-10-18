"""Readouts for the evaluation report."""
from __future__ import annotations

from pathlib import Path
import logging

log = logging.getLogger("haenv.analytics")


def noise_floor_for(job) -> dict:
    """Per-dimension noise floor from the latest `results/<task>/<job_id>/reliability-*.json`.

    Noise comes from repeated runs (`tools/reliability_passk.py`, k>=3). Returns `{}`
    when unreadable, so every dimension becomes `noise_unknown`.
    """
    import json as _j
    d = getattr(job, "results_dir", None)
    if d is None:
        return {}
    cands = sorted(Path(d).parent.glob("reliability-*.json"), reverse=True)
    if not cands:
        return {}
    try:
        rep = _j.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:
        return {}
    # A noise floor measured only on deterministic offline stubs is 0.0 and is not
    # used; an artefact without the field counts as unknown.
    if rep.get("stochastic_source") is False:
        log.warning("[report] noise floor %s was measured only on an offline stub (solvers=%s) and is not used; "
                    "a floor of 0 would make any difference read as a real effect. To use it for judging an effect, rerun it on real models.",
                    cands[0].name, rep.get("solvers_measured"))
        return {}
    return {k: v.get("noise_max_abs_delta") for k, v in (rep.get("dims") or {}).items()
            if isinstance(v, dict) and v.get("noise_max_abs_delta") is not None}


def disputed_gold_hits(built: dict | None) -> dict:
    """Cases in this batch that sit on an open entry of `registry/disputed_gold.yaml`.

    Affected cases stay in every score; the report names them and the affected
    dimensions. Returns `{"ok", "error", "n_open", "hits"}`; `ok=False` means the
    register could not be read, which the caller prints.
    """
    try:
        from haenv.resources.registry import load_disputed_gold
        from haenv.resources.registry import _load_yaml
        entries = load_disputed_gold()
        applies = (_load_yaml("disputed_gold.yaml").get("applies_to") or {})
    except Exception as e:                       # noqa: BLE001 -- surfaced, not swallowed
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "n_open": None, "hits": []}
    opened = [e for e in entries if str(e.get("status") or "") == "open"]
    from haenv.world.overlay import spec_id_of
    by_spec: dict[str, list[str]] = {}
    for cid, raw in (built or {}).items():
        sid, _how = spec_id_of(raw, allow_case_id_lookup=False)
        if sid:
            by_spec.setdefault(sid, []).append(str(cid))
    hits, unkeyed = [], []
    for e in opened:
        ap = applies.get(e["id"]) or {}
        specs = [str(s) for s in (ap.get("specs") or ())]
        if not specs:
            unkeyed.append(e["id"])
            continue
        cases = sorted(c for s in specs for c in by_spec.get(s, ()))
        hits.append({"id": e["id"], "field": e.get("field"), "specs": specs,
                     "dims": list(ap.get("dims") or ()), "cases": cases,
                     "question": str(e.get("question") or "").strip().splitlines()[0],
                     "who_can_resolve": str(e.get("who_can_resolve") or "").strip()})
    return {"ok": True, "error": None, "n_open": len(opened), "hits": hits,
            "unkeyed": unkeyed}
