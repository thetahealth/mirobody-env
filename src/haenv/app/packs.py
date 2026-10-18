"""One registration path for a plugin pack (`register_pack`), shared by every pack.

A pack hands over its gold blocks, its emission declaration (`shared_audit.Emission`), its question
template and probe, its judge and the geometries it mounts on; `register_pack` registers them
through `haenv/external_gold.py`, mounts the judge, and declares the hooks explicitly called by
the application for every pack:

* the generation-model premise of a pack's case without the extra gold `meta` keys the pack declares
  (the production builder already drops the external slot and `build.GOLD_META` for every plugin
  case);
* the publish gate: a `haenv run` cell of a pack case is refused (`ABORT(<pack>_unaudited)`) when
  its batch has no audit marker (`shared_audit.marker_name`), written by `tools/validation/pack_audit.py`
  after the batch audit passed; every row of a pack case carries `<pack>_judging_sha16`.

* the acute-event coupling (`couple_acute`): a pack whose item sets acute vital signs on a day
  (`Pack.acute`) has them written into the same-day wearable readings, so a 38.5 degC visit does
  not sit next to a 36.6 degC wrist reading of the same day;
* the solver view (`apply_view`): a pack declares the payload fields its frame reads (`PayloadView`);
  the builder drops the rest from the emitted case after the emission gates and before the final
  leak scan, so `cases.jsonl`, `payloads.jsonl` and the prompt all carry the same, smaller view.

`segment_fingerprint` is the semantic fingerprint of a `tools/release/make_freeze.py` segment (the rule of
`judging_sha16`), used for each pack's judging and world stamps.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import pathlib

# Compatibility imports: existing task plugins keep their public declaration path.
from haenv.contracts.pack import Pack
from haenv.contracts.pack import PackRegistry
from haenv.contracts.pack import PayloadView
from haenv.contracts.pack import WEIGHT_FORECAST_CONTEXT
from haenv.contracts.pack import WEIGHT_FORECAST_PROFILE

PACKS: dict[str, Pack] = {}
# Kept for existing plugin-state snapshot consumers; no execution installation state.
_INSTALLED: list[bool] = []


#: Kernel gates that read the answer's action class, review flag, tests to order or action text
#: (`haenv_kernel.verifier._hard_gates`); `unsafe_action:` is a prefix.
ACTION_FIELD_GATES = ("premature_closure", "missing_clinician_review_flag", "missed_emergency_red_flag",
                      "over_triage", "treatment_before_exclusion", "invasive_before_firstline",
                      "med_change_without_clinician", "unsafe_action:")


def drop_action_gates(row: dict) -> dict:
    """The row without the gates its pack's answer cannot touch; `overall` re-read the way the row
    builder writes it (`FAIL(gate)` with a gate left, else `SCORED`)."""
    g = row.get("gates")
    if not isinstance(g, list):
        return row
    keep = [x for x in g if not any(str(x) == a or (a.endswith(":") and str(x).startswith(a))
                                    for a in ACTION_FIELD_GATES)]
    if len(keep) == len(g):
        return row
    out = {**row, "gates": keep}
    if row.get("overall") in ("FAIL(gate)", "SCORED"):
        out["overall"] = "FAIL(gate)" if keep else "SCORED"
    return out


@functools.lru_cache(maxsize=None)
def _segments_module():
    from haenv.provenance import declarations
    return declarations


def segment(name: str) -> tuple[str, ...]:
    declarations = _segments_module()
    if not hasattr(declarations, name):
        raise ValueError(f"unknown source segment: {name!r}")
    files = getattr(declarations, name)
    if not isinstance(files, tuple) or not all(isinstance(p, str) for p in files):
        raise ValueError(f"not a source segment: {name!r}")
    return files


@functools.lru_cache(maxsize=None)
def segment_fingerprint(name: str) -> str:
    from haenv.provenance.anchor import _fingerprint_of
    return _fingerprint_of(segment(name))


def register_pack(P: Pack) -> list:
    """Idempotent. Returns the pack's judge (a one-element list) the first time, else []."""
    import haenv.contracts.external_gold as EG
    from haenv.world.validation.shared_audit import emission_gate
    P = PackRegistry(PACKS).register(P)
    if P.history_dependency is not None:
        from haenv.world.history.history_dependency import install as register_history
        register_history(P)
    for name, fn, pfn, keys, classes, why in P.blocks:
        if name not in EG.BLOCKS:
            EG.register_gold_block(name, fn, latent_keys=keys, latent_classes=classes, probe_fn=pfn,
                                   source=P.source, why=why)
    E = P.emission
    if E is not None and E.gate not in EG.GATES:
        EG.register_gate(E.gate, emission_gate(E), source="haenv.shared_audit (SA-1, SA-8)",
                         why=f"{P.name}: unrealizable, class != plan, gold re-derived from the records "
                             f"!= gold, pack callbacks, block missing, gold literal, other-sex word => not emitted")
    if P.framing and P.framing[0] not in EG.FRAMINGS:
        EG.register_framing(P.framing[0], P.framing[1], source=P.source, why=P.framing[2])
    if P.probe is not None:
        d = P.probe()
        if d["probe_id"] not in EG.PROBES:
            EG.register_probe(d, source=P.source)
    if P.judge is None:
        return []
    import haenv.scoring.mounts.table as MT
    from haenv.judges import JUDGES
    from haenv.judges import Judge
    jname, fn, kinds, when = P.judge
    if jname not in MT.MOUNT:
        MT.mount(jname, dict(P.mounts), why_not=dict(P.why_not))
    if jname in {j.name for j in JUDGES}:
        return []
    return [Judge(name=jname, fn=fn, kinds=kinds, category="deterministic", when=when)]


def graded_base_labels(built: dict, labels: set) -> set:
    """The base-world labels the build's A5 shortcut scan grades: when every case belongs to a pack that
    declares `base_labels`, only those (a label no score reads is reported, not graded)."""
    owners = [owner(r) for r in built.values()]
    if not owners or any(P is None or P.base_labels is None for P in owners):
        return set(labels)
    return set(labels) & {x for P in owners for x in P.base_labels}


def grades_base_world(built: dict) -> bool:
    """Whether some case's score reads a base-world label (the ddx tier included): False when every
    case belongs to a pack whose score reads none (`base_labels == ()`), so the build's ddx-tier
    surface check (GEN8t) is reported, not enforced, on such a batch."""
    owners = [owner(r) for r in built.values()]
    return not owners or any(P is None or P.base_labels is None or P.base_labels for P in owners)


def owner(raw) -> Pack | None:
    return PackRegistry(PACKS).owner(raw)


#: Heart rate rises about 10 bpm per degC of fever (Liebermeister's rule; Karjalainen & Viitasalo,
#: Acta Med Scand 1986;220:179, 8.5 bpm/degC). provisional, review: pending.
HR_PER_DEGC = 10.0
#: The wearable's resting heart rate is the day's lowest sustained rate, below the visit's rate:
#: the day's resting rate moves this share of the way from its own value to the visit's.
#: provisional, review: pending.
RESTING_HR_SHARE = 0.6


def couple_acute(raw) -> list[dict]:
    """Write the pack's acute vitals into the same-day wearable points of `raw` (in place).

    body_temp takes the visit's temperature; resting_hr moves `RESTING_HR_SHARE` of the way to the
    visit's heart rate when that is higher (fever alone raises it by at least `HR_PER_DEGC` per
    degC above 37); spo2 is capped one point above the visit's. Only a stream that has a point on
    that day is touched; returns the changes made."""
    P = owner(raw)
    if P is None or P.acute is None:
        return []
    ld = raw.longitudinal_data or {}
    out = []
    for a in P.acute(raw) or ():
        day = int(a["day"])
        for name, fn in (("body_temp", lambda v: round(float(a["temp_c"]), 1) if a.get("temp_c") is not None else v),
                         ("resting_hr", lambda v: _resting_hr(v, a)),
                         ("spo2", lambda v: (min(v, round(float(a["spo2_pct"]) + 1.0, 1))
                                             if a.get("spo2_pct") is not None else v))):
            for q in ld.get(name) or ():
                v = q.get("value")
                if int(q["ts"]) != day or not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                new = fn(v)
                if new != v:
                    out.append({"stream": name, "day": day, "from": v, "to": new})
                    q["value"] = new
    return out


def _resting_hr(v: float, a: dict) -> float:
    lo = v
    if a.get("temp_c") is not None and float(a["temp_c"]) > 37.0:
        lo = v + HR_PER_DEGC * (float(a["temp_c"]) - 37.0)
    if a.get("hr_bpm") is not None and float(a["hr_bpm"]) > v:
        lo = max(lo, v + RESTING_HR_SHARE * (float(a["hr_bpm"]) - v))
    lo = min(lo, float(a["hr_bpm"])) if a.get("hr_bpm") is not None else lo
    if lo == v:
        return v
    # at the stream's declared precision, so the coupled day does not stand out by its digits
    from haenv.world.clinical.indicators import _declared_ndigits
    nd = _declared_ndigits("resting_hr")
    return round(lo, nd) if nd else float(round(lo))


def ledger_streams(raw) -> set[str]:
    """Streams an evidence-ledger value of `raw` is read from."""
    from haenv.world.validation.gate_tables import LEDGER_VALUE_STREAMS
    ld = getattr(raw, "longitudinal_data", None) or {}
    out: set[str] = set()
    for e in getattr(raw, "evidence_ledger", None) or ():
        v = e.get("measured_value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out |= set(LEDGER_VALUE_STREAMS.get(str(e.get("source_type") or "")) or ()) & set(ld)
    return out


def apply_view(raw) -> dict | None:
    """Trim `raw` in place to its pack's solver view; returns what was dropped (None: no view)."""
    P = owner(raw)
    V = P.view if P else None
    if V is None or (V.applies is not None and not V.applies(raw)):
        return None
    pc, up = raw.prediction_context or {}, raw.user_profile or {}
    ld = raw.longitudinal_data or {}
    drop_pc = [k for k in V.drop_context if k in pc]
    drop_up = [k for k in V.drop_profile if k in up]
    keep = None if V.streams is None else set(V.streams) | ledger_streams(raw)
    drop_ld = [] if keep is None else sorted(k for k in ld if k not in keep)
    raw.prediction_context = {k: v for k, v in pc.items() if k not in drop_pc}
    raw.user_profile = {k: v for k, v in up.items() if k not in drop_up}
    raw.longitudinal_data = {k: v for k, v in ld.items() if k not in drop_ld}
    return {"pack": P.name, "context": drop_pc, "profile": drop_up, "streams": drop_ld}


def filter_prompt_meta(p, premise: dict) -> dict:
    """Apply pack gold-meta declarations using the unfiltered source slot."""
    import haenv.contracts.external_gold as EG
    src = p.dumps() if hasattr(p, "dumps") else (p or {})
    slot = (src.get("meta") or {}).get(EG.SLOT) or {}
    drop = {k for P in PACKS.values() if set(slot) & {x for b in P.blocks for x in b[3]}
            for k in P.prompt_gold_meta}
    if drop:
        premise["meta"] = {k: v for k, v in (premise.get("meta") or {}).items() if k not in drop}
    return premise


def run_row(geometry, base_row, cid, sname, raw, T, solver, *, ctx):
    """Explicit row boundary: audit before execution, task atoms, action gates, then stamp."""
    from haenv.world.validation.shared_audit import marker_name
    P = owner(raw)
    rp = getattr(ctx, "resp_path", None)
    bdir = pathlib.Path(rp).parent if (P and rp) else None
    if bdir is not None and not (bdir / marker_name(P.name)).is_file():
        out = {"case": cid, "solver": sname, "overall": f"ABORT({P.name}_unaudited)"}
    else:
        # Row callbacks retain their own applicability checks. In historical M2
        # data a diagnostic_visit frame may exist without a verifier m2_frame;
        # its atoms were still computed, while ownership/audit gating was absent.
        row = base_row
        if geometry == "gated":
            for pack in tuple(PACKS.values()):
                if pack.gated_row:
                    row = functools.partial(pack.gated_row, row)
        out = row(cid, sname, raw, T, solver, ctx=ctx)
    if P and not P.answers_action and isinstance(out, dict):
        out = drop_action_gates(out)
    if P and P.judging_segment and isinstance(out, dict):
        out.setdefault(f"{P.name}_judging_sha16", segment_fingerprint(P.judging_segment))
    return out


def install() -> None:
    """Compatibility entry point. Main flows call the registered hooks explicitly."""
