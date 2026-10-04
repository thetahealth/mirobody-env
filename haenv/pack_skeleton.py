"""One registration path for a plugin pack (`register_pack`), shared by every pack.

A pack hands over its gold blocks, its emission declaration (`shared_audit.Emission`), its question
template and probe, its judge and the geometries it mounts on; `register_pack` registers them
through `haenv/external_gold.py`, mounts the judge, and installs once, for every pack:

* the generation-model premise of a pack's case without the extra gold `meta` keys the pack declares
  (the production builder already drops the external slot and `build.GOLD_META` for every plugin
  case);
* the publish gate: a `haenv run` cell of a pack case is refused (`ABORT(<pack>_unaudited)`) when
  its batch has no audit marker (`shared_audit.marker_name`), written by `tools/pack_audit.py`
  after the batch audit passed; every row of a pack case carries `<pack>_judging_sha16`.

`segment_fingerprint` is the semantic fingerprint of a `tools/make_freeze.py` segment (the rule of
`judging_sha16`), used for each pack's judging and world stamps.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import importlib.util
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Callable

#: pack name -> Pack, in registration order
PACKS: dict[str, "Pack"] = {}
_INSTALLED: list[bool] = []
_LOCK = threading.Lock()


@dataclass
class Pack:
    name: str                                   # marker / stamp prefix ("p4", "pack2", "m2")
    owns: Callable                              # raw case -> bool: a case of this pack
    blocks: tuple = ()                          # (name, gold_fn, probe_fn, latent_keys, latent_classes, why)
    emission: object | None = None              # shared_audit.Emission
    framing: tuple | None = None                # (name, template, why)
    probe: Callable | None = None               # () -> probe dict
    judge: tuple | None = None                  # (name, fn, kinds, when)
    mounts: dict = field(default_factory=dict)  # geometry -> "out"
    why_not: dict = field(default_factory=dict)
    prompt_gold_meta: tuple = ()                # extra premise `meta` keys a pack case must not show
    judging_segment: str | None = None          # make_freeze segment stamped on rows
    source: str = ""


@functools.lru_cache(maxsize=None)
def _segments_module():
    from .anchor import ROOT
    spec = importlib.util.spec_from_file_location("_mf_for_packs", pathlib.Path(ROOT) / "tools" / "make_freeze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                # noqa: S102
    return mod


def segment(name: str) -> tuple[str, ...]:
    return tuple(getattr(_segments_module(), name, ()))


@functools.lru_cache(maxsize=None)
def segment_fingerprint(name: str) -> str:
    from .anchor import _fingerprint_of
    return _fingerprint_of(segment(name))


def register_pack(P: Pack) -> list:
    """Idempotent. Returns the pack's judge (a one-element list) the first time, else []."""
    from . import external_gold as EG
    from .shared_audit import emission_gate
    install()
    PACKS.setdefault(P.name, P)
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
    from . import mount_table as MT
    from .judges import JUDGES, Judge
    jname, fn, kinds, when = P.judge
    if jname not in MT.MOUNT:
        MT.mount(jname, dict(P.mounts), why_not=dict(P.why_not))
    if jname in {j.name for j in JUDGES}:
        return []
    return [Judge(name=jname, fn=fn, kinds=kinds, category="deterministic", when=when)]


def owner(raw) -> Pack | None:
    for P in PACKS.values():
        if P.owns(raw):
            return P
    return None


def install() -> None:
    """Idempotent. Wraps `build._world_prompt_premise` and `evaluate._row_single` / `_row_gated`."""
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        from . import build as B
        from . import evaluate as EV
        from . import external_gold as EG
        from .shared_audit import marker_name
        o_premise = B._world_prompt_premise

        @functools.wraps(o_premise)
        def _world_prompt_premise(p, _o=o_premise):
            d = _o(p)
            src = p.dumps() if hasattr(p, "dumps") else (p or {})
            slot = (src.get("meta") or {}).get(EG.SLOT) or {}
            drop = {k for P in PACKS.values() if set(slot) & {x for b in P.blocks for x in b[3]}
                    for k in P.prompt_gold_meta}
            if drop:
                d["meta"] = {k: v for k, v in (d.get("meta") or {}).items() if k not in drop}
            return d

        def _guard(o_row):
            @functools.wraps(o_row)
            def row(cid, sname, raw, *a, **kw):
                P = owner(raw)
                rp = getattr(kw.get("ctx"), "resp_path", None)
                bdir = pathlib.Path(rp).parent if (P and rp) else None
                if bdir is not None and not (bdir / marker_name(P.name)).is_file():
                    out = {"case": cid, "solver": sname, "overall": f"ABORT({P.name}_unaudited)"}
                else:
                    out = o_row(cid, sname, raw, *a, **kw)
                if P and P.judging_segment and isinstance(out, dict):
                    out.setdefault(f"{P.name}_judging_sha16", segment_fingerprint(P.judging_segment))
                return out
            return row

        B._world_prompt_premise = _world_prompt_premise
        EV._row_single = _guard(EV._row_single)
        EV._row_gated = _guard(EV._row_gated)
        _INSTALLED.append(True)
