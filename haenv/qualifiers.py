"""Split a gold diagnosis into core + qualifiers, and record whether each qualifier can be
established from what the solver sees (generation side).

A gold such as "原发性甲状腺功能减退(桥本)" carries a qualifier whose confirming evidence may be
absent from the visible case. `annotate` runs once per emitted case on the visible instance at
T and writes two keys into `adjudication.ddx`, only when a registered family matches:

* `core` -- the gold with each matched qualifier removed (comorbid segments joined by " + ");
* `qualifiers` -- one record per matched qualifier: family, core, aliases, the confirming
  evidence found (ids / streams and the first day), `derivable`, and `required`
  (= `derivable`). A qualifier that cannot be established is not a required component.

Diagnosis text, aliases and threads are left as they are, so cases without a matched family
keep a byte-identical gold. Families live in `registry/gold_qualifiers.yaml`.

SYNTHETIC, for evaluation only, not medical advice.
"""
from __future__ import annotations

import json
import re

_REGISTRY = "gold_qualifiers.yaml"
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def families() -> dict[str, dict]:
    """The registered qualifier families (validated)."""
    from .regpath import load_registry
    d = load_registry(_REGISTRY) or {}
    fams = d.get("families") or {}
    for name, f in fams.items():
        for k in ("match", "qualifier", "meaning", "core_aliases", "confirming"):
            if k not in f:
                raise ValueError(f"{_REGISTRY}: family {name!r} lacks {k!r}")
        conf = f["confirming"] or {}
        if not (conf.get("keywords") or conf.get("thresholds")):
            raise ValueError(f"{_REGISTRY}: family {name!r} declares no confirming evidence")
    return fams


def _ledger_value(text: str, name: str) -> float | None:
    """The first number after `name` in a lab line such as "空腹血糖 6.61 mmol/L(参考 3.9–6.1)"."""
    t = str(text or "")
    if not t.startswith(name):
        return None
    m = _NUM.search(t[len(name):])
    return float(m.group(0)) if m else None


def confirming_evidence(conf: dict, sp) -> list[dict]:
    """Confirming evidence for one qualifier in the visible instance `sp`, sorted by day."""
    hits: list[dict] = []
    ledger = list(getattr(sp, "evidence_ledger", None) or [])
    streams = dict(getattr(sp, "longitudinal_data", None) or {})
    for kw in (conf.get("keywords") or ()):
        k = str(kw).lower()
        for e in ledger:
            if k in json.dumps(e, ensure_ascii=False).lower():
                hits.append({"evidence_id": e.get("evidence_id"),
                             "day": int(e.get("source_timestamp", -1)), "what": str(kw)})
        for name, pts in streams.items():
            if k in str(name).lower() and pts:
                hits.append({"stream": name, "day": min(int(p.get("ts", 0)) for p in pts
                                                        if isinstance(p, dict)),
                             "what": str(kw)})
    for th in (conf.get("thresholds") or ()):
        lo = float(th["min"])
        if th.get("ledger"):
            for e in ledger:
                v = _ledger_value(e.get("symptom") or e.get("note") or "", str(th["ledger"]))
                if v is not None and v >= lo:
                    hits.append({"evidence_id": e.get("evidence_id"),
                                 "day": int(e.get("source_timestamp", -1)),
                                 "what": f"{th['ledger']} {v:g} >= {lo:g}"})
        if th.get("stream"):
            for p in (streams.get(str(th["stream"])) or ()):
                if not isinstance(p, dict):
                    continue
                try:
                    v = float(p.get("value"))
                except (TypeError, ValueError):
                    continue
                if v >= lo:
                    hits.append({"stream": th["stream"], "day": int(p.get("ts", -1)),
                                 "what": f"{th['stream']} {v:g} >= {lo:g}"})
    return sorted(hits, key=lambda h: (h["day"], str(h.get("evidence_id") or h.get("stream"))))


def _thread_of(seg: str, threads) -> str | None:
    low = seg.lower()
    for t in threads or ():
        if any(str(a).lower() in low for a in (t.get("aliases") or ()) if str(a).strip()):
            return t.get("name")
    return None


def qualifiers_for(ddx: dict, sp) -> tuple[str, list[dict]] | None:
    """`(core, records)` for one gold `ddx` block on the visible instance `sp`; None when no
    registered family matches."""
    diag = str((ddx or {}).get("diagnosis") or "")
    if not diag:
        return None
    fams = families()
    tests_blob = " ".join(str(t) for t in ((ddx or {}).get("tests") or ())).lower()
    cores: list[str] = []
    recs: list[dict] = []
    for seg in [s.strip() for s in diag.split(" + ")]:
        core = seg
        for name, f in fams.items():
            if str(f["match"]) not in seg:
                continue
            seg_core = str(f.get("core") or seg.replace(str(f["match"]), "").strip())
            core = str(f.get("core") or core.replace(str(f["match"]), "").strip())
            ev = confirming_evidence(f["confirming"] or {}, sp)
            derivable = bool(ev)
            recs.append({
                "family": name, "segment": seg, "qualifier": f["qualifier"],
                "meaning": f["meaning"], "core": seg_core,
                "core_aliases": list(f.get("core_aliases") or ()),
                "qualifier_aliases": list(f.get("qualifier_aliases") or ()),
                "thread": _thread_of(seg, (ddx or {}).get("threads")),
                "derivable": derivable,
                "derivable_from_day": (ev[0]["day"] if ev else None),
                "evidence": ev[:5],
                "required": derivable,
                "confirming_test_in_gold": any(str(x).lower() in tests_blob
                                               for x in (f.get("confirming_tests") or ())),
            })
        cores.append(core)
    if not recs:
        return None
    return " + ".join(cores), recs


def annotate(raw, sp) -> list[dict] | None:
    """Write `core` and `qualifiers` into `raw.adjudication.ddx` (a copy of the block, so the
    job spec it came from is not mutated). Returns the records, or None when nothing matched."""
    adj = getattr(raw, "adjudication", None) or {}
    ddx = adj.get("ddx")
    if not isinstance(ddx, dict):
        return None
    got = qualifiers_for(ddx, sp)
    if got is None:
        return None
    core, recs = got
    raw.adjudication = {**adj, "ddx": {**ddx, "core": core, "qualifiers": recs}}
    return recs
