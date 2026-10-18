"""Optional, read-only Pack capabilities, outside the frozen scoring skeleton.

Plugins return a ReadoutPack through the haenv.pack_readouts entry-point group.
No callback participates in generation, grading or solver payloads.
"""
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Callable
from haenv.app.packs import Pack


@dataclass
class ReadoutPack(Pack):
    dimensions: dict[str, Callable] = field(default_factory=dict)  # saved case -> category
    base_key: Callable | None = None                             # saved case -> base identity
    raw_correct: Callable | None = None                          # eval row -> bool / None
    main_score: Callable | None = None                           # eval rows -> score / None


def declaration(name: str) -> ReadoutPack:
    matches = {ep.value: ep for ep in entry_points(group="haenv.pack_readouts") if ep.name == name}
    if len(matches) != 1:
        raise ValueError(f"Pack {name!r} needs exactly one haenv.pack_readouts declaration")
    pack = next(iter(matches.values())).load()()
    if not isinstance(pack, ReadoutPack) or pack.name != name:
        raise ValueError(f"Invalid readout declaration for {name!r}")
    return pack


def path(case: dict, dotted: str):
    value = case
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def history_label(case: dict):
    adj = case.get("adjudication") or {}
    for value in [*adj.values(), adj]:
        hd = value.get("history_dependency") if isinstance(value, dict) else None
        if hd:
            return hd.get("active")
    return None


def category(value) -> str:
    import json
    return "unmarked" if value is None else (value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False))


def diversity(pack: ReadoutPack, cases: dict, threshold: float) -> dict:
    from collections import Counter
    readings = {}
    for name, fn in pack.dimensions.items():
        counts = Counter(category(fn(c)) for c in cases.values())
        readings[name] = {"kinds": len(counts), "counts": dict(sorted(counts.items())),
                          "max_share": max(counts.values()) / len(cases) if cases else None}
    warnings = [name for name, r in readings.items() if r["max_share"] is not None and r["max_share"] > threshold]
    bases = {category(pack.base_key(c)) for c in cases.values()} if pack.base_key else None
    return {"n": len(cases), "base_count": len(bases) if bases is not None else None,
            "dimensions": readings, "threshold": threshold, "warnings": warnings}


def seed_readout(batch_dirs, cfg: dict) -> dict:
    """Report the recorded seed's visibility; never read or emit the seed itself."""
    import hashlib
    import json
    from pathlib import Path
    import re
    public = hashlib.sha256(b"").hexdigest()
    hashes, unknown = set(), False
    for directory in batch_dirs:
        metadata = Path(directory) / "batch.json"
        if not metadata.is_file():
            unknown = True
            continue
        digest = (json.loads(metadata.read_text()).get("pack") or {}).get("seed_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            unknown = True
        else:
            hashes.add(digest)
    kinds = {"public" if digest == public else "private" for digest in hashes}
    if unknown or not hashes:
        kinds.add("unknown")
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
    official = cfg.get("board", {}).get("official", False)
    warnings = (["Official board uses a public seed; generate a private-seed pack for an official evaluation."]
                if official and public in hashes else [])
    return {"kind": kind, "seed_sha256": sorted(hashes), "official": official, "warnings": warnings}
