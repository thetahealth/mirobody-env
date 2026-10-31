"""Sourced alternative evidence paths; gold is derived before surface rendering.

Each thread is a disjunction of complete sets. Different threads remain a
conjunction. Legacy threads retain their original single set.
"""
from __future__ import annotations

from haenv.resources.regpath import load_registry, registry_cached


@registry_cached("clinical_expansion.yaml")
def registry():
    return load_registry("clinical_expansion.yaml")


def gold(meta):
    from haenv.m2.core import threads_of
    ext = meta.get("_external") or {}
    if not (ext.get("m2") or ext.get("m2_bridge")):
        return None
    return {"threads": threads_of(meta.get("ddx") or {}),
            "source": "registry/clinical_expansion.yaml#evidence_paths"}


def paths_for(component, fallback):
    row = ((registry().get("evidence_paths") or {}).get(component))
    if row is None:
        return [list(fallback)] if fallback else []
    paths = row.get("sets")
    if not row.get("source") or not isinstance(paths, list) or not paths or any(
            not isinstance(p, list) or not p or len(set(p)) != len(p) for p in paths):
        raise ValueError(f"{component}: evidence paths require sourced nonempty sets")
    return [list(p) for p in paths]


def paths_of(thread):
    return thread.get("key_signal_sets") or ([thread["key_signals"]] if thread.get("key_signals") else [])


def coverage(threads, acquired):
    """Best complete-path fraction, pooled by path size across diagnostic lines.

    A partial mixture of two alternatives never becomes a complete path.
    Empty evidence is inapplicable, never automatically complete.
    """
    got = set(acquired)
    chosen = []
    for thread in threads:
        paths = paths_of(thread)
        if paths:
            chosen.append(max(paths, key=lambda p: (len(set(p) & got) / len(p), -len(p))))
    needed = set(k for p in chosen for k in p)
    missed = sorted(needed - got)
    return {"fraction": round(len(needed & got) / len(needed), 4) if needed else None,
            "complete": not missed if needed else None, "missed": missed,
            "selected_sets": chosen}


def with_paths(ddx, thread):
    paths = paths_for(thread["component"], thread["key_signals"])
    swap = ddx.get("historical_key_signals") or {}
    return {**thread, "key_signal_sets": [[swap.get(k, k) for k in p] for p in paths],
            # Union retained for menu reachability, audit and push-item discovery.
            "key_signals": list(dict.fromkeys(swap.get(k, k) for p in paths for k in p))}
