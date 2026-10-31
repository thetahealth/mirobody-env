"""Tier R single-fact twins for the decisive premises emitted by G3.

Twins change records on a private copy. Interval changes need a world rebuild,
so they are reported as world-level rather than certified on a gold block.
"""
from __future__ import annotations

from copy import deepcopy

from haenv.p3 import interactions
from haenv.p3.gold import accepted, g3


def twin(blk: dict, premise: str) -> dict | None:
    if premise == "interval":
        return None
    b = deepcopy(blk)
    if premise == "control":
        interactions._control(b, "on" if g3(blk).get("control") in ("off", "over") else "off")
    elif premise == "contraindication":
        from haenv.p3.history import acei_contraindicated
        b["adverse_event_record"] = [r for r in b.get("adverse_event_record", ())
                                     if not acei_contraindicated([r], b["drug"])]
    elif premise == "comorbidity":
        b.pop("comorbidity", None)
    else:
        operators = {"coverage": interactions._coverage, "administration": interactions._administration,
                     "dose_limit": interactions._dose_limit, "headroom": interactions._headroom,
                     "symptom_course": interactions._symptom_course, "symptom_onset": interactions._symptom_onset}
        operators[premise](b)
    return b


def necessity(blk: dict) -> dict[str, str]:
    base = g3(blk)
    original = set(accepted(blk, base))
    out = {}
    for premise in base["decisive"]:
        changed = twin(blk, premise)
        if changed is None:
            out[premise] = "world_level"
            continue
        gold = g3(changed)
        if gold["class"] is None:
            out[premise] = "undetermined"
            continue
        alternate = set(accepted(changed, gold))
        out[premise] = ("same" if original == alternate else "partial" if original & alternate else "flip")
    return out


def failures(blk: dict) -> dict[str, str]:
    return {premise: outcome for premise, outcome in necessity(blk).items()
            if outcome in ("partial", "same", "undetermined")}
