"""Factor catalogue of the lab world (`registry/labworld.yaml:factors`).

A factor is something visible on a lab report or in the record around it: a dose change, a
weight change, an ACEI start, a meal before a "fasting" draw, a delayed specimen, a bedside
meter, a creatinine method. Each has a `kind` (true_change / preanalytical / method /
background) and the analytes it acts on; on any other analyte it is a decoy with zero bias.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from . import tables

KINDS = ("true_change", "preanalytical", "method", "background")


def catalogue() -> dict[str, dict]:
    return dict(tables().get("factors") or {})


def factor(name: str) -> dict:
    f = catalogue().get(name)
    if not isinstance(f, dict):
        raise KeyError(f"labworld: unknown factor {name!r}")
    return f


def kind(name: str) -> str:
    return str(factor(name)["kind"])


def acts_on(name: str, analyte: str) -> bool:
    return analyte in (factor(name).get("acts_on") or ())


def factors_of(analyte: str, k: str) -> list[str]:
    """Factors of kind `k` that act on `analyte`, in table order."""
    return [n for n, f in catalogue().items() if f.get("kind") == k and analyte in (f.get("acts_on") or ())]


def decoys_for(analyte: str) -> list[str]:
    """Factors that act on another analyte and not on this one (design 4.2: a never-acting note
    carries no confusion, so pure background factors are not decoys)."""
    out = []
    for n, f in catalogue().items():
        on = list(f.get("acts_on") or ())
        if on and analyte not in on and f.get("kind") != "background":
            out.append(n)
    return out


def effect_range(name: str) -> tuple[float, float]:
    f = factor(name)
    lo, hi = f["effect"]
    return float(lo), float(hi)


def typical_effect_log(name: str) -> float:
    """|ln(biased/true)| at the literature point `typical: [true, biased]` (feasibility check)."""
    import math
    a, b = factor(name)["typical"]
    return abs(math.log(float(b) / float(a)))
