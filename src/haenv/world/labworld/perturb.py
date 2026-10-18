"""Factor catalogue of the lab world (`registry/labworld.yaml:factors`).

A factor is something visible on a lab report or in the record around it: a dose change, a
weight change, an ACEI start, a meal before a "fasting" draw, a delayed specimen, a bedside
meter, a creatinine method. Each has a `kind` (true_change / preanalytical / method /
background) and the analytes it acts on; on any other analyte it is a decoy with zero bias.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.world.labworld import tables

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
    return [n for n, f in catalogue().items() if not f.get("diversity_only") and
            f.get("kind") == k and analyte in (f.get("acts_on") or ())]


def decoys_for(analyte: str) -> list[str]:
    """Factors that act on another analyte and not on this one (design 4.2: a never-acting note
    carries no confusion, so pure background factors are not decoys); a factor that lists `decoy_on` is a
    decoy only on those analytes."""
    from haenv.world.generation.world_knobs import background_enabled
    background = background_enabled()
    out = []
    for n, f in catalogue().items():
        if f.get("diversity_only"):
            continue
        on = list(f.get("acts_on") or ())
        # The scope restriction was added with the background layer; off replays its old catalogue.
        only = f.get("decoy_on") if background else None
        if on and analyte not in on and f.get("kind") != "background" and (only is None or analyte in only):
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


def stability_h(analyte: str) -> float | None:
    """The wait before centrifugation (hours) within which `analyte`'s value is stable
    (`registry/labworld.yaml:specimen_stability`); None when the table has no bound for it."""
    e = (tables().get("specimen_stability") or {}).get(analyte)
    return float(e["max_h"]) if isinstance(e, dict) and e.get("max_h") is not None else None


def no_effect_violation(p: dict, analyte: str) -> str | None:
    """Why a perturbation recorded as not acting would act under the world's model (None = it does
    not): its factor is registered on `analyte`, a wait before centrifugation is at or past the
    analyte's stability bound, or a cooked-meat meal sits closer to the draw than the evening meal two
    days before. A method that runs on every draw (`method_throughout`) moves both readings alike and
    acts on no change."""
    f = str(p.get("factor"))
    params = p.get("params") or {}
    if f == "delayed_processing":
        lim, h = stability_h(analyte), params.get("hours")
        if lim is None:
            return f"{f} acts on {analyte}" if acts_on(f, analyte) else None
        if h is not None and float(h) >= lim:
            return f"wait of {float(h):g} h before centrifugation is at or past the {lim:g} h stability bound"
        return None
    if f == "meat_meal":
        rests = int(params.get("days_before", 0)) >= 2 or params.get("meal") == "dinner"
        return None if rests else "cooked-meat meal within the window in which it acts"
    if params.get("method_throughout"):
        return None
    if acts_on(f, analyte):
        return f"{f} acts on {analyte}"
    return None
