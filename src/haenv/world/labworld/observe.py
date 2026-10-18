"""Measurement layer of the lab world, with a record for every point.

    observed = clamp(truth * exp(e + sum(b_mult)) + sum(delta_add), range), printed at ndigits

`e` is the per-draw total variation (EFLM CV_total on the log scale, clipped at +-3 sigma, the
same form `build.observe_labs` uses). `b_mult` / `delta_add` are the acting perturbations of the
point (`perturb`); a decoy perturbation is recorded with `acts: false` and zero bias. Every point
keeps `{day, time, truth, e, perturbations, printed}`, so a judging fix or an audit is a
recompute from the record, never a rebuild.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

TRUTH_NDIGITS = 4          # same as build.TRUTH_NDIGITS
E_NDIGITS = 6


def print_value(v: float, ndigits: int, rng: tuple[float, float]) -> float:
    lo, hi = rng
    v = min(max(float(v), float(lo)), float(hi))
    return round(v, int(ndigits)) if ndigits else float(round(v))


def raw_observed(truth: float, e: float, perturbations) -> float:
    mult = sum(float(p.get("bias_log") or 0.0) for p in perturbations or () if p.get("acts"))
    add = sum(float(p.get("additive") or 0.0) for p in perturbations or () if p.get("acts"))
    return float(truth) * math.exp(float(e) + mult) + add


def net_bias_log(truth: float, e: float, perturbations) -> float:
    """ln(observed with the acting perturbations / observed without them), before rounding."""
    base = float(truth) * math.exp(float(e))
    return math.log(raw_observed(truth, e, perturbations) / base)


def make_record(day: int, time: str, truth: float, e: float, perturbations, ndigits: int,
                rng: tuple[float, float]) -> dict:
    truth = round(float(truth), TRUTH_NDIGITS)
    e = round(float(e), E_NDIGITS)
    pts = [dict(p) for p in perturbations or ()]
    return {"day": int(day), "time": str(time), "truth": truth, "e": e, "perturbations": pts,
            "printed": print_value(raw_observed(truth, e, pts), ndigits, rng)}


def record_consistent(rec: dict, ndigits: int, rng: tuple[float, float]) -> str | None:
    """`None` when the printed value follows from the record; else what is wrong."""
    for p in rec.get("perturbations") or ():
        if not p.get("acts") and (float(p.get("bias_log") or 0.0) != 0.0 or float(p.get("additive") or 0.0) != 0.0):
            return f"day {rec['day']}: non-acting perturbation {p.get('factor')} carries a bias"
    want = print_value(raw_observed(rec["truth"], rec["e"], rec.get("perturbations")), ndigits, rng)
    if want != rec["printed"]:
        return f"day {rec['day']}: printed {rec['printed']} != recomputed {want}"
    return None
