"""superpose.py -- multi-event superposition with a tanh soft ceiling.

    u     = Sum_e beta_e * g_e(t)        (including events in their decay tail)
    Delta = M * tanh(u / M)

`M` is the largest plausible offset an indicator can take from overlapping
events: near-additive when `|u| << M`, bounded by `M` otherwise, so
overlapping events do not reach the hard clamp. `M` has no default and must
be > 0.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import math
from typing import Iterable

# tanh(x) equals +-1 in double precision beyond |x| = 20.
_TANH_SAT = 20.0


class SaturationError(ValueError):
    """The saturation ceiling is missing or invalid (no default)."""


def saturate(contributions: Iterable[float], m: float) -> float:
    """Sum the signed contributions (indicator units) and apply the soft
    ceiling `m` (> 0). Returns a value with `|result| <= m` (exactly +-m at
    extreme overlap, in double precision)."""
    if not (isinstance(m, (int, float)) and math.isfinite(m) and m > 0):
        raise SaturationError(f"saturation cap m must be finite and > 0: {m!r}; no defaults are provided")

    u = 0.0
    for c in contributions:
        c = float(c)
        if not math.isfinite(c):
            raise SaturationError(f"contribution must be finite, got {c!r}")
        u += c

    x = u / m
    if x >= _TANH_SAT:
        return m
    if x <= -_TANH_SAT:
        return -m
    return m * math.tanh(x)


def linear_sum(contributions: Iterable[float]) -> float:
    """Unsaturated sum, for auditing how much the ceiling absorbed; production
    always uses `saturate`."""
    return sum(float(c) for c in contributions)
