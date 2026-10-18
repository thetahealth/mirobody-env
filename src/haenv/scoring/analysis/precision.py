"""Shared exact/display precision state and deterministic means."""
from __future__ import annotations

import math as _math


def _fsum_mean(xs, nd: int = 3):
    return _rd(_math.fsum(xs) / len(xs), nd) if xs else None


_EXACT = {"on": False}


def _rd(x, nd: int):
    """`round(x, nd)` for display; the unrounded value inside `exact_precision()`."""
    return x if _EXACT["on"] else round(x, nd)


class exact_precision:
    """Context in which the scoring aggregation (`rank_ddx`) keeps full float precision."""

    def __enter__(self):
        self._prev = _EXACT["on"]
        _EXACT["on"] = True
        return self

    def __exit__(self, *exc):
        _EXACT["on"] = self._prev
        return False


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return _fsum_mean(xs)
