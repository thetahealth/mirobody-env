"""bounds.py -- physiological constraints: soft saturation -> transform-domain update ->
hard projection of value and slope.

Bounded or non-negative indicators update in log or logit coordinates, so they stay in
range by construction. Violations are counted on the pre-projection proposal `y_hat`;
`clip_rate` must be 0 at the emission gate (`audit.assert_clean`).

`BoundSpec.lo/hi` is the physiologically possible range, not a reference interval: an
HbA1c of 14% is real and must survive.

SYNTHETIC, evaluation only; not medical advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Transform = Literal["identity", "log", "logit"]

_EXP_CAP = 60.0  # cap on exp() input, avoids overflow
_EPS = 1e-9       # open-interval margin for logit


class BoundError(ValueError):
    """Bound spec missing or invalid. Deliberately has no default value."""


@dataclass(frozen=True)
class BoundSpec:
    """Physiological constraint for one indicator.

    `lo`/`hi` is the possible range; `max_step` (> 0) is the maximum day-over-day change.
    """

    indicator: str
    lo: float
    hi: float
    max_step: float
    transform: Transform = "identity"
    source: str = ""
    #: Operating point, required for non-identity transforms: noise is converted between the
    #: transform domain and raw units at this declared point (e.g. spo2 at 97.4), not over the
    #: whole range.
    at: float | None = None

    def __post_init__(self) -> None:
        if not str(self.indicator).strip():
            raise BoundError("indicator must not be empty")
        for name in ("lo", "hi", "max_step"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise BoundError(f"{self.indicator}: {name} must be finite, got {v!r}")
        if not self.lo < self.hi:
            raise BoundError(f"{self.indicator}: requires lo < hi, got {self.lo} / {self.hi}")
        if self.max_step <= 0:
            raise BoundError(
                f"{self.indicator}: max_step must be > 0 (without a slope limit the indicator "
                "can cross its full range in one day)"
            )
        if self.transform not in ("identity", "log", "logit"):
            raise BoundError(f"{self.indicator}: unknown transform {self.transform!r}")
        if self.transform == "log" and self.lo <= 0:
            raise BoundError(f"{self.indicator}: log transform requires lo > 0, got {self.lo}")
        if self.transform != "identity":
            if self.at is None:
                raise BoundError(
                    f"{self.indicator}: transform={self.transform!r} requires `at` (reference point). "
                    "The raw-unit step of transform-domain noise depends on the value; "
                    "use a citable reference point such as the measured median, noted in source")
            if not (self.lo < float(self.at) < self.hi):
                raise BoundError(
                    f"{self.indicator}: at={self.at} must lie strictly inside ({self.lo}, {self.hi})")
        elif self.at is not None:
            raise BoundError(
                f"{self.indicator}: transform=identity takes no `at` (the reference point "
                "would have no effect)")
        if not str(self.source).strip():
            raise BoundError(f"{self.indicator}: source must not be empty; every bound needs a citation")


@dataclass
class ProjectionReport:
    """Projection statistics, all counted on the pre-projection proposal `y_hat`."""

    n_total: int = 0
    n_range_violation: int = 0     # y_hat fell outside [lo, hi]
    n_slope_violation: int = 0     # |y_hat - y_prev| exceeded max_step
    n_clipped: int = 0             # projection actually changed the value (hard clamp fired)
    by_indicator: dict[str, int] = field(default_factory=dict)

    @property
    def range_rate(self) -> float:
        return self.n_range_violation / self.n_total if self.n_total else 0.0

    @property
    def slope_rate(self) -> float:
        return self.n_slope_violation / self.n_total if self.n_total else 0.0

    @property
    def clip_rate(self) -> float:
        return self.n_clipped / self.n_total if self.n_total else 0.0


def to_transform(y: float, spec: BoundSpec) -> float:
    """Value -> transform-domain coordinate."""
    if spec.transform == "identity":
        return y
    if spec.transform == "log":
        if y <= 0:
            raise BoundError(f"{spec.indicator}: log transform requires y > 0, got {y}")
        return math.log(y)
    # logit: normalize to (0,1) first, then take the logit
    span = spec.hi - spec.lo
    p = min(1.0 - _EPS, max(_EPS, (y - spec.lo) / span))
    return math.log(p / (1.0 - p))


def from_transform(x: float, spec: BoundSpec) -> float:
    """Transform-domain coordinate -> value. Inverse of `to_transform`."""
    if spec.transform == "identity":
        return x
    if spec.transform == "log":
        return math.exp(min(_EXP_CAP, x))
    p = 1.0 / (1.0 + math.exp(min(_EXP_CAP, -x)))
    return spec.lo + p * (spec.hi - spec.lo)


def raw_sd_to_transform_sd(sd_raw: float, at: float, spec: BoundSpec) -> float:
    """Convert a raw-unit noise scale into transform-domain scale (first order, at `at`).

    `loading` / `idio_sd` are in transform-domain units, since noise is added in that domain:

        identity : sd
        log      : sd / at
        logit    : sd * span / ((at - lo) * (hi - at))
    """
    if not (math.isfinite(sd_raw) and sd_raw > 0):
        raise BoundError(f"{spec.indicator}: sd_raw must be > 0, got {sd_raw!r}")
    if spec.transform == "identity":
        return sd_raw
    if not (spec.lo < at < spec.hi):
        raise BoundError(f"{spec.indicator}: reference point {at} is not inside ({spec.lo}, {spec.hi})")
    if spec.transform == "log":
        return sd_raw / at
    span = spec.hi - spec.lo
    return sd_raw * span / ((at - spec.lo) * (spec.hi - at))


def induced_step(budget: float, spec: BoundSpec) -> float:
    """Maximum raw-unit daily step induced by a transform-domain noise budget, at `at`.

    Makes `max_step` (raw units) comparable with `noise.bound` (transform units). Takes the
    larger of both directions, since log and logit are asymmetric.
    """
    if not (math.isfinite(budget) and budget >= 0):
        raise BoundError(f"{spec.indicator}: budget must be finite and >= 0, got {budget!r}")
    if spec.transform == "identity":
        return budget
    at = float(spec.at)                     # __post_init__ guarantees non-None when non-identity
    x = to_transform(at, spec)
    up, dn = from_transform(x + budget, spec), from_transform(x - budget, spec)
    return max(abs(up - at), abs(at - dn))


def project(
    proposal: float,
    prev: float | None,
    spec: BoundSpec,
    report: ProjectionReport | None = None,
) -> float:
    """Project the proposal into the feasible set, recording violations before projection.

    `prev=None` marks the first point (no slope check); `report=None` means no tracking.
    Returns a value with `lo <= y <= hi` and, when `prev` is given, `|y - prev| <= max_step`.
    """
    if not math.isfinite(proposal):
        raise BoundError(f"{spec.indicator}: proposal must be finite, got {proposal!r}")

    if report is not None:
        report.n_total += 1
        if proposal < spec.lo or proposal > spec.hi:
            report.n_range_violation += 1
        if prev is not None and abs(proposal - prev) > spec.max_step:
            report.n_slope_violation += 1

    y = min(spec.hi, max(spec.lo, proposal))
    if prev is not None:
        # Range first, then slope.
        y = min(prev + spec.max_step, max(prev - spec.max_step, y))
        # The slope clamp can leave the range when prev is pinned to a bound; re-clamp.
        y = min(spec.hi, max(spec.lo, y))

    if report is not None and y != proposal:
        report.n_clipped += 1
        report.by_indicator[spec.indicator] = report.by_indicator.get(spec.indicator, 0) + 1
    return y
