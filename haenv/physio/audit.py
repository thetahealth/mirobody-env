"""audit.py -- distribution-shape audit of generated streams: ACF, missing rate, clip rate, saturation.

`verify.py` checks range, mean, drift, step size and density but not shape.
Any hard-clamp firing (`clip_rate > 0`) fails emission, because the clamp is a
safety net for misregistered soft parameters. ACF checks catch deterministic
periodic structure; they do not show that a stream is indistinguishable from
real data.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .bounds import ProjectionReport


class AuditError(ValueError):
    """Audit failed. Caught by the emission gate => not emitted."""


def acf(series: Sequence[float | None], lag: int) -> float:
    """Sample autocorrelation at `lag` calendar days, using only pairs where both points exist.

    Missing points are never zero-filled or compacted out (compacting shifts the
    time axis). Raises when the series is too short, has too few pairs, or has zero
    variance; it never returns 0.0 for "could not be computed".
    """
    if lag <= 0:
        raise AuditError(f"lag must be > 0, got {lag}")
    xs: list[float | None] = [None if v is None else float(v) for v in series]
    n = len(xs)
    if n <= lag + 1:
        raise AuditError(f"series length {n} is too short for autocorrelation at lag={lag}")

    pairs = [(xs[i], xs[i - lag]) for i in range(lag, n)
             if xs[i] is not None and xs[i - lag] is not None]
    if len(pairs) < lag + 2:
        raise AuditError(
            f"lag={lag} has only {len(pairs)} valid pairs (series length {n}; "
            "too many or too clustered missing values)"
        )

    a_vals = [a for a, _ in pairs]
    b_vals = [b for _, b in pairs]
    ma = sum(a_vals) / len(a_vals)
    mb = sum(b_vals) / len(b_vals)
    da = sum((a - ma) ** 2 for a in a_vals)
    db = sum((b - mb) ** 2 for b in b_vals)
    if da <= 0 or db <= 0:
        raise AuditError(
            "paired values have zero variance: a constant series has no autocorrelation "
            "(usually the noise is not wired in)"
        )
    num = sum((a - ma) * (b - mb) for a, b in pairs)
    return num / math.sqrt(da * db)


def missing_rate(series: Sequence[float | None]) -> float:
    """Fraction of `None` points."""
    xs = list(series)
    if not xs:
        raise AuditError("empty series has no missing rate")
    return sum(1 for v in xs if v is None) / len(xs)


@dataclass
class DistributionReport:
    """Distribution readings for one indicator stream."""

    indicator: str
    acf1: float
    acf7: float
    missing: float
    n: int
    #: ACF after first differencing, used by `check_policy`; `None` when it cannot be computed.
    acf1_detrended: float | None = None
    acf7_detrended: float | None = None
    #: Valid pair counts on the detrended series; thresholds scale with 1/sqrt(n).
    n_pairs_lag1: int = 0
    n_pairs_lag7: int = 0

    def check(
        self,
        acf1_range: tuple[float, float],
        acf7_max_abs: float,
        missing_range: tuple[float, float],
    ) -> list[str]:
        """Fixed-range check on the raw series (flags any trending stream; production uses `check_policy`)."""
        bad: list[str] = []
        lo, hi = acf1_range
        if not (lo <= self.acf1 <= hi):
            bad.append(f"{self.indicator}: ACF(1)={self.acf1:.3f} not in [{lo}, {hi}]")
        if abs(self.acf7) > acf7_max_abs:
            bad.append(
                f"{self.indicator}: |ACF(7)|={abs(self.acf7):.3f} > {acf7_max_abs} "
                "(strong lag-7 autocorrelation indicates a purely periodic structure)"
            )
        mlo, mhi = missing_range
        if not (mlo <= self.missing <= mhi):
            bad.append(f"{self.indicator}: missing rate={self.missing:.3f} not in [{mlo}, {mhi}]")
        return bad

    def check_policy(self, policy: "AuditPolicy", phi: float) -> list[str]:
        """Production check on the first-differenced series against the expected value from the stream's phi.

        For the AR(1) noise of `noise.correlated_noise_series`, ACF(1) of the first
        difference is `(phi-1)/2`. `phi` is the registry's `ar_phi` for streams in
        `noise_time_structure.applies_to`, else 0. Deviations are judged as
        `z = |deviation| * sqrt(valid pairs)` against `policy.z_max`.
        """
        bad: list[str] = []
        exp1 = (phi - 1.0) / 2.0
        if self.acf1_detrended is None:
            return [f"{self.indicator}: ACF cannot be computed after detrending (not measured, not a pass)"]
        if self.n_pairs_lag1 > 10:
            z1 = abs(self.acf1_detrended - exp1) * math.sqrt(self.n_pairs_lag1)
            if z1 > policy.z_max:
                bad.append(
                    f"{self.indicator}: detrended ACF(1)={self.acf1_detrended:.3f}, "
                    f"expected (phi-1)/2={exp1:.3f} (phi={phi}), "
                    f"z={z1:.2f} > {policy.z_max} (valid pairs {self.n_pairs_lag1})")
        if self.acf7_detrended is not None and self.n_pairs_lag7 > 10:
            z7 = abs(self.acf7_detrended) * math.sqrt(self.n_pairs_lag7)
            if z7 > policy.z_max:
                bad.append(
                    f"{self.indicator}: detrended |ACF(7)|={abs(self.acf7_detrended):.3f}, "
                    f"z={z7:.2f} > {policy.z_max} (valid pairs {self.n_pairs_lag7})"
                    "; strong lag-7 autocorrelation indicates a purely periodic structure")
        mmax = policy.missing_max_by_stream.get(self.indicator, policy.missing_max_default)
        if self.missing > mmax:
            bad.append(f"{self.indicator}: missing rate={self.missing:.3f} > {mmax}")
        if policy.missing_min_default is not None and self.missing < policy.missing_min_default:
            bad.append(f"{self.indicator}: missing rate={self.missing:.3f} < "
                       f"{policy.missing_min_default} (a fully complete stream is itself a signal)")
        return bad


@dataclass(frozen=True)
class AuditPolicy:
    """Thresholds for the distribution audit, read only from `distribution_audit` in `registry/physio_streams.yaml` (no defaults in code)."""

    severity: str
    z_max: float
    missing_max_default: float
    missing_max_by_stream: dict
    missing_min_default: float | None

    @staticmethod
    def from_registry(doc: dict, *, required: bool = True) -> "AuditPolicy | None":
        """`required=False` => returns `None` if the section is missing; the caller then fails when an audit is required."""
        d = (doc or {}).get("distribution_audit")
        if not isinstance(d, dict):
            if not required:
                return None
            raise AuditError(
                "registry/physio_streams.yaml has no `distribution_audit:` section; "
                "distribution audit thresholds are undefined (no built-in defaults).")
        if str(d.get("detrend")) != "first_difference":
            raise AuditError(
                f"distribution_audit.detrend={d.get('detrend')!r} is not an allowed value; "
                "only first_difference is supported (the expected (phi-1)/2 assumes it)")
        if str(d.get("severity")) not in ("warn", "gate"):
            raise AuditError(f"distribution_audit.severity={d.get('severity')!r} must be warn or gate")
        return AuditPolicy(
            severity=str(d["severity"]),
            z_max=float(d["z_max"]),
            missing_max_default=float(d["missing_max_default"]),
            missing_max_by_stream={str(k): float(v)
                                   for k, v in (d.get("missing_max_by_stream") or {}).items()},
            missing_min_default=(None if d.get("missing_min_default") is None
                                 else float(d["missing_min_default"])),
        )


def first_difference(series: Sequence[float | None]) -> list[float | None]:
    """First difference; `None` wherever either adjacent point is missing (never differences across a gap)."""
    xs = list(series)
    out: list[float | None] = [None] * len(xs)
    for i in range(1, len(xs)):
        a, b = xs[i - 1], xs[i]
        if a is not None and b is not None:
            out[i] = b - a
    return out


def describe(indicator: str, series: Sequence[float | None]) -> DistributionReport:
    """Compute the readings for one stream (raw and detrended ACF, missing rate)."""
    d1 = d7 = None
    n1 = n7 = 0
    dif = first_difference(series)
    n1 = sum(1 for i in range(1, len(dif)) if dif[i] is not None and dif[i - 1] is not None)
    n7 = sum(1 for i in range(7, len(dif)) if dif[i] is not None and dif[i - 7] is not None)
    try:
        d1, d7 = acf(dif, 1), acf(dif, 7)
    except AuditError:
        pass                    # constant or too few points after differencing => record None, never 0.0
    return DistributionReport(
        indicator=indicator,
        acf1=acf(series, 1),
        acf7=acf(series, 7),
        missing=missing_rate(series),
        n=len(series),
        acf1_detrended=d1,
        acf7_detrended=d7,
        n_pairs_lag1=n1,
        n_pairs_lag7=n7,
    )


@dataclass
class AuditSummary:
    """Summary for one case (or one batch). Non-empty `violations` => the emission gate fails it."""

    projection: ProjectionReport
    distributions: list[DistributionReport] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, object]:
        """Persisted shape. The readings go into the artifact so they can be recomputed."""
        return {
            "projection": {
                "n_total": self.projection.n_total,
                "range_rate": self.projection.range_rate,
                "slope_rate": self.projection.slope_rate,
                "clip_rate": self.projection.clip_rate,
                "clipped_by_indicator": dict(self.projection.by_indicator),
            },
            "distributions": [
                {"indicator": d.indicator, "acf1": d.acf1, "acf7": d.acf7,
                 "missing": d.missing, "n": d.n,
                 "acf1_detrended": d.acf1_detrended, "acf7_detrended": d.acf7_detrended,
                 "n_pairs_lag1": d.n_pairs_lag1, "n_pairs_lag7": d.n_pairs_lag7}
                for d in self.distributions
            ],
            "violations": list(self.violations),
        }


def assert_clean(summary: AuditSummary) -> None:
    """Emission gate: raise `AuditError` on `clip_rate > 0` or any distribution violation."""
    problems = list(summary.violations)
    if summary.projection.clip_rate > 0:
        problems.append(
            f"hard projection applied {summary.projection.n_clipped}/{summary.projection.n_total} times "
            f"(clip_rate={summary.projection.clip_rate:.4f}); "
            "the hard clamp firing means the soft mechanism is misconfigured. "
            f"By indicator: {summary.projection.by_indicator}"
        )
    if problems:
        raise AuditError("audit failed:\n  - " + "\n  - ".join(problems))


def saturation_bite(linear: float, saturated: float) -> float:
    """Relative saturation `(|linear| - |saturated|) / |linear|`, for audit use.

    Near 0 means `M` is registered too loosely; near 1 means the soft ceiling does
    almost all the work (check event density or `beta`).
    """
    if not math.isfinite(linear) or not math.isfinite(saturated):
        raise AuditError(f"saturation_bite inputs must be finite: {linear!r} / {saturated!r}")
    if linear == 0:
        return 0.0
    return (abs(linear) - abs(saturated)) / abs(linear)
