"""noise.py -- low-rank correlated observation noise.

    eps_k(t) = Sum_g L[k,g] * z_g(t)  +  eta_k(t)

Indicators co-move through shared factors z_g (global / cardiovascular /
metabolic ...) plus idiosyncratic noise eta_k. This is co-movement from a
shared perturbation, not comorbidity interaction. All draws are pure functions
of a path through the counter RNG in `haenv/rng.py` (no `random` import), so
adding one stream never shifts another.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..rng import unit

# Box-Muller needs u in (0, 1]; `unit()` can return 0.0.
_U_FLOOR = 1e-12


class NoiseSpecError(ValueError):
    """A noise spec is missing or invalid. There is no default value."""


@dataclass(frozen=True)
class NoiseSpec:
    """The noise spec for one indicator.

    `loadings` (`{factor: loading}`) may be empty, but `idio_sd` must be > 0,
    unless `external` names an upstream renderer that supplies the noise.
    """

    indicator: str
    loadings: Mapping[str, float]
    idio_sd: float
    trunc: float = 3.0  # cutoff for the standardized deviate: noise is bounded by construction
    #: Upstream calibrated renderer that supplies this stream's noise; this layer then adds none.
    external: str = ""

    def __post_init__(self) -> None:
        if not str(self.indicator).strip():
            raise NoiseSpecError("indicator must not be empty")
        if self.external:
            if self.idio_sd != 0 or any(float(w) != 0 for w in self.loadings.values()):
                raise NoiseSpecError(
                    f"{self.indicator}: noise is declared external ({self.external}) but "
                    "this layer still adds some; that counts the variance twice")
            return
        if not (math.isfinite(self.idio_sd) and self.idio_sd > 0):
            raise NoiseSpecError(
                f"{self.indicator}: idio_sd must be finite and > 0, got {self.idio_sd!r} "
                "(with no loadings the series would be deterministic)"
            )
        for g, w in self.loadings.items():
            if not math.isfinite(float(w)):
                raise NoiseSpecError(f"{self.indicator}: loading on factor {g} is not finite: {w!r}")
        if not (math.isfinite(self.trunc) and self.trunc > 0):
            raise NoiseSpecError(
                f"{self.indicator}: trunc must be > 0. Unbounded noise under a hard slope limit gets clamped,\n"
                "and audit.assert_clean fails on clip_rate > 0"
            )

    @property
    def marginal_sd(self) -> float:
        """Marginal standard deviation, sqrt(Sum wg^2 + idio_sd^2) (untruncated)."""
        return math.sqrt(sum(float(w) ** 2 for w in self.loadings.values())
                         + float(self.idio_sd) ** 2)

    @property
    def bound(self) -> float:
        """Supremum of a single day's noise; `max_step` must be >= 2*bound plus the signal's own step, or the hard clamp fires."""
        return self.trunc * (sum(abs(float(w)) for w in self.loadings.values()) + self.idio_sd)


def normal_dev(*path) -> float:
    """Path -> standard normal deviate via Box-Muller; a pure function of the path."""
    u1 = max(_U_FLOOR, unit(*path, "bm0"))
    u2 = unit(*path, "bm1")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _truncated(z: float, trunc: float) -> float:
    """Clip a standardized deviate to +-trunc (clipping, not resampling, keeps the value a pure function of the path)."""
    return min(trunc, max(-trunc, z))


def factor_value(case_id: str, group: str, t: int) -> float:
    """The shared factor `z_g(t)`; every indicator in the group reads the same value."""
    return normal_dev(case_id, "factor", group, t)


def correlated_noise(
    case_id: str,
    t: int,
    specs: Sequence[NoiseSpec],
) -> dict[str, float]:
    """Correlated noise `{indicator: eps}` for a group of indicators on a single day."""
    if not specs:
        raise NoiseSpecError("specs is empty: no indicator to add noise to")

    groups = sorted({g for s in specs for g in s.loadings})
    z = {g: factor_value(case_id, g, t) for g in groups}

    out: dict[str, float] = {}
    for s in specs:
        if s.indicator in out:
            raise NoiseSpecError(f"duplicate indicator: {s.indicator}")
        shared = sum(float(w) * _truncated(z[g], s.trunc) for g, w in s.loadings.items())
        idio = s.idio_sd * _truncated(normal_dev(case_id, "idio", s.indicator, t), s.trunc)
        out[s.indicator] = shared + idio
    return out


def load_noise_specs(raw: Mapping[str, object]) -> list[NoiseSpec]:
    """Build `NoiseSpec`s from a registry fragment. Any non-conforming entry raises."""
    if not isinstance(raw, Mapping) or not raw:
        raise NoiseSpecError("noise registry must be a non-empty mapping")
    specs: list[NoiseSpec] = []
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            raise NoiseSpecError(f"{name}: entry must be a mapping")
        if "idio_sd" not in item:
            raise NoiseSpecError(f"{name}: missing idio_sd; no defaults are provided")
        loadings = item.get("loadings") or {}
        if not isinstance(loadings, Mapping):
            raise NoiseSpecError(f"{name}: loadings must be a mapping")
        specs.append(
            NoiseSpec(
                indicator=str(name),
                loadings={str(g): float(w) for g, w in loadings.items()},
                idio_sd=float(item["idio_sd"]),
                trunc=float(item.get("trunc", 3.0)),
            )
        )
    return specs


def correlated_noise_series(
    case_id: str,
    days: Sequence[int],
    specs: Sequence[NoiseSpec],
    phi: Mapping[str, float] | None = None,
    scale: Mapping[str, float] | None = None,
) -> dict[str, dict[int, float]]:
    """Temporally correlated observation noise across a sequence of days (AR(1) on top of `correlated_noise`).

        a      = phi^dt                         # dt = days since the previous sample
        eps(t) = a*eps(prev) + sqrt(1-a^2)*u(t)  # u(t) = correlated_noise for that day

    `sqrt(1-a^2)` keeps the marginal variance unchanged; phi = 0 reproduces
    `correlated_noise` exactly. `phi` absent for an indicator means 0 (no
    extrapolation). `scale` multiplies each indicator's noise, e.g. to express it as
    a fraction of body weight. `days` must be ascending.
    """
    ds = [int(d) for d in days]
    if ds != sorted(ds):
        raise NoiseSpecError("days must be in ascending order (the AR recursion depends on order)")
    out: dict[str, dict[int, float]] = {s.indicator: {} for s in specs}
    prev_day: int | None = None
    prev_eps: dict[str, float] = {}
    for d in ds:
        u = correlated_noise(case_id, d, specs)
        for s in specs:
            k = s.indicator
            p = float((phi or {}).get(k, 0.0))
            if not (0.0 <= p < 1.0):
                raise NoiseSpecError(f"{k}: φ must be in [0, 1), got {p!r}")
            if prev_day is None or p == 0.0:
                e = u[k]
            else:
                a = p ** max(1, d - prev_day)
                e = a * prev_eps[k] + math.sqrt(max(0.0, 1.0 - a * a)) * u[k]
            prev_eps[k] = e
            out[k][d] = e * float((scale or {}).get(k, 1.0))
        prev_day = d
    return out


# ---------------------------------------------------------------- weight noise shape
#: Weekly weight rhythm, as a fraction of body mass by weekday (Mon..Sun): the shape of the
#: men's row of Turicchi 2020 Table 2 (PMID 32353079, PMC7192384; Mon +0.256 ... Fri -0.156 %,
#: peak to trough 0.41 %), centred and rescaled to the study's overall 0.35 %; Monday
#: heaviest, Friday lightest. The same table gives 0.29 % for women and 0.31 % / 0.26 % for
#: obesity classes 1 / 2-3; one shape and one amplitude are used for every case. Sums to zero
#: over the week (to 2e-5).
WEIGHT_WEEKDAY_FRAC = (0.00226, 0.00057, -0.00060, -0.00099, -0.00124, -0.00093, 0.00091)


def weight_weekday(case_id: str, day: int) -> int:
    """Weekday (0 = Monday) of course day `day`; the same calendar phase as
    `build._weighed_on`, where days 5 and 6 are the weekend."""
    import hashlib
    phase = int(hashlib.sha256(str(case_id).encode()).hexdigest()[:8], 16)
    return (int(day) + phase) % 7


def weight_noise_shaped(case_id: str, day: int, eps: float, mass: float,
                        sd_frac: float) -> float:
    """The weight observation noise on `day` (kg): `eps` (at the calibrated marginal SD
    `sd_frac x mass`) plus the weekly rhythm, with `eps` scaled down so the rhythm's
    variance fits inside the calibrated total. A pure function of `(case_id, day)`."""
    week_var = sum(f * f for f in WEIGHT_WEEKDAY_FRAC) / len(WEIGHT_WEEKDAY_FRAC)
    keep = math.sqrt(max(0.0, 1.0 - week_var / max(1e-12, sd_frac * sd_frac)))
    return eps * keep + mass * WEIGHT_WEEKDAY_FRAC[weight_weekday(case_id, day)]
