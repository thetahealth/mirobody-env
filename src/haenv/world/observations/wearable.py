"""Wearable-stream realism: empirical calibration, non-wear masking and derived streams.

`CALIBRATION` holds per-stream shape parameters (log or linear scale, spread, lag-1
persistence, availability, non-wear transitions); `ar1` renders a stationary AR(1)
series; `worn_on` draws a bursty two-state non-wear mask; the derived-stream helpers
compute streams that are functions of another stream or of patient facts. Every draw is
`blake2b(path)`, so values are pure functions of (case_id, stream, day).

The parameters were measured on the LifeSnaps dataset (Yfantidou et al., *Scientific
Data* 9:663, 2022; doi:10.5281/zenodo.7229547; CC BY 4.0; attribution in `NOTICE.md`),
the daily Fitbit table cut into 28-day windows. Only aggregate shape statistics are used,
never levels: the reference cohort is healthy and haenv's is metabolic. `steps` is
measured on full-wear days only (days that also carry `resting_hr`), because the corpus
records a partly worn day as a low step count rather than as missing. `stress_score` is
reversed from the corpus convention.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import haenv.world.observations.streams as _streams

#: Longest run of consecutive missing days (the reference tops out at 17 over 28 days);
#: longer holes come only from explicit noise windows.
MAX_GAP_DAYS = 17

#: Availability floor. Availability is drawn from `case_id` alone, so a stream's presence
#: carries no label information.
AVAIL_MIN = 0.20


def _unit(*path: object) -> float:
    """Deterministic uniform draw in [0, 1) keyed by an arbitrary path."""
    key = "|".join(str(p) for p in path).encode("utf-8")
    h = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def _gauss(*path: object) -> float:
    """Deterministic standard normal draw, Box-Muller over two independent paths."""
    u1 = min(max(_unit("bm0", *path), 1e-12), 1.0 - 1e-12)
    u2 = _unit("bm1", *path)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@dataclass(frozen=True)
class Calibration:
    """Empirical shape parameters for one wearable stream.

    `sigma` and `phi` are the within-window SD and lag-1 autocorrelation on the working scale
    (dimensionless on a log scale). `avail` is the share of subjects reporting the stream.
    The stationary non-wear rate is `p_off_worn / (p_off_worn + 1 - p_off_off)`.
    """

    log_scale: bool
    sigma: float
    phi: float
    avail: float
    p_off_worn: float
    p_off_off: float

    @property
    def steady_off(self) -> float:
        denom = self.p_off_worn + (1.0 - self.p_off_off)
        return self.p_off_worn / denom if denom > 0 else 0.0


#: Per-stream parameters, keyed by haenv stream name. Reference columns: resting_hr,
#: rmssd (hrv), steps, minutesAsleep (sleep_hours), stress_score (reversed),
#: nightly_temperature centred per subject (skin_temp), spo2. Exact zeros in steps, rmssd,
#: minutesAsleep and stress_score are the corpus's "not recorded" marker and count as missing.
#: Streams absent from this table use the global default behaviour.
#: W3 (world 1.4) refit of non-wear, review: pending. The 28-day windows of the W3 instrument
#: (every calendar day between a person's first and last record is a nominal day)
#: put LifeSnaps' per-person non-wear at resting_hr 0.119, steps
#: 0.083, sleep 0.268, hrv 0.310, spo2 0.366, skin_temp 0.259, against 0.06-0.17 in the board
#: world in the aggregate non-wear comparison. `p_off_worn` of the six streams was raised so
#: the realised rates match (stationary rate x 1.25-1.93, fitted by simulating `worn_on` and
#: reading it with the instrument); `avail` is the share of the LifeSnaps calibration set that
#: reports the stream (`wear_presence.real`). `p_off_off` and the spreads are unchanged.
CALIBRATION: dict[str, Calibration] = {
    # W3 round 13 (review: pending): resting_hr sigma 1.8707 -> 1.971, phi 0.8055 -> 0.7664 and
    # hrv phi 0.2308 -> 0.15, fitted jointly with `SHARED_MA` and resting HR's `SLOW_LOADING`
    # on the LifeSnaps calibration split using a joint moving-average calibration.
    "resting_hr":   Calibration(log_scale=False, sigma=1.971, phi=0.7664,
                                avail=1.000, p_off_worn=0.0536, p_off_off=0.5972),
    # hrv phi 0.15 -> 0.08 in W3 round 19: the mean of the per-person persistence (`PHI_SPREAD`),
    # set so the window lag-1 median stays near LifeSnaps (0.26 vs 0.23); review: pending.
    "hrv":          Calibration(log_scale=True,  sigma=0.2110, phi=0.08,
                                avail=0.620, p_off_worn=0.3945, p_off_off=0.4378),
    "steps":        Calibration(log_scale=True,  sigma=0.5675, phi=0.1066,
                                avail=1.000, p_off_worn=0.02, p_off_off=0.6957),
    "stress_score": Calibration(log_scale=False, sigma=5.0891, phi=0.1929,
                                avail=0.352, p_off_worn=0.3355, p_off_off=0.5621),
    # W3 round 16 (review: pending): sleep_hours phi -0.0659 -> -0.09 and spo2 phi -0.0369 ->
    # 0.13. Read in 28-day windows, the lag-1 autocorrelation is -0.112 (sleep) and 0.045 (spo2)
    # in the LifeSnaps calibration split against -0.085 and -0.085 in the board world; acf1_sleep
    # ranked second and acf1_spo2 seventh in the persist group
    # in a permutation-importance comparison. After: -0.098, 0.057.
    "sleep_hours":  Calibration(log_scale=False, sigma=1.2631, phi=-0.09,
                                avail=0.980, p_off_worn=0.2277, p_off_off=0.4705),
    # W3 round 15 (review: pending): skin_temp p_off_worn 0.2789 -> 0.33. The LifeSnaps
    # calibration split misses skin temperature on 21.4% of a median window's days (per-person
    # median 0.259); the board world read 17.9% (per person 0.20), with too many fully observed
    # windows (12.6% vs 8.3%), and off_skin_temp ranked second in the missing group
    # in a permutation-importance comparison. Calibration trials:
    # 0.33 improved the missing group on all four CV seeds, 0.37 did not.
    "skin_temp":    Calibration(log_scale=False, sigma=0.4373, phi=0.1128,
                                avail=0.890, p_off_worn=0.33, p_off_off=0.4670),
    "spo2":         Calibration(log_scale=False, sigma=0.7454, phi=0.13,
                                avail=0.380, p_off_worn=0.4078, p_off_off=0.5064),
}


#: Streams whose parameters are reasoned rather than measured, with the reason.
UNANCHORED = {
    "body_temp":      "the reference corpus measures skin temperature at the wrist, "
                      "not core body temperature, and NHANES carries no wearable "
                      "series at all; distribution shape is reasoned from the "
                      "sensor's stated resolution, not measured",
    "activity_index": "a composite index with no public counterpart, so there is "
                      "nothing to calibrate against even in principle",
}


def is_calibrated(name: str) -> bool:
    return name in CALIBRATION


#: Floor on the non-wear rate tolerance, so a very short window is not held to a
#: band narrower than one day in fifty.
OFF_RATE_TOL_FLOOR = 0.02
#: Width of the sampling bands below, in standard errors (about 0.3% false alarms).
BAND_SE = 3.0


def off_rate_tolerance(name: str, span: int, rate: float | None = None,
                       case_id: str | None = None) -> float:
    """Allowed deviation of the realised non-wear rate from the declared one.

    `BAND_SE` standard errors of a two-state chain's rate over `span` days:
    pi (1 - pi) / span * (1 + lam) / (1 - lam), with lam = p_off_off - p_off_worn; for a
    patient in the long-gap class (`case_id`), plus the variance of the episode share, whose
    runs average `_episode_mean_len()` days.
    """
    cal = CALIBRATION[name]
    pi = cal.steady_off if rate is None else rate
    # Persistence of the chain that is actually rendered: the stream's own chain
    # (`OWN_P_OFF_OFF`) can be more persistent than the marginal `p_off_off` (W3 round 3: steps
    # 0.85 vs 0.70), and longer runs make the realised rate noisier.
    lam = max(cal.p_off_off, OWN_P_OFF_OFF.get(name, cal.p_off_off)) - cal.p_off_worn
    if span <= 0 or not (-1.0 < lam < 1.0):
        return 1.0
    var = pi * (1.0 - pi) / span * (1.0 + lam) / (1.0 - lam)
    e = episode_off_rate(case_id) if case_id is not None else 0.0
    if e > 0:
        var += e * (1.0 - e) / span * (2.0 * _episode_mean_len() - 1.0)
    var += DEVICE_OFF_DAY * (1.0 - DEVICE_OFF_DAY) / span       # independent single days
    return max(OFF_RATE_TOL_FLOOR, BAND_SE * math.sqrt(var))


#: Person-to-person spread of a stream's lag-1 persistence (W3 round 19, review: pending): the
#: patient's phi is `CALIBRATION.phi` + `PHI_SPREAD[name]` x N(0, 1), cut to [-0.9, 0.95] and
#: keyed on (case_id, stream) only. hrv: read in 28-day windows, LifeSnaps (calibration split)
#: has HRV lag-1 quantiles (10/25/50/75/90) -0.14 / -0.03 / 0.23 / 0.45 / 0.57, the board world
#: 0.02 / 0.12 / 0.28 / 0.44 / 0.56 (too uniform). Spread 0.2 with `CALIBRATION["hrv"].phi`
#: 0.15 -> 0.08, which holds the median, gives -0.11 / 0.08 / 0.26 / 0.41 / 0.56 in calibration
#: trials. Steps' window spread already
#: matches (IQR 0.30 vs 0.30), so it has none.
PHI_SPREAD: dict[str, float] = {"hrv": 0.2}


def person_phi(case_id: str | None, name: str) -> float:
    """The patient's lag-1 persistence for `name` (the calibrated value when there is no spread
    or no case)."""
    phi = CALIBRATION[name].phi
    sp = PHI_SPREAD.get(name, 0.0)
    if sp and case_id is not None:
        z = max(-2.5, min(2.5, _gauss(case_id, "person_phi", name)))
        phi = max(-0.9, min(0.95, phi + sp * z))
    return phi


def mean_shift_tolerance(name: str, va: list[float], vb: list[float], case_id: str | None = None) -> float:
    """`BAND_SE` standard errors of the difference between two segment means, using the pooled
    within-segment SD and the effective sample size n (1 - phi) / (1 + phi), with the patient's
    own phi (`person_phi`) when `case_id` is given.
    """
    phi = person_phi(case_id, name)
    shrink = (1.0 - phi) / (1.0 + phi) if -1.0 < phi < 1.0 else 1.0

    def _ss(xs: list[float]) -> float:
        mu = sum(xs) / len(xs)
        return sum((x - mu) ** 2 for x in xs)

    dof = len(va) + len(vb) - 2
    if dof <= 0:
        return float("inf")
    sd = math.sqrt((_ss(va) + _ss(vb)) / dof)
    n_a, n_b = max(1.0, len(va) * shrink), max(1.0, len(vb) * shrink)
    return BAND_SE * sd * math.sqrt(1.0 / n_a + 1.0 / n_b)


#: Head-room (in SD) the process centre keeps from each hard bound.
CENTRE_HEADROOM_SD = 2.5


def centre_for(name: str, base: float,
               hard_range: tuple[float, float]) -> float | None:
    """Process centre for a calibrated stream, in its working scale: the baseline, pulled
    inward so the calibrated spread fits between the hard bounds.
    """
    cal = CALIBRATION.get(name)
    if cal is None:
        return None
    lo, hi = hard_range
    if cal.log_scale:
        lo, hi = math.log(max(1e-6, lo)), math.log(max(1e-6, hi))
        centre = math.log(max(1e-6, base))
    else:
        centre = float(base)
    pad = CENTRE_HEADROOM_SD * cal.sigma
    if hi - lo <= 2.0 * pad:
        return 0.5 * (lo + hi)
    return min(max(centre, lo + pad), hi - pad)


#: Width of the soft shoulder at each bound, in standard deviations.
SOFT_SHOULDER_SD = 0.5


def _soft_bound(y: float, lo: float, hi: float, margin: float) -> tuple[float, bool]:
    """Compress the far tail into `[lo, hi]` with a continuous, unit-slope shoulder instead of
    clipping. Returns the value and whether the shoulder was used.
    """
    if margin <= 0 or hi - lo <= 2.0 * margin:
        return min(max(y, lo), hi), not (lo <= y <= hi)
    top, bot = hi - margin, lo + margin
    if y > top:
        return hi - margin * math.exp(-(y - top) / margin), True
    if y < bot:
        return lo + margin * math.exp(-(bot - y) / margin), True
    return y, False


#: Loadings on two shared daily factors: innovations are a1 * F1 + a2 * F2 +
#: sqrt(1 - a1^2 - a2^2) * own, which keeps each stream's marginal spread and persistence.
#: Fitted to the reference corpus's median within-subject correlation of daily values
#: (21 pairs, RMSE 0.041). A plain float is a one-factor loading.
SHARED_LOADING: dict[str, float | tuple[float, float]] = {
    "resting_hr": (0.908, 0.279),
    "hrv": (-0.813, 0.0),
    "steps": (0.042, 0.087),
    "stress_score": (0.347, -0.884),
    "sleep_hours": (-0.179, 0.281),
    "skin_temp": (0.188, 0.04),
    "spo2": (0.035, -0.021),
}


#: Day-of-week shift of the process centre on the working scale (log for `steps`, hours for
#: `sleep_hours`), Monday first, summing to zero over the week. The calendar phase is the one
#: the weight rhythm uses (`physio.noise.weight_weekday`: days 5 and 6 are the weekend).
#: `steps`: about 21% fewer steps at the weekend, Sunday lowest (pedometer and wrist-device
#: cohorts put weekend counts roughly 9-26% under weekday counts, Sunday usually lowest).
#: `sleep_hours` is the night ending on that morning: 0.63 h longer on Saturday and Sunday
#: mornings (weekend catch-up sleep; social jet lag of about 44 min in working adults,
#: Roenneberg et al. 2012, Curr Biol 22:939-943). Other streams carry no weekday shift.
WEEKDAY_SHIFT: dict[str, tuple[float, ...]] = {
    "steps":       (0.068, 0.068, 0.068, 0.068, 0.068, -0.120, -0.220),
    "sleep_hours": (-0.18, -0.18, -0.18, -0.18, -0.18, 0.45, 0.45),
}


#: Shared slow factor (W3 round 9, review: pending). In LifeSnaps (calibration split) resting HR
#: moves with skin temperature and sleep over days, not only from day to day: the value-level
#: correlations (0.18, -0.12) exceed what the daily-innovation coupling implies, and resting
#: HR~skin temperature is still 0.15 one day apart in the cross-stream lag comparison.
#: Each listed stream adds `SLOW_LOADING` x a shared,
#: unit-variance AR(1) with persistence `SLOW_PSI` (one path per patient, on the absolute day
#: index); its own AR(1) is scaled by sqrt(1 - c^2) and its persistence set to
#: (phi - c^2 psi) / (1 - c^2), so the marginal spread and the lag-1 autocorrelation stay at
#: `CALIBRATION`. Fitted to the real lag-0/1/7 cross-correlations with `SHARED_LOADING` held
#: fixed (joint slow-factor fit: psi 0.9, resting_hr 0.45 at its bound, skin 0.165, sleep
#: -0.118, hrv -0.082, stress 0.061), then scaled by 0.6 with psi 0.8 after board trials: the
#: full fit lowered the within-window spread of resting HR and made the noise and shape groups
#: worse by 0.016 / 0.027; the joint refit of the daily loadings
#: as well made shape worse by 0.020. With these values the coupling group improves on
#: all four CV fold seeds tried (-0.005 to -0.016) and shape costs +0.010 to +0.014.
SLOW_PSI = 0.8
#: resting_hr 0.27 -> 0.4325 in the joint moving-average fit with `SHARED_MA`; review: pending.
SLOW_LOADING: dict[str, float] = {"resting_hr": 0.4325, "skin_temp": 0.1, "sleep_hours": -0.07, "hrv": -0.05, "stress_score": 0.037}


def _slow_path(seed: str, last_day: int) -> list[float]:
    """The patient's shared slow factor on days 0..last_day (unit variance, stationary start)."""
    out = [_gauss(seed, "slow", "init")]
    k = math.sqrt(max(0.0, 1.0 - SLOW_PSI * SLOW_PSI))
    for d in range(1, last_day + 1):
        out.append(SLOW_PSI * out[-1] + k * _gauss(seed, "slow", d))
    return out


#: Person-specific day-of-week deviation (W3 round 20, review: pending): on top of the
#: population profile (`WEEKDAY_SHIFT`), each patient has their own seven-day deviation, drawn
#: as N(0, sd) per weekday, centred to sum to zero (scaled by sqrt(7/6) to keep the per-day SD)
#: and keyed on (case_id, stream). The SD is the person-by-weekday variance of the LifeSnaps
#: calibration split's person-demeaned weekday means, less its sampling noise (per-cell variance
#: over count): steps 0.079 (log scale), sleep 0.25 h; for HRV the noise exceeds it (no
#: support), so HRV has none. Kept: steps 0.065 (board trials over four CV seeds: 0.079 alone or
#: with sleep 0.25 cost shape 0.017-0.020 on a seed; 0.065 improved weekday on all four seeds).
PERSON_WEEKDAY_SD: dict[str, float] = {"steps": 0.065}


def person_weekday(seed: str, name: str) -> tuple[float, ...] | None:
    sd = PERSON_WEEKDAY_SD.get(name, 0.0)
    if not sd:
        return None
    z = [_gauss(seed, "wk_dev", name, i) for i in range(7)]
    m = sum(z) / 7.0
    k = sd * math.sqrt(7.0 / 6.0)
    return tuple(k * (x - m) for x in z)


def weekday_of(seed: str, day: int) -> int:
    """Weekday (0 = Monday) of course day `day` for case `seed`; the weight rhythm's phase."""
    from haenv.world.physio.noise import weight_weekday
    return weight_weekday(seed, day)


#: W3 (world 1.4), review: pending. Day-to-day changes in LifeSnaps are heavier-tailed than a
#: Gaussian AR(1): the median per-window ratio q95/median of |change| is 3.80 for steps, 2.57
#: resting_hr, 2.80 hrv, 3.13 sleep against 2.42-2.48 / 2.00 / 2.44 / 2.46 in the board world
#: (calibration-window change comparison; frozen target for steps >= 3.0).
#: Each stream's own innovation is a Student t with `INNOV_T_DOF` degrees of freedom,
#: scaled to unit variance, so the spread and persistence are unchanged.
INNOV_T_DOF = 4
#: Streams with t innovations. Not `steps`: its heavy tail in the reference is downward (partly
#: worn days, `PARTIAL_WEAR`), while a symmetric t tail on the log scale pushed 27 rendered
#: values to the 60,000 hard bound in the full-wear steps check.
INNOV_T_STREAMS = frozenset({"resting_hr", "hrv", "sleep_hours", "skin_temp", "spo2", "stress_score",
                             "steps"})
#: W3 round 4 (review: pending): `steps` takes t innovations again, with the upward side cut at
#: `INNOV_UP_CUT` SD, so the heavy tail is the downward one LifeSnaps shows. Permutation
#: importance put the steps tail ratio second in the noise group (real 3.80, synthetic 2.80;
#: permutation-importance comparison); symmetric t tails had pushed
#: steps to the 60,000 hard bound.
INNOV_UP_CUT: dict[str, float] = {"steps": 3.0}
#: Scale mixture on a stream's own innovation (W3 round 11, review: pending): with probability p
#: the shock is multiplied by m, and the whole is rescaled to unit variance, so the spread and
#: persistence are unchanged and only the tails get heavier. LifeSnaps (calibration split),
#: per-person day-to-day changes in units of their median absolute change: hrv symmetric (skew
#: -0.01, excess kurtosis 2.3, 1.1% / 0.9% beyond +5 / -5 units), steps kurtosis 10 (2.0% / 2.4%);
#: a Gaussian puts 0.04% there. With the mixture the window tail ratio reads hrv 2.29 -> 2.76
#: (real 2.94 at display resolution) and steps 2.86 -> 3.14 (real 3.80; frozen limit >= 3.0).
#: Steps keeps `INNOV_UP_CUT`, so its upward tail stays bounded.
#: The same mixture on the first shared daily factor (one draw per patient and day, so the
#: streams that load on it -- resting_hr, hrv, stress -- have their unusual days together).
INNOV_MIX_SHARED: tuple[float, float] | None = (0.08, 4)
INNOV_MIX: dict[str, tuple[float, float]] = {"hrv": (0.05, 4), "steps": (0.05, 4)}
#: Lowest value a worn day can show (W3 round 4, review: pending). In LifeSnaps 0 steps is the
#: "not recorded" marker, and the lowest recorded counts are single digits (0.1% quantile of
#: nonzero days 7 steps, 4,691 days); the heavier downward tail must not print a 0.
LOW_FLOOR: dict[str, float] = {"steps": 5.0}
#: W3, review: pending. Real windows differ far more from person to person than ours: the SD
#: across person-windows of the within-window SD is 0.29 vs 0.11 (spo2) and 0.44 vs 0.06
#: (skin_temp), of resting_hr's 0.73 vs 0.61 (same file, univariate readings). A per-person,
#: per-stream factor exp(N(0, `NOISE_SCALE_LOG_SD`)) on the spread, cut at +-2.5 SD, keyed on
#: (case_id, stream) only.
NOISE_SCALE_LOG_SD = 0.35
#: Per-stream overrides of `NOISE_SCALE_LOG_SD`. `steps` is narrower: on the log scale a wide
#: factor reaches the hard bound (see `INNOV_T_STREAMS`).
NOISE_SCALE_LOG_SD_BY_STREAM: dict[str, float] = {"steps": 0.15}
#: W3, review: pending. Partly worn days: LifeSnaps records them as a low count, not as missing;
#: 6.1% of its observed step days are under 1,500 against 2.4% in the board world
#: in the partial-wear comparison. Probability per worn day, and the range of the worn fraction.
#: 0.04 / 0.10-0.40 gave 8.2% low days and a tail ratio of 2.94 in calibration trials.
PARTIAL_WEAR: dict[str, tuple[float, float, float]] = {"steps": (0.03, 0.03, 0.30)}


def _student_t(*path: object) -> float:
    """Deterministic unit-variance Student t draw with `INNOV_T_DOF` (even) degrees of freedom."""
    k = INNOV_T_DOF
    chi2 = -2.0 * sum(math.log(max(_unit("chi", i, *path), 1e-300)) for i in range(k // 2))
    return _gauss(*path) / math.sqrt(chi2 / k) * math.sqrt((k - 2.0) / k)


def noise_scale(seed: str, name: str) -> float:
    """Per-person factor on a stream's day-to-day spread (`NOISE_SCALE_LOG_SD`)."""
    z = max(-2.5, min(2.5, _gauss(seed, "noise_scale", name)))
    return math.exp(NOISE_SCALE_LOG_SD_BY_STREAM.get(name, NOISE_SCALE_LOG_SD) * z)


#: MA(1) on the first shared daily factor (W3 round 13, review: pending): the factor on day d is
#: (z_d + theta z_{d-1}) / sqrt(1 + theta^2), so every stream that loads on it -- resting HR, HRV,
#: stress -- also carries yesterday's shared shock, with unit variance kept. In LifeSnaps
#: (calibration split) today's resting HR correlates -0.48 with yesterday's HRV and today's HRV
#: -0.16 with yesterday's resting HR (28-day windows). Fitted jointly with resting HR's and
#: HRV's persistence, resting HR's spread and its slow-factor loading (joint fit: theta 0.386,
#: loss 197 -> 68); 0.25 is the calibration-tested candidate, which improved noise and shape on
#: four CV seeds where the fitted 0.386 did not.
SHARED_MA = 0.25


def _shared_z(seed: str, key: object) -> float:
    z = _gauss(seed, "shared", key)
    if INNOV_MIX_SHARED:
        pm, mm = INNOV_MIX_SHARED
        z *= (mm if _unit(seed, "shared_mix", key) < pm else 1.0) / math.sqrt(1.0 - pm + pm * mm * mm)
    return z


def _shared_f1(seed: str, key: object) -> float:
    """The first shared daily factor on `key` (a day index, or "init"), unit variance."""
    z = _shared_z(seed, key)
    if not SHARED_MA:
        return z
    prev = _shared_z(seed, key - 1) if isinstance(key, int) else _shared_z(seed, ("pre", key))
    return (z + SHARED_MA * prev) / math.sqrt(1.0 + SHARED_MA * SHARED_MA)


def ar1(name: str, base: float, days: list[int], seed: str,
        hard_range: tuple[float, float],
        stats: dict | None = None) -> list[float] | None:
    """Stationary AR(1) series for a calibrated stream, one value per day, started from the
    stationary distribution. Returns None for uncalibrated streams.
    """
    cal = CALIBRATION.get(name)
    if cal is None or not days:
        return None
    centre = centre_for(name, base, hard_range)
    if centre is None:
        return None
    lo, hi = hard_range
    if cal.log_scale:
        lo, hi = math.log(max(1e-6, lo)), math.log(max(1e-6, hi))
    margin = SOFT_SHOULDER_SD * cal.sigma
    c_slow = SLOW_LOADING.get(name, 0.0)
    phi = max(-0.95, min(0.95, person_phi(seed, name)))
    if c_slow:
        phi = max(-0.95, min(0.95, (phi - c_slow * c_slow * SLOW_PSI) / (1.0 - c_slow * c_slow)))
    sd_ar = cal.sigma * math.sqrt(1.0 - c_slow * c_slow)
    innov = sd_ar * math.sqrt(max(1e-12, 1.0 - phi * phi))
    slow = _slow_path(seed, max(days)) if c_slow else None
    load = SHARED_LOADING.get(name, 0.0)
    a1, a2 = (load, 0.0) if isinstance(load, (int, float)) else load
    b = math.sqrt(max(0.0, 1.0 - a1 * a1 - a2 * a2))

    def _shock(key: object) -> float:
        z2 = a2 * _gauss(seed, "shared2", key) if a2 else 0.0
        own = _student_t(seed, name, key) if name in INNOV_T_STREAMS else _gauss(seed, name, key)
        mx = INNOV_MIX.get(name)
        if mx:
            pm, mm = mx
            own *= (mm if _unit(seed, "innov_mix", name, key) < pm else 1.0) / math.sqrt(1.0 - pm + pm * mm * mm)
        if name in INNOV_UP_CUT:
            own = min(own, INNOV_UP_CUT[name])
        f1 = _shared_f1(seed, key)
        return a1 * f1 + z2 + b * own

    wk = WEEKDAY_SHIFT.get(name)
    pwk = person_weekday(seed, name)
    if pwk is not None:
        wk = tuple((wk[i] if wk else 0.0) + pwk[i] for i in range(7))
    scale = noise_scale(seed, name)
    innov *= scale
    pw = PARTIAL_WEAR.get(name)
    x = sd_ar * scale * _shock("init")
    out: list[float] = []
    n_soft = 0
    for i, d in enumerate(days):
        if i:
            x = phi * x + innov * _shock(d)
        xs = x + (c_slow * cal.sigma * scale * slow[d] if slow is not None else 0.0)
        y, softened = _soft_bound(centre + xs + (wk[weekday_of(seed, d)] if wk else 0.0),
                                  lo, hi, margin)
        n_soft += int(softened)
        if pw and _unit(seed, "partial_wear", name, d) < pw[0]:
            frac = pw[1] + (pw[2] - pw[1]) * _unit(seed, "partial_frac", name, d)
            y = y + math.log(frac) if cal.log_scale else y * frac
        if name in LOW_FLOOR:
            fl = LOW_FLOOR[name]
            y = max(y, math.log(fl) if cal.log_scale else fl)
        out.append(math.exp(y) if cal.log_scale else y)
    if stats is not None:
        stats["n_soft"] = stats.get("n_soft", 0) + n_soft
        stats["n_soft_total"] = stats.get("n_soft_total", 0) + len(out)
    return out


def stream_available(case_id: str, name: str) -> bool:
    """Whether this patient's device reports `name` at all (depends on `case_id` only)."""
    cal = CALIBRATION.get(name)
    if cal is None:
        return True
    return _unit(case_id, "wear_avail", name) < max(AVAIL_MIN, cal.avail)


#: Share of the rarest stream's non-wear that is device-level (shared by all streams on a
#: day); fitted to the reference corpus's co-missingness.
SHARED_OFF_FRACTION = 0.9
#: Persistence of the device-level chain, and of each stream's own chain (`OWN_P_OFF_OFF`;
#: `CALIBRATION.p_off_off` is the stream's marginal persistence in the corpus). A day is
#: missing when either chain is off, so two chains each at the marginal persistence merge
#: into runs longer than the corpus's. Fitted jointly on the corpus's 28-day windows (share
#: of one-day runs, share of runs over three days, non-wear rate, for all seven streams, and
#: as counter-metrics the pairwise phi of the missing indicators, 0.519 against 0.572, and
#: the within-window spread of per-stream rates, 0.071 against 0.061), on synthetic case ids;
#: with 0.6957 and the marginal persistences, one-day runs were 0.27-0.48 of all runs against
#: 0.42-0.68 in the corpus.
#: W3 round 3 (review: pending): the day-scale chains, the compliance multiplier and the
#: long-gap class below were fitted jointly by random search on simulated masks
#: to the LifeSnaps calibration split read on calendar days: pooled shares of 1-day and
#: >3-day runs, quartiles of the per-person missing rate, and the per-person median missing rate
#: of the W3 instrument (frozen tolerance, plan appendix B). `p_off_worn` in `CALIBRATION` is from
#: the same fit. Round-1 values: SHARED_P_OFF_OFF 0.55, COMPLIANCE_SHAPE 1.20.
SHARED_P_OFF_OFF = 0.37
OWN_P_OFF_OFF: dict[str, float] = {
    "resting_hr": 0.25,   # W3 round 6 (0.68 before): runs 0.264/0.477 -> 0.402/0.352 (ref 0.433/0.348); review: pending
    "hrv": 0.36,
    "steps": 0.85,   # tuned after the joint fit: >3-day share 0.33 -> 0.44 (reference 0.468)
    "stress_score": 0.72,
    "sleep_hours": 0.45,  # W3 round 6 (0.40 before): longer sleep gaps, runs 0.438/0.270 -> 0.402/0.299 (ref 0.453/0.336); review: pending
    "skin_temp": 0.34,    # W3 round 6 (0.40 before): runs 0.443/0.271 -> 0.490/0.249 (ref 0.500/0.284), instrument one-day share still <= 0.60; review: pending
    "spo2": 0.5,
}


#: Shape of the per-patient compliance multiplier (Gamma, mean 1); W3 round 3 fit (see above).
COMPLIANCE_SHAPE = 0.30
#: Shape of the per-stream compliance multiplier (Gamma, mean 1). It was `COMPLIANCE_SHAPE`
#: until W3 round 5 (review: pending): with 0.30 the per-stream multipliers of one patient
#: scattered so widely that the spread of missing rates across streams within a window read
#: 0.125 against 0.090 in LifeSnaps, the top feature of the missing group
#: in a permutation-importance comparison. 1.0 together with
#: `COMPLIANCE_TRAIT_WEIGHT` 0.70 reads 0.0876 on the board world; chosen
#: among calibration trials as the one that matches the spread, with
#: test_world_wear unchanged.
STREAM_COMPLIANCE_SHAPE = 1.0
#: Ceiling on a patient's non-wear rate.
OFF_RATE_CAP = 0.60
#: Weight of the patient-level multiplier in each stream's own non-wear; 0.2 matches the
#: reference's dispersion of per-stream missing rates within a window.
COMPLIANCE_TRAIT_WEIGHT = 0.70   # W3 round 5 (0.42 before), see `STREAM_COMPLIANCE_SHAPE`; review: pending


def _shared_rate() -> float:
    return SHARED_OFF_FRACTION * min(c.steady_off for c in CALIBRATION.values())


def _gamma_unit_mean(case_id: str, shape: float) -> float:
    """Deterministic Gamma(shape, 1/shape) draw: mean 1, CV 1/sqrt(shape).

    Marsaglia-Tsang on shape + 1, boosted by U^(1/shape) for shape < 1. Draws are
    indexed by attempt so the result depends on case_id only.
    """
    k = shape + 1.0 if shape < 1.0 else shape
    d = k - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    g = d
    for attempt in range(64):
        x = _gauss(case_id, "compliance_x", attempt)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = _unit(case_id, "compliance_u", attempt)
        if math.log(max(u, 1e-300)) < 0.5 * x * x + d - d * v + d * math.log(v):
            g = d * v
            break
    if shape < 1.0:
        g *= max(_unit(case_id, "compliance_boost"), 1e-300) ** (1.0 / shape)
    return g / shape


def compliance(case_id: str, name: str | None = None) -> float:
    """Multiplier on a non-wear rate (depends on case_id and stream name only). `name=None` is
    the patient's device-level multiplier.
    """
    if name is None:
        return _gamma_unit_mean(case_id, COMPLIANCE_SHAPE)
    return _gamma_unit_mean(f"{case_id}|{name}", STREAM_COMPLIANCE_SHAPE)


def case_rates(case_id: str, name: str) -> tuple[float, float, float]:
    """(device-level, stream-level, combined) stationary non-wear rates for one patient."""
    cal = CALIBRATION[name]
    shared = _shared_rate()
    own = max(0.0, (cal.steady_off - shared) / (1.0 - shared))
    s_c = min(OFF_RATE_CAP, compliance(case_id) * shared)
    g_own = (COMPLIANCE_TRAIT_WEIGHT * compliance(case_id)
             + (1.0 - COMPLIANCE_TRAIT_WEIGHT) * compliance(case_id, name))
    o_c = min(OFF_RATE_CAP, g_own * own)
    e_c = episode_off_rate(case_id)
    return s_c, o_c, 1.0 - (1.0 - s_c) * (1.0 - o_c) * (1.0 - e_c) * (1.0 - DEVICE_OFF_DAY)


def expected_off_runs(case_id: str, name: str, span: int) -> float:
    """Expected number of non-wear runs over `span` days for this patient: device-chain runs,
    own-chain runs and episodes (`verify` fails a stream with no missing day only when at least
    five runs are expected)."""
    cal = CALIBRATION[name]
    s_c, o_c, _ = case_rates(case_id, name)
    own_pp = OWN_P_OFF_OFF.get(name, cal.p_off_off)
    return span * (s_c * (1.0 - SHARED_P_OFF_OFF) + o_c * (1.0 - own_pp) + episode_start_rate(case_id)
                   + DEVICE_OFF_DAY * (1.0 - DEVICE_OFF_DAY))


def _chain(case_id: str, key: tuple, days: list[int], rate: float,
           p_off_off: float) -> list[bool]:
    """Two-state chain with stationary off-rate `rate`; True means off."""
    if rate <= 0.0 or not days:
        return [False] * len(days)
    p_off_worn = rate * (1.0 - p_off_off) / (1.0 - rate)
    off = _unit(case_id, "wear_init", *key) < rate
    out = []
    for i, d in enumerate(days):
        if i:
            off = _unit(case_id, "wear", *key, d) < (p_off_off if off else p_off_worn)
        out.append(off)
    return out


#: Shape check on a calibrated stream's missing days (`verify.check_stream`). Non-wear from
#: `worn_on` is bursty and has no calendar rhythm, so a residue class of the day index modulo k
#: that is (almost) always missing is a sampling comb (every other day, every fourth day), not
#: non-wear. A class needs `COMB_MIN_N` nominal days and `COMB_SHARE` of them missing, while the
#: stream as a whole misses at most `COMB_MAX_OFF`; with the patient cap `OFF_RATE_CAP` = 0.6,
#: 23 of 24 days missing in one class is about 1e-4 per class even for independent days, and
#: non-wear runs (<= `MAX_GAP_DAYS`) cannot fill a class spread over the whole span.
COMB_PERIODS = (2, 3, 4, 5, 6, 7)
COMB_MIN_N = 24
COMB_SHARE = 0.95
COMB_MAX_OFF = 0.75


def sampling_comb(observed_days: list[int]) -> tuple[int, int, float] | None:
    """`(k, residue, missing share)` of the first day-of-index-mod-k class that is a sampling
    comb (see `COMB_SHARE`), or None. `observed_days` are the days with a value; the nominal
    days are every day from the first to the last of them."""
    if len(observed_days) < 2:
        return None
    obs = set(int(d) for d in observed_days)
    first, last = min(obs), max(obs)
    span = last - first + 1
    if 1.0 - len(obs) / span > COMB_MAX_OFF:
        return None
    for k in COMB_PERIODS:
        for r in range(k):
            nominal = [d for d in range(first, last + 1) if d % k == r]
            if len(nominal) < COMB_MIN_N:
                continue
            share = sum(d not in obs for d in nominal) / len(nominal)
            if share >= COMB_SHARE:
                return k, r, share
    return None


#: Person-level non-wear class (W3, world 1.4; review: pending). LifeSnaps on calendar days is
#: a mixture: most people rarely miss, a minority have device-off episodes of days to weeks.
#: A patient is in the long-gap
#: class with probability `LONG_GAP_SHARE`; only that class has episodes. An episode starts on
#: a worn day with probability `EPISODE_RATE` x a Gamma(`EPISODE_SHAPE`) propensity (mean 1)
#: and lasts `EPISODE_MIN` + Exp(`EPISODE_MEAN_EXTRA`) days; `worn_on` cuts runs at
#: `MAX_GAP_DAYS`. Everything depends on case_id only, so it carries no label.
#: Fitted with the chains above (W3 round 3, review: pending).
LONG_GAP_SHARE = 0.78
EPISODE_RATE = 0.195
EPISODE_MIN = 4
EPISODE_MEAN_EXTRA = 12.2
EPISODE_SHAPE = 0.15
EPISODE_RATE_CAP = 0.5


def long_gap(case_id: str) -> bool:
    """Whether the patient is in the long-gap non-wear class."""
    return _unit(case_id, "nonwear_class") < LONG_GAP_SHARE


def episode_start_rate(case_id: str) -> float:
    """Daily episode start probability of this patient (0 outside the long-gap class)."""
    if EPISODE_RATE <= 0 or not long_gap(case_id):
        return 0.0
    return min(EPISODE_RATE_CAP, EPISODE_RATE * _gamma_unit_mean(f"{case_id}|episode", EPISODE_SHAPE))


def _episode_mean_len() -> float:
    return min(float(MAX_GAP_DAYS), EPISODE_MIN + EPISODE_MEAN_EXTRA)


def episode_off_rate(case_id: str) -> float:
    """Stationary share of this patient's days inside an episode."""
    r = episode_start_rate(case_id)
    return r * _episode_mean_len() / (1.0 + r * _episode_mean_len()) if r > 0 else 0.0


def _episodes(case_id: str, days: list[int]) -> list[bool]:
    """Episode mask over `days`; True means inside an episode."""
    r = episode_start_rate(case_id)
    if r <= 0 or not days:
        return [False] * len(days)
    out: list[bool] = []
    left = 0
    for d in days:
        if left > 0:
            out.append(True)
            left -= 1
        elif _unit(case_id, "episode_start", d) < r:
            u = _unit(case_id, "episode_len", d)
            n = EPISODE_MIN + int(-EPISODE_MEAN_EXTRA * math.log(max(1e-12, 1.0 - u)))
            out.append(True)
            left = min(n, MAX_GAP_DAYS) - 1
        else:
            out.append(False)
    return out


#: Whole-device-off days (W3 round 7, review: pending): single days on which every stream of
#: the patient is missing (left on the charger, not put on), drawn for every patient with
#: daily probability `DEVICE_OFF_DAY`, independent of the compliance classes. In LifeSnaps
#: (calibration split) the median 28-day window has 7.5% of its missing days missing on every
#: stream; the round-6 world had 0 in the joint-missingness comparison,
#: because its device-level chain sits on a few low-compliance patients.
DEVICE_OFF_DAY = 0.01   # board trials: 0.02 broke the resting-HR one-day limit and test_world_wear (steps), 0.035 more so


def _device_off_days(case_id: str, days: list[int]) -> list[bool]:
    return [_unit(case_id, "device_off_day", d) < DEVICE_OFF_DAY for d in days]


def worn_on(case_id: str, name: str, days: list[int]) -> list[bool]:
    """Non-wear mask over `days`, in order.

    A day is missing when the device chain or this stream's own chain is off; both are scaled
    by the patient's compliance and start from their stationary distribution. Runs are cut
    at `MAX_GAP_DAYS`.
    """
    cal = CALIBRATION.get(name)
    if cal is None or not days:
        return [True] * len(days)
    shared, own, _ = case_rates(case_id, name)
    dev_off = _chain(case_id, ("device",), days, shared, SHARED_P_OFF_OFF)
    own_off = _chain(case_id, (name,), days, own, OWN_P_OFF_OFF.get(name, cal.p_off_off))
    ep_off = _episodes(case_id, days)
    day_off = _device_off_days(case_id, days)
    out: list[bool] = []
    gap = 0
    for a, b, e, z in zip(dev_off, own_off, ep_off, day_off):
        worn = not (a or b or e or z)
        if worn:
            gap = 0
        else:
            gap += 1
            # `>=` so a run never exceeds MAX_GAP_DAYS.
            if gap >= MAX_GAP_DAYS:
                worn, gap = True, 0
        out.append(worn)
    return out


# --------------------------------------------------------------------------
# Derived streams
# --------------------------------------------------------------------------
# Each derived stream is a function of an existing quantity. Formulas are written in the
# units this repository registers.


def _height_m(bmi: float | None, weight_kg: float) -> float:
    """Height implied by BMI and weight; falls back to a population median."""
    if bmi and bmi > 0 and weight_kg > 0:
        return math.sqrt(weight_kg / bmi)
    return 1.68


def bmr_mifflin(weight_kg: float, bmi: float | None, age: int, sex: str) -> float:
    """Basal metabolic rate, kcal/day.

    Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO,
    Am J Clin Nutr 1990;51:241-7:
        BMR = 10 * weight(kg) + 6.25 * height(cm) - 5 * age + s,
        s = +5 for men, -161 for women.
    Height is not recorded directly, so it comes from BMI and weight.
    """
    h_cm = _height_m(bmi, weight_kg) * 100.0
    s = 5.0 if str(sex).upper().startswith("M") else -161.0
    return 10.0 * weight_kg + 6.25 * h_cm - 5.0 * float(age) + s


def vo2max_uth(resting_hr: float, age: int) -> float:
    """VO2max, mL/kg/min, from the heart-rate ratio.

    Uth N, Sorensen H, Overgaard K, Pedersen PK, Eur J Appl Physiol 2004;91:111-5:
        VO2max = 15.3 * HRmax / HRrest,
    with HRmax approximated as 220 - age.
    """
    hr_max = max(120.0, 220.0 - float(age))
    return 15.3 * hr_max / max(35.0, float(resting_hr))


#: Multiplier on the walking-only energy estimate, for activity a step count does not see.
#: Fitted, not from a textbook; the reference corpus cannot check it.
ACTIVE_BURN_NON_AMBULATORY = 2.35


def active_burn_from_steps(steps: float, weight_kg: float) -> float:
    """Activity energy expenditure, kcal/day, from step count: about 0.5 kcal per kg per km at
    a 0.72 m stride, scaled by `ACTIVE_BURN_NON_AMBULATORY`.
    """
    walking = 0.5 * max(40.0, weight_kg) * (steps * 0.00072)
    return ACTIVE_BURN_NON_AMBULATORY * walking


def exercise_hr_from_resting(resting_hr: float, age: int, intensity: float) -> float:
    """Average exercise heart rate from the heart-rate reserve.

    Karvonen MJ, Kentala E, Mustala O, Ann Med Exp Biol Fenn 1957;35:307-15:
        HR_target = HR_rest + intensity * (HR_max - HR_rest),
    with `intensity` the session intensity in [0, 1].
    """
    hr_max = max(120.0, 220.0 - float(age))
    return resting_hr + intensity * max(0.0, hr_max - resting_hr)


def mets_from_exercise_hr(exercise_hr: float, resting_hr: float, age: int) -> float:
    """Metabolic equivalents from the fraction of heart-rate reserve in use (1 at rest, about
    12 at maximum).
    """
    hr_max = max(120.0, 220.0 - float(age))
    frac = (exercise_hr - resting_hr) / max(1.0, hr_max - resting_hr)
    return 1.0 + 11.0 * min(1.0, max(0.0, frac))


def sleep_stage_split(sleep_hours: float, case_id: str, day: int) -> tuple[float, float]:
    """Deep and REM percentages for one night, shortening with total sleep; centres 17.25% and
    19.50% are common adult values.
    """
    short = max(0.0, min(1.0, (8.0 - sleep_hours) / 4.0))
    deep = 17.25 * (1.0 - 0.35 * short) + 2.6 * _gauss(case_id, "deep", day)
    rem = 19.50 * (1.0 - 0.45 * short) + 3.0 * _gauss(case_id, "rem", day)
    deep = min(45.0, max(2.0, deep))
    rem = min(45.0, max(2.0, rem))
    if deep + rem > 75.0:                      # light sleep must keep a real share
        scale = 75.0 / (deep + rem)
        deep, rem = deep * scale, rem * scale
    return deep, rem


def sleep_efficiency_from_hours(sleep_hours: float, case_id: str, day: int) -> float:
    """Sleep efficiency as a fraction, lower on short nights."""
    short = max(0.0, min(1.0, (8.0 - sleep_hours) / 4.0))
    eff = 0.93 - 0.06 * short + 0.02 * _gauss(case_id, "eff", day)
    return min(0.99, max(0.60, eff))


def derive_streams(facts, rendered: dict, seed: str) -> dict[str, list[dict]]:
    """Build the derived streams from already-rendered parents and patient facts.

    Each derived stream sits on its parent's observation days. Basal metabolic rate uses the
    starting weight, so it cannot mirror the weight series (the gold evidence).
    """
    out: dict[str, list[dict]] = {}
    age = int(getattr(facts, "age_lo", 40) or 40)
    sex = str(getattr(facts, "sex", "F") or "F")
    w_kg = float(getattr(facts, "start_weight", 75.0) or 75.0)
    bmi = getattr(facts, "bmi", None)
    cid = str(getattr(facts, "case_id", "") or seed)

    steps = rendered.get("steps") or []
    rhr = rendered.get("resting_hr") or []
    sleep = rendered.get("sleep_hours") or []

    if steps:
        out["active_burn"] = [
            {"ts": q["ts"], "value": int(round(active_burn_from_steps(q["value"], w_kg)))}
            for q in steps]
        bmr = bmr_mifflin(w_kg, bmi, age, sex)
        out["calories_bmr"] = [
            {"ts": q["ts"], "value": int(round(bmr + 18.0 * _gauss(cid, "bmr", q["ts"])))}
            for q in steps]
    if rhr:
        out["vo2_max"] = [
            {"ts": q["ts"], "value": round(vo2max_uth(q["value"], age), 1)} for q in rhr]
        ex_hr, mets = [], []
        for q in rhr:
            # Session intensity is drawn from (case, day) only, so it cannot encode the label.
            inten = 0.33 + 0.25 * _unit(cid, "intensity", q["ts"])
            h = exercise_hr_from_resting(q["value"], age, inten)
            ex_hr.append({"ts": q["ts"], "value": int(round(h))})
            mets.append({"ts": q["ts"], "value": round(mets_from_exercise_hr(h, q["value"], age), 1)})
        out["exercise_avg_hr"], out["exercise_mets"] = ex_hr, mets
    if sleep:
        eff, deep, rem = [], [], []
        for q in sleep:
            eff.append({"ts": q["ts"],
                        "value": round(sleep_efficiency_from_hours(q["value"], cid, q["ts"]), 3)})
            d, r = sleep_stage_split(q["value"], cid, q["ts"])
            deep.append({"ts": q["ts"], "value": round(d, 1)})
            rem.append({"ts": q["ts"], "value": round(r, 1)})
        out["sleep_efficiency"], out["deep_percentage"], out["rem_percentage"] = eff, deep, rem
    return out


#: Reference-corpus streams deliberately not generated, with the reason.
NOT_ADMITTED = {
    "distance":   "an affine restatement of steps at a fixed stride -- the two "
                  "correlate at 0.985 over 4,691 days at a median 0.704 m per "
                  "step -- so it would add a number without adding a constraint",
    "bpm":        "daily mean heart rate, a blend of the resting_hr and "
                  "exercise_avg_hr this module already generates (it correlates "
                  "with resting_hr at 0.643); a third heart-rate stream binds "
                  "nothing the first two do not",
    "scl_avg":    "electrodermal activity, recorded on 380 of 7,410 days and by "
                  "2 of 71 subjects; two subjects cannot support a distribution, "
                  "so there is nothing to calibrate against",
}

#: Derived streams and what they are functions of.
DERIVED_BINDINGS = _streams.derived_bindings()      # `derived_from` in registry/streams.yaml

#: Which stream's observation days a derived stream sits on. This can differ from its
#: binding: basal metabolic rate depends on weight but is observed when the watch is worn.
GRID_PARENT = _streams.grid_parents()               # `grid_parent` in registry/streams.yaml


def prune_orphans(longitudinal: dict) -> list[str]:
    """Drop derived streams whose grid parent is gone. Returns what was removed."""
    removed = []
    for name, parent in GRID_PARENT.items():
        if name in longitudinal and not longitudinal.get(parent):
            longitudinal.pop(name, None)
            removed.append(name)
    return removed
