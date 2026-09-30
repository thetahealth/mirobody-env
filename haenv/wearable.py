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

from . import streams as _streams

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
CALIBRATION: dict[str, Calibration] = {
    "resting_hr":   Calibration(log_scale=False, sigma=1.8707, phi=0.8055,
                                avail=0.927, p_off_worn=0.0545, p_off_off=0.5972),
    "hrv":          Calibration(log_scale=True,  sigma=0.2110, phi=0.2308,
                                avail=0.533, p_off_worn=0.1074, p_off_off=0.4378),
    "steps":        Calibration(log_scale=True,  sigma=0.5675, phi=0.1066,
                                avail=0.952, p_off_worn=0.0322, p_off_off=0.6957),
    "stress_score": Calibration(log_scale=False, sigma=5.0891, phi=0.1929,
                                avail=0.352, p_off_worn=0.1085, p_off_off=0.5621),
    "sleep_hours":  Calibration(log_scale=False, sigma=1.2631, phi=-0.0659,
                                avail=0.770, p_off_worn=0.0964, p_off_off=0.4705),
    "skin_temp":    Calibration(log_scale=False, sigma=0.4373, phi=0.1128,
                                avail=0.752, p_off_worn=0.1128, p_off_off=0.4670),
    "spo2":         Calibration(log_scale=False, sigma=0.7454, phi=-0.0369,
                                avail=0.309, p_off_worn=0.1373, p_off_off=0.5064),
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


def off_rate_tolerance(name: str, span: int, rate: float | None = None) -> float:
    """Allowed deviation of the realised non-wear rate from the declared one.

    `BAND_SE` standard errors of a two-state chain's rate over `span` days:
    pi (1 - pi) / span * (1 + lam) / (1 - lam), with lam = p_off_off - p_off_worn.
    """
    cal = CALIBRATION[name]
    pi = cal.steady_off if rate is None else rate
    lam = cal.p_off_off - cal.p_off_worn
    if span <= 0 or not (-1.0 < lam < 1.0):
        return 1.0
    var = pi * (1.0 - pi) / span * (1.0 + lam) / (1.0 - lam)
    return max(OFF_RATE_TOL_FLOOR, BAND_SE * math.sqrt(var))


def mean_shift_tolerance(name: str, va: list[float], vb: list[float]) -> float:
    """`BAND_SE` standard errors of the difference between two segment means, using the pooled
    within-segment SD and the effective sample size n (1 - phi) / (1 + phi).
    """
    phi = CALIBRATION[name].phi
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
    phi = max(-0.95, min(0.95, cal.phi))
    innov = cal.sigma * math.sqrt(max(1e-12, 1.0 - phi * phi))
    load = SHARED_LOADING.get(name, 0.0)
    a1, a2 = (load, 0.0) if isinstance(load, (int, float)) else load
    b = math.sqrt(max(0.0, 1.0 - a1 * a1 - a2 * a2))

    def _shock(key: object) -> float:
        z2 = a2 * _gauss(seed, "shared2", key) if a2 else 0.0
        return a1 * _gauss(seed, "shared", key) + z2 + b * _gauss(seed, name, key)

    x = cal.sigma * _shock("init")
    out: list[float] = []
    n_soft = 0
    for i, d in enumerate(days):
        if i:
            x = phi * x + innov * _shock(d)
        y, softened = _soft_bound(centre + x, lo, hi, margin)
        n_soft += int(softened)
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
#: Persistence of the shared chain, taken from `steps`, whose non-wear is almost all shared.
SHARED_P_OFF_OFF = 0.6957


#: Shape of the per-patient compliance multiplier (Gamma, mean 1): 1 / 0.91^2 matches the
#: reference's median coefficient of variation of per-subject missing rates.
COMPLIANCE_SHAPE = 1.20
#: Ceiling on a patient's non-wear rate.
OFF_RATE_CAP = 0.60
#: Weight of the patient-level multiplier in each stream's own non-wear; 0.2 matches the
#: reference's dispersion of per-stream missing rates within a window.
COMPLIANCE_TRAIT_WEIGHT = 0.20


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
    key = case_id if name is None else f"{case_id}|{name}"
    return _gamma_unit_mean(key, COMPLIANCE_SHAPE)


def case_rates(case_id: str, name: str) -> tuple[float, float, float]:
    """(device-level, stream-level, combined) stationary non-wear rates for one patient."""
    cal = CALIBRATION[name]
    shared = _shared_rate()
    own = max(0.0, (cal.steady_off - shared) / (1.0 - shared))
    s_c = min(OFF_RATE_CAP, compliance(case_id) * shared)
    g_own = (COMPLIANCE_TRAIT_WEIGHT * compliance(case_id)
             + (1.0 - COMPLIANCE_TRAIT_WEIGHT) * compliance(case_id, name))
    o_c = min(OFF_RATE_CAP, g_own * own)
    return s_c, o_c, 1.0 - (1.0 - s_c) * (1.0 - o_c)


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
    own_off = _chain(case_id, (name,), days, own, cal.p_off_off)
    out: list[bool] = []
    gap = 0
    for a, b in zip(dev_off, own_off):
        worn = not (a or b)
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
