"""The daily-metric catalog and stream planning: which wearable and clinical streams a case
carries, at what cadence, and how each is rendered.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from . import wearable as _wear
import hashlib
import math
from dataclasses import dataclass
from haenv import streams as _streams
from .events_pools import Facts, proxy_tags


# ============================================================ daily metric catalog
@dataclass
class MetricSpec:
    name: str
    devices: tuple[str, ...]          # any one in inventory suffices for daily
                                       # sampling (empty = self-reported diary,
                                       # needs no device)
    unit: str
    base: float                       # default healthy-person baseline
    amp: float                        # intraday/day-to-day physiological
                                       # fluctuation amplitude
    period: int                       # deterministic waveform period (days;
                                       # unrelated to T / the reversal week)
    ndigits: int                      # 0 = round to integer
    hard_range: tuple[float, float]   # hard physiological range: any point
                                       # outside it fails validation
    tol: float                        # tolerance between the mean and the
                                       # profile's expected baseline
    vital_key: str | None = None      # if the raw case states this vital's
                                       # baseline, defer to it
    tags: tuple[str, ...] = ()        # used for "driver proxy" exclusion


#: Baseline adjustments per population trait. Order matters (floating-point
#: addition is not associative). Entry shapes:
#:   `("<a boolean attribute on Facts>", delta)`  -- add delta when it matches
#:   `("age_decade_over", (anchor, delta))`      -- add delta per decade past anchor
#:   `("drug_prefix", (prefix, delta))`          -- add delta when the drug name
#:                                                  matches this prefix
BASELINE_ADJUST: dict[str, tuple[tuple[str, object], ...]] = {
    "steps":            (("heavy", -2200), ("elderly", -1500), ("osa", -600)),
    "activity_index":   (("heavy", -16), ("elderly", -12)),
    "resting_hr":       (("hyperthyroid", 30), ("hypothyroid", -6),
                         ("heavy", 4), ("elderly", -3)),
    "hrv":              (("age_decade_over", (40, -5)), ("hyperthyroid", -12), ("heavy", -5)),
    "sleep_hours":      (("osa", -0.7), ("elderly", -0.4)),
    "spo2":             (("osa", -1.6), ("heavy", -0.6)),
    "body_temp":        (("hyperthyroid", 0.2),),
    "skin_temp":        (("hyperthyroid", 0.15),),
    "gi_symptom_score": (("drug_prefix", ("metformin", 0.8)),),
}


#: The entry shapes that are not boolean attributes of `Facts`.
BASELINE_ADJUST_FORMS = ("age_decade_over", "drug_prefix")


def _adj(name: str, base: float, f: Facts) -> float:
    """Adjust a metric's baseline by the raw case's baseline facts (`BASELINE_ADJUST`)."""
    v = base
    for kind, arg in BASELINE_ADJUST.get(name, ()):
        if kind == "age_decade_over":
            anchor, delta = arg                                   # type: ignore[misc]
            v += delta * max(0, (f.age_lo - anchor) // 10)
        elif kind == "drug_prefix":
            prefix, delta = arg                                   # type: ignore[misc]
            v += delta if f.drug.startswith(prefix) else 0.0
        else:
            # No default: a misspelled trait name must raise, not read as False.
            if getattr(f, kind):
                v += arg                                          # type: ignore[operator]
    return v


#: Person-level baseline terms, added on top of the population profile (`BASELINE_ADJUST`).
#: They are kept out of `BASELINE_ADJUST` because that profile is what the event-planning
#: prompt shows (`_catalog_for`); `plan_streams` and `verify.check_stream` use the sum
#: (`_adj_full`). Entry shapes: a boolean `Facts` attribute (`delta` when true),
#: `("age_decade_over", (anchor, delta))`, `("level", delta)`, and `("case_offset", sd)`: a
#: per-case draw N(0, sd) cut at +-2.5 sd, keyed on (case_id, metric), so it carries no label.
#: `resting_hr`: NHANES pulse medians by sex and age band are 70/72/68 bpm for men and 76/74/72
#: for women (18-39 / 40-59 / 60+), against the profile's 68: women +4, +2 overall, -1 per
#: decade past 50 (with `elderly` already -3 from 65). Quer et al. 2020 (PLoS One 15:e0227709,
#: 92,457 Fitbit users) put the SD of the person's mean resting rate at 7.7 bpm; the population
#: terms give 4.7 bpm in the v1.0.1 cohort, so the person offset has SD 6.5 (expected total 8.0).
INDIVIDUAL_ADJUST: dict[str, tuple[tuple[str, object], ...]] = {
    "resting_hr": (("female", 4.0), ("level", 2.0), ("age_decade_over", (50, -1.0)),
                   ("case_offset", 6.5)),
}


#: The entry shapes of `INDIVIDUAL_ADJUST` that are not boolean attributes of `Facts`.
INDIVIDUAL_ADJUST_FORMS = ("age_decade_over", "level", "case_offset")


#: Cut of the per-case offset, in SD.
CASE_OFFSET_CUT_SD = 2.5


def individual_shift(name: str, f: Facts) -> float:
    """Person-level shift of a metric's baseline (`INDIVIDUAL_ADJUST`); 0 when the raw case
    states this vital's baseline itself."""
    spec = METRIC_BY_NAME.get(name)
    if spec is not None and spec.vital_key and f.baseline_vitals.get(spec.vital_key) is not None:
        return 0.0
    v = 0.0
    for kind, arg in INDIVIDUAL_ADJUST.get(name, ()):
        if kind == "age_decade_over":
            anchor, delta = arg                                   # type: ignore[misc]
            v += delta * max(0, (f.age_lo - anchor) // 10)
        elif kind == "level":
            v += float(arg)                                       # type: ignore[arg-type]
        elif kind == "case_offset":
            z = _wear._gauss(f.case_id, "level_offset", name)
            v += float(arg) * max(-CASE_OFFSET_CUT_SD, min(CASE_OFFSET_CUT_SD, z))  # type: ignore[arg-type]
        elif getattr(f, kind):
            v += arg                                              # type: ignore[operator]
    return v


def _adj_full(name: str, base: float, f: Facts) -> float:
    """Population profile plus the person-level shift."""
    return _adj(name, base, f) + individual_shift(name, f)


# The daily metrics, in manifest order (`registry/streams.yaml`, `render`).
METRICS: tuple[MetricSpec, ...] = tuple(MetricSpec(**f) for f in _streams.metric_fields())


METRIC_BY_NAME = {m.name: m for m in METRICS}


# ============================================================ Daily-metric stream planning/rendering
@dataclass
class StreamPlan:
    spec: MetricSpec
    base_eff: float                  # actual baseline derived from profile (+ the raw case's baseline vitals)
    step_days: int                   # sampling step (derived from event_density.measure_per_week)
    source: str                      # baseline source: profile | raw_case_vital
    phase: int = 0


def _gold_signal_of_driver(driver: str) -> str | None:
    """The gold-evidence stream for this driver, from `build.GOLD_EVIDENCE`
    (same source as `verify.GOLD_SIGNAL_OF`); imported lazily to avoid a cycle."""
    try:
        from .registry import GOLD_EVIDENCE
    except Exception:                                  # noqa: BLE001
        return None
    g = (GOLD_EVIDENCE or {}).get(driver) or {}
    return g.get("signal")


def _wearable_cadence(f: Facts, m: MetricSpec, step: int) -> tuple[int, str | None]:
    """Sampling step for a planned stream, or the reason it is not planned.

    A calibrated wearable stream is sampled daily (a property of the device,
    not the follow-up plan). Whether a device reports a metric is drawn from
    case_id only.
    """
    if not _wear.is_calibrated(m.name):
        return step, None
    if not _wear.stream_available(str(getattr(f, "case_id", "") or ""), m.name):
        return 0, "device_does_not_report_metric"
    return 1, None


def world_layer_base_signals(raw, world_signals=()) -> set[str]:
    """Names in `raw.longitudinal_data` that `plan_streams` must leave alone.

    Everything upstream wrote, except a calibrated stream that the kernel's
    `inject_distractors` wrote as a distractor and that is not gold evidence:
    that copy is replaced by the calibrated render.
    """
    upstream = set(getattr(raw, "longitudinal_data", None) or {})
    adj = getattr(raw, "adjudication", None) or {}
    distractor = set(adj.get("distractor_signals") or ()) if isinstance(adj, dict) else set()
    world = set(world_signals or ())
    free = {n for n in distractor & upstream if _wear.is_calibrated(n) and n not in world}
    return upstream - free


def plan_streams(f: Facts, event_density: dict, driver: str,
                 drop: set[str] | None = None,
                 base_signals=None) -> tuple[list[StreamPlan], list[dict]]:
    """Select the daily metrics to inject: device in the kit, not a driver
    proxy, not already provided by the world layer (`base_signals`, which
    holds gold evidence the injector must not touch)."""
    drop = drop or set()
    banned = proxy_tags(f, driver)
    mpw = float(event_density.get("measure_per_week", 7) or 7)
    step = max(1, int(round(7.0 / max(0.5, mpw))))
    plans, skipped = [], []
    for i, m in enumerate(METRICS):
        if m.name in _wear.DERIVED_BINDINGS:
            # Derived streams are produced from their parents after rendering.
            continue
        if m.name in drop:
            skipped.append({"item": m.name, "reason": "dropped_by_verifier"})
            continue
        if m.name in (base_signals or ()):
            # Gold evidence from the world layer; do not overwrite.
            skipped.append({"item": m.name, "reason": "provided_by_world_layer"})
            continue
        if m.devices and not (set(m.devices) & set(f.devices)):
            skipped.append({"item": m.name, "reason": f"no_device{list(m.devices)}"})
            continue
        if set(m.tags) & banned:
            skipped.append({"item": m.name, "reason": "driver_or_comorbid_proxy"})
            continue
        base = _adj_full(m.name, m.base, f)
        src = "profile"
        vk = m.vital_key
        if vk and f.baseline_vitals.get(vk) is not None:   # the raw case wrote an actual level for this vital -> use it
            base, src = float(f.baseline_vitals[vk]), "raw_case_vital"
        lo, hi = m.hard_range
        base = min(max(base, lo + m.amp + 1e-9), hi - m.amp - 1e-9)
        step_eff, why = _wearable_cadence(f, m, step)
        if why:
            skipped.append({"item": m.name, "reason": why})
            continue
        plans.append(StreamPlan(spec=m, base_eff=round(base, m.ndigits or 0) if m.ndigits else round(base),
                                step_days=step_eff, source=src, phase=(i * 3) % 7))
    return plans, skipped


#: AR(1) day-scale coefficient. ACF(k) = phi^k, so phi^7 < 0.35 requires
#: phi < 0.862; 0.80 gives ACF(1)=0.80, ACF(7)=0.21.
_AR_PHI = 0.80


#: Stationary sd = `_AR_SD_FRAC x amp`, kept small because AR(1) has
#: Gaussian-like tails and `base_eff` only reserves `amp` headroom.
_AR_SD_FRAC = 0.35


def _det_shock(seed: str, k: int) -> float:
    """A deterministic [-1, 1) uniform shock from `blake2b(seed|k)`; no RNG
    state, so results do not depend on call order."""
    h = hashlib.blake2b(f"{seed}|{k}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 63) - 1.0


def _render_calibrated(p: "StreamPlan", end_day: int, seed: str,
                       stats: dict | None) -> list[dict]:
    """Render a stream calibrated in `haenv.wearable`: empirical distribution
    family, lag-1 persistence and spread, plus a two-state non-wear mask."""
    m = p.spec
    days = list(range(0, end_day + 1, max(1, int(p.step_days))))
    series = _wear.ar1(m.name, float(p.base_eff), days, seed, m.hard_range, stats)
    if series is None:                                   # not calibrated after all
        return []
    mask = _wear.worn_on(seed, m.name, days)
    lo, hi = m.hard_range
    out, n_clip = [], 0
    for d, v, worn in zip(days, series, mask):
        if not worn:
            continue
        if v < lo or v > hi:
            n_clip += 1
        v = min(max(v, lo), hi)
        out.append({"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(round(v))})
    if stats is not None:
        stats["n"] = stats.get("n", 0) + len(out)
        stats["n_clipped"] = stats.get("n_clipped", 0) + n_clip
    return out


def render_stream(p: StreamPlan, end_day: int, seed: str = "",
                  stats: dict | None = None) -> list[dict]:
    """A deterministic AR(1) waveform over [0, end_day]; never reads the outcome.

    AR(1) rather than fixed sines, which would give the corpus a lag-7
    fingerprint. `seed` must carry the case identity (phases vary only by
    stream index). When `stats` is given, `{"n", "n_clipped"}` are recorded.
    """
    m, out = p.spec, []
    if _wear.is_calibrated(m.name):
        return _render_calibrated(p, end_day, seed, stats)
    sd = _AR_SD_FRAC * m.amp
    phi_s = _AR_PHI ** max(1, int(p.step_days))          # scaled by the sampling step
    sig_s = sd * math.sqrt(max(0.0, 1.0 - phi_s * phi_s))
    bound = sig_s * math.sqrt(3.0)                       # uniform distribution: sd = bound/sqrt(3)
    key = f"{seed}|{m.name}|{p.phase}"
    # Start from the stationary distribution (no visible warm-up).
    x = sd * _det_shock(key, -1)
    n_clip = 0
    for k, d in enumerate(range(0, end_day + 1, p.step_days)):
        if k:
            x = phi_s * x + bound * _det_shock(key, k)
        v = p.base_eff + x
        if m.name == "scale_qc_flag":                     # binary QC: one calibration failure every 17 days
            # Binary QC flag: a fixed comb (period 17, phase per patient),
            # replacing the AR(1) value. A random draw would exceed
            # `gates.A5_MIN_REL_SPAN`, which is calibrated against this comb.
            t = d + p.phase
            v = 0.0 if (t % m.period == 0 and d > 0) else 1.0
        lo, hi = m.hard_range
        if v < lo or v > hi:
            n_clip += 1
        v = min(max(v, lo), hi)
        out.append({"ts": d, "value": round(v, m.ndigits) if m.ndigits else int(round(v))})
    if stats is not None:
        stats["n"] = stats.get("n", 0) + len(out)
        stats["n_clipped"] = stats.get("n_clipped", 0) + n_clip
    return out
