"""build.py -- question generation: raw case + hidden control variables ->
latent premise -> conditioned generation -> iterative validation -> emit.

Follows this order: (1) premise + validation, (2) GV-1
generate-validate, (3) noise/distractors derived from the premise, (3b)
day-by-day events with per-item validation and the physiology layer, (3c)
reading artifacts and post-injection on the observed series, (4) recheck +
leakage probe. A
case that fails any stage is not emitted. The kernel lives in
`core/`; this module does task-level assembly. SYNTHETIC, evaluation only.
"""
from __future__ import annotations

from .regpath import registry_cached as _registry_cached
from .yamlcache import load_yaml as _cached_yaml

from . import latent_rules as _LR

import copy
import functools
import json
import math
import statistics
import logging
import pathlib

from schema import RawCase
from latent import LatentPremise, make_premise
from synth import synthesize, premise_conflicts
from noise import inject as inject_noise, inject_distractors
from build import build_instance, leakage_probe

from . import indicators as _indicators
from . import artifact_params as _artifact_params
from . import drug_effects as _drug_effects
from . import demographics as _demographics
from . import post_inject as _post_inject
from .job import raw_field as _raw_field
from . import events as events_mod
from .events import EVENT_RATE_DEFAULTS as _EVENT_RATE_DEFAULTS
from . import gates
from . import verify as verify_mod
from . import wq
from .job import CaseSpec
from .job import driver_of as _job_driver_of, outcome_of as _job_outcome_of

log = logging.getLogger("haenv.build")

STEP, END = 7, 365

# Day the disease course ends. Equal to `END` (the extent of the world-layer
# streams) but a separate quantity; per-case end days come from `course_end_of`.
COURSE_END = END

# Domain of the course end day. Held at one value: a varying end day changes
# `prediction_window` on the question surface, and a shorter course can flip
# `outcome_label` via `label_rule.minimum_change_magnitude`.
COURSE_END_DOMAIN: tuple[int, ...] = (365,)


def course_end_of(case_id: str, latent: dict | None = None) -> int:
    """This case's course end day: the explicit declaration, else sampled from
    `case_id` alone (never from diagnosis/outcome).
    """
    if latent and latent.get("course_end_day") is not None:
        return int(latent["course_end_day"])
    from . import rng
    return int(rng.pick(list(COURSE_END_DOMAIN), case_id, "course_end"))

# --------------------------------------------------------------- Primary-signal sampling and missingness
# Weight is sampled daily with missed weigh-ins. Missingness depends only on the
# day of week (answer-neutral MAR), never on driver/outcome.
WEIGHT_STEP = 1
WEIGH_P_WEEKDAY = 0.82           # weigh-in probability on weekdays
WEIGH_P_WEEKEND = 0.45           # weigh-in probability on Sat/Sun
WEIGHT_MAX_GAP = 6               # longest everyday gap (days); longer gaps come only from a noise window
WOBBLE_PERIOD = 21.0             # period of the physiological wobble (days)


# --------------------------------------------------------------- Gold-standard evidence stream (world layer)
# Every case carries every registered evidence stream; only the gold driver's
# stream is abnormal. `GOLD_EVIDENCE` lives in `registry.py` and is
# re-exported here.
from .prompts import PROMPT  # noqa: E402,F401  (question-surface templates, re-exported)
from .registry import GOLD_EVIDENCE  # noqa: E402,F401  (re-exported)
from . import external_gold as _EG   # noqa: E402  external gold-standard registration surface (empty by default => strictly a no-op)
from haenv import data_root as _dr   # resource root: source tree = repo root, wheel = haenv/_data


def ledger_weight_point(ld: dict, T: int, cid: str) -> tuple[int, float]:
    """The ledger weight EV: the last point of the `weight` stream with ts ≤ T.

    Not the nadir, which can fall after T, and not the minimum, since a scale
    reports its current reading. Called both at construction and after noise
    injection, so the ledger matches the final noised stream.
    """
    pts = ld.get("weight")
    if not isinstance(pts, list) or not pts:
        raise ValueError(f"{cid}: the case has no weight stream; cannot build the weight-ledger EV")
    le_T = [q for q in pts
            if isinstance(q, dict) and isinstance(q.get("ts"), (int, float))
            and int(q["ts"]) <= int(T) and q.get("value") is not None]
    if not le_T:
        raise ValueError(f"{cid}: the weight stream has no points at ts <= T={T}; cannot build the weight-ledger EV")
    last = max(le_T, key=lambda q: int(q["ts"]))
    return int(last["ts"]), float(last["value"])


def gold_evidence_streams(driver: str, rev_week: int, occurred: bool,
                          T: int, end_day: int = END,
                          dens_step: int = 1, case_id: str = "C") -> dict[str, list[dict]]:
    """The gold-standard evidence streams: present in every case, abnormal only
    when this driver is the gold standard.

    Plateau then a 4-week rise whose onset is `min(rev_day, T) - lead_days`, so
    the evidence is visible before T. Fluctuation is AR(1) with a per-case,
    per-stream seed and amplitude `max(0.04 × normal, 0.8 tick)`, small enough
    to stay under GEN14's "flat" threshold (`gates.GEN14_FLAT_REL_SPAN`). The rise does not also lower
    `medication_adherence`.
    """
    out: dict[str, list[dict]] = {}
    rev_day = rev_week * 7
    step_eff = max(STEP, dens_step)
    for drv, spec in GOLD_EVIDENCE.items():
        is_gold = (drv == driver) and occurred
        onset = max(0, min(rev_day, int(T)) - int(spec["lead_days"]))
        lo, hi = spec["range"]
        pts = []
        # The 0.8-tick floor keeps coarse, small-valued signals from rounding to a
        # constant.
        _tick = 10.0 ** (-int(spec["ndigits"]))
        amp = max(0.04 * float(spec["normal"]), 0.8 * _tick)
        phi = events_mod._AR_PHI ** max(1, int(step_eff))
        sd = events_mod._AR_SD_FRAC * amp
        bound = sd * math.sqrt(max(0.0, 1.0 - phi * phi)) * math.sqrt(3.0)
        key = f"{case_id}|{spec['signal']}|gold"
        # Start from the stationary distribution, so there is no warm-up segment.
        x = sd * events_mod._det_shock(key, -1)
        for k, d in enumerate(range(0, end_day + 1, step_eff)):
            if k:
                x = phi * x + bound * events_mod._det_shock(key, k)
            v = spec["normal"]
            if is_gold and d >= onset:
                ramp = min(1.0, (d - onset) / 28.0)
                v = spec["normal"] + (spec["abnormal"] - spec["normal"]) * ramp
            v += x
            pts.append({"ts": d, "value": round(min(max(v, lo), hi), spec["ndigits"])})
        out[spec["signal"]] = pts
    return out


# --------------------------------------------------------------- Generator
@_registry_cached("physio_streams.yaml")
def _trajectory_shape_registered() -> tuple[float, float, float]:
    """`registry/physio_streams.yaml:trajectory_shape` -> `(descent_k, rebound_k,
    descent_k_spread)`; `(0, 0, 0)` (piecewise linear) if the entry is missing.
    """
    from .regpath import load_registry as _lr
    doc = _lr("physio_streams.yaml") or {}
    s = doc.get("trajectory_shape") or {}
    return (float(s.get("descent_k") or 0.0), float(s.get("rebound_k") or 0.0),
            float(s.get("descent_k_spread") or 0.0))


def _trajectory_shape(case_id: str) -> tuple[float, float]:
    """This case's desired `(descent_k, rebound_k)`, capped later by `_feasible_k`.

    The descent curvature is sampled per case from `case_id` only; the gold
    label reads the start, the minimum and the span after it, never the shape
    of the descent, so curvature cannot change it.
    """
    k_desc, k_reb, spread = _trajectory_shape_registered()
    if spread <= 0.0:
        return k_desc, k_reb
    from .rng import unit as _unit
    return k_desc * (1.0 - spread + 2.0 * spread * _unit(case_id, "traj_k", "descent")), k_reb


def _feasible_k(trend_per_day: float, budget_per_day: float, k_want: float) -> float:
    """Largest curvature `k <= k_want` whose initial slope stays within the budget.

    The exponential's initial slope is `m(k) = k/(1-e^-k)` times the linear
    slope (m=1 at k=0, strictly increasing), so `m(k) <= budget / trend` is
    solved by bisection.
    """
    if k_want <= 1e-9 or trend_per_day <= 1e-12:
        return k_want
    allow = budget_per_day / trend_per_day
    if allow <= 1.0:
        return 0.0
    if k_want / (1.0 - math.exp(-k_want)) <= allow:
        return k_want
    lo, hi = 0.0, k_want
    for _ in range(40):
        mid = (lo + hi) / 2.0
        m = 1.0 if mid <= 1e-9 else mid / (1.0 - math.exp(-mid))
        if m <= allow:
            lo = mid
        else:
            hi = mid
    return lo


def _shape(x: float, k: float) -> float:
    """Normalized exponential `f(x) = (1 - e^(-kx)) / (1 - e^(-k))`: `f(0)=0`,
    `f(1)=1`, linear at `k=0`. Exact endpoints keep the segment end at `nadir`,
    which `label_rule.baseline_window` reads.
    """
    x = min(1.0, max(0.0, x))
    if k <= 1e-9:
        return x
    return (1.0 - math.exp(-k * x)) / (1.0 - math.exp(-k))


def _weighed_on(day: int, case_id: str) -> bool:
    """Whether the patient weighed in on this day: deterministic, from `day % 7`
    and a hash of `case_id` only.
    """
    import hashlib
    phase = int(hashlib.sha256(str(case_id).encode()).hexdigest()[:8], 16)
    p = WEIGH_P_WEEKEND if (day + phase) % 7 in (5, 6) else WEIGH_P_WEEKDAY
    h = int(hashlib.sha256(f"{case_id}|{day}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < p


# --------------------------------------------------------------- weight skeleton
# Four irregularities on top of the three-segment skeleton: an uneven descent,
# plateau episodes, a per-case rebound shape, and transient ups and downs. Every
# random number comes from `_skeleton_draws(case_id)`; the rest is deterministic.

#: Descent phases per case: 3..SKEL_DESCENT_MAX_PHASES, equal length, own pace each.
SKEL_DESCENT_MAX_PHASES = 6
#: Relative pace of a stalled descent phase.
SKEL_STALL_PACE = 0.08
#: Relative pace of the hold between two rebound steps.
SKEL_HOLD_PACE = 0.15
#: Episodes drawn per case; enough to cover the longest course.
SKEL_MAX_EPISODES = 16
#: Longest episode window (days), rise plus the first recovery. The residual after it
#: fades slowly, so an episode can leave the course above the label threshold; the label
#: guard decides whether that stands.
SKEL_EPISODE_MAX_DAYS = 45
#: A no-regain series may hold above the label threshold this long (days), or as
#: long as its plain version does; half of `MIN_PERSIST_DAYS`, leaving room for the
#: observation noise added later.
SKEL_MAX_RUN_DAYS = 28
#: The run limit counts days within this margin (kg) under the label threshold,
#: since the observation noise added later reaches that far.
SKEL_RUN_MARGIN_KG = 0.5
#: A declared regain must hold this far (kg) above the label threshold.
SKEL_REGAIN_RUN_MARGIN_KG = 0.6
#: Shortest gap between two episodes (days).
SKEL_VALLEY_MIN_DAYS = 10
#: Holiday-type gains as a fraction of body mass: 0.4 % (Helander 2016, US Christmas) to
#: 1.35 % (Turicchi 2020, PMID 32353079, Christmas). Only part comes off again: Turicchi
#: keeps >= 0.35 of 1.35 % in March, Helander about half, Yanovski 2000 (PMID 10727591) none
#: by the next autumn; the residual share and its fading timescale (days) are drawn here.
SKEL_GAIN_FRAC = (0.004, 0.0135)
SKEL_GAIN_RESIDUAL = (0.25, 0.5)
SKEL_GAIN_TAU_DAYS = (90.0, 270.0)
#: Share of the descent budget an episode's own rise may use.
SKEL_EPISODE_PACE_FRAC = 0.8
#: Rebound shapes, drawn per case; the endpoint and so the mean slope stay fixed.
SKEL_REBOUND_VARIANTS = ("gradual", "accelerating", "steps")
#: Slow drift: an Ornstein-Uhlenbeck process on weekly knots. Timescale (days) and
#: stationary SD (kg) are drawn per case from these ranges; the drift is soft-bounded
#: at `SKEL_DRIFT_BOUND_KG`, below `latent_rules.MIN_CHANGE_KG`, and moves at most
#: `SKEL_DRIFT_SLOPE_FRAC` of the descent budget per day.
SKEL_DRIFT_TAU_DAYS = (28.0, 84.0)
SKEL_DRIFT_SD_KG = (0.12, 0.25)
SKEL_DRIFT_BOUND_KG = 0.4
#: Soft bound below zero (kg): drift barely dips under the course, so the low point,
#: which GEN22 holds within 0.5 kg of the declared nadir, stays where it was.
SKEL_DRIFT_BOUND_DOWN_KG = 0.1
SKEL_DRIFT_SLOPE_FRAC = 0.25
#: Days over which drift fades in or out at T when it is on at one side only.
SKEL_FADE_DAYS = 28
#: A declared regain must hold at least this far (kg) above the label threshold, for
#: `MIN_PERSIST_DAYS + SKEL_REGAIN_MARGIN_DAYS` days, on the rendered series.
SKEL_REGAIN_MARGIN_KG = 0.5
SKEL_REGAIN_MARGIN_DAYS = 7
#: A declared regain on a course with no loss before the reversal first dips this far
#: (kg) under its first value, in a smooth dip of at least this many days between T
#: and the reversal.
SKEL_SETTLE_KG = 0.35
SKEL_SETTLE_DAYS = 28
#: Guard levels (post-T only): 0 full, 1 gradual rebound, 2 no episodes after T,
#: 3 no drift after T, 4 the plain skeleton after T.
SKEL_LEVELS = (0, 1, 2, 3, 4)

_SKEL_N_DESCENT = 1 + SKEL_DESCENT_MAX_PHASES + 3
_SKEL_N_REBOUND = 8
_SKEL_N_EPISODE = 7
#: Drift draws: two parameters plus one shock per weekly knot, for courses up to 1113 days.
_SKEL_N_DRIFT = 2 + 160


def _skeleton_draws(case_id: str) -> dict[str, list]:
    """Every uniform draw the weight skeleton uses, from `case_id` only.

    Never reads outcome, reversal week or driver, so all outcome arms share them.
    The demo exports this table per case, as it does `events._det_shock`.
    """
    from .rng import unit as _u
    cid = str(case_id)
    return {
        "descent": [_u(cid, "skel", "descent", i) for i in range(_SKEL_N_DESCENT)],
        "rebound": [_u(cid, "skel", "rebound", i) for i in range(_SKEL_N_REBOUND)],
        "episodes": [[_u(cid, "skel", "episode", j, i) for i in range(_SKEL_N_EPISODE)]
                     for j in range(SKEL_MAX_EPISODES)],
        "drift": [_u(cid, "skel", "drift", i) for i in range(_SKEL_N_DRIFT)],
    }


def _pace_pieces(pieces: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`(end_x, pace)` from `(duration, progress)` pieces: equal area under the pace
    means equal progress, and the mean pace over [0, 1] is 1."""
    td = sum(d for d, _ in pieces)
    tp = sum(q for _, q in pieces)
    out, ad = [], 0.0
    for d, q in pieces:
        ad += d
        out.append((ad / td, (q / tp) / (d / td)))
    return out


def _pace_at(pieces: list[tuple[float, float]], x: float) -> float:
    for end_x, pace in pieces:
        if x < end_x:
            return pace
    return pieces[-1][1]


def _descent_pieces(dr: list[float]) -> list[tuple[float, float]]:
    """Descent pace: 3..6 equal phases, 0..2 of them stalled (never the first or
    the last, so the descent leaves the start and reaches the nadir moving)."""
    k = 3 + int(dr[0] * (SKEL_DESCENT_MAX_PHASES - 2))
    paces = [0.35 + 1.3 * dr[1 + i] for i in range(k)]
    for j in range(int(dr[1 + SKEL_DESCENT_MAX_PHASES] * 3)):
        paces[1 + int(dr[2 + SKEL_DESCENT_MAX_PHASES + j] * (k - 2))] = SKEL_STALL_PACE
    return _pace_pieces([(1.0, q) for q in paces])


def _rebound_pieces(dr: list[float]) -> list[tuple[float, float]]:
    """Rebound steps: 2..3 fast rises, each followed by a hold."""
    pieces = []
    for i in range(2 + int(dr[1] * 2)):
        rise = 0.25 + 0.2 * dr[2 + i]
        share = 0.6 + 0.8 * dr[5 + i]
        pieces += [(rise, share * (1.0 - SKEL_HOLD_PACE)), (1.0 - rise, share * SKEL_HOLD_PACE)]
    return _pace_pieces(pieces)


def _paced_progress(plain: list[float], pieces: list[tuple[float, float]],
                    cap: float) -> list[float]:
    """Re-pace a progress curve (`plain[0] = 0` .. `plain[-1] = 1`).

    Each day's plain increment is multiplied by its phase's pace, then the
    increments are water-filled under `cap` and renormalized to sum to 1: mass
    a capped day cannot take moves to days with room. The endpoints stay exact.
    """
    n = len(plain) - 1
    if n <= 0:
        return list(plain)
    inc = [(plain[d + 1] - plain[d]) * _pace_at(pieces, (d + 0.5) / n) for d in range(n)]
    cap = max(cap, max(plain[d + 1] - plain[d] for d in range(n)))
    fixed = [False] * n
    for _ in range(n + 1):
        free = sum(v for v, f in zip(inc, fixed) if not f)
        room = 1.0 - sum(v for v, f in zip(inc, fixed) if f)
        if free <= 0.0:
            break
        inc = [v if f else v * room / free for v, f in zip(inc, fixed)]
        over = [i for i in range(n) if not fixed[i] and inc[i] > cap]
        if not over:
            break
        for i in over:
            inc[i], fixed[i] = cap, True
    out, acc = [0.0], 0.0
    for v in inc[:-1]:
        acc += v
        out.append(min(1.0, acc))
    out.append(1.0)
    return out


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _episode_plan(ep: list[float], budget_day: float, mass: float = 80.0) -> dict:
    """One episode from its seven draws.

    Up (a holiday gain) or down (an illness dip): a fast move, then a slower,
    incomplete recovery that leaves `residual` of the height, which fades with
    timescale `tau` days. The rise is stretched only as far as the descent budget
    requires. An up episode is `SKEL_GAIN_FRAC` of body mass; published holiday gains
    are only partly reversed (`SKEL_GAIN_RESIDUAL`).
    """
    kind = "up" if ep[0] < 0.75 else "down"
    lo, hi = SKEL_GAIN_FRAC
    h = (mass * (lo + (hi - lo) * ep[1])) if kind == "up" else (0.3 + 0.5 * ep[1])
    rise = max(2 + int(4 * ep[2]),
               int(math.ceil(1.5 * h / max(1e-9, SKEL_EPISODE_PACE_FRAC * budget_day))))
    rise = min(rise, SKEL_EPISODE_MAX_DAYS // 3)
    n = min(SKEL_EPISODE_MAX_DAYS, rise + int(math.ceil(rise * (2.0 + 3.0 * ep[3]))))
    return {"kind": kind, "height": h, "rise": rise, "n": n,
            "residual": SKEL_GAIN_RESIDUAL[0] + (SKEL_GAIN_RESIDUAL[1] - SKEL_GAIN_RESIDUAL[0]) * ep[5],
            "tau": SKEL_GAIN_TAU_DAYS[0] + (SKEL_GAIN_TAU_DAYS[1] - SKEL_GAIN_TAU_DAYS[0]) * ep[6],
            "valley": SKEL_VALLEY_MIN_DAYS + int(25 * ep[4])}


def _episode_profile(plan: dict, length: int) -> list[float]:
    """Unit-height profile over days 0..length: fast rise, slower recovery to
    `residual` at day `n`, then an exponential fade of the residual."""
    rise, n, r, tau = plan["rise"], plan["n"], plan["residual"], plan["tau"]
    out = []
    for i in range(length + 1):
        if i <= rise:
            out.append(_smoothstep(i / rise))
        elif i <= n:
            out.append(1.0 - (1.0 - r) * _smoothstep((i - rise) / (n - rise)))
        else:
            out.append(r * math.exp(-(i - n) / tau))
    return out


def _max_step(vals: list[float], lo: int, hi: int) -> float:
    lo, hi = max(0, lo), min(len(vals) - 1, hi)
    return max((abs(vals[d + 1] - vals[d]) for d in range(lo, hi)), default=0.0)


def _largest_ok(ok) -> float:
    """Largest `s` in [0, 1] with `ok(s)`, by bisection; `ok(0)` is assumed."""
    if ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _drift_series(draws: dict, end_day: int, budget_day: float) -> list[float]:
    """The case's slow drift over days 0..end_day: an OU process on weekly knots,
    soft-bounded by `SKEL_DRIFT_BOUND_KG` above and `SKEL_DRIFT_BOUND_DOWN_KG` below,
    smooth between knots, 0 on day 0."""
    dr = draws["drift"]
    tau = SKEL_DRIFT_TAU_DAYS[0] + (SKEL_DRIFT_TAU_DAYS[1] - SKEL_DRIFT_TAU_DAYS[0]) * dr[0]
    sd = SKEL_DRIFT_SD_KG[0] + (SKEL_DRIFT_SD_KG[1] - SKEL_DRIFT_SD_KG[0]) * dr[1]
    a = math.exp(-7.0 / tau)
    step_cap = SKEL_DRIFT_SLOPE_FRAC * budget_day * 7.0 / 1.5
    x, knots = 0.0, [0.0]
    if end_day // 7 + 1 > len(dr) - 2:
        raise ValueError(f"course of {end_day} days is longer than the drift draws cover")
    for k in range(1, (end_day // 7) + 2):
        u = dr[2 + (k - 1)]
        shock = sd * math.sqrt(1.0 - a * a) * math.sqrt(3.0) * (2.0 * u - 1.0)
        x += max(-step_cap, min(step_cap, (a - 1.0) * x + shock))
        bound = SKEL_DRIFT_BOUND_KG if x >= 0.0 else SKEL_DRIFT_BOUND_DOWN_KG
        knots.append(bound * math.tanh(x / bound))
    return [knots[d // 7] + (knots[d // 7 + 1] - knots[d // 7]) * _smoothstep((d % 7) / 7.0)
            for d in range(end_day + 1)]


def _step_caps(ref: list[float], budget_day: float) -> list[float]:
    """Per-day step cap: `max(descent budget, the reference series' own step)`; entry `d`
    caps the step from day `d` to day `d + 1`."""
    return [max(budget_day, abs(ref[d + 1] - ref[d])) + 1e-9 for d in range(len(ref) - 1)]


def _within_caps(vals: list[float], caps: list[float], lo: int, hi: int) -> bool:
    """Every step from day `lo` to day `hi` stays within its cap."""
    lo, hi = max(0, lo), min(len(vals) - 1, hi)
    return all(abs(vals[d + 1] - vals[d]) <= caps[d] for d in range(lo, hi))


def _apply_irregularity(base: list[float], draws: dict, budget_day: float, floor: float,
                        T: int, end_day: int, *, pre_drift: bool = True,
                        post_drift: bool = True, pre_eps: bool = True, post_eps: bool = True,
                        post_t_until: int | None = None) -> dict:
    """Add the case's drift, then its episodes, to `base` in place.

    Every daily step stays within `max(descent budget, the step of base as passed in)`:
    the drift follows its target only as far as that cap allows, each episode is scaled
    until its window fits, and a final forward pass holds the residual tails to the cap. Drift on one side of T only fades in (or out)
    over `SKEL_FADE_DAYS` after T, so the choice for the post-T side never moves a pre-T
    value; everything runs forward in time, so no later value reaches an earlier one.
    Episodes sit on the absolute day axis and none starts before T and ends after it:
    an episode's scale reads `base` across its footprint, so one crossing T would carry
    post-T values into pre-T ones; the residual tail of a pre-T episode may run past T.
    A dip keeps at least half its local distance above `floor`. `post_t_until` stops
    post-T episodes that would end after that day.
    """
    caps = _step_caps(base, budget_day)
    drift = _drift_series(draws, end_day, budget_day)
    dT = drift[min(T, end_day)]

    def target(d):
        if d <= T:
            return drift[d] if pre_drift else 0.0
        fade = _smoothstep((d - T) / SKEL_FADE_DAYS)
        if pre_drift and post_drift:
            return drift[d]
        if pre_drift:
            return dT * (1.0 - fade)
        if post_drift:
            return drift[d] - dT * (1.0 - fade)
        return 0.0

    ref = list(base)
    off = target(0)
    base[0] = ref[0] + off
    for d in range(1, end_day + 1):
        step = ref[d] - ref[d - 1] + target(d) - off
        step = max(-caps[d - 1], min(caps[d - 1], step))
        off = base[d - 1] + step - ref[d]
        base[d] = ref[d] + off
    added = []
    t = 7 + int(28 * draws["episodes"][0][4])
    for ep in draws["episodes"]:
        if t >= end_day:
            break
        plan = _episode_plan(ep, budget_day, base[0])
        n = plan["n"]
        t0, t1 = t, min(end_day, t + n)
        t = t + n + plan["valley"]
        if (t0 <= T <= t1 + 1 or (not post_eps and t0 > T) or (not pre_eps and t1 < T)
                or (post_t_until is not None and t0 > T and t1 >= post_t_until)):
            continue
        h = plan["height"]
        if plan["kind"] == "down":
            h = min(h, 0.5 * (min(base[t0:t1 + 1]) - floor))
            if h < 0.05:
                continue
        prof = _episode_profile(plan, end_day - t0)
        sign = -1.0 if plan["kind"] == "down" else 1.0

        def with_scale(s, _t0=t0, _t1=t1, _p=prof, _sg=sign, _h=h):
            v = list(base)
            for i in range(min(end_day, _t1 + 1) - _t0 + 1):
                v[_t0 + i] += _sg * s * _h * _p[i]
            return v

        s = _largest_ok(lambda s: _within_caps(with_scale(s), caps, t0 - 1, t1 + 1))
        if s * h < 0.05:
            continue
        for i in range(end_day - t0 + 1):
            base[t0 + i] += sign * s * h * prof[i]
        added.append({"kind": plan["kind"], "t0": t0, "t1": t1, "height": round(s * h, 4),
                      "residual": round(plan["residual"], 3)})
    # Residual tails and the seams between layers: one forward pass holds every step to
    # its cap, catching up later where it has to. Forward only, so a day never depends
    # on a later one.
    for d in range(1, end_day + 1):
        step = max(-caps[d - 1], min(caps[d - 1], base[d] - base[d - 1]))
        base[d] = base[d - 1] + step
    return {"episodes": added, "drift_sd": round(statistics.pstdev(drift), 4) if drift else 0.0,
            "caps": caps}


def _ensure_regain(base: list[float], rev_day: int, T: int, end_day: int,
                   budget_day: float, declared_gain: float, declared_lost: float,
                   caps: list[float] | None = None, nadir: float | None = None) -> dict:
    """Make a declared regain hold on `base`: after `max(rev_day, T)`, keep the series
    at least `SKEL_REGAIN_MARGIN_KG` above the label threshold for
    `MIN_PERSIST_DAYS + SKEL_REGAIN_MARGIN_DAYS` days, lifting it with a ramp no
    steeper than half the descent budget where it falls short. Only days after T
    change, and no step leaves `caps`.

    Returns `{"state", "dip_kg", "lift_kg", "end_over_declared_kg"}`. `state` is
    `"held"`, `"lifted"`, `"no_room"`, or `"declared_below_rule"` when the declared
    course itself regains less than the rule asks (`declared_gain < max(MIN_CHANGE_FRAC x
    declared_lost, MIN_CHANGE_KG)`): an inconsistent latent, which GEN13 must refuse,
    not something to paper over. `dip_kg` is how far the low point moved down,
    `lift_kg` the largest lift, `end_over_declared_kg` the last day's value minus the
    declared end (`first value - declared_lost + declared_gain`).
    """
    out = {"state": "n/a", "dip_kg": 0.0, "lift_kg": 0.0, "end_over_declared_kg": 0.0}
    caps = caps or _step_caps(base, budget_day)
    before = list(base)

    def done(state):
        out["state"] = state
        s0_ = min(max(rev_day, T) + 1, end_day)
        out["dip_kg"] = round(max(0.0, min(before[:s0_]) - min(base[:s0_])), 3)
        out["lift_kg"] = round(max(0.0, max(b - a for a, b in zip(before, base))), 3)
        out["end_over_declared_kg"] = round(
            base[end_day] - (base[0] - declared_lost + declared_gain), 3)
        return out

    if declared_gain < max(_LR.MIN_CHANGE_FRAC * max(0.0, declared_lost), _LR.MIN_CHANGE_KG):
        return done("declared_below_rule")
    s0 = max(rev_day, T) + 1
    if s0 >= end_day:
        return done("no_room")
    # A course that never went `SKEL_SETTLE_KG` below its first value has lost too
    # little for the rule to read a regain through observation noise: a smooth dip
    # between T and the reversal brings the low point there, inside GEN22's 0.5 kg
    # nadir tolerance.
    target = base[0] - SKEL_SETTLE_KG
    if nadir is not None:
        target = max(target, nadir - SKEL_ANCHOR_MARGIN_KG + 0.05)
    room = s0 - 1 - T
    if room >= SKEL_SETTLE_DAYS and min(base[:s0]) > target:
        for width in range(SKEL_SETTLE_DAYS, room + 1):
            # The dip ends up to four weeks before the reversal, so the weeks just
            # before it stay level and the turn itself stays visible.
            end = s0 - 1 - min(28, room - width)
            trial = list(base)
            for i in range(width + 1):
                d = end - width + i
                trial[d] -= max(0.0, trial[d] - target) * math.sin(math.pi * i / width)
            if _within_caps(trial, caps, end - 1 - width, end + 1):
                base[:] = trial
                break
    low = min(base[:s0])
    lost = max(0.0, base[0] - low)
    gap = max(_LR.MIN_CHANGE_FRAC * lost, _LR.MIN_CHANGE_KG)

    def run_above(vals, level):
        best, run = 0, None
        for d in range(s0, end_day + 1):
            if vals[d] >= level:
                run = d if run is None else run
                best = max(best, d - run)
            else:
                run = None
        return best

    # With the margin first; where the course is too short for it, the bare rule.
    for margin_kg, need in ((SKEL_REGAIN_MARGIN_KG, _LR.MIN_PERSIST_DAYS + SKEL_REGAIN_MARGIN_DAYS),
                            (0.1, _LR.MIN_PERSIST_DAYS + 1)):
        level = low + gap + margin_kg
        if run_above(base, level) >= need:
            return done("held")
        rise = max(0.0, level - base[s0 - 1])
        # The ramp never outpaces the descent budget.
        width = max(14, int(math.ceil(1.5 * rise / max(1e-9, 0.5 * budget_day))))
        if end_day - (s0 + width) < need:
            width = max(int(math.ceil(1.5 * rise / max(1e-9, budget_day))), end_day - s0 - need)
        if end_day - (s0 + width) < need:
            continue
        for d in range(s0, end_day + 1):
            base[d] = max(base[d], base[s0 - 1] + rise * _smoothstep((d - s0 + 1) / width))
        return done("lifted")
    return done("no_room")


def _skeleton_base(start: float, nadir: float, desc_end: int, rev_day: int | None,
                   slope: float, end_day: int, k_desc: float, k_reb: float,
                   budget_day: float, draws: dict, level: int,
                   T: int, pre_pace: bool = True) -> tuple[list[float], dict]:
    """Daily three-segment skeleton over days 0..end_day, and what shaped it.

    Level 4 is the plain skeleton. Lower levels re-pace the descent (when
    `pre_pace`) and, at level 0, draw the rebound shape. The daily step stays
    within `max(budget_day, the plain skeleton's own step)`. Drift, episodes and
    the regain check come after, in `_weight_render`.
    """
    span = max(1, end_day - rev_day) if rev_day is not None else 1
    end_v = nadir + slope * span / 7.0 if rev_day is not None else nadir
    drop = nadir - start
    d_hi = min(desc_end, end_day)

    plain = []
    for d in range(0, end_day + 1):
        if d <= desc_end:
            plain.append(start + drop * _shape(d / max(1, desc_end), k_desc))
        elif rev_day is None or d <= rev_day:
            plain.append(nadir)
        else:
            plain.append(nadir + (end_v - nadir) * _shape((d - rev_day) / span, k_reb))
    meta = {"level": level, "n_stall": 0, "stalls": [], "rebound": "gradual", "episodes": [],
            "base": plain}
    if level >= 4:
        return plain, meta
    base = list(plain)

    # Descent: the same start, nadir and end day, with an uneven pace.
    if pre_pace and abs(drop) > 1e-9 and d_hi >= 2:
        pieces = _descent_pieces(draws["descent"])
        prog = _paced_progress([_shape(d / max(1, desc_end), k_desc) for d in range(d_hi + 1)],
                               pieces, budget_day / abs(drop))
        for d in range(d_hi + 1):
            base[d] = start + drop * prog[d]
        # A stall: a phase whose realised progress is under a third of the plain one.
        stalls, lo_x = [], 0.0
        for end_x, _pace in pieces:
            lo, hi = int(round(lo_x * d_hi)), int(round(end_x * d_hi))
            lo_x = end_x
            if hi > lo and (prog[hi] - prog[lo]) * 3.0 < (plain[hi] - plain[lo]) / drop:
                stalls.append((lo, hi))
        meta["n_stall"], meta["stalls"] = len(stalls), stalls

    # Rebound: one of three shapes; the endpoint, and so the mean slope, stay fixed.
    # Only a rebound that starts at or after T has a drawn shape, so no fallback on
    # it can reach a pre-T value.
    if level == 0 and rev_day is not None and T <= rev_day < end_day:
        variant = SKEL_REBOUND_VARIANTS[int(draws["rebound"][0] * len(SKEL_REBOUND_VARIANTS))]
        meta["rebound"] = variant
        us = [(d - rev_day) / span for d in range(rev_day, end_day + 1)]
        if variant == "accelerating":
            # Half accelerating, half gradual: slower at first, yet the regain still
            # starts at the reversal.
            frac = [0.5 * (1.0 - _shape(1.0 - u, k_reb)) + 0.5 * _shape(u, k_reb) for u in us]
        elif variant == "steps":
            frac = _paced_progress([_shape(u, k_reb) for u in us],
                                   _rebound_pieces(draws["rebound"]),
                                   budget_day / max(1e-9, abs(end_v - nadir)))
        else:
            frac = [_shape(u, k_reb) for u in us]
        for i, d in enumerate(range(rev_day, end_day + 1)):
            if d > desc_end:
                base[d] = nadir + (end_v - nadir) * frac[i]

    meta["base"] = base
    return base, meta


def _no_regain_until(plain: list[dict], outcome: str) -> int | None:
    """For a no-regain case, the plain series' low day: post-T episodes end before it.

    After the low point the observation noise alone already brings runs near
    `MIN_PERSIST_DAYS`; lifting that stretch would let it meet the regain rule.
    Only post-T episodes read this, so it never reaches a pre-T value.
    """
    if outcome == "regain" or not plain:
        return None
    return int(min(plain, key=lambda q: float(q["value"]))["ts"])


def _steps_within(pts: list[dict], max_weekly: float, upto: int | None = None) -> bool:
    """Every step between consecutive readings stays within `max_weekly` kg per week,
    the limit GV-1 checks (without its 5 % tolerance)."""
    v = sorted((int(q["ts"]), float(q["value"])) for q in pts
               if upto is None or int(q["ts"]) <= upto)
    return all(abs(y1 - y0) <= max_weekly * (t1 - t0) / 7.0 + 1e-9
               for (t0, y0), (t1, y1) in zip(v, v[1:]))


#: At a declared reversal the weekly slope over the four weeks after must exceed the
#: slope over the four weeks before by at least this much (kg/week).
SKEL_REVERSAL_SLOPE_GAIN = 0.15


def _slope_kg_week(pts: list[dict], a: int, z: int) -> float | None:
    """Least-squares weight slope in kg/week over days [a, z]; None under five readings."""
    v = [(int(q["ts"]), float(q["value"])) for q in pts if a <= int(q["ts"]) <= z]
    if len(v) < 5:
        return None
    mx = sum(d for d, _ in v) / len(v)
    my = sum(y for _, y in v) / len(v)
    sxx = sum((d - mx) ** 2 for d, _ in v)
    return 7.0 * sum((d - mx) * (y - my) for d, y in v) / sxx if sxx else None


def _reversal_gain(pts: list[dict], rev_day: int) -> float | None:
    """Slope over the four weeks after `rev_day` minus the slope over the four before."""
    before = _slope_kg_week(pts, rev_day - 28, rev_day)
    after = _slope_kg_week(pts, rev_day, rev_day + 28)
    return None if before is None or after is None else after - before


def _reversal_visible(pts: list[dict], rev_day: int | None,
                      ref: list[dict] | None = None) -> bool:
    """Whether the regain visibly begins at `rev_day`: the slope gain there is at least
    `SKEL_REVERSAL_SLOPE_GAIN`, or as large as `ref`'s (the plain series) when that is
    smaller. True when there is no reversal or too few readings."""
    if rev_day is None:
        return True
    gain = _reversal_gain(pts, rev_day)
    if gain is None:
        return True
    need = SKEL_REVERSAL_SLOPE_GAIN
    ref_gain = _reversal_gain(ref, rev_day) if ref is not None else None
    if ref_gain is not None:
        need = min(need, ref_gain)
    return gain >= need - 1e-9


#: The emitted low point must stay this close (kg) to the declared nadir, inside GEN22's
#: 0.5 kg with room for the physiology layer.
SKEL_ANCHOR_MARGIN_KG = 0.35


def _label_guard(plain: list[dict], T: int, outcome: str, nadir: float | None = None):
    """`ok(series, upto=None)`: whether `series` keeps the label rule as `plain` does.

    The verdict must be the declared outcome or the plain series' verdict (they
    differ only where the plain loss is wobble-sized). The observation noise added
    later moves values by up to `SKEL_RUN_MARGIN_KG`, so runs are also read against
    the threshold shifted by that margin: a no-regain series may not stay above
    `threshold - margin` longer than `max(SKEL_MAX_RUN_DAYS, plain's run)`, and a
    regain series must stay above `threshold + SKEL_REGAIN_RUN_MARGIN_KG` at least as
    long as `min(MIN_PERSIST_DAYS, plain's run)`.
    With `upto`, both series are cut at that day and only the no-regain limit
    applies, to any outcome, so the pre-T check never reads the outcome.
    Either way the low point may not fall more than `SKEL_ANCHOR_MARGIN_KG` under the
    declared `nadir`, nor under the plain series' own low point where that is lower.
    """
    rule = {"min_change_frac": _LR.MIN_CHANGE_FRAC, "min_change_kg": _LR.MIN_CHANGE_KG,
            "smoothing": _LR.LABEL_SMOOTHING, "min_persist_days": _LR.MIN_PERSIST_DAYS}

    def read(series, upto, shift):
        """Verdict and the longest run after the low point above `threshold + shift`."""
        if upto is not None:
            series = [q for q in series if int(q["ts"]) <= upto]
        got, det = gates.derive_outcome({"weight": series}, rule)
        if got is None or "threshold" not in det:
            return got, 0
        # The run is read on the series the rule reads, from the same low point.
        vals = gates.label_series(series, rule)
        i_min = min(range(len(vals)), key=lambda i: float(vals[i]["value"]))
        line = float(det["threshold"]) + shift
        run, best = None, 0
        for q in vals[i_min:]:
            if float(q["value"]) >= line:
                run = int(q["ts"]) if run is None else run
                best = max(best, int(q["ts"]) - run)
            else:
                run = None
        return got, best

    def anchor_ok(series, upto):
        if nadir is None:
            return True
        vals = [float(q["value"]) for q in series if upto is None or int(q["ts"]) <= upto]
        pvals = [float(q["value"]) for q in plain if upto is None or int(q["ts"]) <= upto]
        # Never lower than the plain series already goes: where it sits at the edge of
        # GEN22's tolerance, any further dip would take the case out.
        line = nadir - SKEL_ANCHOR_MARGIN_KG
        if pvals:
            line = min(line, min(pvals))
        return not vals or min(vals) >= line - 1e-9

    def ok(series, upto=None):
        if not anchor_ok(series, upto):
            return False
        got, low = read(series, upto, -SKEL_RUN_MARGIN_KG)
        p_got, p_low = read(plain, upto, -SKEL_RUN_MARGIN_KG)
        if upto is not None:
            return got == p_got and low <= max(SKEL_MAX_RUN_DAYS, p_low)
        declared = "event_occurred" if outcome == "regain" else "event_not_occurred"
        if got not in (p_got, declared):
            return False
        if outcome != "regain":
            return low <= max(SKEL_MAX_RUN_DAYS, p_low)
        high = read(series, None, SKEL_REGAIN_RUN_MARGIN_KG)[1]
        p_high = read(plain, None, SKEL_REGAIN_RUN_MARGIN_KG)[1]
        return high >= min(_LR.MIN_PERSIST_DAYS, p_high)

    return ok


def _max_weekly_delta(disease: str) -> float:
    """`max_weekly_delta` of this disease's weight domain (1.5 kg/wk if absent)."""
    try:
        from latent import DISEASE_SIGNAL_DOMAIN as _DOM
        return float(((_DOM.get(disease) or {}).get("weight") or {}).get("max_weekly_delta", 1.5))
    except Exception:
        return 1.5


#: The last weight-skeleton summary per case, read into the build audit. The build runs
#: cases on a thread pool; each case writes and reads only its own key.
_SKELETON_AUDIT: dict[str, dict] = {}


def _record_skeleton(case_id: str, path: str, meta: dict) -> None:
    """Keep what the audit needs from a render: path, guard level and outcome, pre-T
    choices, the regain check and its deviations, the episode count."""
    _SKELETON_AUDIT[str(case_id)] = {
        "path": path, "level": meta.get("level"), "guard_ok": meta.get("guard_ok"),
        "pre": meta.get("pre"), "regain": meta.get("regain"),
        "regain_detail": meta.get("regain_detail"),
        "n_episodes": len(meta.get("episodes") or [])}


def _pchip_daily(src: list[tuple[int, float]], end: int, cap: float) -> list[float]:
    """Daily values over days 0..end through the anchors `src` (sorted `(ts, value)`),
    shape-preserving (Fritsch-Carlson): no overshoot, flat where two anchors are equal,
    exact at every anchor. An interval whose daily step would exceed
    `max(its chord slope, cap)` stays linear. Outside the anchors the end values hold.
    """
    xs = [x for x, _ in src]
    ys = [y for _, y in src]
    n = len(xs)
    h = [xs[k + 1] - xs[k] for k in range(n - 1)]
    dl = [(ys[k + 1] - ys[k]) / h[k] for k in range(n - 1)]
    der = [dl[0]] + [0.0] * (n - 2) + [dl[-1]]
    for k in range(1, n - 1):
        if dl[k - 1] * dl[k] > 0.0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            der[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])
    out = []
    for d in range(end + 1):
        if d <= xs[0]:
            out.append(ys[0])
        elif d >= xs[-1]:
            out.append(ys[-1])
        else:
            out.append(None)
    for k in range(n - 1):
        x0, x1, y0, y1 = xs[k], xs[k + 1], ys[k], ys[k + 1]
        seg = []
        for d in range(x0, x1 + 1):
            t = (d - x0) / h[k]
            seg.append((2 * t ** 3 - 3 * t ** 2 + 1) * y0 + (t ** 3 - 2 * t ** 2 + t) * h[k] * der[k]
                       + (-2 * t ** 3 + 3 * t ** 2) * y1 + (t ** 3 - t ** 2) * h[k] * der[k + 1])
        limit = max(abs(dl[k]), cap) + 1e-9
        if any(abs(seg[i + 1] - seg[i]) > limit for i in range(len(seg) - 1)):
            seg = [y0 + dl[k] * (d - x0) for d in range(x0, x1 + 1)]
        for i, d in enumerate(range(x0, x1 + 1)):
            if 0 <= d <= end:
                out[d] = seg[i]
    return out


def _weight_overlay(lin_pts: list[dict], model_pts: list, case_id: str, T: int, outcome: str,
                    nadir: float, disease: str = "obesity",
                    rev_week: int | None = None, declared_gain: float = 0.0,
                    declared_lost: float = 0.0) -> tuple[list[dict], dict]:
    """The LLM generator's weight series: a shape-preserving curve through the
    model's points, then the case's drift and episodes, then the regain check.

    `lin_pts` is `_resample_to_grid`'s linear version (the reference the label guard
    compares with); `model_pts` are the model's own points, which the curve passes
    through exactly. Up to T the choices (curve, drift, episodes) read pre-T readings
    only; after T the guard steps back one layer at a time and never moves a reading
    at or before T. Timestamps never change.
    """
    meta = {"level": 5, "episodes": [], "guard_ok": True}
    src: dict[int, float] = {}
    for q in model_pts or []:
        if isinstance(q, dict) and "ts" in q and "value" in q:
            src.setdefault(int(q["ts"]), float(q["value"]))
    src_l = sorted(src.items())
    if len(src_l) < 2 or len(lin_pts) < 3:
        return lin_pts, meta
    # When T is not a model point, the days from the last point before T up to T hold
    # that point's value. Interpolating toward the first point after T would let a point
    # written with the outcome in view shape days before T; extrapolating the pre-T slope
    # could run past the declared nadir or the descent budget, while a hold of under one
    # model step stays within both.
    before = [p for p in src_l if p[0] < T]
    if T not in src and before and src_l[-1][0] > T:
        src[T] = before[-1][1]
        src_l = sorted(src.items())
        lin_pts = _resample_to_grid(lin_pts, [{"ts": x, "value": y} for x, y in src_l])
    from .latent_rules import DESCENT_BUDGET_FRAC as _DESC_FRAC
    budget_day = _max_weekly_delta(disease) / 7.0 * _DESC_FRAC
    grid = sorted((int(q["ts"]), float(q["value"])) for q in lin_pts)
    end = grid[-1][0]
    lin, j = [], 0
    for d in range(end + 1):
        while j + 1 < len(grid) and grid[j + 1][0] <= d:
            j += 1
        (x0, y0), (x1, y1) = grid[j], grid[min(j + 1, len(grid) - 1)]
        lin.append(y0 if d <= x0 or x1 == x0 else y0 + (y1 - y0) * (d - x0) / (x1 - x0))
    # The pieces up to T and from T (a model point or the held point above) are
    # interpolated separately, each with one-sided end slopes, so no model point after T
    # shapes a day before it.
    pch = list(lin)
    for piece in ([p for p in src_l if p[0] <= T], [p for p in src_l if p[0] >= T]):
        if len(piece) >= 2:
            daily = _pchip_daily(piece, end, budget_day)
            lo, hi = piece[0][0], piece[-1][0]
            for d in range(max(0, lo), min(end, hi) + 1):
                pch[d] = daily[d]
    split = next((x for x, _ in src_l if x >= T), end)
    draws = _skeleton_draws(case_id)
    until = _no_regain_until(lin_pts, outcome)
    rev_day = int(rev_week) * 7 if (outcome == "regain" and rev_week is not None) else None

    def compose(pre, post):
        base = [(pch if (pre["pchip"] if d <= split else post["pchip"]) else lin)[d]
                for d in range(end + 1)]
        info = _apply_irregularity(base, draws, budget_day, nadir, T, end,
                                   pre_drift=pre["drift"], post_drift=post["drift"],
                                   pre_eps=pre["eps"], post_eps=post["eps"], post_t_until=until)
        caps = info.pop("caps")
        rg = (_ensure_regain(base, rev_day, T, end, budget_day, declared_gain=declared_gain,
                             declared_lost=declared_lost, caps=caps, nadir=nadir)
              if rev_day is not None else {"state": "n/a"})
        info["regain"], info["regain_detail"] = rg["state"], rg
        return [{**q, "value": round(base[int(q["ts"])], 3)} for q in lin_pts], info

    _rule_ok = _label_guard(lin_pts, T, outcome, nadir)
    _mw = _max_weekly_delta(disease)

    def ok(series, upto=None):
        return (_rule_ok(series, upto) and _steps_within(series, _mw, upto)
                and (upto is not None or _reversal_visible(series, rev_day, lin_pts)))
    off = {"pchip": False, "drift": False, "eps": False}
    pre = dict(off)
    for key in ("pchip", "drift", "eps"):
        trial = {**pre, key: True}
        if ok(compose(trial, trial)[0], upto=T):
            pre = trial
    for level, post in ((0, {"pchip": True, "drift": True, "eps": True}),
                        (2, {"pchip": True, "drift": True, "eps": False}),
                        (3, {"pchip": True, "drift": False, "eps": False}),
                        (4, off)):
        out, info = compose(pre, post)
        if ok(out):
            break
    meta = {"level": level, "pre": pre, "guard_ok": ok(out), **info}
    _record_skeleton(case_id, "llm", meta)
    return out, meta


def _weight_render(start: float, nadir: float, T: int, outcome: str,
                   rev_week: int, slope: float, case_id: str = "C",
                   step: int = WEIGHT_STEP, disease: str = "obesity",
                   end_day: int = END, nadir_day: int | None = None) -> tuple[list[dict], dict]:
    """`_weight_series` plus the skeleton's metadata (guard level, stalls, rebound
    variant, episodes, daily skeleton)."""
    step = max(1, int(step))
    rev_day = rev_week * 7 if outcome == "regain" else None
    # The low point's day comes from the latent layer (`nadir_day`, default T); with
    # a reversal point the descent ends at `min(T, rev_day)`.
    if rev_day is None:
        desc_end = int(nadir_day) if nadir_day else T
    else:
        desc_end = max(7, min(T, rev_day))

    # The wobble amplitude uses only the slope budget the trend leaves unused, so
    # the daily series stays within `max_weekly_delta`. Day-to-day fluid-balance
    # noise is not modeled.
    mw = _max_weekly_delta(disease)
    # The descent lasts at least as long as losing this much weight physically
    # takes; `latent_rules.DESCENT_BUDGET_FRAC` must equal the fraction used here.
    from .latent_rules import DESCENT_BUDGET_FRAC as _DESC_FRAC
    _need = abs(nadir - start) / max(1e-9, mw / 7.0 * _DESC_FRAC)
    desc_end = max(desc_end, int(math.ceil(_need)))
    trend = max(abs(nadir - start) / max(1, desc_end), abs(slope) / 7.0)
    budget = max(0.0, mw / 7.0 * 0.9 - trend)
    amp = min(0.12, budget * WOBBLE_PERIOD / (2 * math.pi))
    _k_want, _k_reb_want = _trajectory_shape(case_id)
    _budget_day = mw / 7.0 * _DESC_FRAC
    _k_desc = _feasible_k(abs(nadir - start) / max(1, desc_end), _budget_day, _k_want)
    _k_reb = _feasible_k(abs(slope) / 7.0, _budget_day, _k_reb_want)

    def emit(base: list[float]) -> list[dict]:
        pts, gap = [], 0
        for d in range(0, end_day + 1, step):
            # Fluctuation is AR(1), seeded per case (as in `events.render_stream`),
            # within the `amp` budget.
            if d == 0:
                _w_phi = events_mod._AR_PHI ** max(1, int(step))
                _w_sd = events_mod._AR_SD_FRAC * amp
                _w_sig = _w_sd * math.sqrt(max(0.0, 1.0 - _w_phi * _w_phi))
                _w_bound = _w_sig * math.sqrt(3.0)
                _w_key = f"{case_id}|weight|wobble"
                _w_x = _w_sd * events_mod._det_shock(_w_key, -1)
                _w_k = 0
            else:
                _w_k += 1
                _w_x = _w_phi * _w_x + _w_bound * events_mod._det_shock(_w_key, _w_k)
            v = base[d] + _w_x
            keep = (step != WEIGHT_STEP or d == 0
                    or gap >= WEIGHT_MAX_GAP or _weighed_on(d, case_id))
            if not keep:
                gap += 1
                continue
            gap = 0
            pts.append({"ts": d, "value": round(v, 2)})
        return pts

    draws = _skeleton_draws(case_id)
    args = (start, nadir, desc_end, rev_day, slope, end_day, _k_desc, _k_reb, _budget_day, draws)
    plain, _ = _skeleton_base(*args, level=4, T=T)
    plain_pts = emit(plain)
    _rule_ok = _label_guard(plain_pts, T, outcome, nadir)

    def ok(series, upto=None):
        return (_rule_ok(series, upto) and _steps_within(series, mw, upto)
                and (upto is not None or _reversal_visible(series, rev_day, plain_pts)))
    until = _no_regain_until(plain_pts, outcome)

    def compose(level, pre, post_drift, post_eps):
        base, meta = _skeleton_base(*args, level=min(level, 1), T=T, pre_pace=pre["pace"])
        if level >= 4:
            base = _post_t_plain(base, plain, T, _budget_day)
        meta.update(_apply_irregularity(base, draws, _budget_day, nadir, T, end_day,
                                        pre_drift=pre["drift"], post_drift=post_drift,
                                        pre_eps=pre["eps"], post_eps=post_eps,
                                        post_t_until=until))
        caps = meta.pop("caps")
        rg = (_ensure_regain(base, rev_day, T, end_day, _budget_day,
                             declared_gain=slope * max(1, end_day - rev_day) / 7.0,
                             declared_lost=start - nadir, caps=caps, nadir=nadir)
              if rev_day is not None else {"state": "n/a"})
        meta["regain"], meta["regain_detail"] = rg["state"], rg
        meta.update(level=level, base=base, plain=plain, pre=dict(pre))
        return emit(base), meta

    # Pre-T choices read pre-T readings only, the same way in every arm, one layer
    # at a time: descent pacing, drift, episodes.
    pre = {"pace": False, "drift": False, "eps": False}
    for key in ("pace", "drift", "eps"):
        trial = {**pre, key: True}
        if ok(compose(1, trial, trial["drift"], trial["eps"])[0], upto=T):
            pre = trial
    # Later levels change post-T days only.
    for level, post_drift, post_eps in ((0, True, True), (1, True, True), (2, True, False),
                                        (3, False, False), (4, False, False)):
        pts, meta = compose(level, pre, post_drift, post_eps)
        if ok(pts):
            break
    meta["guard_ok"] = ok(pts)
    _record_skeleton(case_id, "det", meta)
    return pts, meta


def _post_t_plain(base: list[float], plain: list[float], T: int, budget_day: float) -> list[float]:
    """`base` up to T, then the plain skeleton, joined by a smooth fade of the
    offset at T; slow enough to add at most half the descent budget per day."""
    out = list(base)
    if T >= len(base) - 1:
        return out
    off = base[T] - plain[T]
    width = max(28, int(math.ceil(3.0 * abs(off) / max(1e-9, budget_day))))
    for d in range(T + 1, len(base)):
        out[d] = plain[d] + off * (1.0 - _smoothstep((d - T) / width))
    return out


def _weight_series(start: float, nadir: float, T: int, outcome: str,
                   rev_week: int, slope: float, case_id: str = "C",
                   step: int = WEIGHT_STEP, disease: str = "obesity",
                   end_day: int = END, nadir_day: int | None = None) -> list[dict]:
    """Weight trajectory: descent -> plateau -> (if regain) rebound from `rev_day`,
    continuous at every seam, with the per-case irregularities of `_skeleton_base`.

    `step` is the declared `sampling_days` (GEN6). Under daily sampling, day 0
    is always kept and everyday gaps never exceed `WEIGHT_MAX_GAP`.
    """
    return _weight_render(start, nadir, T, outcome, rev_week, slope, case_id=case_id,
                          step=step, disease=disease, end_day=end_day,
                          nadir_day=nadir_day)[0]


class HaenvGenerator:
    """Deterministically generates the disease course from the latent premise;
    conflict checking is left to synth (GV-1).
    """

    def generate(self, p: LatentPremise, original_case, feedback) -> RawCase:
        m = p.meta
        T = int(m.get("prediction_time_T", 84))
        start, nadir = float(m["start"]), float(m["nadir"])
        # Density can only thin a stream: `step_eff = max(declared step,
        # round(7 / measure_per_week))`, a no-op at the default `mpw=7`.
        _ed_mpw = float((getattr(p, "event_density", None) or {}).get("measure_per_week", 7) or 7)
        _dens_step = max(1, int(round(7.0 / max(0.1, _ed_mpw))))
        _cid = str(m.get("case_id") or "HAENV")
        outcome = _job_outcome_of(_cid, m)
        rev_week = int(m.get("reversal_week", (T + 56) // 7))
        slope = float(m.get("regain_slope", 0.35))
        occurred = outcome == "regain"
        driver = _job_driver_of(_cid, m)
        ce = int(m.get("course_end_day", END))
        # With `regain_end_kg`, the rebound slope is back-solved from the endpoint
        # over the rebound weeks; it is not clamped, so an infeasible endpoint fails
        # GV-1 visibly.
        _end_kg = m.get("regain_end_kg")
        if _end_kg is not None and outcome == "regain":
            slope = (float(_end_kg) - nadir) / max(1e-6, (ce - rev_week * 7) / 7.0)
        w_step = int(_raw_field((p.device_signals.get("signals") or {}).get("weight") or {},
                                "sampling_days"))
        wpts = _weight_series(start, nadir, T, outcome, rev_week, slope,
                              case_id=m.get("case_id", "HAENV"),
                              step=max(w_step, _dens_step), end_day=ce,
                              disease=_raw_field(p.patient_basics, "disease"),
                              nadir_day=m.get("nadir_day"))

        # The wearable EV comes from the weight stream at ≤T (see `ledger_weight_point`).
        _ev_ts, _ev_val = ledger_weight_point({"weight": wpts}, T, _cid)

        adh = [{"ts": pt["day"], "value": pt["level"]}
               for pt in p.adherence.get("trajectory", [])] or \
              [{"ts": T, "value": p.adherence.get("baseline", 0.95)}]
        steps = p.patient_basics.get("regimen", {}).get("dose_steps", [2.5, 5.0, 7.5])
        # Maintenance dose after the ladder, through the course end day.
        dose = [{"ts": i * 28, "value": v} for i, v in enumerate(steps)]
        _last = (len(steps) - 1) * 28
        dose += [{"ts": d, "value": steps[-1]}
                 for d in (_last + (ce - _last) // 2, ce) if d > _last]

        return RawCase(
            case_id=m.get("case_id", "HAENV"),
            user_profile={"age_range": _raw_field(p.patient_basics, "age_range"),
                          "sex": _raw_field(p.patient_basics, "sex"),
                          "known_conditions": [_raw_field(p.patient_basics, "disease")],
                          "treatment_goals": p.patient_basics.get("goals", ["sustained_weight_loss"]),
                          "device_inventory": p.device_signals.get("devices", [])},
            prediction_context={
                                # External task types' solver-visible block; `{}` when none is registered.
                                **_EG.probe_blocks_for(m),
                                "prediction_time_T": T,
                                "target_event_type": m.get("target_event_type", "weight_regain"),
                                "prediction_window": f"{ce - T}d",
                                "available_history_window": f"{T}d"},
            longitudinal_data={"weight": wpts, "dose_timeline": dose,
                               "medication_adherence": adh,
                               **{s: render_clinical(
                                       s, c, wpts, ce,
                                       drug=_drug_of(p),
                                       dose_mg=_last_dose_mg(dose),
                                       adherence_pts=adh, case_id=_cid,
                                       response=_drug_response_of(p))
                                  for s, c in (m.get("clinical") or {}).items()},
                               **gold_evidence_streams(driver, rev_week, occurred, T,
                                                       end_day=ce,
                                                       dens_step=_dens_step,
                                                       case_id=_cid)},
            evidence_ledger=[{"evidence_id": f"EV-{m.get('case_id','C')}-1",
                              "source_type": "wearable", "source_timestamp": _ev_ts,
                              "measured_value": _ev_val, "claim_supported": True,
                              "reliability_status": "reliable"}],
            # A declared outcome_label (diagnosis cases) takes precedence; otherwise from the weight direction.
            outcome_label=(m.get("outcome_label")
                           or ("event_occurred" if occurred else "event_not_occurred")),
            # The judge reads the structured fields; the string is for humans.
            label_rule={"minimum_change_magnitude":
                            f"regain >= max({_LR.MIN_CHANGE_FRAC:.0%} of lost weight, "
                            f"{_LR.MIN_CHANGE_KG} kg), read on the 7-reading centred rolling median; "
                            f"with less than {_LR.MIN_CHANGE_KG} kg lost it is a gain of "
                            f"{_LR.MIN_CHANGE_KG} kg above the low point",
                        "minimum_persistence": "sustained >=8w", "baseline_window": "nadir",
                        # Thresholds come from `latent_rules`, which also checks them before generation.
                        "min_change_frac": _LR.MIN_CHANGE_FRAC,
                        "min_change_kg": _LR.MIN_CHANGE_KG,
                        "smoothing": _LR.LABEL_SMOOTHING,
                        "min_persist_days": _LR.MIN_PERSIST_DAYS},
            gold_drivers=[driver],
            adjudication={"adjudication_protocol_present": True, "primary_driver": driver,
                          # From latent.ddx_red_flag; a hardcoded False would disarm missed_emergency_red_flag.
                          "red_flag_present": bool(m.get("red_flag", False)),
                          "clinician_action_warranted": (occurred
                                                         if m.get("clinician_warranted") is None
                                                         else bool(m.get("clinician_warranted"))),
                          **({"ddx": m["ddx"]} if m.get("ddx") else {}),
                          # External task types' gold block; `{}` when none is registered.
                          **_EG.blocks_for(m)},
            reversal_points=([{"week": rev_week, "type": "real", "flip": "risk_low->elevated",
                               "trigger": f"day{rev_week*7} 起回升,证伪'能维持'"}] if occurred else []),
        )


# --------------------------------------------------------------- LLM generator
_CASE_PROMPT = """你是"合成病程"生成器。依据下面的**隐变量前提**与**原始病例事实**,生成一段纵向病程观测。
只输出一个 JSON 对象(无多余文字、无 markdown 代码围栏)。

## 硬约束(违反会被确定性校验器判负并要求你重出)
1. 信号名只能用:{signals};单位与取值必须落在前提声明的 plausible_range 内。
2. 相邻采样点的**周变化**不得超过生理上限:{deltas}。
3. `dose_timeline` 只能取该药剂量梯级 {ladder} 中的值,相邻两次**变更**间隔 ≥ {titration} 天;
   首次给药之前主信号不得已明显偏离基线(药效不能早于用药)。
4. 主信号必须走出前提要求的形态:从 {start} 起降至约 {nadir}(第 {nadir_day} 天前后),
   {shape}
5. `medication_adherence` 取 0–1,最低点须与前提的依从轨迹大致相符(不得比前提最低值再低 0.2 以上)。
6. 时间范围:ts 从 0 到 {end} 天;主信号按每 7 天一个点即可(不必逐天)。
7. **答案侧信息只能出现在 outcome_label/gold_drivers/adjudication/reversal_points 这些字段里**;
   `evidence_ledger` 的任何文字(症状、备注)**不得**出现结局词/归因词(如"复胖""停药后""甲亢""无效""依从性差"),
   因为 evidence_ledger 会原样交给被测模型看。

## 隐变量前提(verifier 侧,含本例的真结局与真驱动 —— 据此生成,但别写进 evidence_ledger)
{premise}

## 原始病例事实(必须与之一致)
{original}

## 上一轮被校验器判负的冲突(必须逐条修掉)
{feedback}

## 输出 JSON 结构
{{"longitudinal_data":{{"<主信号>":[{{"ts":0,"value":..}},...],
   "dose_timeline":[{{"ts":0,"value":..}},...],"medication_adherence":[{{"ts":..,"value":0-1}},...]}},
 "evidence_ledger":[{{"evidence_id":"EV-{cid}-1","source_type":"smart_scale|wearable|lab_panel|clinic_scale|cgm|bp_cuff",
   "source_timestamp":<=T,"measured_value":..,"claim_supported":true,"reliability_status":"reliable"}}],
 "label_rule":{{"minimum_change_magnitude":"..","minimum_persistence":"..","baseline_window":"nadir"}}}}"""


class LLMCaseGenerator:
    """Generates the disease course with a real model: structural fields come from
    the premise, numeric sequences from the model.

    Output goes through `synth.synthesize`'s GV-1 loop, with conflicts fed back
    for regeneration. The gold standard (outcome_label/gold_drivers/
    adjudication/reversal_points) always comes from the hidden control
    variables, never from the model.
    """

    def __init__(self, dispatch, cs: CaseSpec):
        self.dispatch, self.cs = dispatch, cs

    def generate(self, p: LatentPremise, original_case, feedback) -> RawCase:
        from latent import DISEASE_SIGNAL_DOMAIN, drug_pkpd
        from solver import _extract_json

        m = p.meta
        T = int(m.get("prediction_time_T", 84))
        disease = _raw_field(p.patient_basics, "disease")
        domain = DISEASE_SIGNAL_DOMAIN.get(disease, {})
        sigs = list(p.device_signals.get("signals", {}))
        pk = drug_pkpd(p.patient_basics.get("regimen", {}).get("drug", "")) or {}
        # Same single source as `HaenvGenerator.generate`.
        outcome = _job_outcome_of(str(m.get("case_id") or self.cs.case_id), m)
        rev_week = int(m.get("reversal_week", (T + 56) // 7))
        _ce_llm = int(m.get("course_end_day", END))
        shape = (f"随后在第 {rev_week * 7} 天前后转为回升,到第 {_ce_llm} 天累计回升 "
                 f"≥ 已减体重的 5%(斜率约 {m.get('regain_slope', 0.35)}/周)。"
                 if outcome == "regain" else
                 f"随后一直维持在该水平附近(到第 {_ce_llm} 天不得出现持续 8 周以上的明显回升)。")
        prompt = _CASE_PROMPT.format(
            signals=sigs, cid=self.cs.case_id,
            deltas={s: domain.get(s, {}).get("max_weekly_delta") for s in sigs},
            ladder=list(pk.get("ladder", ())), titration=pk.get("min_titration_days", 28),
            start=m.get("start"), nadir=m.get("nadir"), nadir_day=T, shape=shape, end=_ce_llm,
            premise=json.dumps(_world_prompt_premise(p), ensure_ascii=False),
            original=json.dumps(original_case or {}, ensure_ascii=False),
            feedback=json.dumps(sorted({x.get("kind") for x in (feedback or [])}),
                                ensure_ascii=False) or "(首轮,无)")
        data = _extract_json(self.dispatch(prompt, require_json=True))

        raw = HaenvGenerator().generate(p, original_case, feedback)   # deterministic skeleton for structure + gold standard
        ld = data.get("longitudinal_data") or {}
        for name in list(raw.longitudinal_data):                     # only replace the values the model provided
            pts = ld.get(name)
            if isinstance(pts, list) and len(pts) >= 3:
                raw.longitudinal_data[name] = _resample_to_grid(
                    raw.longitudinal_data[name], pts)
                if name == "weight":
                    raw.longitudinal_data[name] = _weight_overlay(
                        raw.longitudinal_data[name], pts,
                        str(m.get("case_id") or self.cs.case_id),
                        T, outcome, float(m["nadir"]), disease=disease,
                        rev_week=rev_week,
                        declared_gain=((float(m["regain_end_kg"]) - float(m["nadir"]))
                                       if m.get("regain_end_kg") is not None else
                                       float(m.get("regain_slope", 0.35))
                                       * max(1, _ce_llm - rev_week * 7) / 7.0),
                        declared_lost=float(m["start"]) - float(m["nadir"]))[0]
        evs = data.get("evidence_ledger")
        if isinstance(evs, list) and evs:
            raw.evidence_ledger = [e for e in evs if isinstance(e, dict) and e.get("evidence_id")]
        # The model's `label_rule` is discarded: it decides `outcome_label`, and the
        # gold label is never handed to the generation model.
        if isinstance(data.get("label_rule"), dict) and data["label_rule"]:
            log.debug("[build] %s the model supplied label_rule; dropped (it would decide the gold label)",
                      self.cs.case_id)
        log.info("[build] %s course generated by the LLM (signals %s, %d evidence entries)", self.cs.case_id,
                 {k: len(v) for k, v in raw.longitudinal_data.items()}, len(raw.evidence_ledger))
        return raw



def _resample_to_grid(grid: list[dict], pts: list) -> list[dict]:
    """Resamples the model's `(ts, value)` sequence onto the code-defined grid.

    Timestamps and missingness come from the premise (so GEN6's cadence check
    holds); only values come from the model. Points outside the model's range
    take the nearest endpoint value, never an extrapolation.
    """
    src = sorted(
        ({"ts": int(x["ts"]), "value": float(x["value"])}
         for x in pts if isinstance(x, dict) and "ts" in x and "value" in x),
        key=lambda d: d["ts"])
    if len(src) < 2 or not grid:
        return grid                                   # too few points: keep the deterministic sequence
    xs = [d["ts"] for d in src]
    ys = [d["value"] for d in src]

    def _at(t: int) -> float:
        if t <= xs[0]:
            return ys[0]
        if t >= xs[-1]:
            return ys[-1]
        import bisect
        i = bisect.bisect_left(xs, t)
        if xs[i] == t:
            return ys[i]
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        return y0 + (y1 - y0) * (t - x0) / max(1, x1 - x0)

    return [{"ts": int(g["ts"]), "value": round(_at(int(g["ts"])), 3)} for g in grid]


def original_facts(cs: CaseSpec, T: int) -> dict:
    """The case facts handed to the generator (no outcome); anchor values also feed
    premise_conflicts for the recheck.
    """
    raw = cs.raw
    start = float(_raw_field(raw, "start_weight"))
    nadir = float(_raw_field(raw, "nadir_weight"))
    return {"disease": raw.get("disease"), "drug": raw.get("drug"),
            "dose_steps": raw.get("dose_steps"), "devices": raw.get("devices"),
            "age_range": raw.get("age_range"), "sex": raw.get("sex"),
            "comorbidities": _raw_field(raw, "comorbidities"), "bmi": raw.get("bmi"),
            "baseline_vitals": raw.get("baseline_vitals", {}),
            "index_time_T": T, "course_end_day": course_end_of(cs.case_id, cs.latent),
            "anchor_values": {"weight": {"day": 0, "value": start, "tol": 1.0}},
            "nadir_weight": nadir}


# --------------------------------------------------------------- Clinical signal streams (GEN7)
#
# Every declared device produces its streams (lab_panel, cgm, bp_cuff). Names
# must match the kernel's `DISEASE_SIGNAL_DOMAIN`, be declared in the premise
# with the actual step (GEN6), and respect `max_weekly_delta`. Values follow
# weight with a lag and attenuation, so they never disclose more than weight.
CLINICAL_BY_DEVICE: dict[str, set[str]] = {
    "lab_panel": {"HbA1c", "LDL", "ALT", "AST", "FIB4", "triglycerides", "fasting_glucose"},
    "cgm": {"CGM_TIR", "fasting_glucose"},
    "bp_cuff": {"systolic_bp", "diastolic_bp"},
}
# Per signal: (`per_kg`, `step`). Rendered as
# base + per_kg x ATTEN x (w(t-lag) - w0), so `per_kg` is the change per 1 kg
# gained (GEN15 checks its sign). Baselines come per cohort from
# `registry/indicators.yaml` and `ndigits` from the indicator dossier. `FIB4`
# is an independent stream, not computed from ALT/AST.
CLINICAL_SPEC: dict[str, tuple[float, int]] = {
    #                 per_kg   step
    "HbA1c":           (0.06,   90),
    "fasting_glucose": (0.05,   90),
    "CGM_TIR":         (-1.20,   7),    # weight gain -> time-in-range drops
    "LDL":             (0.010,  90),
    "triglycerides":   (0.020,  90),
    "ALT":             (0.80,   90),
    "AST":             (0.50,   90),
    "FIB4":            (0.012,  90),
    # BP coefficients per kg: Neter JE et al., Hypertension 2003;42(5):878-84
    # (PMID 12975389): -1.05 mmHg systolic, -0.92 mmHg diastolic per kg lost.
    "systolic_bp":     (1.05,    7),
    "diastolic_bp":    (0.92,    7),
}
CLINICAL_LAG_DAYS = 30          # lab values reflect roughly a month-old state
CLINICAL_ATTEN = 0.75           # lab values only partly track weight


@_registry_cached("clinical_baselines.yaml")
def _clinical_baselines() -> dict:
    """`registry/clinical_baselines.yaml` -- cohort tiers for clinical baselines."""
    from .regpath import load_registry as _lr
    return _lr("clinical_baselines.yaml") or {}


def clinical_cohort(sig: str, disease: str, comorbidities) -> str | None:
    """Which cohort this case belongs to on `sig` (via the registry's
    `diabetes_markers`, matched against disease and comorbidities); `None` if the
    signal has no cohort tiers.
    """
    reg = _clinical_baselines()
    entry = ((reg.get("signals") or {}).get(sig) or {})
    if not entry.get("cohorts"):
        return None
    hay = " ".join([str(disease or "")] + [str(c) for c in (comorbidities or [])]).lower()
    if any(str(m).lower() in hay for m in (reg.get("diabetes_markers") or [])):
        return "diabetic"
    return str(entry.get("default_cohort") or "non_diabetic")


def clinical_plan(disease: str, devices: list[str],
                  comorbidities=(), case_id: str = "") -> dict[str, dict]:
    """(disease + comorbidity domains ∩ what the devices can produce) - weight -> the
    clinical signals this case generates, with their parameters. Baselines are
    sampled per cohort.
    """
    from latent import DISEASE_SIGNAL_DOMAIN
    # The domain includes comorbidities; a signal still needs a device that
    # produces it (`CLINICAL_BY_DEVICE`).
    domain = dict(DISEASE_SIGNAL_DOMAIN.get(disease, {}))
    for _c in (comorbidities or ()):
        domain.update(DISEASE_SIGNAL_DOMAIN.get(str(_c), {}))
    producible: set[str] = set()
    for dev in devices or []:
        producible |= CLINICAL_BY_DEVICE.get(dev, set())
    _sigs = [s for s in sorted(set(domain) & producible)
             if s != "weight" and s in CLINICAL_SPEC]
    # Sampled as one group, so an umbrella diagnosis (e.g. dyslipidemia: LDL or TG)
    # shows up in at least one signal.
    _bases = _indicators.sample_case(disease, _sigs, comorbidities, case_id=str(case_id))
    out: dict[str, dict] = {}
    for sig in _sigs:
        per_kg, step = CLINICAL_SPEC[sig]
        nd = _indicators.of(sig)["ndigits"]
        base = _bases[sig]
        out[sig] = {"base": base, "per_kg": per_kg, "step": step, "ndigits": nd,
                    # drug effects apply only to the cohort they were measured on
                    "cohort": _indicators.cohort_of(sig, disease, comorbidities),
                    "unit": domain[sig]["unit"], "range": list(domain[sig]["range"]),
                    "max_weekly_delta": domain[sig]["max_weekly_delta"]}
    return out


def _drug_of(p) -> str:
    return str((p.patient_basics.get("regimen") or {}).get("drug") or "")


def _drug_response_of(p) -> float:
    """This case's drug-response multiplier (`drug_effects.response_for`); shared by
    rendering and GEN15.
    """
    m = p.meta or {}
    return _drug_effects.response_for(str(m.get("case_id") or "HAENV"), m.get("driver"),
                                      _drug_of(p), m.get("drug_response"))


def _drug_terms_for(raw, p, cplan: dict) -> dict[str, float]:
    """Net drug-effect increment per clinical signal over the observation window,
    for GEN15's expected direction; same `effect_at` as `render_clinical`.
    """
    ld = raw.longitudinal_data or {}
    drug = _drug_of(p)
    resp = _drug_response_of(p)
    dose = _last_dose_mg(ld.get("dose_timeline"))
    adh = ld.get("medication_adherence") or []

    def _adh_at(day: int) -> float:
        q = [x for x in adh if isinstance(x, dict) and int(x.get("ts", -1)) <= day]
        return float(q[-1]["value"]) if q else 1.0

    out: dict[str, float] = {}
    for sig, spec in (cplan or {}).items():
        pts = ld.get(sig) or []
        if len(pts) < 2:
            continue
        d0, d1 = int(pts[0]["ts"]), int(pts[-1]["ts"])
        kw = dict(per_kg=spec["per_kg"], atten=CLINICAL_ATTEN, dose_mg=dose,
                  cohort=spec.get("cohort"), response=resp)
        out[sig] = (_drug_effects.effect_at(drug, sig, d1, _adh_at, **kw)
                    - _drug_effects.effect_at(drug, sig, d0, _adh_at, **kw))
    return out


def _last_dose_mg(dose_pts) -> float | None:
    """The dose (mg) at the last point of the dose trajectory (current dose, not the
    maximum); `None` if empty.
    """
    pts = [q for q in (dose_pts or [])
           if isinstance(q, dict) and q.get("value") is not None]
    return float(pts[-1]["value"]) if pts else None


@functools.lru_cache(maxsize=1)
def _assert_kernel_params_in_sync() -> bool:
    """Checks the artifact-amplitude registry against the kernel's default
    arguments, once per process on the generation path.
    """
    from noise import NOISE_CLASSES
    _artifact_params.assert_matches_kernel(NOISE_CLASSES)
    return True


def world_layer_gaps() -> dict[str, object]:
    """The world layer's known gaps (pending cohorts, uncalibrated parameters,
    drugs without effect data), recorded in each case's `audit`.
    """
    return {
        "pending_cohorts": _indicators.pending_cohorts(),
        "known_defects": sorted(_indicators.known_defects()),
        "median_only_cohorts": {k: v for k, v in _indicators.distribution_status().items()
                                if not str(v).startswith("分布")},
        "uncalibrated_artifact_params": _artifact_params.missing_calibration(),
        # drug pool read live, so a newly added drug is covered
        "drugs_without_effect_data": _drug_effects.missing_drugs(
            {d for dis, _w in _demographics.disease_pool()
             for d, _s in _demographics.drugs_for(dis)}),
        "drug_effect_cohorts": sorted(
            {c for c in (_drug_effects._doc().get("applies_to_cohorts_default") or [])}),
    }


@_registry_cached("physio_streams.yaml")
def _clinical_cv() -> dict:
    """`registry/physio_streams.yaml:clinical_measurement.cv` -- within-person
    measurement-to-measurement CV per indicator. `null` entries add no noise.
    """
    from .regpath import load_registry as _lr
    doc = _lr("physio_streams.yaml") or {}
    raw = ((doc.get("clinical_measurement") or {}).get("cv") or {})
    return {k: float(v) for k, v in raw.items() if v is not None}


def _quantize_within_slope(prev: dict | None, v: float, day: int,
                           mwd: float, ndigits: int) -> float:
    """Rounds to `ndigits` while keeping the weekly-slope limit.

    When the allowed change over the interval is smaller than one quantization
    step (e.g. HbA1c 3 days apart), the only legal value is `prev`. `prev is
    None` (first point) => just rounded.
    """
    step = 10.0 ** (-int(ndigits)) if ndigits else 1.0
    q = round(v, int(ndigits)) if ndigits else float(round(v))
    if prev is None:
        return q
    gap = max(1e-9, (int(day) - int(prev["ts"])) / 7.0)
    n_step = math.floor(float(mwd) * gap / step + 1e-9)
    p = float(prev["value"])
    q = min(max(q, p - n_step * step), p + n_step * step)
    return round(q, int(ndigits)) if ndigits else float(round(q))


def render_clinical(sig: str, spec: dict, weight_pts: list[dict], end_day: int,
                    drug: str = "", dose_mg: float | None = None,
                    adherence_pts: list[dict] | None = None,
                    case_id: str = "C", response: float = 1.0) -> list[dict]:
    """Derives one clinical signal from the weight trajectory: lag + attenuation +
    drug effect + value-range clamp + weekly-slope clamp.

        v = base + per_kg×ATTEN×Δw  +  effect_at(drug, sig, day, adherence(day))

    `effect_at` excludes the weight-mediated part, so it is not counted twice.
    `weight` itself never gets a drug term (`drug_effects.FORBIDDEN`).
    """
    if not weight_pts:
        return []
    w0 = float(weight_pts[0]["value"])

    def adh_at(day: int) -> float:
        """Adherence on this day; 1.0 when unknown (trial effect sizes assume full
        adherence).
        """
        pts = [q for q in (adherence_pts or [])
               if isinstance(q, dict) and int(q.get("ts", -1)) <= day]
        return float(pts[-1]["value"]) if pts else 1.0

    def w_at(day: int) -> float:
        d = max(0, day - CLINICAL_LAG_DAYS)
        prev = [p for p in weight_pts if int(p["ts"]) <= d]
        return float((prev[-1] if prev else weight_pts[0])["value"])

    lo, hi = spec["range"]
    mwd = float(spec["max_weekly_delta"])
    _cv = _clinical_cv().get(sig, 0.0)
    pts: list[dict] = []
    for _k, day in enumerate(range(0, end_day + 1, int(spec["step"]))):
        v = spec["base"] + spec["per_kg"] * CLINICAL_ATTEN * (w_at(day) - w0)
        v += _drug_effects.effect_at(drug, sig, day, adh_at,
                                     per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                     dose_mg=dose_mg, cohort=spec.get("cohort"),
                                     response=response)
        # Measurement variation: independent per draw (indexed by draw, not day).
        if _cv:
            v *= 1.0 + _cv * math.sqrt(3.0) * events_mod._det_shock(
                f"{case_id}|{sig}|meas", _k)
        v = min(max(v, lo), hi)
        pts.append({"ts": day,
                    "value": _quantize_within_slope(
                        pts[-1] if pts else None, v, day, mwd, spec["ndigits"])})
    # Add a final sample on `end_day`, which a coarse step would otherwise miss.
    if pts and int(pts[-1]["ts"]) < int(end_day):
        v = spec["base"] + spec["per_kg"] * CLINICAL_ATTEN * (w_at(int(end_day)) - w0)
        v += _drug_effects.effect_at(drug, sig, int(end_day), adh_at,
                                     per_kg=spec["per_kg"], atten=CLINICAL_ATTEN,
                                     dose_mg=dose_mg, cohort=spec.get("cohort"),
                                     response=response)
        if _cv:
            v *= 1.0 + _cv * math.sqrt(3.0) * events_mod._det_shock(
                f"{case_id}|{sig}|meas", len(pts))
        v = min(max(v, lo), hi)
        pts.append({"ts": int(end_day),
                    "value": _quantize_within_slope(
                        pts[-1], v, int(end_day), mwd, spec["ndigits"])})
    return pts


# --------------------------------------------------------------- Premise construction
#: Premise `meta` keys about delivery to the solver, not the patient; kept out
#: of the world-course prompt so they cannot change the patient's course.
OBSERVATION_ONLY_META = ("rhythm_gap",)

#: Premise `meta` keys the world-course prompt may see; all others are
#: withheld. Every key `premise_spec` emits is in exactly one of the two tuples.
WORLD_PROMPT_META = (
    _EG.SLOT, "case_id", "difficulty_class", "target_event_type", "prediction_time_T",
    "start", "nadir", "course_end_day", "nadir_day", "outcome", "driver",
    "reversal_week", "regain_slope", "regain_end_kg", "clinical", "ddx", "red_flag",
    "clinician_warranted", "outcome_label", "drug_response",
)


def _world_prompt_premise(p) -> dict:
    d = p.dumps()
    d["meta"] = {k: v for k, v in (d.get("meta") or {}).items() if k in WORLD_PROMPT_META}
    return d


def _register_comorbid_domain(disease: str, comorbidities):
    """Adds the comorbidities' signals to `DISEASE_SIGNAL_DOMAIN[disease]` for the
    duration of one case, without editing kernel source.

    The kernel's premise checks only look at the primary condition's domain.
    The addition is scoped to the current thread and removed afterwards, so it
    never leaks into other cases.
    """
    from contextlib import contextmanager                     # noqa: PLC0415

    @contextmanager
    def _noop():
        yield

    if not comorbidities:
        return _noop()
    from latent import DISEASE_SIGNAL_DOMAIN
    dom = DISEASE_SIGNAL_DOMAIN.get(str(disease))
    if dom is None:
        return _noop()

    extra: dict = {}
    for c in comorbidities:
        for sig, spec in (DISEASE_SIGNAL_DOMAIN.get(str(c)) or {}).items():
            if sig not in dom:
                extra.setdefault(sig, spec)
    return DISEASE_SIGNAL_DOMAIN.scoped(str(disease), extra)


def premise_spec(cs: CaseSpec, T: int = 84) -> dict:
    """raw (case facts) + latent (hidden control variables) -> the latent premise
    spec.
    """
    raw, lat = cs.raw, cs.latent
    disease = _raw_field(raw, "disease")
    drug = _raw_field(raw, "drug")
    steps = list(raw.get("dose_steps", [2.5, 5.0, 7.5]))
    devices = list(raw.get("devices", ["smart_scale", "wearable"]))
    start = float(_raw_field(raw, "start_weight"))
    nadir = float(_raw_field(raw, "nadir_weight"))
    outcome, driver = cs.outcome, cs.driver

    adh_low = float(lat.get("adherence_low", 0.95))
    # Post-T adherence points sit at the midpoint and the end of the course.
    _ce = course_end_of(cs.case_id, lat)
    _mid = T + (_ce - T) // 2
    if driver == "poor_medication_adherence" and outcome == "regain":
        traj = [{"day": 42, "level": 0.95}, {"day": 56, "level": 0.88}, {"day": T, "level": 0.80},
                {"day": _mid, "level": round((0.80 + adh_low) / 2, 2)},
                {"day": _ce, "level": adh_low}]
    else:
        traj = [{"day": 42, "level": 0.96}, {"day": T, "level": 0.94},
                {"day": _mid, "level": 0.93},
                {"day": _ce, "level": max(0.90, adh_low)}]

    ed = lat.get("event_density", {}) or {}
    # The value range covers the whole trajectory, including the regain tail.
    rev_week = int(lat.get("reversal_week", (T + 56) // 7))
    slope = float(lat.get("regain_slope", 0.35))
    # Same endpoint rule as `HaenvGenerator.generate`.
    _end_kg = lat.get("regain_end_kg")
    proj_end = (float(_end_kg) if (_end_kg is not None and outcome == "regain")
                else (nadir + slope * max(0, (_ce - rev_week * 7)) / 7.0
                      if outcome == "regain" else nadir))
    lo = max(40.0, min(nadir, start) - 4)
    hi = min(250.0, max(start, nadir, proj_end) + 4)
    # Declared sampling interval, using the same density expression as the
    # rendered stream (GEN6 compares the two).
    _ed_mpw0 = float((lat.get("event_density") or {}).get("measure_per_week", 7) or 7)
    sampling_days = max(int(_raw_field(raw, "sampling_days")),
                        max(1, int(round(7.0 / max(0.1, _ed_mpw0)))))
    _comorb = _raw_field(raw, "comorbidities")
    return {
        "patient_basics": {"disease": disease, "comorbidities": _raw_field(raw, "comorbidities"),
                           "age_range": _raw_field(raw, "age_range"), "sex": _raw_field(raw, "sex"),
                           "regimen": {"drug": drug, "dose_steps": steps},
                           "goals": raw.get("goals", ["sustained_weight_loss"])},
        # Rate defaults come from `events.EVENT_RATE_DEFAULTS`, shared with the injector and gates.
        "event_density": {"measure_per_week": ed.get("measure_per_week", 7),
                          "dosing_per_week": ed.get("dosing_per_week", 1),
                          **{k: ed.get(k, d) if ed.get(k, d) is not None else d
                             for k, d in _EVENT_RATE_DEFAULTS.items()}},
        "device_signals": {"devices": devices,
                           # Primary signal plus the clinical signals of `clinical_plan`, each declared
                           # with its actual step (GEN6).
                           "signals": {"weight": {"unit": "kg", "sampling_days": sampling_days,
                                                  "plausible_range": [lo, hi],
                                                  # Declared missingness for GEN6; everyday gaps exist only under daily sampling.
                                                  **({"expected_missing_rate": round(
                                                      1 - (5 * WEIGH_P_WEEKDAY
                                                           + 2 * WEIGH_P_WEEKEND) / 7, 3),
                                                      "max_gap_days": WEIGHT_MAX_GAP}
                                                     if sampling_days == WEIGHT_STEP else {})},
                                       **{s: {"unit": c["unit"], "sampling_days": c["step"],
                                              "plausible_range": c["range"]}
                                          for s, c in clinical_plan(disease, devices, _comorb, cs.case_id).items()}},
                           # Gold evidence streams are not declared here: they are kernel `AUX_SIGNALS`,
                           # checked by GEN14 rather than GEN6.
                           "n_signals": 1 + len(clinical_plan(disease, devices, _comorb, cs.case_id))},
        "adherence": {"baseline": 0.95, "trajectory": traj,
                      "missingness_mechanism": lat.get("missingness", "MAR")},
        # The first entry passes external task types' latent keys through (see
        # `haenv/external_gold.py`); it must sit inside `meta`, since `LatentPremise`
        # drops unknown top-level keys.
        "meta": {**({_EG.SLOT: {k: lat[k] for k in _EG.passthrough_keys() if k in lat}}
                     if _EG.passthrough_keys() else {}),
                 "case_id": cs.case_id, "difficulty_class": lat.get("difficulty", "B"),
                 "target_event_type": lat.get("target_event", "weight_regain"),
                 "prediction_time_T": T, "start": start, "nadir": nadir,
                 "course_end_day": _ce,
                 # Low point's day; `None` => `T`. Keys reach the generator only if listed here.
                 "nadir_day": lat.get("nadir_day"),
                 "outcome": outcome, "driver": driver,
                 # Drug-response override, emitted only when declared (an always-present `null`
                 # would change every cached world prompt).
                 **({"drug_response": lat["drug_response"]}
                    if lat.get("drug_response") is not None else {}),
                 "reversal_week": int(lat.get("reversal_week", (T + 56) // 7)),
                 "regain_slope": float(lat.get("regain_slope", 0.35)),
                 # Rebound endpoint (kg); `None` => `regain_slope`.
                 "regain_end_kg": lat.get("regain_end_kg"),
                 "clinical": clinical_plan(disease, devices, _comorb, cs.case_id),
                 # Diagnosis ground truth, landed into adjudication.ddx.
                 "ddx": cs.ddx,
                 "red_flag": bool(lat.get("ddx_red_flag", False)),
                 "clinician_warranted": lat.get("ddx_clinician_warranted"),
                 "outcome_label": lat.get("ddx_outcome_label"),
                 # Per-case "information gap" flag (read by `evaluate.slices_for` under
                 # `slices: real-rhythm+gap`), so a batch can mix gap and non-gap cases.
                 "rhythm_gap": bool(lat.get("rhythm_gap", False))},
    }


# --------------------------------------------------------------- (3b) day-by-day event closed loop
def _prompt_frame() -> str:
    """The prompt template sent to the solver (no payload), for the text-leakage scan."""
    return PROMPT


def anonymize_evidence_ids(raw: RawCase, report: dict | None = None) -> dict[str, str]:
    """Replaces EV ids with category-free sequence numbers; returns the old -> new
    mapping.

    Planning ids carry the category (`-S` real symptom, `-B`/`-L` injected
    benign/life event, `-D` upstream distractor), which would leak the
    signal/noise split. Renaming happens after the drop-and-reinject loop (which
    uses planning ids as handles) and before `build_instance`. It updates the
    ledger, the Q-side injection manifest and the validation report together.
    Numbering follows `(source_timestamp, planning id)`, which adds no
    information.
    """
    led = list(raw.evidence_ledger or [])
    order = sorted(range(len(led)),
                   key=lambda i: (int(led[i].get("source_timestamp", 10 ** 9)),
                                  str(led[i].get("evidence_id", ""))))
    cid = raw.case_id
    id_map = {str(led[i].get("evidence_id", "")): f"EV-{cid}-{n:02d}"
              for n, i in enumerate(order, 1)}
    for e in led:
        old = str(e.get("evidence_id", ""))
        if old in id_map:
            e["evidence_id"] = id_map[old]
    man = wq.injected_manifest(cid, required=False)
    # Rename every matching string in the manifest, whatever the field's name or shape.
    def _rename(x):
        if isinstance(x, str):
            return id_map.get(x, x)
        if isinstance(x, list):
            return [_rename(i) for i in x]
        if isinstance(x, dict):
            return {k: _rename(v) for k, v in x.items()}
        return x

    man = _rename(man)
    # Added after `_rename` (its keys are the old ids); verifier-only.
    man["id_map"] = id_map
    wq.register_injection(cid, man)
    if report:
        for it in (report.get("items") or []):
            if str(it.get("item", "")) in id_map:
                it["item"] = id_map[str(it["item"])]
    return id_map


def _inject_events_verified(raw: RawCase, cs: CaseSpec, p: LatentPremise, T: int,
                            max_rounds: int, dispatch=None) -> tuple[RawCase, dict]:
    """Injects day-by-day events, validates them per item, and re-injects without
    the failing items until it converges or rounds run out.

    Every round starts from the same pre-injection `base`. With `dispatch`, an
    LLM plans the injection and receives the previous round's failure reasons.
    """
    base = copy.deepcopy(raw)
    rev_day = (int(cs.latent.get("reversal_week", (T + 56) // 7)) * 7
               if cs.outcome == "regain" else None)
    drop: set[str] = set()
    reasons: dict[str, list[str]] = {}             # per-item drop reasons, kept for the audit
    frame = _prompt_frame()
    # Pre-initialized for `max_rounds <= 0`.
    cand, rep, injected, manifest = base, {}, {}, {}
    for r in range(1, max_rounds + 1):
        cand = copy.deepcopy(base)
        cand, manifest = events_mod.inject(cand, cs, p, cs.driver, drop,
                                           dispatch=dispatch, feedback=reasons)
        injected = manifest.get("injected") or {}   # Q-side ledger; registered only after convergence
        rep = verify_mod.verify_case(base, cand, cs, p, manifest, cs.driver, rev_day, frame)
        rep["round"] = r
        rep["dropped"] = sorted(drop)
        rep["drop_reasons"] = reasons
        if rep["ok"]:
            break
        if rep["bad_items"]:                       # per-item failure -> drop then re-inject
            for it in rep["items"]:
                if not it["ok"]:
                    reasons[it["item"]] = [k for k, v in it["checks"].items() if not v["ok"]]
            log.warning("[build] %s round %d: dropped non-conforming items %s", cs.case_id, r, rep["bad_items"])
            drop |= set(rep["bad_items"])
            continue
        break                                      # case-level failure (invariance/text leakage): dropping items cannot fix it
    bad = rep.get("bad_items", []) + [k for k, v in (rep.get("case_checks") or {}).items()
                                      if not v.get("ok")]
    items = rep.get("items", [])
    metric_kinds = {"daily_metric", "inherited_metric"}
    return cand, {"ok": bool(rep.get("ok")), "rounds": rep.get("round"), "bad": bad,
                  "dropped": sorted(drop),
                  "n_streams": sum(1 for i in items if i["kind"] in metric_kinds),
                  "n_events": sum(1 for i in items if i["kind"] not in metric_kinds),
                  "report": rep, "injected": injected,
                  # Pre-noise ground-truth trajectory when the physiology layer is on
                  # (verifier-only, for step (4)); `None` when off.
                  "physio_clean_ld": (manifest or {}).get("_clean_longitudinal_data")}


#: `premise_conflicts` kinds about what the solver sees before T. With the physiology layer on,
#: they are judged on the observed series: the truth series carries neither the reading
#: artifacts nor the clinic-scale reference.
OBSERVED_CONFLICT_KINDS = frozenset({"window_trap_not_visible_pre_T"})


def last_shown_day(raw: RawCase, geometry: dict | None) -> int:
    """The last day whose readings the solver is shown, for the geometry the case runs in
    (`CaseSpec.geometry`): T for single-shot and gated; the last slice for slices
    (`evaluate.slices_for`; fewer than two slices run single-shot); the last round's pointer
    for multi-round (`runner.run_multiround`: a round every `cadence_days` from T while the
    pointer stays within T + prediction window, at most 60 rounds)."""
    T = int(raw.prediction_context["prediction_time_T"])
    name = (geometry or {}).get("name", "single")
    if name == "slices":
        from .evaluate import slices_for
        days = slices_for(raw, geometry["slices"])
        return int(max(days)) if len(days) >= 2 else T
    if name == "multi":
        cadence = int(geometry["cadence_days"])
        w = str(raw.prediction_context.get("prediction_window", "180d")).rstrip("d")
        window = int(w) if w.isdigit() else 180
        return T + cadence * min(window // cadence, min(60, window // cadence + 2) - 1)
    return T


def trap_visibility_conflicts(raw: RawCase, last_day: int) -> list[dict]:
    """`trap_not_visible_to_solver` (gate) for a trap whose evidence is not in the readings the
    solver is shown (ts <= `last_day`).

    Point artifacts (`transient_spike`, `unit_error`): GEN27 (`gates.check_trap_evidence`)
    on the primary series cut at `last_day` -- the trap's reading is shown and stands out
    from the neighbours shown with it. `device_switch`: at least one reading from the switch
    to `last_day`, the same standard as a single-reading artifact. `context_confound` has its
    own gate (`window_trap_not_visible_pre_T`)."""
    from noise import POINT_ARTIFACTS, PRIMARY
    shown = [q for q in raw.longitudinal_data.get(PRIMARY) or [] if int(q["ts"]) <= last_day]
    traps = [rp for rp in raw.reversal_points or [] if rp.get("type") == "trap"]
    out: list[dict] = []
    if any(rp.get("noise_class") in POINT_ARTIFACTS for rp in traps):
        cut = copy.copy(raw)
        cut.longitudinal_data = {PRIMARY: shown}
        out += [{"kind": "trap_not_visible_to_solver", "severity": "gate",
                 "detail": f"shown to day {last_day}: {h['detail']}"}
                for h in gates.check_trap_evidence(cut)]
    for rp in traps:
        if rp.get("noise_class") != "device_switch":
            continue
        d0 = int(rp["day"])
        if not any(int(q["ts"]) >= d0 for q in shown):
            out.append({"kind": "trap_not_visible_to_solver", "severity": "gate",
                        "detail": f"device_switch@day{d0}: no {PRIMARY} reading from the "
                                  f"switch to day {last_day}"})
    return out


#: Kernel noise classes applied before the events and physiology layers: `adherence_gap`
#: changes the patient (true adherence) and `mnar_missing` removes readings. Every other
#: class moves readings, which is a property of the measurement: those go onto the observed
#: series after the physiology layer (`_inject_observation_artifacts`), so its truth series
#: never carries them.
NOISE_BEFORE_PHYSIO = frozenset({"adherence_gap", "mnar_missing"})


def _inject_kernel_noise(raw: RawCase, nz: dict) -> RawCase:
    """One declared noise item, amplitude from `registry/artifact_rates.yaml`."""
    d0 = int(nz["week"]) * 7
    return inject_noise(raw, nz["class"], d0, d0 + int(nz.get("span_days", 10)),
                        **_artifact_params.kwargs_for(nz["class"]))


def _inject_observation_artifacts(raw: RawCase, cs: CaseSpec,
                                  applied: list[str]) -> tuple[RawCase, list[str]]:
    """Puts the declared reading artifacts on the observed series, at their injected size.

    Returns the case and the items whose injector found no reading to move (no trap).
    A kernel injector also adds a clinic-scale reference (`weight_ref`) sampled from the
    series it is given, here the observed one; that copy is dropped, so every case's
    reference comes from the `post_inject` backfill of the truth series.
    """
    todo = [nz for nz in cs.noise if nz["class"] not in NOISE_BEFORE_PHYSIO]
    if not todo:
        return raw, []
    had_ref = "weight_ref" in raw.longitudinal_data
    unplaced: list[str] = []
    for nz in todo:
        n_rp = len(raw.reversal_points)
        raw = _inject_kernel_noise(raw, nz)
        (applied if len(raw.reversal_points) > n_rp else unplaced).append(
            f"{nz['class']}@wk{nz['week']}")
    if not had_ref:
        raw.longitudinal_data.pop("weight_ref", None)
    raw.case_id = cs.case_id                                    # the injector mutated id, restore it
    return raw, unplaced


# --------------------------------------------------------------- Main pipeline
def build_case(cs: CaseSpec, max_rounds: int = 6, T: int | None = None,
               dispatch=None, event_dispatch=None) -> tuple[RawCase | None, dict]:
    """The complete question-generation loop for one case. Returns
    `(RawCase | None, audit)`; `None` = not emitted.

    `T` defaults to the case's `latent.index_time_T`. `dispatch` / `event_dispatch`
    are the LLM dispatchers for course generation / event planning (`None` =
    deterministic).
    """
    # The comorbidity domain must be in scope for every gate, not just `make_premise`.
    with _register_comorbid_domain(_raw_field(cs.raw, "disease"),
                                   _raw_field(cs.raw, "comorbidities") or []):
        return _build_case_inner(cs, max_rounds, T, dispatch, event_dispatch)


def _build_case_inner(cs: CaseSpec, max_rounds: int = 6, T: int | None = None,
                      dispatch=None, event_dispatch=None) -> tuple[RawCase | None, dict]:
    """The body of `build_case`."""
    T = int(T if T is not None else cs.index_time_T)
    audit: dict = {"case_id": cs.case_id, "emitted": False, "T": T,
                   "generator": "llm" if dispatch else "deterministic",
                   "event_generator": "llm" if event_dispatch else "deterministic"}
    # Validate the active registry once per process, fail-closed; the result is
    # recorded in `audit`.
    from .overlay import validate_registry as _vreg
    audit["registry_validated"] = _vreg()        # raises RegistryInvalid on failure
    # Latent-key consistency (`latent_rules`): every contradiction is recorded in
    # the audit so a failed case points at the keys to fix.
    from .latent_rules import audit_latent, blocking_contradictions, contradictions
    _lat = {**(cs.latent or {}), "index_time_T": T}
    _lc = audit_latent(dict(cs.raw or {}), _lat)
    audit["latent_checks"] = [{"rule": c.rule, "ok": c.ok, "detail": c.detail} for c in _lc]
    # Only the blocking subset stops the case.
    _bad = contradictions(dict(cs.raw or {}), _lat)
    _block = blocking_contradictions(dict(cs.raw or {}), _lat)
    if _bad:
        audit["latent_contradictions"] = [c.rule for c in _bad]
        audit["latent_blocking"] = [c.rule for c in _block]
        for c in _bad:
            log.error("[latent] %s %s: %s", cs.case_id, c.rule, c.detail)
    if _block:
        # Blocking contradictions stop the case before any model call.
        log.error("[build] %s latent contradiction -> not emitted (stopped before generation, no model call spent) %s",
                  cs.case_id, [c.rule for c in _block])
        return None, audit
    try:
        p = make_premise("human", premise_spec(cs, T))          # (1) premise + validation
    except ValueError as e:
        audit["premise_error"] = str(e)
        log.error("[build] %s premise validation failed: %s", cs.case_id, e)
        return None, audit
    audit["premise_ok"] = True
    audit["world_layer_gaps"] = world_layer_gaps()

    # ---- (1b) Declaration-only gates, run before any model call ----
    _decl = (gates.check_drug_indication(p)  # GEN27: does the drug have an indication for this disease
             + gates.check_comorbidity_vocab(p)  # GEN28: does the comorbidity name have a physiological consumer
             + gates.check_rhythm_gap_feasible(cs)  # GEN29: the gap tier's declaration must actually fit
             + gates.check_dosing_consistency(p)  # GEN20b: dosing frequency and route are self-consistent
             + gates.check_premise_registry(p.dumps())  # meta-rule: premise fields must be registered
             + gates.check_demographic_plausibility(None, cs)  # GEN24: the declared age range is plausible for the condition
             # GEN24b: `known_conditions` must not give away a gold line
             + gates.check_gold_line_in_known_conditions(None, cs))
    audit["declaration_gates"] = [f"{h['kind']}:{h['detail']}" for h in _decl]
    _decl_block = gates.batch_gate_blockers(_decl)
    for h in _decl:
        if h not in _decl_block:
            log.warning("[gates] %s %s: %s", cs.case_id, h["kind"], h["detail"])
    if _decl_block:
        audit["declaration_blocking"] = [h["kind"] for h in _decl_block]
        log.error("[build] %s declaration gate failed -> not emitted (stopped before generation, no model call spent) %s",
                  cs.case_id, [h["kind"] for h in _decl_block])
        return None, audit

    gen = LLMCaseGenerator(dispatch, cs) if dispatch else HaenvGenerator()
    raw, rep = synthesize(p, original_case=original_facts(cs, T), generator=gen,  # (2) generate + GV-1
                          max_rounds=max_rounds, apply_noise=False)
    audit["synth_rounds"] = rep.get("rounds")
    audit["synth_history"] = rep.get("history")
    if raw is None:                                             # failed to converge -> not emitted
        audit["last_conflicts"] = [c["kind"] for c in rep.get("last_conflicts", [])]
        audit["last_conflicts_detail"] = [f"{c['kind']}:{c.get('detail')}"
                                          for c in rep.get("last_conflicts", []) if c.get("detail")]
        log.error("[build] %s GV-1 did not converge -> not emitted %s", cs.case_id,
                  audit.get("last_conflicts_detail") or audit["last_conflicts"])
        return None, audit

    applied: list[str] = []                                     # (3) inject, derived from the premise
    # Noise amplitudes come from `registry/artifact_rates.yaml` (no fallback),
    # checked against the kernel's defaults. Reading artifacts wait for (3c).
    _assert_kernel_params_in_sync()
    for nz in cs.noise:
        if nz["class"] in NOISE_BEFORE_PHYSIO:
            raw = _inject_kernel_noise(raw, nz)
            applied.append(f"{nz['class']}@wk{nz['week']}")
    if cs.distractor_level != "none":
        # `seed_shift` rotates the kernel's distractor pool; without it the `low` tier
        # always draws index 0.
        from . import rng as _rng
        _shift = int(_rng.unit(cs.case_id, "distractor", "shift") * 10)
        raw = inject_distractors(raw, p, level=cs.distractor_level, seed_shift=_shift)
        applied.append(f"distractor:{cs.distractor_level}@shift{_shift}")
    raw.case_id = cs.case_id                                    # the injector mutated id, restore it
    raw.latent_premise = p.dumps()                              # verifier-only bookkeeping
    audit["noise_applied"] = applied
    audit["weight_skeleton"] = _SKELETON_AUDIT.pop(str(cs.case_id), None)

    raw, vrep = _inject_events_verified(raw, cs, p, T, max_rounds,   # (3b) day-by-day events + per-item validation
                                       dispatch=event_dispatch)
    # (3c) Observation layer, after the ground-truth trajectory is captured: the
    # kernel's reading artifacts, then post-kernel dirty data (e.g. carried-forward
    # values). Both are judged as observations, never as physiological slope.
    # `clean_weight` is the truth series every clinic-scale reference is sampled from;
    # with the physiology layer off, the course before the reading artifacts.
    _truth_w = ((vrep.get("physio_clean_ld") or {}).get("weight")
                or copy.deepcopy(raw.longitudinal_data.get("weight")))
    # The label rule read on the truth series, for the record (the gold is set elsewhere).
    _tl, _td = gates.derive_outcome({"weight": _truth_w or []}, raw.label_rule)
    audit["weight_truth_label"] = {"verdict": _tl, "sustained_days": _td.get("sustained_days")}
    raw, _unplaced = _inject_observation_artifacts(raw, cs, applied)
    _post = _post_inject.apply_post_injection(raw, case_id=cs.case_id, clean_weight=_truth_w)
    applied += _post
    audit["post_injected"] = _post

    audit.update(event_rounds=vrep["rounds"], event_streams=vrep["n_streams"],
                 event_evidence=vrep["n_events"], event_dropped=vrep["dropped"],
                 verify_ok=vrep["ok"], verify_bad=vrep["bad"])
    audit["_verify_report"] = vrep["report"]
    # Coupling rules fired, and those still pending physician review, as of generation.
    _cpl = (vrep.get("injected") or {}).get("coupling") or {}
    audit["coupling_fired"] = list(_cpl.get("fired_rules") or [])
    audit["coupling_pending_review"] = list(_cpl.get("pending_review") or [])
    if not vrep["ok"]:                                          # not emitted
        log.error("[build] %s daily-event verification did not converge -> not emitted bad=%s", cs.case_id, vrep["bad"])
        return None, audit
    # (3b') Register the Q-side injection ledger (wq law 2) once, after convergence. Diagnosis
    # cases also record the answer contract their framing asks for (question side, not gold).
    _qinj = vrep.get("injected") or {}
    if cs.ddx:
        from .framings import DDX_ANSWER_CONTRACT as _DAC
        _qinj = {**_qinj, "answer_contract": copy.deepcopy(_DAC)}
    wq.register_injection(cs.case_id, _qinj)
    # (3d) EV id anonymization, after registration and before `build_instance`.
    audit["ev_id_map"] = anonymize_evidence_ids(raw, vrep.get("report"))

    # (3e) Realign the weight ledger to the final (noised) stream; the change is
    # recorded in the audit.
    _lw_ts, _lw_val = ledger_weight_point(raw.longitudinal_data, T, cs.case_id)
    _lw_moved = []
    for _ev in (raw.evidence_ledger or []):
        if gates.LEDGER_VALUE_STREAMS.get(str(_ev.get("source_type") or "")) != {"weight"}:
            continue
        _old = _ev.get("measured_value")
        if not isinstance(_old, (int, float)) or isinstance(_old, bool):
            continue
        if abs(float(_old) - _lw_val) > 1e-9 or int(_ev.get("source_timestamp", -1)) != _lw_ts:
            _lw_moved.append({"evidence_id": str(_ev.get("evidence_id")),
                              "from": [_ev.get("source_timestamp"), float(_old)],
                              "to": [_lw_ts, _lw_val]})
        _ev["source_timestamp"], _ev["measured_value"] = _lw_ts, _lw_val
    audit["ledger_weight_realigned"] = _lw_moved

    # ---- (4) Recheck + leakage gate ------------------------------------------------------
    #
    # Slope and anchors are judged on the ground-truth trajectory, value range on
    # the observed sequence. At full descent pace the kernel's weekly-slope limit
    # (1.5 x 1.05 / 7 kg/day) leaves about 11 g/day for observation noise, so it
    # cannot apply to the physiology layer's observed stream.
    _clean_ld = vrep.get("physio_clean_ld")
    if _clean_ld is None:
        conflicts = premise_conflicts(raw, p)
    else:
        _truth = copy.copy(raw)
        _truth.longitudinal_data = _clean_ld
        conflicts = premise_conflicts(_truth, p)     # slope / value range on the ground-truth trajectory
        # value range also on the observed sequence
        conflicts += gates.check_observed_in_domain(raw, p)
        # what the solver can see is judged on the observed series
        conflicts = ([c for c in conflicts if c["kind"] not in OBSERVED_CONFLICT_KINDS]
                     + [c for c in premise_conflicts(raw, p) if c["kind"] in OBSERVED_CONFLICT_KINDS])
    sp_probe, _ = build_instance(raw, T)                        # the copy the solver sees
    hits = (gates.check_cadence(raw.longitudinal_data, p, cs)   # (4b) haenv-side assertions (GEN6/7/13/14)
            + gates.check_device_inventory(raw.user_profile, raw.longitudinal_data)
            + gates.check_clinical_coupling(raw.longitudinal_data,     # GEN15 coupling direction
                                            _cplan := clinical_plan(
                                                p.patient_basics.get('disease', ''),
                                                p.device_signals.get('devices', []),
                                                p.patient_basics.get('comorbidities') or [],
                                                cs.case_id),
                                            # expected direction includes the drug effect
                                            drug_terms=_drug_terms_for(raw, p, _cplan))
            # GEN13: outcome derivable by label_rule, judged on the ground-truth trajectory
            + gates.check_outcome_derivable(
                _truth if _clean_ld is not None else raw, cs)
            + gates.check_gold_coverage(sp_probe, raw.gold_drivers,  # GEN14: the gold standard must be derivable from the question surface
                                        (raw.adjudication or {}).get("ddx"))
            + gates.check_symptom_day_separability(raw, T)  # GEN18: timestamps must not let signal/noise be separated at a glance
            + gates.check_ev_id_opaque(sp_probe)                # GEN21: ids must not carry a category prefix
            + gates.check_sex_consistency(raw, cs)  # GEN19: the question surface must not contradict the declared sex
            # GEN22: declared anchors honored, judged on the ground-truth trajectory
            + gates.check_anchors_honored(_truth if _clean_ld is not None else raw, cs)
            + gates.check_stream_horizons(raw)                  # GEN23: stream endpoints must not exceed the declared course end day
            + gates.check_trap_evidence(raw)                    # GEN27: a single-reading trap must stay visible on the observed series
            + gates.check_ledger_values_traceable(raw, T)       # GEN25: ledger values must be traceable to the same-named stream at ≤T
            + gates.check_clinical_baseline_cohort(raw, p)      # GEN26: the baseline must not portray an undiagnosed person as sick
            + gates.check_event_density(raw, p, T)              # GEN20a: however much noise is declared, that much must be injected
            + gates.check_missingness(raw.longitudinal_data, p, cs, T)   # GEN20c: missingness must be answer-neutral
            + _EG.gates_for(raw, cs, sp_probe, T))                     # external task-type gates; [] when none
    conflicts += [h for h in hits if h["severity"] == "gate"]  # gate-severity hits merge into the emission gate
    # A declared reading artifact that found no reading to move (e.g. inside a window
    # `mnar_missing` emptied) would leave the case without the artifact it declares.
    conflicts += [{"kind": "noise_not_placed", "severity": "gate",
                   "detail": f"{x}: no reading in its window"} for x in _unplaced]
    # A trap is only fair if the solver is shown its evidence.
    conflicts += trap_visibility_conflicts(raw, last_shown_day(raw, cs.geometry))
    warns = [h for h in hits if h["severity"] != "gate"]
    for w in warns:
        log.warning("[gates] %s %s: %s", cs.case_id, w["kind"], w["detail"])
    sp, _ = build_instance(raw, T)
    leak_ok, leak_viol = leakage_probe(sp, T)
    # The item verifier scanned before the final observation and payload steps.
    # Check the exact payload handed to the solver after those steps as well.
    _final_text = verify_mod.scan_solver_text(
        raw, T, ddx_aliases=verify_mod._ddx_aliases(cs), solver_payload=sp)
    leak_viol = list(leak_viol) + [f for f in _final_text["findings"]
                                    if not f.startswith("kernel_leakage_probe:")]
    leak_ok = leak_ok and _final_text["ok"]
    audit.update(post_noise_conflicts=[c["kind"] for c in conflicts],
                 post_noise_conflict_details=[
                     f"{c['kind']}:{c['detail']}" for c in conflicts if c.get("detail")],
                 gate_warnings=[f"{w['kind']}:{w['detail']}" for w in warns],
                 leak_ok=leak_ok, leak_violations=leak_viol,
                 final_text_scan={"ok": _final_text["ok"],
                                  "findings": _final_text["findings"],
                                  "n_chars": _final_text["n_chars"]},
                 solver_visible_signals=len(sp.longitudinal_data),
                 solver_visible_evidence=len(sp.evidence_ledger))
    if conflicts or not leak_ok:
        log.error("[build] %s blocked by the emission gate: conflicts=%s leak=%s\n          %s",
                  cs.case_id, audit["post_noise_conflicts"], leak_viol,
                  "\n          ".join(audit["post_noise_conflict_details"]) or "(kernel conflict carries no detail)")
        return None, audit

    # Gold qualifiers: split core + qualifier and record, on what the solver sees at T, whether
    # each qualifier can be established (after every gate, so emission is unaffected).
    from . import qualifiers as _qualifiers
    _qrecs = _qualifiers.annotate(raw, sp)
    if _qrecs:
        audit["gold_qualifiers"] = [{"family": q["family"], "derivable": q["derivable"]}
                                    for q in _qrecs]
    audit["emitted"] = True
    audit["outcome_label"] = raw.outcome_label
    audit["gold_drivers"] = list(raw.gold_drivers)
    log.info("[build] %s emitted ✓ (rounds=%s noise=%s)", cs.case_id, rep.get("rounds"), applied)
    return raw, audit
