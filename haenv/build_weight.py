"""The weight series: the trajectory skeleton (descent, plateau, regain, irregularity), the label
guard that keeps the declared outcome derivable, and daily rendering onto the sampling grid.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .regpath import registry_cached as _registry_cached
from . import latent_rules as _LR
import math
import statistics
from . import events as events_mod
from . import gates
from .world_knobs import (  # noqa: F401
    END,
    STEP,
)


# --------------------------------------------------------------- Primary-signal sampling and missingness
# Weight is sampled daily with missed weigh-ins. Missingness depends only on the
# day of week (answer-neutral MAR), never on driver/outcome.
WEIGHT_STEP = 1


WEIGH_P_WEEKDAY = 0.82           # weigh-in probability on weekdays


WEIGH_P_WEEKEND = 0.45           # weigh-in probability on Sat/Sun


WEIGHT_MAX_GAP = 6               # longest everyday gap (days); longer gaps come only from a noise window


WOBBLE_PERIOD = 21.0             # period of the physiological wobble (days)


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
    declared `nadir`, nor under the plain series' own low point where that is lower; over
    the whole course it may not stay more than `SKEL_ANCHOR_MARGIN_KG` above it either,
    unless the plain series' low point is higher.
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
        if vals and min(vals) < line - 1e-9:
            return False
        # Over the whole course the low point must also come down to the declared nadir
        # (GEN22 reads it from above too), or as far as the plain series does.
        top = nadir + SKEL_ANCHOR_MARGIN_KG
        if pvals:
            top = max(top, min(pvals))
        return upto is not None or not vals or min(vals) <= top + 1e-9

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
        from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN as _DOM
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
    """A weight series through model points: a shape-preserving curve through them,
    then the case's drift and episodes, then the regain check. `LLMCaseGenerator` does
    not call it (the release channel's weight is the deterministic skeleton); the skeleton
    tools and the demo page do.

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
