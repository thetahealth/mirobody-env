"""Noise-free set points of the follow-up analytes and their visible-cause effect terms.

    ln FG*(t) = ln b + beta * ln(w(t) / w(0)) - kappa * sum_k dlevel_k * onset(t - t_k)
    ln Cr*(t) = ln b + A * on(t)

`onset(s) = 1 - exp(-s / tau)` for s > 0, else 0; `level` is the exposure on the log-dose scale
(`dose_level`: 0 off drug, `start_level` at the line's first dose). The GLP-1 term runs along the whole visible
dose line with one per-case response `kappa`, so a dose step earlier in the history moves the
history readings the same way the step between the two compared visits moves the follow-up.
`on(t)` is the ACEI on-state with first-order onset and offset. The weight series is the
world's noise-free weight (`build._LAB_CTX[case]["wpts"]`), smoothed over `smooth_days`.

These are the drug-effect pieces pack 3 is meant to reuse. All coefficients are provisional
(`registry/labworld.yaml:effects`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math
import statistics

from . import tables


def effect(name: str) -> dict:
    e = (tables().get("effects") or {}).get(name)
    if not isinstance(e, dict):
        raise KeyError(f"labworld: unknown effect {name!r}")
    return e


def onset(s: float, tau: float) -> float:
    return 0.0 if s <= 0 else 1.0 - math.exp(-float(s) / float(tau))


def dose_level(dose: float, first: float, start_level: float | None = None) -> float:
    """Exposure level of a dose on the log-dose scale: 0 off drug, `start_level` at the first
    dose of the line, `start_level + ln(dose / first)` above it (a pause returns it to 0)."""
    if start_level is None:
        start_level = float(effect("glp1_on_fasting_glucose").get("start_level", 1.0))
    return 0.0 if not dose or dose <= 0 else float(start_level) + math.log(float(dose) / float(first))


def dose_steps(dose_pts) -> list[tuple[int, float]]:
    """(day, change of `dose_level`) at every change of the dose line, in day order, starting
    from off-drug before the first point (a line that starts on day 0 has its first step there)."""
    pts = sorted((q for q in dose_pts or [] if isinstance(q.get("value"), (int, float))),
                 key=lambda x: int(x["ts"]))
    first = next((float(q["value"]) for q in pts if q["value"] > 0), None)
    if first is None:
        return []
    out, prev = [], 0.0
    for q in pts:
        lv = dose_level(float(q["value"]), first)
        if lv != prev:
            out.append((int(q["ts"]), lv - prev))
        prev = lv
    return out


def glp1_unit_term(dose_pts, day: int, tau: float) -> float:
    """sum_k dlevel_k * onset(day - t_k): the GLP-1 term per unit kappa (sign not applied)."""
    return sum(d * onset(day - t, tau) for t, d in dose_steps(dose_pts))


def smoothed_weight(weight_pts, day: int, smooth_days: int) -> float | None:
    """Median of the weight readings in [day - smooth_days, day]; the last reading before when
    that window is empty."""
    win = [float(q["value"]) for q in weight_pts or []
           if day - smooth_days <= int(q["ts"]) <= day and isinstance(q.get("value"), (int, float))]
    if win:
        return float(statistics.median(win))
    prev = [q for q in weight_pts or [] if int(q["ts"]) <= day and isinstance(q.get("value"), (int, float))]
    return float(prev[-1]["value"]) if prev else None


def weight_log_term(weight_pts, day: int, smooth_days: int) -> float:
    w0 = smoothed_weight(weight_pts, 0, smooth_days)
    wt = smoothed_weight(weight_pts, day, smooth_days)
    if not w0 or not wt:
        return 0.0
    return math.log(wt / w0)


def fg_log(day: int, *, base: float, kappa: float, beta: float, tau: float,
           dose_pts, weight_pts, smooth_days: int) -> float:
    return (math.log(base) + beta * weight_log_term(weight_pts, day, smooth_days)
            - kappa * glp1_unit_term(dose_pts, day, tau))


def acei_on(day: int, events, tau: float, initial_on: bool) -> float:
    """ACEI on-state in [0, 1] on `day`. `events` = [{day, action: start|stop}]; before the first
    event the state is settled at `initial_on`; after each event it relaxes toward the new
    target with time constant `tau` (days), starting from the level it had reached."""
    target = 1.0 if initial_on else 0.0
    level_at, t_at = target, None
    for ev in sorted(events or [], key=lambda e: int(e["day"])):
        d = int(ev["day"])
        if d > day:
            break
        level_now = level_at if t_at is None else target + (level_at - target) * math.exp(-(d - t_at) / float(tau))
        target, level_at, t_at = (1.0 if ev["action"] == "start" else 0.0), level_now, d
    if t_at is None:
        return level_at
    return target + (level_at - target) * math.exp(-(day - t_at) / float(tau))


def cr_log(day: int, *, base: float, A: float, events, tau: float, initial_on: bool) -> float:
    return math.log(base) + A * acei_on(day, events, tau, initial_on)


def egfr_ckd_epi(cr_umol: float, age: int, sex: str) -> float:
    """CKD-EPI 2021 creatinine equation (race-free), mL/min/1.73 m2 (packs 3 and 4: a CKD patient at a dealt eGFR)."""
    k, a = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    x = float(cr_umol) / 88.4 / k
    return 142.0 * min(x, 1.0) ** a * max(x, 1.0) ** -1.200 * 0.9938 ** age * (1.012 if sex == "F" else 1.0)


def cr_for_egfr(egfr: float, age: int, sex: str) -> float:
    """The creatinine (umol/L) at which CKD-EPI 2021 gives `egfr` (above the knot)."""
    k = 0.7 if sex == "F" else 0.9
    x = (float(egfr) / (142.0 * 0.9938 ** age * (1.012 if sex == "F" else 1.0))) ** (-1.0 / 1.2)
    return max(x, 1.0) * k * 88.4
