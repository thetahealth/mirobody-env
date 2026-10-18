"""Drug effects of the chronic-medication world (pack 3): a managed drug's dose line and its
adherence drive one control axis (HbA1c, LDL-C, home systolic pressure, TSH).

    u(t) = effect[dose(t)] * a(t)                     piecewise constant
    X(t) = sum_k du_k * onset(t - t_k, tau_truth)     first-order onset (`truth.onset`, shared with pack 4)
    ln y(t) = ln y_u - r * X(t)    (model: log)       y(t) = y_u - r * X(t)    (model: additive)

`a(t)` is the coverage of the refill record (30-day supplies; between two pickups the share of
days covered), so the adherence truth and its reading are one record. `r` is the per-case
response, `y_u` the untreated level. Tables: `registry/p3_meds.yaml`, `registry/control_targets.yaml`
(provisional, design doc K4). Segment: `tools/release/make_freeze.py:P3_WORLD`.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import math

from haenv.world.labworld.truth import onset

MEDS, TARGETS = "p3_meds.yaml", "control_targets.yaml"
SUPPLY_DAYS = 30


@functools.lru_cache(maxsize=4)
def _load(path: str, mtime: int) -> dict:
    from haenv.resources.yamlcache import load_yaml
    return load_yaml(path) or {}


def _table(name: str) -> dict:
    from haenv.resources.regpath import registry_path
    p = registry_path(name)
    return _load(str(p), p.stat().st_mtime_ns)


def meds() -> dict:
    return _table(MEDS)


def targets() -> dict:
    return _table(TARGETS)


def drug(name: str, known=()) -> dict:
    """The drug's table; a known comorbidity listed under its `if_known` overrides the entries it
    names (the comorbidity changes the guideline's next step, e.g. CKD: titrate the ACEI)."""
    d = meds()["drugs"][name]
    over = [d["if_known"][k] for k in known or () if k in (d.get("if_known") or {})]
    return {**d, **{k: v for o in over for k, v in o.items()}} if over else d


def comorbidity(code: str) -> dict:
    """A comorbidity pack 3 can place on a patient (`registry/p3_meds.yaml:comorbidities`)."""
    return meds()["comorbidities"][code]


def axis(name: str) -> dict:
    return targets()["axes"][name]


def rule() -> dict:
    return meds()["rule"]


def rcv(axis_name: str) -> float:
    """Two-sided 95% reference change value on the log scale."""
    return 1.96 * math.sqrt(2.0) * float(axis(axis_name)["cv_total"])


def band(axis_name: str) -> float:
    """One reading's reference change against a fixed threshold, log scale: 1.96 x CV (the RCV
    bounds the difference of two readings, 1.96 x sqrt 2 x CV; `rule.control_gray_rcv` = 1/sqrt 2)."""
    return rcv(axis_name) * float(rule()["control_gray_rcv"])


def effect_of(drug_name: str, dose: float) -> float:
    if not dose:
        return 0.0
    d = drug(drug_name)
    return float(d["effect"][list(d["ladder"]).index(dose)])


def coverage_segments(pickups: list[int], a_before: float, a_last: float,
                      supply: int = SUPPLY_DAYS, reset=()) -> list[tuple[int, float]]:
    """(day, adherence from that day on): `a_before` before the first pickup of the record,
    then the share of days covered between consecutive pickups, an early refill's surplus carried
    into the next interval (proportion of days covered), then `a_last` after the last pickup
    (today's adherence, read through the pill count). `supply`: days per dispensing; `reset`: the
    pickup days of a new strength, where the old strength's stock no longer counts."""
    out = [(-10 ** 6, float(a_before))]
    for (p, q), c in zip(zip(pickups, pickups[1:]), carry_in(pickups, supply, reset)):
        out.append((int(p), min(1.0, (float(supply) + c) / float(q - p))))
    out.append((int(pickups[-1]), float(a_last)))
    return out


def carry_in(pickups: list[int], supply: int = SUPPLY_DAYS, reset=()) -> list[float]:
    """Days of supply left over at each pickup from the earlier ones (0 at the first, and at a
    pickup day in `reset`: a new strength, the old one's stock set aside)."""
    out, c = [], 0.0
    reset = {int(x) for x in reset}
    for p, q in zip([None] + list(pickups[:-1]), pickups):
        if p is not None:
            c = 0.0 if int(q) in reset else max(0.0, float(supply) + c - float(int(q) - int(p)))
        out.append(c)
    return out


def adherence_T(segments: list[tuple[int, float]], T: int, window: int, start: int | None = None) -> float:
    """Mean adherence over (T - window, T], from `start` on when the drug started inside it: the
    a(T) the gold rule reads."""
    lo = T - window + 1 if start is None else max(T - window + 1, int(start))
    return sum(level_at(segments, d) for d in range(lo, T + 1)) / float(T + 1 - lo)


def level_at(segments: list[tuple[int, float]], day: int) -> float:
    v = segments[0][1]
    for d, x in segments:
        if d <= day:
            v = x
    return v


def input_steps(drug_name: str, dose_line: list[list], adherence: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """(day, du) at every change of u = effect x adherence; u = 0 before the first dose."""
    days = sorted({int(d) for d, _ in dose_line} | {int(d) for d, _ in adherence if d > -10 ** 6})
    start = int(min(d for d, _ in dose_line))
    out, prev = [], 0.0
    for t in days:
        if t < start:
            continue
        dose = [v for d, v in sorted(dose_line) if int(d) <= t][-1]
        u = effect_of(drug_name, dose) * level_at(adherence, t)
        if u != prev:
            out.append((t, u - prev))
        prev = u
    return out


def response(steps: list[tuple[int, float]], day: int, tau: float) -> float:
    return sum(du * onset(day - t, tau) for t, du in steps)


def truth(drug_name: str, day: int, y_u: float, r: float, steps) -> float:
    d = drug(drug_name)
    X = response(steps, day, float(d["tau_truth_days"]))
    if d["model"] == "log":
        return float(y_u) * math.exp(-float(r) * X)
    return float(y_u) - float(r) * X


def solve_untreated(drug_name: str, y_T: float, r: float, X_T: float) -> float:
    """The untreated level that puts the truth at `y_T` on day T given the response X(T)."""
    if drug(drug_name)["model"] == "log":
        return float(y_T) * math.exp(float(r) * X_T)
    return float(y_T) + float(r) * X_T
