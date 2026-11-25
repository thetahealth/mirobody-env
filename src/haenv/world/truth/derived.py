"""Pure derived quantities; CKD-EPI 2021 (Inker) and ADAG (Nathan)."""
from __future__ import annotations

from haenv.world.truth.registry import effects


def egfr_ckd_epi_2021(cr_umol, age, sex) -> float:
    e = effects()["egfr_equation"]
    female = str(sex).upper() == "F"
    k = e["k_female" if female else "k_male"]
    alpha = e["alpha_female" if female else "alpha_male"]
    x = float(cr_umol) / e["scr_umol_per_mg_dl"] / k
    return (e["coefficient"] * min(x, 1)**alpha * max(x, 1)**e["upper_power"]
            * e["age_base"]**float(age) * (e["female_factor"] if female else 1))


def cr_at_egfr(egfr, age, sex) -> float:
    if egfr <= 0:
        raise ValueError("eGFR must be positive")
    lo, hi = 1e-9, 100.0
    while egfr_ckd_epi_2021(hi, age, sex) > egfr:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if egfr_ckd_epi_2021(mid, age, sex) > egfr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def adag_band(a1c) -> tuple[float, float]:
    rows = effects()["glycaemia"]["adag_table"]["value"]
    if a1c <= rows[0][0]:
        return tuple(rows[0][1:])
    for a, b in zip(rows, rows[1:]):
        if a1c <= b[0]:
            f = (a1c - a[0]) / (b[0] - a[0])
            return tuple(a[i] + f * (b[i] - a[i]) for i in (1, 2))
    return tuple(rows[-1][1:])


def adag_eag(a1c):
    return 1.5944 * a1c - 2.594


def approach(base, target, t, onset, half_time):
    if t < onset:
        return base
    return base + (target - base) * (1 - 2**(-(t - onset) / half_time)) if half_time else target


def reflect_below(v, limit, step, floor):
    return max(floor, 2 * limit - v - step) if v >= limit else v
