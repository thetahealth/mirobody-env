"""The thyroid axis of the coupled world (B4): free T4 read from TSH, person set points and the steady state
of established levothyroxine users. SYNTHETIC, evaluation only.

`ft4` is the one reader of the TSH-FT4 relation (the routine panels call it). The population line is
Spencer 1990's steady-state inverse log-linear relation, log10 TSH = 2.56 - 0.022 FT4I, scaled so that the
geometric middle of Spencer's normal TSH range (0.39-4.6 mU/L) maps to the middle of the euthyroid FT4
interval of the Elecsys FT4 II (0.93-1.7 ng/dL, FDA K131244), in pmol/L (`registry/routine_panels.yaml`
`thyroid`). With a case id the person's own curve is read: the population line scaled by the person's FT4
set point, exp(CVG_FT4 * z) with the EFLM between-person CV of FT4 (`registry/biological_variation.yaml`);
the TSH set point is the person's general-population TSH draw (Hollowell 2002 table of the panels). Within a
person FT4 and TSH then move along one curve; between persons they scatter by the set points (individual
ranges narrower than the population's: CVI < CVG for both analytes).

Established levothyroxine users (`lt4_steady_tsh`): the person's steady-state TSH at a stable dose is one of
the sourced states of `registry/drug_effects.yaml` levothyroxine `on_treatment_tsh` (over-replacement below
0.1 and 0.1-0.5 mU/L after 54-60 months, Taylor 2014; under-replacement above 4.5 mU/L, Somwaru 2009), drawn
log-uniform inside its band (our form); otherwise the person's own TSH set point (full replacement).
"""
from __future__ import annotations

import math

#: states of an established levothyroxine user, in the order their shares are cut from one uniform
LT4_STATES = ("suppressed", "low", "under")


def _relation() -> tuple[float, float, float]:
    from haenv.world.truth.panels import registry
    t = registry()["thyroid"]
    a, b = float(t["log10_tsh_intercept"]["value"]), float(t["log10_tsh_slope_per_ft4i"]["value"])
    lo, hi = (float(x) for x in t["normal_tsh_miu_l"]["value"])
    f_lo, f_hi = (float(x) for x in t["euthyroid_ft4_ng_dl"]["value"])
    mid_ft4i = (a - math.log10(math.sqrt(lo * hi))) / b
    scale = (f_lo + f_hi) / 2 * float(t["pmol_l_per_ng_dl"]["value"]) / mid_ft4i
    return a, b, scale


def set_point_scale(case_id: str) -> float:
    """The person's FT4 set point relative to the population line: exp(CVG_FT4 * z), z drawn once per case."""
    from haenv.world.truth import variation as V
    return math.exp(float(V.analyte("FT4")["cvg"]) * V.person_z(str(case_id), "FT4"))


def ft4(tsh_miu_l: float, case_id: str | None = None) -> float:
    """Free T4 (pmol/L) at steady state for a serum TSH (mIU/L); inverse log-linear in TSH.

    Without a case id: the population line. With one: the person's curve (the line times the person's FT4
    set point)."""
    if not tsh_miu_l > 0:
        raise ValueError("TSH must be positive")
    a, b, scale = _relation()
    if not math.log10(float(tsh_miu_l)) < a:
        raise ValueError("TSH beyond the relation's range (FT4 would not be positive)")
    line = scale * (a - math.log10(float(tsh_miu_l))) / b
    return line * (set_point_scale(case_id) if case_id is not None else 1.0)


def lt4_row() -> dict:
    from haenv.world.truth.drugs import _doc
    return next(r for r in _doc()["analyte_effects"] if r["drug"] == "levothyroxine")


def lt4_state(case_id: str) -> str:
    """The established user's replacement state: one of `LT4_STATES` or "replaced" (sourced shares)."""
    from haenv.world.generation import rng
    bands = lt4_row()["on_treatment_tsh"]
    u, edge = rng.unit(str(case_id), "b4-thyroid", "lt4-state"), 0.0
    for state in LT4_STATES:
        if state in bands:
            edge += float(bands[state]["share"])
            if u < edge:
                return state
    return "replaced"


def lt4_steady_tsh(case_id: str, set_point_tsh: float) -> tuple[str, float]:
    """(state, steady-state TSH in mIU/L) of an established levothyroxine user at a stable dose."""
    from haenv.world.generation import rng
    state = lt4_state(case_id)
    if state == "replaced":
        return state, float(set_point_tsh)
    lo, hi = (float(x) for x in lt4_row()["on_treatment_tsh"][state]["band_miu_l"])
    u = rng.unit(str(case_id), "b4-thyroid", "lt4-tsh")
    return state, math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def approach_log(start: float, target: float, days: float) -> float:
    """TSH after `days` at a new dose: first-order approach on the log scale with the registered T4 half-time
    (`kinetics.half_time_days`; steady state of T4 and TSH in about 6 weeks, ATA 2014)."""
    half = float(lt4_row()["kinetics"]["half_time_days"])
    f = 1.0 - 0.5 ** (max(float(days), 0.0) / half)
    return math.exp(math.log(start) + f * (math.log(target) - math.log(start)))
