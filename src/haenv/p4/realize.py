"""Pack 4 item realization: turn a case's plan into a follow-up item on its built world.

Runs inside the build, after the world's own lab measurement layer (`build.observe_labs`, wrapped
by `hooks.install`), so the world (dose line, noise-free weight, profile) is final. For one
analyte it writes:

* the history of the analyte on the case's lab calendar up to the previous draw t0, and today's
  draw at T, each with a per-point record `{day, time, truth, e, perturbations, printed}`;
* the visible context around the two compared draws: draw time, fasting label, specimen
  handling, lab and method, the meals before each draw and the medication record;
* the gold (`class`, `true_change_pct`, `factor`, `next_step`) re-derived by `gold.g4` from the
  records; an attempt whose re-derived class is not the planned one is discarded and redrawn.

Magnitude: `|dobs| = r * RCV`, r uniform in the analyte's band (PREREG 2) for the three classes with a
source, and in `NOISE_BAND` (inside the RCV) for noise: a change beyond the RCV with no source is a
significant change by the standard reading, so the gold reads the RCV comparison and the magnitude
is decisive evidence, not a surface cue. The true-change size is
`dtrue = dobs - eps`, eps a draw of the pair noise (the probe's rule); the noise pair of the
other classes is drawn from the noise model conditioned on the difference the item needs
(exact rejection sampling); the visible cause's per-case response (kappa, A) is solved from
dtrue and must fall in its registered range.

R15: a CKD patient's creatinine sits at the level of a dealt eGFR (CKD-EPI 2021), and one of the
two compared draws runs Jaffe while the other runs the enzymatic method: an absolute offset that acts
but carries at most `max_carry` of the observed change, so the switch is not the source. A noise item
is the exception: the same method at both draws, so no method offset can share its change.

Returns `None` with a reason when no admissible realization exists; the emission gate then
refuses the case.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from haenv.p4.gold import NOISE_BAND
from haenv.p4.gold import g4
from haenv.p4.gold import gold_block_consistent
from haenv.p4.gold import method_key
from haenv.p4.gold import next_step
from haenv.p4.gold import true_change_pct
from haenv.p4.plan import h01
from haenv.p4.plan import background_on
from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached


@registry_cached("p4_clinical_sources.yaml")
def clinical_sources():
    return load_registry("p4_clinical_sources.yaml")


def clinical_error(case_id: str, day: int, method: str | None = None) -> float:
    """Biological variation belongs to the specimen; analytical error to its assay.

    Splitting the registered components preserves their marginal total CV and
    keeps a patient's biological draw fixed when a different assay is used.
    """
    from haenv.world.labworld import analyte
    from haenv.world.labworld import z
    cfg = analyte("creatinine")
    key = method or cfg["cva_default"]
    return (float(cfg["cvi"]) * z("p4-biological", case_id, int(day)) +
            float(cfg["cva"][key]) * z("p4-analytical", case_id, int(day), key))


def paired_method_scatter(reference: float) -> float:
    """Remove modelled assay imprecision from the published paired-sample scatter.

    CVI cancels in a same-specimen method comparison. The published residual
    is homoscedastic, whereas registered assay CVs are relative; the remainder
    is a synthesis approximation, not an estimated patient-bias distribution.
    """
    from haenv.world.labworld import analyte
    cfg = analyte("creatinine")
    comparison = clinical_sources()["method_comparison"]
    total = float(comparison["residual_sd_mg_dL"]) * float(comparison["umol_L_per_mg_dL"])
    analytical_variance = float(reference) ** 2 * sum(float(v) ** 2 for v in cfg["cva"].values())
    return math.sqrt(max(0.0, total ** 2 - analytical_variance))


def first_acei_review_days(raw, ckd, case_id: str) -> int:
    """Draw the first renal check after initiation, rather than a settled review.

    The interval follows the patient's risk before any observed values are
    generated. The spread within the recommended window is illustrative.
    """
    from haenv.world.labworld import u01
    up = raw.user_profile or {}
    age = str(up.get("age_range") or "0-4")
    older = int(age.split("-")[0]) >= 60 if age[:2].isdigit() else False
    known = set(up.get("known_conditions") or ())
    risk = bool(ckd) or older or bool(known & {"CKD", "T2D"})
    cfg = clinical_sources()["acei_first_review"]
    lo, hi = cfg["higher_risk_days" if risk else "usual_days"]
    return int(lo) + int(u01("p4-first-acei-review", case_id) * (int(hi) - int(lo) + 1))


def clinical_acei_response(case_id: str) -> float:
    """A declared population proxy, not a causal estimate from measured changes.

    Published observed-change bins provide mixture weights. Positive response
    within each bin is illustrative; the existing mechanism's upper limit is
    retained. The data include measurement variation and negative changes.
    """
    from haenv.world.labworld import u01
    cfg = clinical_sources()["acei_response_proxy"]
    upper = float(_tables()["effects"]["acei_on_creatinine"]["A"][1])
    rows = [row for row in cfg["observed_bins"]
            if math.log1p(float(row["lower_pct"]) / 100) < upper]
    total = sum(row["n"] for row in rows)
    value = u01("p4-response", case_id, 0, 0) * total
    for row in rows:
        value -= row["n"]
        if value < 0:
            lo = math.log1p(float(row["lower_pct"]) / 100)
            hi = min(upper, math.log1p(float(row["upper_pct"]) / 100))
            return lo + (hi - lo) * u01("p4-response-within", case_id)
    raise AssertionError("response mixture draw outside its population")

N_GAPS = 64            # candidate (t0, T) intervals per case
GAP_LO, GAP_HI = 28, 182
N_ATTEMPTS = 160       # noise / bias redraws per interval
EDGE_DAYS = 7          # a cause lies at least this far after t0
#: Creatinine background (never acting on the compared change, R7): ACEI events at least this
#: many days before the previous draw (seven time constants: settled); a meat dinner two evenings
#: before a draw (34 h or more: outside the 12-24 h window, so it cannot act); the lab running Jaffe on every draw (no switch).
BG_ACEI_MIN_BEFORE = 49
BG_MEAT_DINNER_RATE = 0.50        # per compared draw (each draw independently), see `_place_context`
#: A non-acting wait before centrifugation is drawn inside [DELAY_DECOY_MIN_H, the analyte's stability bound - 1 h]
#: (`labworld.perturb.stability_h`), so the printed wait always lies inside the bound the report states.
DELAY_DECOY_MIN_H = 2.0
BG_JAFFE_ALL_RATE = 0.45
#: R8: on a glucose item outside the method class, every draw on the bedside meter (one method
#: throughout, no switch). (R8's light breakfast before a draw labelled fasting is gone: a draw is
#: labelled by its clock, and every non-fasting draw shows the meals before it.)
BG_POC_ALL_RATE = 0.40
BG_REAGENT_RATE = 0.30
TAIL_TAUS = 1          # "no net GLP-1 step" looks back this many time constants before t0


def fed_draw_ok(cls: str, clock0: str, clock1: str) -> bool:
    """Whether the interval (draw clocks of the previous and today's draw) may carry an item of `cls`:
    every class other than the meal class needs a compared draw outside the fasting hours."""
    from haenv.p4.render import is_fasting
    return cls == "preanalytical" or not (is_fasting(clock0[:5]) and is_fasting(clock1[:5]))


class Unrealizable(ValueError):
    pass


#: Rejection tally of the attempt loop (diagnostics only; read by tools, never by the gold).
REJECTIONS: dict[str, int] = {}


def _rej(label: str) -> None:
    REJECTIONS[label] = REJECTIONS.get(label, 0) + 1


def _tables():
    from haenv.world.labworld import tables
    data = tables()
    return data if background_on() else {**data, "render": data["render_background_off"]}


def _clock(case_id: str, day: int) -> str:
    from haenv.world.generation.build import _draw_clock
    return _draw_clock(str(case_id), int(day), 0)


def _hhmm_minus(clock: str, hours: float) -> str:
    h, m = int(clock[:2]), int(clock[3:5])
    t = (h * 60 + m - int(round(hours * 60))) % (24 * 60)
    return f"{t // 60:02d}:{t % 60:02d}"


def _visible_dose(raw, T: int) -> list[dict]:
    return [q for q in (raw.longitudinal_data or {}).get("dose_timeline") or [] if int(q["ts"]) <= T]


def _drug(raw) -> str:
    pb = (raw.latent_premise or {}).get("patient_basics") or {}
    return str((pb.get("regimen") or {}).get("drug") or "")


def _indication(raw) -> str | None:
    from haenv.p4.plan import acei_indication
    pb = (raw.latent_premise or {}).get("patient_basics") or {}
    return acei_indication(pb.get("disease"), pb.get("comorbidities"))


def calendar(case_id: str, analyte: str, ce: int, dose_pts) -> list[int]:
    """The analyte's draw days on the case's lab calendar (`build.lab_draw_days`)."""
    from haenv.world.generation.build import _drawn
    from haenv.world.generation.build import lab_draw_days
    from haenv.world.labworld import analyte as A
    days = lab_draw_days(str(case_id), int(ce), dose_pts)
    if analyte == "fasting_glucose":
        return [d for d in days if _drawn(str(case_id), "fasting_glucose", d)]
    p = float(A(analyte).get("draw_inclusion", 1.0))
    from haenv.world.labworld import u01
    return [d for d in days if d == 0 or u01("cr-cal", case_id, d) < p]


def history_records(raw, case_id: str, an: str, world, sh: float, nd: int, rng) -> list[dict]:
    """The analyte's draws in the world's record before day 0 (`med_course.history_visit_days` from
    `world_medication.history_lo`): each visit draws the analyte with its `draw_inclusion`; the value is the
    same world's truth on that day (`world.truth`) with the item's per-draw variation; no perturbation."""
    import haenv.world.clinical.med_course as MC
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    from haenv.world.labworld import analyte as A
    from haenv.world.labworld import u01
    from haenv.world.labworld import z
    from haenv.world.labworld.observe import make_record
    lo = ((getattr(raw, "latent_premise", None) or {}).get(WORLD_MEDICATION_KEY) or {}).get("history_lo")
    if lo is None:
        return []
    p = float(A(an).get("draw_inclusion", 1.0))
    return [make_record(d, _clock(case_id, d), world.truth(d), sh * z("p4prefix", case_id, an, d), [], nd, rng)
            for d in MC.history_visit_days(str(case_id), int(lo)) if u01("cr-prefix", case_id, d) < p]


def ckd_record(raw, case_id: str) -> dict:
    """R15: the CKD patient's age, sex, eGFR, stage and creatinine set point (CKD-EPI 2021)."""
    from haenv.world.labworld import tables
    from haenv.world.labworld import u01
    from haenv.world.labworld.truth import cr_for_egfr
    reg = tables()["comorbidities"]["CKD"]
    up = raw.user_profile or {}
    sex = "M" if str(up.get("sex")) == "M" else "F"
    a = str(up.get("age_range") or "50-54")
    lo, hi = (int(x) for x in a.split("-")) if a[:2].isdigit() and "-" in a else (50, 54)
    age = lo + int(u01("p4ckd-age", case_id) * (hi - lo + 1))
    e_lo, e_hi = (float(x) for x in reg["egfr"])
    egfr = e_lo + (e_hi - e_lo) * u01("p4ckd-egfr", case_id)
    return {"code": "CKD", "display": reg["display"], "age": age, "sex": sex, "egfr0": round(egfr, 1),
            "stage": "G3b" if egfr >= 30 else "G4", "cr0": round(cr_for_egfr(egfr, age, sex), 2)}


def _base(case_id: str, analyte: str, raw, ctx) -> float:
    from haenv.world.labworld import analyte as A
    from haenv.world.labworld import z
    a = A(analyte)
    up = raw.user_profile or {}
    if analyte == "fasting_glucose":
        known = set(up.get("known_conditions") or ())
        co = a["baseline"]["T2D" if "T2D" in known else "default"]
        a1c = ((ctx or {}).get("clinical") or {}).get("HbA1c") or {}
        if a1c.get("base"):
            # anchored on the world's HbA1c set point (ADAG eAG = 1.59 A1c - 2.59; fasting below it)
            return min(max(1.59 * float(a1c["base"]) - 2.59 - 0.6, 4.6), 14.0)
    else:
        co = a["baseline"]["M" if str(up.get("sex")) == "M" else "F"]
    return float(co["median"]) * math.exp(float(co["sd_log"]) * z("p4base", case_id, analyte))


class _World:
    """The truth model of one item (fixed per realization)."""

    def __init__(self, analyte, base, dose_pts, wpts, kappa=0.0, A=0.0, events=(), initial_on=False,
                 glp1=True):
        e = _tables()["effects"]
        self.analyte, self.base = analyte, base
        self.dose_pts, self.wpts = dose_pts, wpts
        self.kappa, self.A = kappa, A
        self.events, self.initial_on = list(events), initial_on
        self.glp1 = glp1
        self.tau_g = float(e["glp1_on_fasting_glucose"]["tau_days"])
        self.beta = float(e["weight_on_fasting_glucose"]["beta"])
        self.smooth = int(e["weight_on_fasting_glucose"]["smooth_days"])
        self.tau_a = float(e["acei_on_creatinine"]["tau_days"])

    def log_truth(self, day: int) -> float:
        from haenv.world.labworld import truth as TR
        if self.analyte == "fasting_glucose":
            return TR.fg_log(day, base=self.base, kappa=(self.kappa if self.glp1 else 0.0), beta=self.beta,
                             tau=self.tau_g, dose_pts=self.dose_pts, weight_pts=self.wpts,
                             smooth_days=self.smooth)
        return TR.cr_log(day, base=self.base, A=self.A, events=self.events, tau=self.tau_a,
                         initial_on=self.initial_on)

    def truth(self, day: int) -> float:
        return round(math.exp(self.log_truth(day)), 4)


def _draw_cvs(an: str, cls: str, s: int, case_id: str, ckd: bool) -> tuple[float, float, float]:
    """Total CV of the previous draw, of today's draw and of the history draws. Creatinine's depends
    on the method (`labworld.cv_total`): the Jaffe draw of a method-difference item is today's when
    the change rises and the previous one when it falls; any other item runs Jaffe on every draw or
    on none (`BG_JAFFE_ALL_RATE`, drawn here and rendered by `_place_context`)."""
    from haenv.world.labworld import cv_total
    if an != "creatinine":
        c = cv_total(an)
        return c, c, c
    if cls == "method_difference":
        return cv_total(an, "jaffe" if s < 0 else None), cv_total(an, "jaffe" if s > 0 else None), cv_total(an)
    c = cv_total(an, "jaffe" if jaffe_throughout(case_id, cls, ckd) else None)
    return c, c, c


#: Lower edge of the change band in RCV units: a change item stays beyond the RCV when the reader's
#: RCV runs up to a fifth above the registry's (a creatinine CVI of 5.5% instead of 4.5%).
CHANGE_BAND_LO = 1.3


def change_band(an: str) -> tuple[float, float]:
    """|dobs|/RCV band of the three classes with a change beyond the RCV."""
    from haenv.world.labworld import analyte as A
    return CHANGE_BAND_LO, float(A(an)["band_upper_rcv"])


def jaffe_throughout(case_id: str, cls: str, ckd: bool) -> bool:
    from haenv.world.labworld import u01
    return cls != "method_difference" and not ckd and u01("p4bg-jaffe", case_id) < BG_JAFFE_ALL_RATE


def _steps_in(dose_pts, lo: int, hi: int) -> list[tuple[int, float]]:
    from haenv.world.labworld.truth import dose_steps
    return [(t, d) for t, d in dose_steps(dose_pts) if lo < t <= hi]


def _perturbation(factor: str, analyte: str, role: str, *, bias_log: float = 0.0, additive=None,
                  params=None) -> dict:
    from haenv.world.labworld.perturb import acts_on
    from haenv.world.labworld.perturb import kind
    acts = role == "cause" and acts_on(factor, analyte)
    return {"factor": factor, "kind": kind(factor), "role": role, "acts": bool(acts),
            "bias_log": round(float(bias_log), 6) if acts else 0.0,
            "additive": (round(float(additive), 6) if (acts and additive is not None) else None),
            "params": dict(params or {})}


def options(plan: dict) -> list[tuple[str, int]]:
    """(analyte, sign) in the order the item tries them: the dealt pair, the other sign, then the
    other eligible analyte with the dealt sign and with the other sign. Which pair a world can
    carry depends on its dose phase (an escalating GLP-1 line lowers glucose on every interval);
    the pack assembly balances the realized pairs inside each class (PREREG 4, revision R1)."""
    from haenv.p4.plan import signs_of
    an, s = plan["analyte"], int(plan["sign"])
    out = [(an, x) for x in (s, -s) if x in signs_of(an)]
    for b in plan.get("analytes_ok") or ():
        if b != an:
            out += [(b, x) for x in (s, -s) if x in signs_of(b)]
    return out


def realize(raw, case_id: str, plan: dict, ctx: dict | None) -> dict:
    """The realized item (gold, records, context) for the first feasible (analyte, sign) of
    `options(plan)`. Raises `Unrealizable` when none is feasible."""
    if plan.get("version") == 2:
        from haenv.p4.diversity import realize as diverse
        return diverse(raw, case_id, plan, ctx)
    why = []
    for rank, (an, s) in enumerate(options(plan)):
        try:
            item = _realize_one(raw, case_id, {**plan, "analyte": an, "sign": s,
                                               "decoys": (plan.get("decoys") or {}).get(an) or [],
                                               "factor": (plan.get("factor") or {}).get(an)}, ctx)
        except Unrealizable as e:
            why.append(f"{an}{'+' if s > 0 else '-'}: {e}")
            continue
        item["option_rank"] = rank
        return item
    raise Unrealizable(" | ".join(why))


def _realize_one(raw, case_id: str, plan: dict, ctx: dict | None) -> dict:
    if plan.get("version") == 2:
        from haenv.p4.diversity import realize as diverse
        return diverse(raw, case_id, plan, ctx)
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        return _realize_clinical(raw, case_id, plan, ctx)
    from haenv.world.labworld import analyte as A
    from haenv.world.labworld import rcv95
    from haenv.world.labworld import u01
    from haenv.world.labworld import z
    from haenv.world.labworld.perturb import factor as F
    tb = _tables()
    an, cls, s = plan["analyte"], plan["class"], int(plan["sign"])
    a = A(an)
    nd, rng = int(a["ndigits"]), tuple(float(x) for x in a["range"])
    ckd_item = plan.get("comorbidity") == "CKD" and an == "creatinine"
    s0, s1, sh = _draw_cvs(an, cls, s, case_id, ckd_item)
    sig = math.sqrt(s0 * s0 + s1 * s1)          # sd of the pair difference
    R = rcv95(s0, s1)
    band = change_band(an)
    lo_r, hi_r = NOISE_BAND if cls == "analytic_biological_noise" else band
    r = lo_r + (hi_r - lo_r) * float(plan["u_ratio"])
    D = s * r * R
    T = int(raw.prediction_context["prediction_time_T"])
    ctx = ctx or {}
    meta = ((raw.latent_premise or {}).get("meta") or {})
    ce = int(ctx.get("ce") or meta.get("course_end_day") or T)
    dose_vis = _visible_dose(raw, T)
    drug = _drug(raw)
    glp1 = drug in (tb["render"].get("glp1_class") or ())
    wpts = [q for q in (ctx.get("wpts") or (raw.longitudinal_data or {}).get("weight") or []) if int(q["ts"]) <= T]
    ckd = ckd_record(raw, case_id) if ckd_item else None
    base = float(ckd["cr0"]) if ckd else _base(case_id, an, raw, ctx)
    eff = tb["effects"]
    k_lo, k_hi = (float(x) for x in eff["glp1_on_fasting_glucose"]["kappa"])
    k_def = float(eff["glp1_on_fasting_glucose"]["kappa_default"])
    a_lo, a_hi = (float(x) for x in eff["acei_on_creatinine"]["A"])
    tau_g = float(eff["glp1_on_fasting_glucose"]["tau_days"])
    chronic_acei = u01("p4acei-chronic", case_id) < 0.25
    cal = calendar(case_id, an, ce, (raw.longitudinal_data or {}).get("dose_timeline"))
    reasons: list[str] = []

    gap_hi = min(GAP_HI, T - 21)
    if gap_hi < GAP_LO:
        raise Unrealizable(f"T {T}: no previous draw at least {GAP_LO} days earlier and after day 21")
    for j in range(N_GAPS):
        gap = GAP_LO + int(h01("p4gap", case_id, j) * (gap_hi - GAP_LO + 1))
        t0 = T - gap
        # A fed draw is on the record of every class, not only of the pre-analytical one (whose meal sits
        # before a non-fasting draw): a creatinine draw does not need fasting, so the fed label names no
        # class. Each non-meal class takes an interval with at least one fed compared draw, the same
        # condition the meal class meets by construction.
        if background_on() and not fed_draw_ok(cls, _clock(case_id, t0), _clock(case_id, T)):
            _rej(f"{cls}:{an}:no_fed_draw")
            continue
        steps_win = _steps_in(dose_vis, t0, T) if glp1 else []
        steps_tail = _steps_in(dose_vis, t0 - int(TAIL_TAUS * tau_g), T) if glp1 else []
        from haenv.world.labworld.truth import weight_log_term
        smooth = int(eff["weight_on_fasting_glucose"]["smooth_days"])
        W = weight_log_term(wpts, T, smooth) - weight_log_term(wpts, t0, smooth)
        events: list[dict] = []
        initial_on = chronic_acei
        cause = None
        world = None
        if an == "fasting_glucose":
            if cls != "true_change":
                # a GLP-1 step acts on glucose: the dose may not change on net in the window or
                # within one time constant before it (a step and its reversal may sit there);
                # the zone check below then keeps intervals on which the set point did not move
                near = _steps_in(dose_vis, t0 - int(TAIL_TAUS * tau_g), T)
                if near and abs(sum(d for _, d in near)) > 1e-9:
                    _rej(f"{cls}:{an}:glp1_net_step_near_window")
                    continue
                world = _World(an, base, dose_vis, wpts, kappa=k_def, glp1=glp1)
            else:
                world = None            # solved per attempt below
        else:
            # an ACEI started well before the previous draw (settled by t0, so it does not move
            # this change): the same factor shown at a timing where it does not act (R7)
            # every followed patient has an ACEI indication and an ACEI history before the window:
            # on it since before day 0, or started well before the previous draw -- and on a
            # non-true item, when there is room, stopped for cough well before it too (R13i), so
            # the number of ACEI rows does not tell a stop inside the window
            bg_ev = []
            d_bg = t0 - BG_ACEI_MIN_BEFORE - int(u01("p4acei-bg-day", case_id) * 60)
            if not initial_on and d_bg >= 7:
                bg_ev = [{"day": d_bg, "action": "start", "role": "background"}]
                room = t0 - BG_ACEI_MIN_BEFORE - (d_bg + 28)
                if cls != "true_change" and room >= 0:
                    d_st = d_bg + 28 + int(u01("p4acei-bg-stop", case_id) * (room + 1))
                    bg_ev.append({"day": d_st, "action": "stop", "role": "background"})
            elif not initial_on:
                initial_on = True
            events = list(bg_ev)
            world = _World(an, base, dose_vis, wpts, A=(a_lo + a_hi) / 2, events=events, initial_on=initial_on)
        # ---- decoys realizable on this interval (substitute in pool order when not) ----
        from haenv.world.labworld.perturb import decoys_for
        dec_plan = list(plan.get("decoys") or ())
        pool = [f for f in decoys_for(an) if f not in dec_plan]
        dec: list[str] = []
        subs: list[dict] = []

        def realizable(f):
            if f == "glp1_dose_change":
                return bool(steps_win)
            if f == "weight_change":
                return abs(W) >= 0.04
            return True
        for f in dec_plan:
            if realizable(f):
                dec.append(f)
                continue
            alt = next((g for g in pool if realizable(g) and g not in dec), None)
            subs.append({"planned": f, "used": alt})
            if alt:
                dec.append(alt)
                pool.remove(alt)
        for attempt in range(N_ATTEMPTS):
            key = (case_id, j, attempt)
            # ---- truth change for this attempt ----
            if cls == "true_change":
                eps = sig * z("p4eps", *key)
                dtrue = D - eps
                if not (abs(dtrue) >= R / 2 * 1.04 and abs(dtrue) >= abs(D) / 2 * 1.04 and dtrue * D > 0):
                    _rej("true:eps")
                    continue
                if an == "fasting_glucose":
                    from haenv.world.labworld.truth import glp1_unit_term
                    beta = float(eff["weight_on_fasting_glucose"]["beta"])
                    G = glp1_unit_term(dose_vis, T, tau_g) - glp1_unit_term(dose_vis, t0, tau_g)
                    cause = None
                    ok_steps = [t for t, d in steps_win if t0 + EDGE_DAYS <= t <= T - 14]
                    if glp1 and ok_steps and abs(G) > 1e-6:
                        kap = (dtrue - beta * W) / (-G)
                        if k_lo <= kap <= k_hi:
                            cause = ("glp1_dose_change", {"kappa": round(kap, 6),
                                                          "steps": [list(x) for x in steps_win]})
                            world = _World(an, base, dose_vis, wpts, kappa=kap, glp1=glp1)
                    if cause is None and abs(W) >= 0.04 and beta * W * D > 0:
                        world = _World(an, base, dose_vis, wpts, kappa=k_def, glp1=glp1)
                        dtrue = world.log_truth(T) - world.log_truth(t0)
                        if (abs(dtrue) >= R / 2 and abs(dtrue) >= abs(D) / 2 and abs(D - dtrue) <= 2.5 * sig
                                and abs(beta * W) > abs(dtrue - beta * W)):
                            cause = ("weight_change", {"dlog_weight": round(W, 6)})
                    if cause is None:
                        _rej("true:fg_no_cause")
                        continue
                else:
                    lo_d, hi_d = t0 + EDGE_DAYS, T - 14
                    if hi_d < lo_d:
                        break
                    d_e = lo_d + int(h01("p4acei-day", case_id, j) * (hi_d - lo_d + 1))
                    if s > 0:
                        ev, init = [{"day": d_e, "action": "start"}], False
                    elif bg_ev:
                        # started well before the window (the background start), stopped inside it
                        ev, init = bg_ev + [{"day": d_e, "action": "stop"}], False
                    else:
                        ev, init = [{"day": d_e, "action": "stop"}], True
                    from haenv.world.labworld.truth import acei_on
                    du = acei_on(T, ev, world.tau_a, init) - acei_on(t0, ev, world.tau_a, init)
                    Aval = dtrue / du if abs(du) > 1e-9 else float("nan")
                    if not (a_lo <= Aval <= a_hi):
                        _rej("true:acei_A")
                        continue
                    world = _World(an, base, dose_vis, wpts, A=Aval, events=ev, initial_on=init)
                    events, initial_on = ev, init
                    cause = ("acei_start_or_stop", {"A": round(Aval, 6), "event": ev[0]})
                dtrue = world.log_truth(T) - world.log_truth(t0)
            else:
                dtrue = world.log_truth(T) - world.log_truth(t0)
                if abs(dtrue) >= 0.8 * R / 4:
                    _rej(f"{cls}:{an}:truth_moved")
                    break
            # ---- the acting perturbation (pre-analytic / method) ----
            pert0: list[dict] = []
            pert1: list[dict] = []
            B = 0.0
            fac = plan.get("factor")
            add_on = None
            if cls in ("preanalytical", "method_difference"):
                f = F(fac)
                need = abs(D) / 2 * 1.06
                fsign = int(f.get("sign", 0))
                if fsign == 0:
                    on_t1 = h01("p4place", case_id, attempt) < 0.5
                    bsign = s if on_t1 else -s
                else:
                    on_t1 = fsign == s
                    bsign = fsign
                if fac == "delayed_processing":
                    rate = float(f["rate_per_h"][0]) + (float(f["rate_per_h"][1]) - float(f["rate_per_h"][0])) * u01("p4rate", *key)
                    hmin = max(float(f["hours"][0]), 0.5 + need / rate)
                    if hmin > float(f["hours"][1]):
                        _rej("pre:hours")
                        continue
                    hours = round(hmin + (float(f["hours"][1]) - hmin) * u01("p4hours", *key) * 0.5, 1)
                    mag = rate * (hours - 0.5)
                    if mag < need:
                        continue
                    params = {"hours": hours, "rate_per_h": round(rate, 4)}
                elif fac == "postprandial_labelled_fasting":
                    mag = need + (min(0.45, need * 2.2) - need) * u01("p4mag", *key)
                    params = {}
                else:
                    if fac == "meat_meal":
                        # a cooked-meat meal at a meal time 1-4 h before a non-fasting draw
                        from haenv.p4.render import meat_slot
                        dday = T if on_t1 else t0
                        slot = meat_slot(_clock(case_id, dday)[:5], case_id, dday)
                        if slot is None:
                            _rej("pre:meat_on_fasting_draw")
                            break
                    lo, hi = (float(x) for x in f["effect"])
                    lo = max(lo, need)
                    if lo > hi:
                        _rej(f"{fac}:need>hi")
                        continue
                    mag = lo + (hi - lo) * u01("p4mag", *key)
                    params = {"meal": slot[0], "meal_time": slot[1]} if fac == "meat_meal" else {}
                b = bsign * mag
                B = b if on_t1 else -b
                add_on = (on_t1, fac, b, params)
            # ---- the noise pair, conditioned on the difference the item needs ----
            diff = D - dtrue - B
            e0 = -diff * s0 * s0 / (sig * sig) + (s0 * s1 / sig) * z("p4e0", *key)
            e1 = e0 + diff
            if abs(e0) > 3 * s0 or abs(e1) > 3 * s1:
                _rej(f"{cls}:{an}:noise_clip")
                continue
            y0, y1 = world.truth(t0), world.truth(T)
            if add_on is not None:
                on_t1, fac, b, params = add_on
                if fac == "postprandial_labelled_fasting":
                    yb = (y1 if on_t1 else y0) * math.exp(e1 if on_t1 else e0)
                    delta = yb * (math.exp(b) - 1.0)
                    lo_a, hi_a = (float(x) for x in F(fac)["additive_mmol"])
                    if not lo_a <= delta <= hi_a:
                        _rej("pp:delta")
                        continue
                    p = _perturbation(fac, an, "cause", additive=delta,
                                      params={"meal_hours_before": round(1.0 + 3.0 * u01("p4meal", *key), 2)})
                else:
                    p = _perturbation(fac, an, "cause", bias_log=b, params=params)
                (pert1 if on_t1 else pert0).append(p)
            from haenv.world.labworld.observe import make_record
            r0 = make_record(t0, _clock(case_id, t0), y0, e0, pert0, nd, rng)
            r1 = make_record(T, _clock(case_id, T), y1, e1, pert1, nd, rng)
            g = g4(r0, r1, R, band)
            if g["class"] != cls:
                _rej(f"{cls}:{an}:g4:{g['reason'][:28]}")
                continue
            # ---- accepted: history, decoys, background, visible context ----
            hist = []
            for d in cal:
                if d >= t0 - 14 or d > T:
                    continue
                e = sh * z("p4hist", case_id, an, d)
                hist.append(make_record(d, _clock(case_id, d), world.truth(d), e, [], nd, rng))
            hist = history_records(raw, case_id, an, world, sh, nd, rng) + hist
            item = {"analyte": an, "class": cls, "sign": s, "t0": t0, "t1": T, "rcv": round(R, 6),
                    "band": list(band), "ratio_target": round(r, 6), "cv_total": [s0, s1],
                    "records": hist + [r0, r1], "decoys": dec, "decoy_substitutions": subs,
                    "cause": (cause[0] if cause else (fac if cls in ("preanalytical", "method_difference") else None)),
                    "cause_params": (cause[1] if cause else (add_on[3] if add_on else {})),
                    "acei": {"events": events, "initial_on": bool(initial_on), "indication": _indication(raw)},
                    "base": round(base, 4), "drug": drug, "glp1_class": glp1,
                    "dlog_weight": round(W, 6), "gap_index": j, "attempt": attempt, "g4": g,
                    **({"comorbidity": ckd} if ckd else {})}
            item["true_change_pct"] = true_change_pct(r0, r1)
            item["factor"] = item["cause"] if cls != "analytic_biological_noise" else None
            _place_context(item, case_id, steps_win, bool(item["acei"]["initial_on"]))
            item["next_step"] = next_step(item)[0]
            # every injected factor must leave the gold where the records put it: the item is re-derived once
            # more with its background in place, and a violation redraws the attempt
            bad = gold_block_consistent(item)
            if bad:
                _rej(f"{cls}:{an}:background:{bad[0][:40]}")
                continue
            return item
        reasons.append(f"gap{j}: no admissible draw")
    raise Unrealizable("; ".join(reasons[-2:]) or "no interval")


def _realize_clinical(raw, case_id, plan, ctx):
    """Draw clinical effects and independent per-draw variation before classifying.

    No amplitude target is used. Parameters and the three-sigma measurement
    envelope are the existing laboratory world, not a classifier calibration.
    """
    from haenv.world.labworld import analyte
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    from haenv.world.labworld import u01
    from haenv.world.labworld import z
    from haenv.world.labworld.perturb import factor
    from haenv.world.labworld.observe import make_record
    from haenv.p4.features import visible_class
    from haenv.p4.render import render_block
    from haenv.p4.render import meat_slot
    an, cls, sign = plan["analyte"], plan["class"], int(plan["sign"])
    if an != "creatinine":
        raise Unrealizable("clinical-source realization requires a registered creatinine mechanism")
    cfg = analyte(an)
    nd, bounds = int(cfg["ndigits"]), tuple(cfg["range"])
    T = int(raw.prediction_context["prediction_time_T"])
    ctx = ctx or {}
    ckd = ckd_record(raw, case_id) if plan.get("comorbidity") == "CKD" else None
    base = float(ckd["cr0"]) if ckd else _base(case_id, an, raw, ctx)
    s0, s1, sh = _draw_cvs(an, cls, sign, case_id, bool(ckd))
    R, band = rcv95(s0, s1), change_band(an)
    doses, weights = _visible_dose(raw, T), ctx.get("wpts") or []
    cal = calendar(case_id, an, int(ctx.get("ce") or T), doses)
    # A is a patient response, not a measurement error. Keep the original
    # first draw when measurement errors are rejected; redrawing both would
    # preferentially accept larger physical effects.
    response = clinical_acei_response(case_id)
    method_slope, method_additive, method_side = None, None, None
    if cls == "method_difference":
        from haenv.world.labworld import cv_total
        comparison = clinical_sources()["method_comparison"]
        conversion = float(comparison["umol_L_per_mg_dL"])
        method_slope = float(comparison["slope"])
        reference = base * math.exp(response) if sign > 0 else base
        method_additive = (conversion * float(comparison["intercept_mg_dL"]) +
            paired_method_scatter(reference) * z("p4-clinical-bias", case_id, 0, 0))
        method_side = 1 if sign * ((method_slope - 1) * reference + method_additive) > 0 else 0
        s0, s1 = (cv_total(an, "jaffe" if method_side == 0 else None),
                  cv_total(an, "jaffe" if method_side == 1 else None))
        R = rcv95(s0, s1)
    # Today's specimen is the same specimen for every candidate comparison.
    # Measurement variation belongs to a dated specimen, not to a requested
    # answer class. An ambiguous pair is ineligible, rather than an invitation
    # to redraw that specimen until it agrees with the planned answer.
    e1 = clinical_error(case_id, T, "jaffe" if s1 == cv_total(an, "jaffe") else None)
    first_review = first_acei_review_days(raw, ckd, case_id) if cls == "true_change" and sign > 0 else None
    for j in range(N_GAPS):
        gap_hi = min(GAP_HI, T - 21)
        if gap_hi < GAP_LO:
            break
        t0 = T - GAP_LO - int(h01("p4gap", case_id, j) * (gap_hi - GAP_LO + 1))
        if background_on() and not fed_draw_ok(cls, _clock(case_id, t0), _clock(case_id, T)):
            continue
        # Earlier ACEI courses are generated on the same calendar for all
        # source roles. The last event can precede the comparison or lie in it;
        # its timing, rather than the existence or count of rows, is clinical evidence.
        event_day = (t0 + EDGE_DAYS + int(h01("p4acei-day", case_id, j) * (T - 14 - t0 - EDGE_DAYS + 1))
                     if cls == "true_change" else t0 - BG_ACEI_MIN_BEFORE - int(60 * u01("p4acei-bg-day", case_id)))
        if first_review is not None:
            event_day = T - first_review
            if event_day < t0 + EDGE_DAYS:
                continue
        initial_on = False
        events = []
        cycles = int(3 * u01("p4-prior-courses", case_id))
        for k in range(cycles):
            d = event_day - 200 * (cycles - k)
            events += [{"day": d, "action": "start", "role": "background"},
                       {"day": d + 56, "action": "stop", "role": "background"}]
        # Establish the state before the final event without inventing a start
        # or stop incompatible with that state.
        if sign < 0:
            events.append({"day": event_day - 112, "action": "start", "role": "background"})
        events.append({"day": event_day, "action": "start" if sign > 0 else "stop",
                       "role": "cause" if cls == "true_change" else "background"})
        fac = plan.get("factor")
        magnitude = None
        if cls == "preanalytical":
            if fac != "meat_meal":
                raise Unrealizable("no registered clinical increment for this preanalytical source")
            lo, hi = map(float, clinical_sources()["cooked_meat"]["increment_umol_L"])
            meal_day = T if sign > 0 else t0
            magnitude = lo + (hi - lo) * u01("p4-clinical-bias", case_id, meal_day, 0)
        for attempt in range(1):
            world = _World(an, base, doses, weights, A=response, events=events, initial_on=initial_on)
            e0 = clinical_error(case_id, t0, "jaffe" if s0 == cv_total(an, "jaffe") else None)
            if abs(e0) > 3 * s0 or abs(e1) > 3 * s1:
                continue
            pert = [[], []]
            params = {}
            if cls in ("preanalytical", "method_difference"):
                side = method_side if cls == "method_difference" else (1 if sign > 0 else 0)
                if fac == "meat_meal":
                    day = T if side else t0
                    slot = meat_slot(_clock(case_id, day)[:5], case_id, day)
                    if slot is None:
                        break
                    params = {"meal": slot[0], "meal_time": slot[1]}
                pert[side].append(_perturbation(fac, an, "cause", params=params,
                                  **({"additive": magnitude} if fac == "meat_meal" else
                                     {"bias_log": math.log(method_slope), "additive": method_additive})))
            r0 = make_record(t0, _clock(case_id, t0), world.truth(t0), e0, pert[0], nd, bounds)
            r1 = make_record(T, _clock(case_id, T), world.truth(T), e1, pert[1], nd, bounds)
            g = g4(r0, r1, R, band, source_policy="clinical")
            if g["class"] != cls or g["dlog_obs"] * sign <= 0:
                continue
            history = history_records(raw, case_id, an, world, sh, nd, bounds)
            history += [make_record(d, _clock(case_id, d), world.truth(d), sh * z("p4hist", case_id, an, d),
                                    [], nd, bounds) for d in cal if d < t0 - 14]
            item = {"analyte": an, "class": cls, "sign": sign, "t0": t0, "t1": T,
                    "source_policy": "clinical", "rcv": round(R, 6), "band": list(band),
                    "cv_total": [s0, s1], "ratio_target": g["ratio"], "records": history + [r0, r1],
                    "decoys": list(plan.get("decoys") or ()), "decoy_substitutions": [],
                    "cause": "acei_start_or_stop" if cls == "true_change" else fac,
                    "cause_params": {"A": round(response, 6)} if cls == "true_change" else params,
                    "acei": {"events": events, "initial_on": initial_on, "indication": _indication(raw)},
                    "base": round(base, 4), "drug": _drug(raw), "glp1_class": False, "dlog_weight": 0.0,
                    "gap_index": j, "attempt": attempt, "g4": g,
                    **({"comorbidity": ckd} if ckd else {})}
            item["true_change_pct"] = true_change_pct(r0, r1)
            item["factor"] = item["cause"] if cls != "analytic_biological_noise" else None
            _place_context(item, case_id, [], initial_on)
            item["next_step"] = next_step(item)[0]
            if gold_block_consistent(item) or visible_class({"prediction_context": {"followup_p4": render_block(item, raw, case_id)}})[0] != cls:
                continue
            return item
    raise Unrealizable("no unambiguous clinical-source realization")


_MEAL_FACTORS = ("postprandial_labelled_fasting", "meat_meal")


def _place_context(item: dict, case_id: str, steps_win, chronic_acei: bool) -> None:
    """Decoys and never-acting background notes, recorded on the points they sit on with
    `acts: false` and zero bias (the printed values do not move), or on the window."""
    an, t0, T = item["analyte"], int(item["t0"]), int(item["t1"])
    recs = {int(r["day"]): r for r in item["records"]}
    r0, r1 = recs[t0], recs[T]
    from haenv.world.labworld import u01
    from haenv.world.labworld.perturb import factor as F
    cause_day = None
    for r in (r0, r1):
        if any(p.get("role") == "cause" for p in r["perturbations"]):
            cause_day = int(r["day"])
    detail: list[dict] = []
    acei_dec: list[dict] = []
    for i, f in enumerate(item["decoys"]):
        if f == "glp1_dose_change":
            detail.append({"factor": f, "where": "window", "acts_on_followed": False,
                           "steps": [list(x) for x in steps_win]})
            continue
        if f == "weight_change":
            detail.append({"factor": f, "where": "window", "acts_on_followed": False,
                           "dlog_weight": item["dlog_weight"]})
            continue
        if f == "acei_start_or_stop":
            lo, hi = t0 + EDGE_DAYS, T - 7
            d = lo + int(u01("p4dec-acei", case_id) * max(1, hi - lo + 1))
            ev = {"day": min(d, hi), "action": "stop" if chronic_acei else "start"}
            acei_dec.append(ev)
            detail.append({"factor": f, "where": "window", "acts_on_followed": False, "event": ev})
            continue
        # point-level decoys: never on the draw whose cause conflicts with them
        cands = [r0, r1]
        if f in _MEAL_FACTORS and cause_day is not None and item.get("cause") in _MEAL_FACTORS:
            cands = [r for r in cands if int(r["day"]) != cause_day]
        if f == "delayed_processing" and item.get("cause") == "delayed_processing":
            cands = [r for r in cands if int(r["day"]) != cause_day]
        cands = [r for r in cands if not any(p["factor"] in _MEAL_FACTORS for p in r["perturbations"])] \
            if f in _MEAL_FACTORS else cands
        if f == "postprandial_labelled_fasting":
            from haenv.p4.render import is_fasting
            cands = [r for r in cands if not is_fasting(r["time"][:5])]      # a meal sits before a non-fasting draw
        if not cands:
            detail.append({"factor": f, "where": None, "dropped": "no free draw"})
            continue
        r = cands[int(u01("p4dec-where", case_id, i) * len(cands))]
        params = {}
        if f == "postprandial_labelled_fasting":
            params = {"meal_hours_before": round(1.0 + 3.0 * u01("p4dec-meal", case_id, i), 2)}
        elif f == "meat_meal":
            params = {"meal": "dinner"}
        elif f == "delayed_processing":
            from haenv.world.labworld.perturb import stability_h
            hl, hh = (float(x) for x in F(f)["hours"])
            lim = stability_h(an)
            lo_h = DELAY_DECOY_MIN_H if lim is not None and background_on() else 2.5
            hi_h = (lim - 1.0) if lim is not None and background_on() else min(hh, 4.0)
            params = {"hours": round(lo_h + (hi_h - lo_h) * u01("p4dec-h", case_id, i), 1)}
        r["perturbations"].append(_perturbation(f, an, "decoy", params=params))
        detail.append({"factor": f, "where": int(r["day"]), "acts_on_followed": False})
    # background, never acting (R7), creatinine items: a meat dinner on the evening before one
    # of the two draws (not the draw whose cause is a meal), and the lab running Jaffe throughout
    if an == "creatinine":
        # a meat meal two days before is 48 h out on a CKD patient too: the effect of cooked meat on creatinine is gone
        # after 12 h in every CKD stage 1-4 participant (Nair et al., Diabetes Care 2014, PMID 24062331), so the rate
        # does not depend on the comorbidity (and through it on the class). Each compared draw independently,
        # whether or not its cause is a meal.
        if background_on():
            for r in (r0, r1):
                if u01("p4bg-meat", case_id, r["day"]) < BG_MEAT_DINNER_RATE:
                    r["perturbations"].append(_perturbation("meat_meal", an, "background", params={"days_before": 2}))
        else:
            if u01("p4bg-meat", case_id) < BG_MEAT_DINNER_RATE and not item.get("comorbidity"):
                free = [r for r in (r0, r1) if not any(p["factor"] in _MEAL_FACTORS for p in r["perturbations"])]
                if free:
                    r = free[int(u01("p4bg-meat-where", case_id) * len(free))]
                    r["perturbations"].append(_perturbation("meat_meal", an, "background", params={"meal": "dinner"}))
        if jaffe_throughout(case_id, item["class"], bool(item.get("comorbidity"))):
            for r in item["records"]:
                r["perturbations"].append(_perturbation("jaffe_enzymatic_switch", an, "background",
                                                        params={"method_throughout": True}))
    if (an == "fasting_glucose" and item["class"] != "method_difference"
            and item.get("cause") != "delayed_processing"         # a bedside meter has no specimen to delay
            and u01("p4bg-poc", case_id) < BG_POC_ALL_RATE):
        for r in item["records"]:
            r["perturbations"].append(_perturbation("poc_meter", an, "background",
                                                    params={"method_throughout": True}))
    # background, never acting: the lab's tube policy (serum or Li-heparin plasma, one per item) on every
    # venous draw (`tube_type`), the same on both compared draws
    if background_on():
        plasma = u01("p4tube", case_id) < float((_tables().get("render") or {}).get("tube_plasma_share", 0.5))
        for r in item["records"]:
            r["perturbations"].append(_perturbation("tube_type", an, "background",
                                                    params={"tube": "li_heparin" if plasma else "serum"}))
    # background, never acting: haemolysis index on any draw, a same-method lab switch today
    rate = float((_tables().get("render") or {}).get("hemolysis_rate", 0.15))
    for r in item["records"]:
        if u01("p4hemo", case_id, r["day"]) < rate:
            r["perturbations"].append(_perturbation("hemolysis_note", an, "background",
                                                    params={"index": "1+"}))
    rs = (r0, r1)[int(u01("p4labsw-where", case_id) * 2)] if background_on() else r1
    if u01("p4labsw", case_id) < 0.2 and not any(p.get("kind") == "method" for p in rs["perturbations"]):
        rs["perturbations"].append(_perturbation("lab_switch_zero_bias", an, "background"))
    # R12: a new reagent lot of the same manufacturer (lot check passed, zero bias) on one of the two
    # draws, so a changed method line is not by itself a method difference
    # (enzymatic draws only: Jaffe reagents differ in their compensation for non-creatinine chromogens)
    if background_on():
        free = [r for r in (r0, r1) if method_key(r) is None]
    else:
        r = (r0, r1)[int(u01("p4reagent-where", case_id) * 2)]
        free = [r] if item["class"] != "method_difference" and method_key(r) is None else []
    if free and u01("p4reagent", case_id) < BG_REAGENT_RATE:
        r = free[int(u01("p4reagent-where", case_id) * len(free))]
        r["perturbations"].append(_perturbation("lab_switch_zero_bias", an, "background",
                                                params={"reagent_change": True}))
    item["decoy_detail"] = detail
    item["acei"]["decoy_events"] = acei_dec
    item["acei"]["chronic"] = bool(chronic_acei)
