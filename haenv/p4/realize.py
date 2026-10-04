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

from .gold import NEXT_STEP, NOISE_BAND, g4, true_change_pct
from .plan import h01

N_GAPS = 64            # candidate (t0, T) intervals per case
GAP_LO, GAP_HI = 28, 182
N_ATTEMPTS = 160       # noise / bias redraws per interval
EDGE_DAYS = 7          # a cause lies at least this far after t0
#: Creatinine background (never acting on the compared change, R7): ACEI events at least this
#: many days before the previous draw (seven time constants: settled); a meat dinner on the
#: evening before a draw (>= 12 h before it); the lab running Jaffe on every draw (no switch).
BG_ACEI_MIN_BEFORE = 49
BG_MEAT_DINNER_RATE = 0.50
BG_JAFFE_ALL_RATE = 0.45
#: R8: on a glucose item outside the method class, every draw on the bedside meter (one method
#: throughout, no switch). (R8's light breakfast before a draw labelled fasting is gone: a draw is
#: labelled by its clock, and every non-fasting draw shows the meals before it.)
BG_POC_ALL_RATE = 0.40
BG_REAGENT_RATE = 0.30
TAIL_TAUS = 1          # "no net GLP-1 step" looks back this many time constants before t0


class Unrealizable(ValueError):
    pass


#: Rejection tally of the attempt loop (diagnostics only; read by tools, never by the gold).
REJECTIONS: dict[str, int] = {}


def _rej(label: str) -> None:
    REJECTIONS[label] = REJECTIONS.get(label, 0) + 1


def _tables():
    from ..labworld import tables
    return tables()


def _clock(case_id: str, day: int) -> str:
    from ..build import _draw_clock
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
    from .plan import acei_indication
    pb = (raw.latent_premise or {}).get("patient_basics") or {}
    return acei_indication(pb.get("disease"), pb.get("comorbidities"))


def calendar(case_id: str, analyte: str, ce: int, dose_pts) -> list[int]:
    """The analyte's draw days on the case's lab calendar (`build.lab_draw_days`)."""
    from ..build import _drawn, lab_draw_days
    from ..labworld import analyte as A
    days = lab_draw_days(str(case_id), int(ce), dose_pts)
    if analyte == "fasting_glucose":
        return [d for d in days if _drawn(str(case_id), "fasting_glucose", d)]
    p = float(A(analyte).get("draw_inclusion", 1.0))
    from ..labworld import u01
    return [d for d in days if d == 0 or u01("cr-cal", case_id, d) < p]


def ckd_record(raw, case_id: str) -> dict:
    """R15: the CKD patient's age, sex, eGFR, stage and creatinine set point (CKD-EPI 2021)."""
    from ..labworld import tables, u01
    from ..labworld.truth import cr_for_egfr
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
    from ..labworld import analyte as A
    from ..labworld import z
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
        from ..labworld import truth as TR
        if self.analyte == "fasting_glucose":
            return TR.fg_log(day, base=self.base, kappa=(self.kappa if self.glp1 else 0.0), beta=self.beta,
                             tau=self.tau_g, dose_pts=self.dose_pts, weight_pts=self.wpts,
                             smooth_days=self.smooth)
        return TR.cr_log(day, base=self.base, A=self.A, events=self.events, tau=self.tau_a,
                         initial_on=self.initial_on)

    def truth(self, day: int) -> float:
        return round(math.exp(self.log_truth(day)), 4)


def _steps_in(dose_pts, lo: int, hi: int) -> list[tuple[int, float]]:
    from ..labworld.truth import dose_steps
    return [(t, d) for t, d in dose_steps(dose_pts) if lo < t <= hi]


def _perturbation(factor: str, analyte: str, role: str, *, bias_log: float = 0.0, additive=None,
                  params=None) -> dict:
    from ..labworld.perturb import acts_on, kind
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
    from .plan import signs_of
    an, s = plan["analyte"], int(plan["sign"])
    out = [(an, x) for x in (s, -s) if x in signs_of(an)]
    for b in plan.get("analytes_ok") or ():
        if b != an:
            out += [(b, x) for x in (s, -s) if x in signs_of(b)]
    return out


def realize(raw, case_id: str, plan: dict, ctx: dict | None) -> dict:
    """The realized item (gold, records, context) for the first feasible (analyte, sign) of
    `options(plan)`. Raises `Unrealizable` when none is feasible."""
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
    from ..labworld import analyte as A
    from ..labworld import cv_total, rcv95, u01, z
    from ..labworld.perturb import factor as F
    tb = _tables()
    an, cls, s = plan["analyte"], plan["class"], int(plan["sign"])
    a = A(an)
    nd, rng = int(a["ndigits"]), tuple(float(x) for x in a["range"])
    sig = cv_total(an)
    R = rcv95(sig)
    band = (1.2, float(a["band_upper_rcv"]))
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
    ckd = ckd_record(raw, case_id) if plan.get("comorbidity") == "CKD" and an == "creatinine" else None
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
        steps_win = _steps_in(dose_vis, t0, T) if glp1 else []
        steps_tail = _steps_in(dose_vis, t0 - int(TAIL_TAUS * tau_g), T) if glp1 else []
        from ..labworld.truth import weight_log_term
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
        from ..labworld.perturb import decoys_for
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
                eps = math.sqrt(2.0) * sig * z("p4eps", *key)
                dtrue = D - eps
                if not (abs(dtrue) >= R / 2 * 1.04 and abs(dtrue) >= abs(D) / 2 * 1.04 and dtrue * D > 0):
                    _rej("true:eps")
                    continue
                if an == "fasting_glucose":
                    from ..labworld.truth import glp1_unit_term
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
                        if (abs(dtrue) >= R / 2 and abs(dtrue) >= abs(D) / 2 and abs(D - dtrue) <= 2.5 * math.sqrt(2) * sig
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
                    from ..labworld.truth import acei_on
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
                        from .render import meat_slot
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
            # ---- R15: a CKD patient's Jaffe/enzymatic switch, an absolute offset on the draw that
            # moves with the change (acts, but too small to be the source). A noise item has none:
            # its judged analyte is measured by the same method at both draws, so the observed change
            # is the noise pair's alone ----
            minor = None
            if ckd and cls != "analytic_biological_noise":
                f = F("jaffe_enzymatic_switch")
                lo_u, hi_u = (float(x) for x in f["additive_umol"])
                du = lo_u + (hi_u - lo_u) * u01("p4ckd-jaffe", *key)
                on_t1 = s > 0
                yb = world.truth(T if on_t1 else t0)
                bm = math.log(1.0 + du / yb)
                minor = (on_t1, du)
                B = B + (bm if on_t1 else -bm)
            # ---- the noise pair, conditioned on the difference the item needs ----
            diff = D - dtrue - B
            e0 = -diff / 2 + (sig / math.sqrt(2.0)) * z("p4e0", *key)
            e1 = e0 + diff
            if abs(e0) > 3 * sig or abs(e1) > 3 * sig:
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
            if minor is not None:
                (pert1 if minor[0] else pert0).append(
                    {"factor": "jaffe_enzymatic_switch", "kind": "method", "role": "minor", "acts": True,
                     "bias_log": 0.0, "additive": round(minor[1], 6), "params": {"offset_umol": round(minor[1], 2)}})
            from ..labworld.observe import make_record, net_bias_log
            r0 = make_record(t0, _clock(case_id, t0), y0, e0, pert0, nd, rng)
            r1 = make_record(T, _clock(case_id, T), y1, e1, pert1, nd, rng)
            g = g4(r0, r1, R, band)
            if g["class"] != cls:
                _rej(f"{cls}:{an}:g4:{g['reason'][:28]}")
                continue
            if minor is not None:
                rm = r1 if minor[0] else r0
                pm = [p for p in rm["perturbations"] if p.get("role") == "minor"]
                if abs(net_bias_log(rm["truth"], rm["e"], pm)) > float(tb["comorbidities"]["CKD"]["max_carry"]) * abs(g["dlog_obs"]):
                    _rej(f"{cls}:{an}:ckd_offset_near_half")
                    continue
            # ---- accepted: history, decoys, background, visible context ----
            hist = []
            for d in cal:
                if d >= t0 - 14 or d > T:
                    continue
                e = sig * z("p4hist", case_id, an, d)
                hist.append(make_record(d, _clock(case_id, d), world.truth(d), e, [], nd, rng))
            item = {"analyte": an, "class": cls, "sign": s, "t0": t0, "t1": T, "rcv": round(R, 6),
                    "band": list(band), "ratio_target": round(r, 6), "cv_total": sig,
                    "records": hist + [r0, r1], "decoys": dec, "decoy_substitutions": subs,
                    "cause": (cause[0] if cause else (fac if cls in ("preanalytical", "method_difference") else None)),
                    "cause_params": (cause[1] if cause else (add_on[3] if add_on else {})),
                    "acei": {"events": events, "initial_on": bool(initial_on), "indication": _indication(raw)},
                    "base": round(base, 4), "drug": drug, "glp1_class": glp1,
                    "dlog_weight": round(W, 6), "gap_index": j, "attempt": attempt, "g4": g,
                    **({"comorbidity": ckd} if ckd else {})}
            item["true_change_pct"] = true_change_pct(r0, r1)
            item["factor"] = item["cause"] if cls != "analytic_biological_noise" else None
            item["next_step"] = NEXT_STEP[cls]
            _place_context(item, case_id, steps_win, bool(item["acei"]["initial_on"]))
            return item
        reasons.append(f"gap{j}: no admissible draw")
    raise Unrealizable("; ".join(reasons[-2:]) or "no interval")


_MEAL_FACTORS = ("postprandial_labelled_fasting", "meat_meal")


def _place_context(item: dict, case_id: str, steps_win, chronic_acei: bool) -> None:
    """Decoys and never-acting background notes, recorded on the points they sit on with
    `acts: false` and zero bias (the printed values do not move), or on the window."""
    an, t0, T = item["analyte"], int(item["t0"]), int(item["t1"])
    recs = {int(r["day"]): r for r in item["records"]}
    r0, r1 = recs[t0], recs[T]
    from ..labworld import u01
    from ..labworld.perturb import factor as F
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
            from .render import is_fasting
            cands = [r for r in cands if not is_fasting(r["time"][:5])]      # a meal sits before a non-fasting draw
        if f == "meat_meal":
            from .render import meat_dinner_time
            cands = [r for r in cands if meat_dinner_time(r["time"][:5], case_id, int(r["day"])) is not None]
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
            hl, hh = (float(x) for x in F(f)["hours"])
            params = {"hours": round(2.5 + (min(hh, 4.0) - 2.5) * u01("p4dec-h", case_id, i), 1)}
        r["perturbations"].append(_perturbation(f, an, "decoy", params=params))
        detail.append({"factor": f, "where": int(r["day"]), "acts_on_followed": False})
    # background, never acting (R7), creatinine items: a meat dinner on the evening before one
    # of the two draws (not the draw whose cause is a meal), and the lab running Jaffe throughout
    if an == "creatinine":
        # R15: not on a CKD patient -- creatinine clears several times slower there, so a meat
        # dinner the evening before would still act on the morning draw
        if u01("p4bg-meat", case_id) < BG_MEAT_DINNER_RATE and not item.get("comorbidity"):
            from .render import meat_dinner_time
            free = [r for r in (r0, r1) if not any(p["factor"] in _MEAL_FACTORS for p in r["perturbations"])
                    and meat_dinner_time(r["time"][:5], case_id, int(r["day"])) is not None]
            if free:
                r = free[int(u01("p4bg-meat-where", case_id) * len(free))]
                r["perturbations"].append(_perturbation("meat_meal", an, "background", params={"meal": "dinner"}))
        if (item["class"] != "method_difference" and not item.get("comorbidity")
                and u01("p4bg-jaffe", case_id) < BG_JAFFE_ALL_RATE):
            for r in item["records"]:
                r["perturbations"].append(_perturbation("jaffe_enzymatic_switch", an, "background",
                                                        params={"method_throughout": True}))
    if (an == "fasting_glucose" and item["class"] != "method_difference"
            and item.get("cause") != "delayed_processing"         # a bedside meter has no specimen to delay
            and u01("p4bg-poc", case_id) < BG_POC_ALL_RATE):
        for r in item["records"]:
            r["perturbations"].append(_perturbation("poc_meter", an, "background",
                                                    params={"method_throughout": True}))
    # background, never acting: haemolysis index on any draw, a same-method lab switch today
    rate = float((_tables().get("render") or {}).get("hemolysis_rate", 0.15))
    for r in item["records"]:
        if u01("p4hemo", case_id, r["day"]) < rate:
            r["perturbations"].append(_perturbation("hemolysis_note", an, "background",
                                                    params={"index": "1+"}))
    if (u01("p4labsw", case_id) < 0.2
            and not any(p.get("kind") == "method" for p in r1["perturbations"])):
        r1["perturbations"].append(_perturbation("lab_switch_zero_bias", an, "background"))
    # R12: a reagent change inside one method (traceable calibration, zero bias) on one of the two
    # draws, so a changed method line is not by itself a method difference
    if (item["class"] != "method_difference" and u01("p4reagent", case_id) < BG_REAGENT_RATE):
        r = (r0, r1)[int(u01("p4reagent-where", case_id) * 2)]
        r["perturbations"].append(_perturbation("lab_switch_zero_bias", an, "background",
                                                params={"reagent_change": True}))
    item["decoy_detail"] = detail
    item["acei"]["decoy_events"] = acei_dec
    item["acei"]["chronic"] = bool(chronic_acei)
