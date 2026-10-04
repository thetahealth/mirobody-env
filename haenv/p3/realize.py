"""Pack 3 item realization: a managed chronic drug line on the built world (PREREG 3).

Runs inside the build after the world's own lab measurement layer (`build.observe_labs`,
wrapped by `hooks.install`). From the plan (class, subtype, drug, dose position, decoys) it
lays out, backwards from T:

* the dose line (start rung, increases at least one reassessment interval apart, the last one
  at s), the follow-up visits (5-8, start and today included) and the refill record (six
  30-day pickups and today's pill count); the refill coverage is the adherence truth;
* the symptom diary: the drug's registered adverse effect with the planned cause, or a predating
  decoy (0-1 entries), and 0-3 unrelated self-limited complaints taken from the case's own
  evidence ledger (same day, a lay sentence of the phrase, resolved after its registered
  duration; A14), so diary and ledger have one source;
* for a drug with `renal_labs`, creatinine and potassium at every visit (normal, A12); for a
  patient with CKD (A24) at the level of the dealt eGFR, 5-15% higher once the ACEI has started,
  with the eGFR on the report and the urine ACR at the start and today;
* the control-axis truth: y(T) is placed in the subtype's zone, the per-case response r is
  drawn, the untreated level is solved from it (`labworld.meds`), and every visit gets a
  per-point record (`labworld.observe.make_record`).

An attempt is accepted only when G3 recomputed from the records (`gold.g3`) gives the planned
decision with exactly the subtype's active states; otherwise the next attempt is drawn.
`Unrealizable` when none is accepted; the emission gate then refuses the case.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from .gold import control_status, expected_states, g3
from .plan import h01

N_LAYOUT = 64
N_LEVEL = 40
SETTLED_SPAN = 150     # a follow-up comes tau to tau + 150 days after the previous visit (A5, A17)
AE_AGO = (30, 90)      # an adverse effect started 30-90 days before T, every subtype (A5)
REFILL_WINDOW = 180    # the refill printout lists every pickup of the last 180 days (A10)
OFF = ("off", "top_off", "early", "adh_low")
ROUTINE = ("on", "ae_dose")   # on target at the visits before today (A23: routine monitoring)
BENIGN_WINDOW = 365    # taken from the ledger's complaints of the last year


class Unrealizable(ValueError):
    pass


def _u(*parts) -> float:
    from ..labworld import u01
    return u01("p3", *parts)


def _clock(case_id: str, day: int) -> str:
    from ..build import _draw_clock
    return _draw_clock(str(case_id), int(day), 0)


def _layout(case_id: str, plan: dict, T: int, j: int):
    """The visit schedule (A17): every follow-up comes tau to tau + 150 days after the previous
    visit, and at each visit the clinic does what the drug's action table says -- a titrated drug
    is increased at every off-target follow-up, an `add` drug is never increased -- except for at
    most one visit (`n_extra`) that renews the prescription on the reading that establishes the
    miss. Today is the first visit at which the table calls for a change (an `early` item: an
    unscheduled visit 7 days to tau/2 after the last increase)."""
    from ..labworld.meds import drug
    from .plan import known_of
    d = drug(plan["drug"], known_of(plan))
    always = d.get("on_target") == "titrate"      # A24: raised at every visit below the top
    L = list(d["ladder"])
    n = len(L)
    tau = int(d["tau_rule_days"])
    timing = d["attribution"] == "timing"
    sub, pos = plan["subtype"], plan["position"]
    key = (case_id, j)
    xi = 0 if pos == "min" or (pos == "mid" and n == 2) else n - 1 if pos == "top" else 1 + int(_u("mid", *key) * (n - 2))
    if (sub == "early" or (sub == "ae_dose" and timing)) and xi == 0:
        xi = 1                          # the decisive change is an increase: not on the lowest rung
    starts = [k for k in d["start_rungs"] if k <= xi]
    want = xi - int(plan["n_inc"])
    si = min(starts, key=lambda k: (abs(k - want), k))
    if (sub == "early" or (sub == "ae_dose" and timing)) and si == xi:
        si = xi - 1                     # the decisive change is an increase
    steps = []                          # rungs per increase, oldest first
    if d.get("titration_history"):
        # A21: every class shows the dealt number of earlier increases, each on a reading above
        # target; A22: an overtreatment item's last increase is one step of `over_step_rungs`
        k = int(plan["n_inc"])
        steps = [1] * k
        if sub == "over":
            big = list(d["over_step_rungs"])
            steps[-1] = big[int(_u("big", *key) * len(big))]
        if sub == "top_off" or (always and xi == n - 1):   # A23: at the top, climbed from the nearest start rung (A24: every CKD item on the top)
            si = min(d["start_rungs"], key=lambda r: (abs(r - (xi - k)), -r))
            steps = [1] * (xi - si)
        else:
            si = min(max(0, xi - sum(steps)), max(d["start_rungs"]))
            xi = si + sum(steps)
    elif (d.get("flat") and sub not in OFF) or d["when_off"] == "add":
        # no single increase crosses the target: started here, never increased; the clinic never
        # increases an `add` drug (A16: past dosing follows the same guideline table)
        xi = min(xi, max(d["start_rungs"])) if pos != "min" else 0
        si = xi
        steps = []
    n_inc = len(steps) if steps else xi - si
    if not steps:
        steps = [1] * n_inc
    fu = d.get("followup_days")

    def gap(tag, after_change=False):
        if fu and after_change:          # A22: the recheck interval after a dose change
            return int(fu[0]) + int(_u("gap", tag, *key) * (int(fu[1]) - int(fu[0]) + 1))
        return tau + int(_u("gap", tag, *key) * (SETTLED_SPAN + 1))
    d_on = AE_AGO[0] + int(_u("don", *key) * (AE_AGO[1] - AE_AGO[0] + 1))     # T - onset of an adverse effect
    onset = T - d_on
    if sub == "early":
        last = 7 + int(_u("gap", "early", *key) * (int(0.5 * tau) - 7 + 1))
    elif sub == "ae_dose" and timing:
        last = d_on + 3 + int(_u("gap", "ae", *key) * 26)
    elif sub == "unattributed":
        last = d_on + 56 + int(_u("gap", "un", *key) * 121)
    else:
        last = None
    extra = int(plan.get("n_extra", 0))
    anchored = sub == "early" or (timing and sub in ("ae_dose", "unattributed"))
    if sub == "ae_dose" and timing and not n_inc:
        extra = 0                       # A20: onset 3-28 days after the start -- no follow-up fits before it
    if fu and sub == "unattributed":
        extra = 1                       # A22: the scheduled recheck after the change, before the onset
    if always:
        # A24: no visit below the top renews the dose; on the top, the 4-week recheck after the last
        # increase comes before today's question
        extra = int(xi == n - 1 and sub in ("ae_dose", "adh_low", "on"))
    seq = ["start"] + ["inc"] * n_inc                    # the visits before today, in order
    if extra:
        if anchored and n_inc and not (fu and sub == "unattributed"):
            seq.insert(len(seq) - 1, "ren")              # a renewal before the change the timing is read from
        else:
            seq.append("ren")                            # the renewal on the reading that establishes the miss
    if sub in ROUTINE and not anchored and not (always and xi < n - 1):
        # A23: an item on target before today fills the dealt visit count with routine monitoring
        # visits after the start, at the drug's usual interval (A22: not before the start)
        seq += ["ren"] * max(0, int(plan.get("n_vis", 0)) - 1 - len(seq) - int(d.get("pre_visits_max", 0)))
    if last is None:
        last = gap("last", seq[-1] != "ren")
    anchor = len(seq) - 1
    if anchored and seq[-1] == "ren":
        anchor = len(seq) - 2                            # the change the timing is read from
    days = [0] * len(seq)
    if anchor == len(seq) - 1:
        days[-1] = T - last
    else:                                                # anchor, renewal, today
        g1 = gap("x", True)
        if last - g1 < tau:
            return None
        days[anchor] = T - last
        days[-1] = days[anchor] + g1
    for k in range(anchor - 1, -1, -1):
        days[k] = days[k + 1] - gap(("v", k), seq[k] != "ren")
    t0 = days[0]
    inc = [dd for dd, kind in zip(days, seq) if kind == "inc"]
    s = inc[-1] if inc else t0
    if sub in ("ae_dose", "unattributed") and not days[-1] < onset:
        return None                      # the adverse effect is reported today, not at an earlier visit
    # A19: the visits the course does not need are monitoring visits before the start, so every
    # class spans the same range of visits (`n_vis`, dealt by the plan)
    n_pre = max(0, min(int(d.get("pre_visits_max", 0)), int(plan.get("n_vis", 0)) - len(days) - 1))
    pre = []
    for k in range(n_pre):
        pre.insert(0, (pre[0] if pre else t0) - gap(("pre", k)))
    days = pre + days + [T]
    if fu:                               # A22: no change waits longer than the recheck interval
        kinds = ["pre"] * len(pre) + seq + ["T"]
        if any(b - a > int(fu[1]) for a, b, kd in zip(days, days[1:], kinds) if kd in ("start", "inc")):
            return None
    line = [[t0, L[si]]]
    for k, dd in enumerate(inc):
        line.append([dd, L[si + sum(steps[:k + 1])]])
    return {"t0": t0, "s": s, "inc": inc, "line": line, "xi": xi, "si": si, "D": T - days[0], "tau": tau,
            "onset": onset, "visits": days, "pre": pre, "extra": extra, "always": always}


def _visits(case_id: str, lay: dict, T: int, sub: str, j: int) -> list[int]:
    return list(lay["visits"])


def _refills(case_id: str, plan: dict, lay: dict, T: int, j: int):
    """Every pickup of the last 180 days (as a pharmacy printout lists them, A10), the days
    supplied per dispensing (30 / 60 / 90, a per-patient habit dealt by the plan, A14, A15), the adherence levels around
    them, the dip decoy (one late refill outside the 90-day window of a(T))."""
    sub = plan["subtype"]
    low = sub == "adh_low"
    supply = int(plan["supply"])
    a_norm = round(0.88 + 0.10 * _u("an", case_id, j), 3)
    a_low = round(0.50 + 0.249 * _u("al", case_id, j), 3)
    drop = T - (100 + int(_u("drop", case_id, j) * 51))
    p = [T - (15 + int(_u("last", case_id, j) * 11))]
    i = 0
    lower = max(T - REFILL_WINDOW, lay["t0"])           # the printout lists the last 180 days or since the start
    while True:                                          # newest interval first
        # an adherent interval is 27-34 days (early refills happen, A12); coverage is capped at 1
        a_i = a_low if (low and p[0] > drop) else a_norm + 0.15 * _u("aj", case_id, j, i)
        nxt = p[0] - int(round(supply / a_i))
        if nxt < lower:
            break
        p.insert(0, nxt)
        i += 1
    if low and drop <= lay["t0"]:
        return None
    dip = bool(plan["decoys"]["dip"]) and len(p) >= 2 and p[1] <= T - 90
    if dip:
        # the oldest interval becomes a late refill (1.23-1.33 supplies); it must end outside the 90-day window
        delta = int(round(supply * (1.23 + 0.1 * _u("dipd", case_id, j)))) - (p[1] - p[0])
        p[0] -= delta
        if p[1] > T - 90:
            return None
    if lay.get("always") and p[0] > lay["t0"]:
        # A24: a short course lists the dispensing on the start day (a pickup within a week of the start is it)
        p = [lay["t0"]] + p[1:] if p[0] - lay["t0"] <= 7 else [lay["t0"]] + p
    a_last = a_low if low else a_norm
    taken = int(round(a_last * (T - p[-1])))
    count = max(0, supply - taken)
    return {"refills": p, "a_before": a_norm, "a_last": a_last, "pill_count": count,
            "dip": dip, "supply": supply}


def _benign(case_id: str, ledger, T: int, drug_name: str, n: int, sex: str = "") -> list[dict] | None:
    """Up to `n` unrelated complaints of the diary: the ledger's own registered benign complaints
    before T (A14); None when the ledger holds none (A20)."""
    from ..events import symptom_lay, symptom_lay_every
    from ..labworld.meds import meds
    from .plan import background_ok
    reg, every, says = meds()["benign_complaints"], symptom_lay_every(), symptom_lay(sex)
    seen, cand = set(), []
    for e in ledger or ():
        t, day = e.get("symptom"), e.get("source_timestamp")
        if (e.get("source_type") != "patient_reported_symptom" or t not in reg or t in seen
                or not isinstance(day, int) or not T - BENIGN_WINDOW <= day < T
                or not background_ok(drug_name, t, e.get("context"), *every.get(t, ()), sex=sex)):
            continue
        seen.add(t)
        cand.append((h01("p3ben", case_id, t), t, day, e.get("context")))
    if not cand:
        return None
    out = []
    for _h, t, day, ctx in sorted(cand)[:n]:
        lay = says.get(t) or [t]
        out.append({"kind": "benign", "symptom": lay[int(_u("lay", case_id, t) * len(lay))], "onset": int(day),
                    "context": ctx, "cause": "unrelated", "severity": "轻",
                    "status": "已缓解" if int(day) + int(reg[t]["days"]) < T else "持续至今"})
    return out


def _diary(case_id: str, plan: dict, lay: dict, T: int, j: int, ledger=None, sex: str = "") -> list[dict]:
    from ..labworld.meds import drug
    d = drug(plan["drug"])
    sub = plan["subtype"]
    out = []
    words = list(d["ae_symptoms"])
    word = words[int(_u("aew", case_id) * len(words))]          # the patient's own wording (A17)
    if sub == "ae_dose":
        out.append({"kind": "drug_symptom", "symptom": word, "onset": lay["onset"], "cause": "dose_related"})
    elif sub == "unattributed":
        out.append({"kind": "drug_symptom", "symptom": word, "onset": lay["onset"], "cause": "unattributed"})
    elif plan["decoys"]["predating"]:
        on = lay["t0"] - 14 - int(_u("pre", case_id, j) * 167)
        ctxs = list(d["predating_contexts"])
        out.append({"kind": "drug_symptom", "symptom": word, "onset": on, "cause": "predating",
                    "context": ctxs[int(_u("prec", case_id) * len(ctxs))]})
    for e in out:
        # an adverse-effect entry is moderate and ongoing; the predating decoy is a resolved
        # episode with a harmless cause (an irrelevant fact)
        e["status"], e["severity"] = ("已缓解" if e["cause"] == "predating" else "持续至今"), "中"
    # A20: the diary holds min(dealt count, the ledger's registered complaints) entries -- both
    # independent of the class -- and a drug symptom takes an ordinary entry's place; a ledger
    # without a registered complaint gives no diary (not emitted)
    ben = _benign(case_id, ledger, T, plan["drug"], 10 ** 6, sex)
    if ben is None:
        return None
    n = min(int(plan.get("n_diary", 1)), len(ben))
    return out[:n] + ben[:n - len(out[:n])]


def _renal(case_id: str, visits: list[int], j: int) -> list[list]:
    """[day, creatinine umol/L, potassium mmol/L] at every visit: a stable normal patient level
    with visit-to-visit variation (CV about 5% / 0.1 mmol/L; 自拟)."""
    from ..labworld import z
    cr0 = 55.0 + 40.0 * _u("cr0", case_id, j)
    k0 = 3.8 + 0.7 * _u("k0", case_id, j)
    return [[int(v), int(round(cr0 * (1.0 + 0.05 * z("p3cr", case_id, v, j)))),
             round(k0 + 0.1 * z("p3k", case_id, v, j), 1)] for v in visits]


def _age(raw, case_id: str) -> int:
    a = str((getattr(raw, "user_profile", None) or {}).get("age_range") or "50-54")
    lo, hi = (int(x) for x in a.split("-")) if a[:2].isdigit() and "-" in a else (50, 54)
    return lo + int(h01("p3age", case_id) * (hi - lo + 1))


def comorbidity_record(raw, case_id: str, code: str) -> dict:
    """The patient's comorbidity as the clinic knows it (A24): CKD stage from the dealt eGFR."""
    from ..labworld.meds import comorbidity
    from ..labworld.truth import cr_for_egfr
    c = comorbidity(code)
    sex = "M" if str((getattr(raw, "user_profile", None) or {}).get("sex")) == "M" else "F"
    age = _age(raw, case_id)
    lo, hi = (float(x) for x in c["egfr"])
    egfr = lo + (hi - lo) * _u("egfr", case_id)
    u_lo, u_hi = (float(x) for x in c["uacr_mg_g"])
    r_lo, r_hi = (float(x) for x in c["cr_rise_on_start"])
    return {"code": code, "display": c["display"], "age": age, "sex": sex, "egfr0": round(egfr, 1),
            "stage": "G3a" if egfr >= 45 else "G3b", "cr0": round(cr_for_egfr(egfr, age, sex), 2),
            "uacr": [int(round(u_lo + (u_hi - u_lo) * _u("uacr", case_id, k))) for k in (0, 1)],
            "rise": round(r_lo + (r_hi - r_lo) * _u("crrise", case_id), 4), "co_medication": c.get("co_medication")}


def _renal_ckd(case_id: str, visits: list[int], start: int, cm: dict, j: int) -> list[list]:
    """[day, creatinine, potassium, eGFR] at every visit of a CKD patient (A24): the stage's
    creatinine, `rise` higher once the ACEI has started, visit-to-visit variation as `_renal`."""
    from ..labworld import z
    from ..labworld.meds import comorbidity
    from ..labworld.truth import egfr_ckd_epi
    k_lo, k_hi = (float(x) for x in comorbidity(cm["code"])["potassium"])
    k0 = k_lo + 0.1 + (k_hi - k_lo - 0.2) * _u("k0", case_id, j)
    out = []
    for v in visits:
        cr = float(cm["cr0"]) * (1.0 + (float(cm["rise"]) if int(v) > int(start) else 0.0))
        cr = int(round(cr * (1.0 + 0.05 * z("p3cr", case_id, v, j))))
        k = round(min(k_hi, max(k_lo, k0 + 0.1 * z("p3k", case_id, v, j))), 1)
        out.append([int(v), cr, k, int(round(egfr_ckd_epi(cr, int(cm["age"]), str(cm["sex"]))))])
    return out


def realize(raw, case_id: str, plan: dict) -> dict:
    from ..labworld import z
    from ..labworld.meds import (axis, coverage_segments, drug, input_steps, rcv, response,
                                 solve_untreated, truth)
    from ..labworld.observe import make_record
    from .plan import known_of
    T = int(raw.prediction_context["prediction_time_T"])
    d = drug(plan["drug"], known_of(plan))
    cm = comorbidity_record(raw, case_id, plan["comorbidity"]) if plan.get("comorbidity") else None
    ax_name = d["axis"]
    ax = axis(ax_name)
    R = rcv(ax_name)
    nd, rng = int(ax["ndigits"]), tuple(float(x) for x in ax["range"])
    sub = plan["subtype"]
    upper = float(plan["upper"])
    zone = "off" if sub in OFF else "over" if sub == "over" else "on"
    r_lo, r_hi = (float(x) for x in d["response"])
    u_lo, u_hi = (float(x) for x in d["untreated"])
    why = {}
    for j in range(N_LAYOUT):
        lay = _layout(case_id, plan, T, j)
        if lay is None:
            why["layout"] = why.get("layout", 0) + 1
            continue
        rf = _refills(case_id, plan, lay, T, j)
        if rf is None:
            why["refills"] = why.get("refills", 0) + 1
            continue
        diary = _diary(case_id, plan, lay, T, j, getattr(raw, "evidence_ledger", None),
                       str((getattr(raw, "user_profile", None) or {}).get("sex") or ""))
        if diary is None:
            why["diary"] = why.get("diary", 0) + 1
            continue
        visits = _visits(case_id, lay, T, sub, j)
        seg = coverage_segments(rf["refills"], rf["a_before"], rf["a_last"], rf["supply"])
        steps = input_steps(plan["drug"], lay["line"], seg)
        X_T = response(steps, T, float(d["tau_truth_days"]))
        for k in range(N_LEVEL):
            rho = 1.1 + 0.9 * _u("rho", case_id, j, k)
            if zone == "over":          # A22: below the range, 1.1-2 RCV under its lower bound
                y_T = float(ax["over_below"]) * math.exp(-rho * R)
            else:
                y_T = upper * math.exp((rho if zone == "off" else -rho) * R)
            r = r_lo + (r_hi - r_lo) * _u("r", case_id, j, k)
            y_u = solve_untreated(plan["drug"], y_T, r, X_T)
            if not u_lo <= y_u <= u_hi:
                why["untreated"] = why.get("untreated", 0) + 1
                continue
            recs = []
            for v in visits:
                tr = truth(plan["drug"], v, y_u, r, steps)
                e = float(ax["cv_total"]) * z("p3e", case_id, v, j, k)
                recs.append(make_record(v, _clock(case_id, v), tr, e, [], nd, rng))
            blk = {"drug": plan["drug"], "axis": ax_name, "T": T, "dose_line": lay["line"], "visits": recs,
                   "refills": rf["refills"], "a_before": rf["a_before"], "a_last": rf["a_last"],
                   "diary": diary, "subtype": sub, "class": plan["class"], "upper": upper, "supply": rf["supply"],
                   **({"comorbidity": cm} if cm else {}),
                   "params": {"r": round(r, 6), "y_u": round(y_u, 4), "y_T": round(y_T, 4), "zone": zone}}
            if not all(u_lo <= float(r["printed"]) <= u_hi for r in recs if int(r["day"]) <= lay["t0"]):
                why["start_printed"] = why.get("start_printed", 0) + 1
                continue                # the visible readings up to the start are in the untreated range (A16: LT4 from TSH > 10, statin below 190 mg/dL; A19)
            misses = sum(1 for r in recs[:-1] if lay["t0"] < int(r["day"]) and int(r["day"]) not in lay["inc"]
                         and control_status(ax_name, float(r["printed"]), upper)[0] != "on")
            if misses > max(int(plan.get("n_extra", 0)), int(lay["extra"])):
                why["silent_renewal"] = why.get("silent_renewal", 0) + 1
                continue                # A17: the clinic follows the table at every earlier visit, but for the one renewal
            if any(float(r["printed"]) <= upper for r in recs
                   if int(r["day"]) == lay["t0"] or (int(r["day"]) in lay["inc"] and not lay["always"])):
                why["increase_at_target"] = why.get("increase_at_target", 0) + 1
                continue                # the start and every past increase were made on an off-target reading (A5, A9)
            g = g3(blk)
            if g["class"] != plan["class"] or set(g["states"]) != expected_states(sub):
                why["g3:" + str(g["reason"])[:40]] = why.get("g3:" + str(g["reason"])[:40], 0) + 1
                continue
            blk.update(pill_count=rf["pill_count"], dip=rf["dip"], position=plan["position"], rcv=round(R, 6),
                       tau_rule=lay["tau"], s=lay["s"], start=lay["t0"], g3=g, layout_attempt=j, level_attempt=k)
            if d.get("renal_labs"):
                blk["renal"] = _renal_ckd(case_id, visits, lay["t0"], cm, j) if cm else _renal(case_id, visits, j)
            return blk
    raise Unrealizable(f"no admissible realization: {why}")
