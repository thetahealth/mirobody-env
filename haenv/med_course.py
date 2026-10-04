"""Medication course of the synthetic patient: dose records, adherence readings, concurrent
medicines and their effects.

The premise fixes the primary drug, its dose ladder and a true adherence trajectory. What the
record shows is an observation of that course, and real medication records are not a clean
titration with four evenly placed adherence readings:

* the primary dose climbs the ladder at irregular intervals; after titration a course has at
  most one dose-down episode (a step back to the previous rung, or a hold at 0), then a restart
  or a stay at the lower rung. Each episode moves in one direction: a dose does not bounce
  between two rungs (real prescribing records show dose-downs, 674 : 557 up to down in coronary
  disease outpatients, but not a back-and-forth every one to two months);
* adherence readings come on irregular days, in a random number, with measurement error; a
  long stretch without a reading is common, and some readings dip briefly and recover (a missed
  refill) without being the case's driver;
* a patient with hypertension, dyslipidemia or type 2 diabetes besides the primary condition
  takes a medicine for it (`CONCURRENT`), already at steady state on day 0 and stepped up the
  ladder now and then (titration towards target, one direction).

Nothing here reads the driver, the outcome or the diagnosis: counts, days and shapes are drawn
from `case_id` and the premise's comorbidities, and the adherence readings follow the premise's
true trajectory. Every draw is `blake2b(path)` (`rng.unit`).

Drug effects follow the dose line day by day (`glucose_term`, `bp_term`): a hold removes the
drive, a step down lowers it. Blood-pressure effects are the class averages of Law MR, Wald NJ,
Morris JK, Jordan RE. Value of low dose combination treatment with blood pressure lowering
drugs: analysis of 354 randomised trials. BMJ 2003;326:1427 (PMID 12829555): 9.1/5.5 mmHg at
standard dose, 7.1/4.4 mmHg at half standard dose, additive across classes. Concurrent
medicines act on their deviation from the day-0 steady state.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from .rng import unit

# ---------------------------------------------------------------- adherence readings
#: Readings at or before T: how many, inclusive; drawn from `case_id` only. At least 3, the
#: number GEN14 needs to read a driver's evidence, so the count never depends on the driver.
VISIBLE_READINGS = (3, 6)
#: The most recent reading at or before T sits within this many days of T, for every case.
LAST_READING_WINDOW = 3
#: The first reading falls in this range of days.
FIRST_READING_DAYS = (7, 42)
MIN_READING_SPACING = 7
#: A stream the premise moves (true relative span at or above `GEN14_FLAT_REL_SPAN`) must show
#: a clearly visible move in the readings; otherwise the error draw is repeated.
VISIBLE_SPAN_FLOOR = 0.16
GEN14_FLAT_REL_SPAN = 0.15
#: Readings after T, inclusive; the last is on the course end day.
LATER_READINGS = (1, 4)
#: Measurement error on one adherence reading (SD).
READING_SD = 0.025
#: Share of cases with one brief missed-refill dip among the readings, and its level.
DIP_SHARE = 0.30
DIP_LEVEL = (0.78, 0.90)

# ---------------------------------------------------------------- primary dose course
#: Dose points at or before T, at least (GEN14 reads a driver's dose evidence only when it has 3).
MIN_VISIBLE_DOSE_POINTS = 3
#: Dose-down episodes after titration, weights for 0..4 episodes (kept so the hash draws of a
#: course stay where they were); a course takes at most `MAX_DOSE_DOWN_EPISODES` of them.
DOSE_EPISODE_WEIGHTS = (0.05, 0.20, 0.30, 0.25, 0.20)
MAX_DOSE_DOWN_EPISODES = 1
#: Share of episodes that are a hold (dose 0) rather than a step to the previous rung.
HOLD_SHARE = 0.12
#: Share of episodes after which the dose stays at the lower rung (no restart); only the last
#: episode of a course may stay down.
STAY_DOWN_SHARE = 0.30

# ---------------------------------------------------------------- concurrent medicines
#: Indication -> candidates `(drug, ladder, weight)`; the second hypertension agent is added
#: with `SECOND_AGENT_P`. Ladders: amlodipine and atorvastatin as `registry/drug_indications.yaml`
#: (hypertension / dyslipidemia rows, plus the 2.5 mg amlodipine starting dose); losartan
#: 25/50/100 mg; metformin as the kernel's `DRUG_PKPD['metformin']` ladder.
CONCURRENT: dict[str, tuple[tuple[str, tuple[float, ...]], ...]] = {
    "hypertension": (("amlodipine", (2.5, 5.0, 10.0)), ("losartan", (25.0, 50.0, 100.0))),
    "dyslipidemia": (("atorvastatin", (10.0, 20.0, 40.0, 80.0)),),
    "T2D": (("metformin", (500.0, 1000.0, 1500.0, 2000.0)),),
    # M2 explainer base conditions (attached only by the M2 job generator). Their effects on TSH,
    # LDL, creatinine and potassium are carried by the on-treatment readings they are calibrated
    # on (`registry/base_condition_findings.yaml`, `indicators.yaml` LDL cohorts), not by a
    # dose-response term: a statin LDL term here would also move the dyslipidemia cases of the
    # v1.0.1 packs, whose worlds the bridge keeps byte-identical.
    "hypothyroidism": (("levothyroxine", (25.0, 50.0, 75.0, 100.0, 125.0)),),
    "CAD": (("atorvastatin", (10.0, 20.0, 40.0, 80.0)),),
    "CKD": (("enalapril", (5.0, 10.0, 20.0)),),
}
#: Indications of `CONCURRENT` in the order they are drawn; the M2 bases come after the v1.0.1
#: three, and each draw is keyed on its own indication, so a case without them draws as before.
CONCURRENT_ORDER = ("hypertension", "dyslipidemia", "T2D", "hypothyroidism", "CAD", "CKD")
SECOND_AGENT_P = 0.40
#: Background metformin under a GLP-1 / GIP agent in type 2 diabetes (the trial effects in
#: `drug_effects.yaml` are add-on to metformin).
T2D_BACKGROUND_METFORMIN_P = 0.80
#: Up-steps of a concurrent medicine over the course: weights for 0..3 (fewer when the top rung
#: is reached).
CONCURRENT_EPISODE_WEIGHTS = (0.30, 0.35, 0.22, 0.13)
#: Days between two changes of one concurrent medicine, at least (a clinic visit apart).
CONCURRENT_MIN_GAP = 28

#: Primary drugs whose indication already covers an indication in `CONCURRENT`.
COVERS = {"amlodipine": "hypertension", "losartan": "hypertension",
          "atorvastatin": "dyslipidemia", "metformin": "T2D",
          "semaglutide": "T2D", "tirzepatide": "T2D", "liraglutide": "T2D",
          "dulaglutide": "T2D"}
GLP1_LIKE = ("semaglutide", "tirzepatide", "liraglutide", "dulaglutide")

# ---------------------------------------------------------------- blood-pressure effects
#: Standard dose (mg/day) per antihypertensive, as in Law 2003 (amlodipine 5 mg, losartan 50 mg).
BP_STANDARD_DOSE = {"amlodipine": 5.0, "losartan": 50.0}
#: Law 2003 pooled class averages, mmHg: {fraction of standard dose: (systolic, diastolic)}.
#: Doses above standard read the standard value (no extrapolation), below half read half.
BP_EFFECT = {0.5: (7.1, 4.4), 1.0: (9.1, 5.5)}
#: First-order half-time (days) of the blood-pressure response to a dose step. The amlodipine
#: and losartan labels place the full antihypertensive effect at 2 to 4 weeks.
BP_HALF_TIME_DAYS = 7.0
BP_SIGNALS = {"systolic_bp": 0, "diastolic_bp": 1}


def _gauss(*path) -> float:
    u1 = min(max(unit("medbm0", *path), 1e-12), 1.0 - 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * unit("medbm1", *path))


def _weighted(weights, *path) -> int:
    u, acc = unit(*path), 0.0
    for k, w in enumerate(weights):
        acc += w
        if u < acc:
            return k
    return len(weights) - 1


def dose_at(pts: list[dict], day: int) -> float:
    """The recorded dose on `day` (the last change at or before it); 0 before the first."""
    cur = 0.0
    for q in sorted(pts or [], key=lambda x: int(x["ts"])):
        if int(q["ts"]) > int(day):
            break
        cur = float(q["value"])
    return cur


# ---------------------------------------------------------------- adherence readings
def _true_level(traj: list[dict], baseline: float, day: int) -> float:
    """The premise's true adherence on `day`: linear between its anchors, flat outside."""
    pts = sorted(((int(q["day"]), float(q["level"])) for q in traj), key=lambda x: x[0])
    if not pts:
        return baseline
    if day <= pts[0][0]:
        return pts[0][1]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d0 <= day <= d1:
            return v0 if d1 == d0 else v0 + (v1 - v0) * (day - d0) / (d1 - d0)
    return pts[-1][1]


def _pick_days(case_id: str, tag: str, lo: int, hi: int, n: int, spacing: int) -> list[int]:
    """n distinct days in [lo, hi], at least `spacing` apart, by ranked hash draws."""
    if n <= 0 or hi < lo:
        return []
    cand = sorted(range(lo, hi + 1), key=lambda d: unit(case_id, "med", tag, d))
    out: list[int] = []
    for d in cand:
        if all(abs(d - e) >= spacing for e in out):
            out.append(d)
            if len(out) == n:
                break
    return sorted(out)


def _rel_span(vals: list[float]) -> float:
    return (max(vals) - min(vals)) / max(1e-9, abs(sum(vals) / len(vals))) if vals else 0.0


def adherence_readings(case_id: str, T: int, course_end: int, traj: list[dict],
                       baseline: float = 0.95) -> list[dict]:
    """Observed adherence readings `[{"ts", "value"}]`, sorted by day. The number and days of
    readings are drawn from `case_id` alone; values are the true level plus error."""
    lo_n, hi_n = VISIBLE_READINGS
    n_vis = lo_n + int(unit(case_id, "med", "n_vis") * (hi_n - lo_n + 1))
    last = max(1, T - int(unit(case_id, "med", "last") * LAST_READING_WINDOW))
    f0, f1 = FIRST_READING_DAYS
    first = min(last, f0 + int(unit(case_id, "med", "first") * (f1 - f0 + 1)))
    mid = _pick_days(case_id, "vis", first + MIN_READING_SPACING, last - MIN_READING_SPACING,
                     n_vis - 2, MIN_READING_SPACING)
    days = sorted({first, last} | set(mid))
    lo_l, hi_l = LATER_READINGS
    n_later = lo_l + int(unit(case_id, "med", "n_later") * (hi_l - lo_l + 1))
    later = _pick_days(case_id, "later", T + 1, course_end - MIN_READING_SPACING, n_later - 1,
                       MIN_READING_SPACING)
    if course_end > T:
        later.append(int(course_end))           # the stream runs to the course end (GEN23)
    days = sorted(set(days) | set(later))
    vis = [d for d in days if d <= T]
    dip_at = None
    inner = [d for d in days if d not in (vis[-1] if vis else None, days[0])]
    if inner and unit(case_id, "med", "dip") < DIP_SHARE:
        dip_at = inner[int(unit(case_id, "med", "dip_pick") * len(inner))]
    true = {d: _true_level(traj, baseline, d) for d in days}
    moves = _rel_span([true[d] for d in vis]) >= GEN14_FLAT_REL_SPAN

    def draw(attempt: int) -> list[dict]:
        out = []
        for d in days:
            if d == dip_at:
                v = DIP_LEVEL[0] + unit(case_id, "med", "dip_lv") * (DIP_LEVEL[1] - DIP_LEVEL[0])
                v = min(v, true[d])
            else:
                z = _gauss(case_id, "med", "err", attempt, d) if attempt >= 0 else 0.0
                v = true[d] + READING_SD * max(-2.5, min(2.5, z))
            out.append({"ts": int(d), "value": round(min(1.0, max(0.3, v)), 2)})
        return out

    out: list[dict] = []
    for attempt in list(range(24)) + [-1]:
        out = draw(attempt)
        if not moves or _rel_span([q["value"] for q in out if q["ts"] <= T]) >= VISIBLE_SPAN_FLOOR:
            return out
    return out


# ---------------------------------------------------------------- primary dose course
def dose_course(case_id: str, steps: list[float], T: int, course_end: int,
                min_titration_days: int = 28) -> list[dict]:
    """Observed primary dose record: titration up the ladder at irregular intervals, then at most
    one dose-down episode (one rung down or a hold at 0, followed by a restart at the top rung or a
    stay at the lower one), and repeat records of the current dose. Two
    consecutive records with different values are at least `min_titration_days` apart
    (`synth._pkpd_conflicts`). At least `MIN_VISIBLE_DOSE_POINTS` records lie at or before T."""
    mt = max(1, int(min_titration_days))
    steps = [float(s) for s in steps] or [0.0]
    pts = [{"ts": 0, "value": steps[0]}]
    t = 0
    for i, v in enumerate(steps[1:], 1):
        t += mt + int(unit(case_id, "med", "titr", i) * 0.75 * mt)
        if t > course_end:
            break
        pts.append({"ts": t, "value": v})
    top = pts[-1]["value"]
    lower = steps[max(0, steps.index(top) - 1)] if top in steps else top
    start, end = t + mt, course_end - mt
    n_ep = min(MAX_DOSE_DOWN_EPISODES, _weighted(DOSE_EPISODE_WEIGHTS, case_id, "med", "n_ep"))
    n_ep = min(n_ep, max(0, (end - start + mt) // (2 * mt)))
    if n_ep:
        slack = max(0, (end - start - n_ep * 2 * mt) // n_ep)
        cur = start
        for j in range(n_ep):
            t_dn = cur + int(unit(case_id, "med", "dn", j) * slack)
            t_up = t_dn + mt + int(unit(case_id, "med", "up", j) * slack)
            if t_dn > end:
                break
            hold = unit(case_id, "med", "hold", j) < HOLD_SHARE or lower == top
            pts.append({"ts": t_dn, "value": 0.0 if hold else lower})
            stay = (j == n_ep - 1 and not hold
                    and unit(case_id, "med", "stay", j) < STAY_DOWN_SHARE)
            if stay or t_up > end:
                break
            pts.append({"ts": t_up, "value": top})
            cur = t_up + mt
    pts.sort(key=lambda q: q["ts"])

    def can_repeat(d: int) -> bool:
        nxt = min((q for q in pts if q["ts"] > d), key=lambda q: q["ts"], default=None)
        cur_v = dose_at(pts, d)
        return nxt is None or nxt["value"] == cur_v or nxt["ts"] - d >= mt

    used = {q["ts"] for q in pts}
    for j in range(MIN_VISIBLE_DOSE_POINTS - len([q for q in pts if q["ts"] <= T])):
        cand = sorted((d for d in range(1, T + 1) if d not in used
                       and all(abs(d - u) >= 7 for u in used) and can_repeat(d)),
                      key=lambda d: unit(case_id, "med", "confirm", j, d))
        if not cand:
            break
        d = cand[0]
        pts.append({"ts": d, "value": dose_at(pts, d)})
        used.add(d)
        pts.sort(key=lambda q: q["ts"])
    last_ts, last_v = pts[-1]["ts"], pts[-1]["value"]
    end_rec = end_record_day(case_id, "primary", last_ts, course_end)
    if end_rec > last_ts:
        mid = last_ts + int((end_rec - last_ts) * (0.3 + 0.4 * unit(case_id, "med", "mid_at")))
        if last_ts < mid < end_rec and unit(case_id, "med", "mid") < 0.6:
            pts.append({"ts": mid, "value": last_v})
        pts.append({"ts": int(end_rec), "value": last_v})
    return pts


#: The last record of a medicine falls on a random day of the final `END_RECORD_WINDOW` days of
#: the course (a record exactly on the course-end day would put every course end, often a
#: multiple of 28 days, into the record). Well inside GEN23's 56-day horizon tolerance.
END_RECORD_WINDOW = 28


def end_record_day(case_id: str, tag: str, last_ts: int, course_end: int) -> int:
    """Day of the closing repeat record: in `(course_end - END_RECORD_WINDOW, course_end]`,
    after `last_ts`; `last_ts` when there is no room."""
    lo = max(int(last_ts) + 1, int(course_end) - END_RECORD_WINDOW + 1)
    if lo > course_end:
        return int(last_ts)
    return lo + int(unit(case_id, "med", "end_rec", tag) * (course_end - lo + 1))


# ---------------------------------------------------------------- concurrent medicines
def _concurrent_course(case_id: str, drug: str, ladder: tuple[float, ...],
                       course_end: int) -> list[dict]:
    """A concurrent medicine already at steady state on day 0 (a rung drawn from the ladder),
    stepped up one rung 0-3 times (titration towards target), never down."""
    ladder = tuple(float(x) for x in ladder)
    k = int(unit(case_id, "cm", drug, "rung") * len(ladder))
    pts = [{"ts": 0, "value": ladder[k]}]
    n = _weighted(CONCURRENT_EPISODE_WEIGHTS, case_id, "cm", drug, "n")
    t = 0
    for j in range(n):
        t += CONCURRENT_MIN_GAP + int(unit(case_id, "cm", drug, "gap", j) * 4 * CONCURRENT_MIN_GAP)
        if t >= course_end or k + 1 >= len(ladder):
            break
        k += 1
        pts.append({"ts": t, "value": ladder[k]})
    end_rec = end_record_day(case_id, f"cm-{drug}", pts[-1]["ts"], course_end)
    if end_rec > pts[-1]["ts"]:
        pts.append({"ts": end_rec, "value": pts[-1]["value"]})
    return pts


def concurrent_medicines(case_id: str, disease: str, comorbidities, primary_drug: str,
                         course_end: int) -> list[dict]:
    """Concurrent medicines for the comorbidities the primary drug does not treat:
    `[{"drug", "indication", "dose_timeline"}]`. Reads only the premise's disease,
    comorbidities and primary drug."""
    conds = [str(disease or "")] + [str(c) for c in (comorbidities or [])]
    prim = str(primary_drug or "")
    covered = {COVERS.get(prim.split("_")[0], "")} | {str(disease or "")}
    out: list[dict] = []
    for ind in CONCURRENT_ORDER:
        opts = CONCURRENT[ind]
        if ind in covered:
            if (ind == "T2D" and str(disease) == "T2D" and prim.startswith(GLP1_LIKE)
                    and unit(case_id, "cm", "bg_metformin") < T2D_BACKGROUND_METFORMIN_P):
                drug, lad = opts[0]
                out.append({"drug": drug, "indication": ind,
                            "dose_timeline": _concurrent_course(case_id, drug, lad, course_end)})
            continue
        if ind not in conds:
            continue
        chosen = [opts[int(unit(case_id, "cm", ind, "first") * len(opts))]]
        if len(opts) > 1 and unit(case_id, "cm", ind, "second") < SECOND_AGENT_P:
            chosen.append(next(o for o in opts if o is not chosen[0]))
        for drug, lad in chosen:
            if drug == prim or any(o["drug"] == drug for o in out):
                continue
            out.append({"drug": drug, "indication": ind,
                        "dose_timeline": _concurrent_course(case_id, drug, lad, course_end)})
    return out


# ---------------------------------------------------------------- effects along the dose line
def _first_order(day: int, drive, half_time: float, start: float) -> float:
    k = 1.0 - 0.5 ** (1.0 / float(half_time))
    x = float(start)
    for d in range(1, int(day) + 1):
        x += k * (drive(d) - x)
    return x


def bp_lowering(drug: str, dose: float, sig: str) -> float:
    """mmHg lowered by `drug` at `dose` (steady state), >= 0; 0 for a non-antihypertensive."""
    std = BP_STANDARD_DOSE.get(str(drug or "").split("_")[0])
    if std is None or sig not in BP_SIGNALS or not dose or dose <= 0:
        return 0.0
    frac = float(dose) / std
    key = 1.0 if frac >= 1.0 - 1e-9 else 0.5
    return BP_EFFECT[key][BP_SIGNALS[sig]]


def bp_term(sig: str, day: int, primary_drug: str, primary_dose: list[dict], adherence,
            concurrent: list[dict]) -> float:
    """Blood-pressure increment on `day` (mmHg, <= 0 when lowered): the primary
    antihypertensive from its start (drive x adherence), plus each concurrent antihypertensive's
    deviation from its day-0 steady state."""
    if sig not in BP_SIGNALS:
        return 0.0
    tot = 0.0
    if BP_STANDARD_DOSE.get(str(primary_drug or "").split("_")[0]) is not None:
        adh = adherence if callable(adherence) else (lambda _d, _v=float(adherence or 1.0): _v)
        tot -= _first_order(day, lambda d: bp_lowering(primary_drug, dose_at(primary_dose, d), sig)
                            * max(0.0, min(1.0, float(adh(d)))), BP_HALF_TIME_DAYS, 0.0)
    for m in concurrent or ():
        if BP_STANDARD_DOSE.get(m["drug"]) is None:
            continue
        pts = m["dose_timeline"]
        e0 = bp_lowering(m["drug"], dose_at(pts, 0), sig)
        tot -= _first_order(day, lambda d, _p=pts, _m=m["drug"]: bp_lowering(_m, dose_at(_p, d), sig),
                            BP_HALF_TIME_DAYS, e0) - e0
    return tot


def concurrent_glucose_term(sig: str, day: int, concurrent: list[dict], per_kg: float,
                            atten: float, cohort) -> float:
    """HbA1c / fasting-glucose increment from concurrent glucose-lowering medicines: the
    deviation from the day-0 steady state, with the two-stage kinetics of
    `drug_effects._kinetic_fraction`."""
    from . import drug_effects as de
    if sig not in de.EFFECT_FIELDS or not de.applies_to(cohort):
        return 0.0
    tot = 0.0
    for m in concurrent or ():
        full = de.direct_effect(m["drug"], per_kg, atten, None, sig)
        if not full:
            continue
        pts = m["dose_timeline"]
        on0 = 1.0 if dose_at(pts, 0) > 0 else 0.0
        frac = de.kinetic_level(sig, day, lambda d, _p=pts: 1.0 if dose_at(_p, d) > 0 else 0.0,
                                start=on0)
        tot += full * (frac - on0)
    return tot
