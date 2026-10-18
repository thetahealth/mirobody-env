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

from haenv.p3.gold import accepted
from haenv.p3.gold import control_status
from haenv.p3.gold import edge_margin_why
from haenv.p3.gold import expected_states
from haenv.p3.gold import g3
from haenv.p3.plan import background_on
from haenv.p3.plan import h01

N_LAYOUT = 64
N_LEVEL = 40
RHO = (1.1, 2.0)       # a zone reading lies 1.1-2 bands beyond its threshold (band = one reading's reference change)
SETTLED_SPAN = 150     # a follow-up comes tau to tau + 150 days after the previous visit (A5, A17)
AE_AGO = (30, 90)      # an adverse effect started 30-90 days before T, every subtype (A5)
REFILL_WINDOW = 180    # the refill printout lists every pickup of the last 180 days (A10)
RENEWAL_MIN = 14       # a renewal visit comes 14 days to half the reassessment interval after the change before it
K_SD = 0.075           # potassium visit-to-visit CV: real calibration split, adjacent-value abs difference median 0.31 mmol/L (SD ~0.33 at 4.2)
K_TWO_DIGITS = 0.63    # share of laboratories reporting potassium to 0.01 mmol/L (real calibration split, decimal mix)
OFF = ("off", "top_off", "early", "adh_low")
BENIGN_WINDOW = 365    # taken from the ledger's complaints of the last year
PREDATING_PERSISTS = 0.4   # share of predating decoys that never stopped (ours: a pre-existing symptom that persists)
AE_MILD = 0.3          # share of adverse-effect entries a patient calls mild (the rest moderate)


class Unrealizable(ValueError):
    pass


def _u(*parts) -> float:
    from haenv.world.labworld import u01
    return u01("p3", *parts)


def _clock(case_id: str, day: int) -> str:
    from haenv.world.generation.build import _draw_clock
    return _draw_clock(str(case_id), int(day), 0)


def _layout(case_id: str, plan: dict, T: int, j: int):
    """The visit schedule (A17): every follow-up comes tau to tau + 150 days after the previous
    visit, and at each visit the clinic does what the drug's action table says -- a titrated drug
    is increased at every off-target follow-up, an `add` drug is never increased -- except for at
    most one visit (`n_extra`) that renews the prescription on the reading that establishes the
    miss. Today is the first visit at which the table calls for a change (an `early` item: an
    unscheduled visit 7 days to tau/2 after the last increase)."""
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    from haenv.p3.plan import known_of
    d = drug(plan["drug"], known_of(plan))
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan) and plan.get("initial_low_start"):
        from haenv.resources.regpath import load_registry
        doses = load_registry("p3_clinical_sources.yaml")["levothyroxine_low_start"]["doses_micrograms"]
        d = {**d, "start_rungs": [k for k in d["start_rungs"]
                                 if float(d["ladder"][k]) in doses]}
    always = d.get("on_target") == "titrate"      # A24: raised at every visit below the top
    L = list(d["ladder"])
    n = len(L)
    tau = int(d["tau_rule_days"])
    timing = d["attribution"] == "timing"
    sub, pos = plan["subtype"], plan["position"]
    key = (case_id, j)
    xi = 0 if pos == "min" or (pos == "mid" and n == 2) else n - 1 if pos == "top" else 1 + int(_u("mid", *key) * (n - 2))
    if "tolerated_rung" in plan:
        xi = int(plan["tolerated_rung"])
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
        from haenv.world.validation.difficulty_r2 import enabled
        if sub == "over" and not (enabled(plan) and plan["drug"] == "levothyroxine"):
            big = list(d["over_step_rungs"])
            steps[-1] = big[int(_u("big", *key) * len(big))]
        if sub == "top_off" or (always and xi == n - 1):   # A23: at the top, climbed from the nearest start rung (A24: every CKD item on the top)
            si = min(d["start_rungs"], key=lambda r: (abs(r - (xi - k)), -r))
            steps = [1] * (xi - si)
        else:
            si = min(max(0, xi - sum(steps)), max(d["start_rungs"]))
            xi = si + sum(steps)
        if enabled(plan) and plan["drug"] in ("enalapril", "gliclazide"):
            # A desired history count cannot silently replace today's dealt
            # dose with a higher one. Realize only the increases reachable
            # from a permitted starting dose to the actual requested rung.
            requested = (int(plan["tolerated_rung"]) if "tolerated_rung" in plan else
                0 if pos == "min" else n - 1 if pos == "top" else
                1 + int(_u("mid", *key) * (n - 2)))
            if sub == "early" and requested == 0:
                requested = 1
            si = min([r for r in d["start_rungs"] if r <= requested],
                     key=lambda r: (abs(r - (requested - k)), r))
            xi = requested
            steps = [1] * (xi - si)
    elif (d.get("flat") and sub not in OFF) or d["when_off"] == "add":
        # no single increase crosses the target: started here, never increased; the clinic never
        # increases an `add` drug (A16: past dosing follows the same guideline table)
        xi = min(xi, max(d["start_rungs"])) if pos != "min" else 0
        si = xi
        steps = []
    if enabled(plan) and "replacement_start_rung" in plan:
        # Full replacement sets the first dose. Later TSH-guided adjustments
        # still use the registered small steps; they cannot move that start
        # backwards or be clamped at the ladder ceiling.
        si = int(plan["replacement_start_rung"])
        xi = si + int(plan["n_inc"])
        if xi >= n:
            return None
        steps = [1] * int(plan["n_inc"])
    n_inc = len(steps) if steps else xi - si
    if not steps:
        steps = [1] * n_inc
    fu = d.get("followup_days")
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config
    if enabled(plan) and plan["drug"] == "atorvastatin":
        # Rechecking a newly started or intensified statin at up to 178
        # days confused a stable maintenance interval with the first review.
        # The guideline's 4-12 weeks applies across current response classes.
        fu = config("p3")["statin_first_review"]["days"]

    def gap(tag, after_change=False):
        if fu and after_change:          # A22: the recheck interval after a dose change
            return int(fu[0]) + int(_u("gap", tag, *key) * (int(fu[1]) - int(fu[0]) + 1))
        return tau + int(_u("gap", tag, *key) * (SETTLED_SPAN + 1))

    def ren_gap(tag):
        if enabled(plan) and fu and not n_inc and plan["drug"] in ("atorvastatin", "levothyroxine"):
            # With no intervening titration this is the first lipid/TSH
            # review, not an early prescription-only renewal. The former
            # calendar printed a lipid/TSH result before its review was due.
            return gap(tag, True)
        # a renewal visit comes inside the early window after the change before it: the control axis
        # is not yet due for its recheck (a home reading is shown and does not call for a change), and
        # for CKD it is the 2-4 week creatinine and potassium check (KDIGO 2024 PP 3.6.2)
        hi = int(float(rule()["early_max_frac"]) * tau)
        lo = min(RENEWAL_MIN, hi)
        return lo + int(_u("ren", tag, *key) * (hi - lo + 1))
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
        if n_inc and (sub == "over" or (anchored and not (fu and sub == "unattributed"))):
            seq.insert(len(seq) - 1, "ren")              # a renewal before the change the timing is read from
        else:
            seq.append("ren")                            # the renewal on the reading that establishes the miss
    # Observation padding must not be conditioned on the response subtype. The old on-target
    # branch filled n_vis with extra renewals and thus lengthened only those patients' course.
    # Keep the actual prescribed course plus the independently dealt optional renewal, and
    # only the registry's permitted pre-treatment observations below. Required early/AE
    # timing and scheduled post-change reviews remain clinical evidence, not padding slots.
    # A renewal moves from the end of its interval into the early window after the change before it;
    # the interval keeps its length, so the calendar of every other visit is that of the renewal at
    # the end (the visit span carries no more than before, SA-5/SA-6)
    shift = {k: gap(("v", k - 1), True) - ren_gap(("v", k - 1)) for k, kd in enumerate(seq) if kd == "ren" and k}
    if last is None:
        last = gap("last", seq[-1] != "ren") + shift.get(len(seq) - 1, 0)
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
        days[-1] = days[anchor] + ren_gap("x")
    for k in range(anchor - 1, -1, -1):
        if seq[k + 1] == "ren":
            g = ren_gap(("v", k))
        else:
            g = gap(("v", k), seq[k] != "ren") + shift.get(k, 0)
        days[k] = days[k + 1] - g
    t0 = days[0]
    inc = [dd for dd, kind in zip(days, seq) if kind == "inc"]
    s = inc[-1] if inc else t0
    if sub in ("ae_dose", "unattributed") and not days[-1] < onset:
        return None                      # the adverse effect is reported today, not at an earlier visit
    # A19: the visits the course does not need are monitoring visits before the start, so every
    # class spans the same range of visits (`n_vis`, dealt by the plan)
    # The paired arm removes unrelated world background, not the managed
    # drug's already prescribed monitoring story. R2 retains that story in
    # both arms, within the existing per-drug observation limits.
    pmax = (d.get("pre_visits_max_long", d.get("pre_visits_max", 0))
            if background_on() or enabled(plan) else d.get("pre_visits_max", 0))
    n_pre = max(0, min(int(pmax), int(plan.get("n_vis", 0)) - len(days) - 1))
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
    """Every pickup of the last 180 days (as a pharmacy printout lists them, A10) -- with the
    start-day dispensing when the drug started inside them, and the pickup before them when the
    first one comes after the start of the adherence window, so the printed record covers every day
    the gold reads -- the days supplied per dispensing (30 / 60 / 90, a per-patient habit dealt by
    the plan, A14, A15), the adherence levels around them, the dip decoy (one late refill outside
    the 90-day window of a(T)), and today's pill count (an early refill's surplus is in the box)."""
    from haenv.world.labworld.meds import carry_in
    from haenv.world.labworld.meds import rule
    sub = plan["subtype"]
    low = sub == "adh_low"
    supply = int(plan["supply"])
    a_norm = round(0.88 + 0.10 * _u("an", case_id, j), 3)
    a_low = round(0.50 + 0.249 * _u("al", case_id, j), 3)
    drop = T - (100 + int(_u("drop", case_id, j) * 51))
    p = [T - (15 + int(_u("last", case_id, j) * 11))]
    i = 0
    t0 = int(lay["t0"])
    lower = max(T - REFILL_WINDOW, t0)                   # the printout lists the last 180 days or since the start
    w0 = T - int(rule()["adherence_window_days"]) + 1
    from haenv.world.validation.difficulty_r2 import enabled
    if low and drop <= t0 and not lay["inc"] and enabled(plan):
        # Taking too little can begin with the initial prescription. The
        # existing clinic rule reads coverage from treatment start for a
        # short course; a hundred-day antecedent dropout is not required.
        # Dispense at initiation, replenish exhausted stock, and compute
        # today's count from the same registered consumption rate.
        period = int(round(supply / a_low))
        pickups = list(range(t0, T + 1, period))
        taken = int(round(a_low * (T - pickups[-1])))
        return {"refills": pickups, "a_before": a_low, "a_last": a_low,
                "pill_count": max(0, supply - taken), "dip": False, "supply": supply}
    while True:                                          # newest interval first
        # an adherent interval is 27-34 days (early refills happen, A12)
        a_i = a_low if (low and p[0] > drop) else a_norm + 0.15 * _u("aj", case_id, j, i)
        nxt = p[0] - int(round(supply / a_i))
        if nxt < lower and not (p[0] > w0 and nxt >= t0):
            break
        p.insert(0, nxt)
        i += 1
    if low and drop <= t0:
        return None
    dip = bool(plan["decoys"]["dip"]) and len(p) >= 2 and p[1] <= T - 90
    if dip:
        # the oldest interval becomes a late refill (1.23-1.33 supplies); it must end outside the 90-day window
        delta = int(round(supply * (1.23 + 0.1 * _u("dipd", case_id, j)))) - (p[1] - p[0])
        dip = p[0] - delta >= t0                         # not before the start
        if dip:
            p[0] -= delta
    if p[0] > t0 and (t0 >= lower or p[0] > w0):
        # the dispensing on the start day (a pickup within half a supply of the start is it), when the
        # printout's 180 days hold the start or the first pickup comes after the 90-day window starts
        p = [t0] + p[1:] if p[0] - t0 < 0.5 * supply else [t0] + p
    # an increase is dispensed at its visit in the new strength (days at the new dose); the old
    # strength's stock is set aside (`carry_in` reset). As for the start, a routine pickup within half
    # a supply of the visit is that dispensing; otherwise the visit adds one
    reset = {int(x) for x in lay["inc"]}
    span = (p[0], p[1]) if dip else None
    for x in sorted(x for x in reset if (p[0] < x or x >= lower) and x < T and x not in p):
        near = [q for q in p if q != t0 and q not in reset and abs(q - x) < 0.5 * supply]
        if near:
            p.remove(min(near, key=lambda q: (abs(q - x), q)))
        p = sorted(p + [x])
    if span and not (span[0] in p and span[1] in p and p.index(span[1]) == p.index(span[0]) + 1):
        dip = False                                      # the late refill is no longer one interval
    while len(p) > 2 and p[0] < lower and p[1] <= w0:
        p = p[1:]                                        # the printout lists the last 180 days (A10)
    if any(q - a < 0.5 * supply for a, q in zip(p, p[1:]) if q not in reset):
        return None                                      # a routine refill comes after half a supply
    a_last = a_low if low else a_norm
    taken = int(round(a_last * (T - p[-1])))
    count = max(0, int(round(supply + carry_in(p, supply, reset)[-1])) - taken)
    return {"refills": p, "a_before": a_norm, "a_last": a_last, "pill_count": count,
            "dip": dip, "supply": supply}


def _benign(case_id: str, ledger, T: int, drug_name: str, n: int, sex: str = "") -> list[dict] | None:
    """Up to `n` unrelated complaints of the diary: the ledger's own registered benign complaints
    before T (A14); None when the ledger holds none (A20)."""
    from haenv.world.observations.events import symptom_lay
    from haenv.world.observations.events import symptom_lay_every
    from haenv.world.labworld.meds import meds
    from haenv.p3.plan import background_ok
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


def context_fits(d: dict, word: str, ctx: str) -> bool:
    """A predating situation is registered with the wordings it fits (`fits`; a bare situation fits
    every wording): orthostatic dizziness is told on standing up, not during a climb."""
    for c in d.get("predating_contexts") or ():
        name, fits = (c, None) if isinstance(c, str) else (c["context"], c.get("fits"))
        if name == ctx:
            return fits is None or word in fits
    return False


def predating_context(d: dict, word: str, case_id: str) -> str:
    """The situation of a predating decoy: the dealt one when it fits the wording, else one of those
    that fit."""
    names = [c if isinstance(c, str) else c["context"] for c in d["predating_contexts"]]
    ctx = names[int(_u("prec", case_id) * len(names))]
    if context_fits(d, word, ctx):
        return ctx
    ok = [c for c in names if context_fits(d, word, c)]
    if not ok:
        raise ValueError(f"no predating situation fits {word!r}")
    return ok[int(_u("prec2", case_id) * len(ok))]


def other_drugs(raw, plan: dict, T: int) -> list[tuple[str, float]]:
    """(drug, dose) of the other medicines the world's record has the patient taking on day T (the list the
    block prints as `other_medications`), the managed drug and the comorbidity's co-medication left out."""
    import haenv.world.clinical.med_course as MC
    from haenv.world.generation.build_clinical import WORLD_MEDICATION_KEY
    lp = getattr(raw, "latent_premise", None)
    wm = (lp if isinstance(lp, dict) else {}).get(WORLD_MEDICATION_KEY) or {}
    return [(d, dose) for d, dose, m in MC.active_medicines(wm, T) if d != plan["drug"] and not m.get("overlay")]


def _tier(rate, dose: float) -> float:
    """A flat rate, or the last [dose, rate] pair whose dose the patient's dose reaches."""
    if not isinstance(rate, list):
        return float(rate)
    return float([r for lo, r in rate if dose >= lo][-1] if any(dose >= lo for lo, _ in rate) else rate[0][1])


def _confounders(case_id: str, plan: dict, lay: dict, T: int, others: list[tuple[str, float]]) -> list[dict]:
    """The conspicuous background entries of the diary (registry `confounders`; sources there): another
    medicine's common adverse effect, a self-limited acute illness, and (statin) the drug's own symptom wording
    after a documented non-drug setting. Dealt by case-id hashes at their real-world frequency, never by the
    item's decision; severity, onset range and status follow the distributions of a true adverse effect (an
    adverse effect has `AE_AGO`, `AE_MILD`), and `gold.confounder_why` refuses any entry that could act."""
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import meds
    from haenv.world.labworld.meds import rule
    from haenv.p3.gold import documented_contexts
    from haenv.p3.gold import natural_status
    reg = meds()["confounders"]
    d = drug(plan["drug"])
    sev = lambda *k: "轻" if _u("cfsev", case_id, *k) < AE_MILD else "中"
    out = []
    for key, dose in others:
        spec = (reg["other_drug"] or {}).get(key)
        if spec and not out and h01("p3od", case_id, key) < _tier(spec["rate"], dose):
            sy = spec["symptoms"]
            out.append({"kind": "confounder", "cause": "other_drug", "source_drug": key,
                        "symptom": sy[int(_u("odw", case_id, key) * len(sy))],
                        "onset": T - AE_AGO[0] - int(_u("odon", case_id, key) * (AE_AGO[1] - AE_AGO[0] + 1)),
                        "context": "无明显诱因", "severity": sev(key), "status": "持续至今"})
    ill = reg["intercurrent"]
    win = int(ill["window_days"])
    if h01("p3cold", case_id) < 1.0 - math.exp(-float(ill["episodes_per_year"]) * win / 365.0):
        x = ill["illnesses"][int(_u("coldw", case_id) * len(ill["illnesses"]))]
        on = T - win + int(_u("coldon", case_id) * win)
        out.append({"kind": "confounder", "cause": "intercurrent", "symptom": x["symptom"], "onset": on,
                    "context": ill["settings"][int(_u("coldc", case_id) * len(ill["settings"]))],
                    "severity": sev("cold"), "status": natural_status(on, x["days"], T)})
    bg = (reg["background_symptom"] or {}).get(plan["drug"])
    names = sorted(documented_contexts(d))
    lo = int(lay["t0"]) + int(rule()["start_gray_days"])
    if bg and T - 1 >= lo and h01("p3bgs", case_id) < float(bg["rate"]):
        on = lo + int(_u("bgon", case_id) * (T - lo))
        out.append({"kind": "confounder", "cause": "background_symptom", "symptom": d["ae_symptoms"][int(_u("bgw", case_id) * len(d["ae_symptoms"]))],
                    "onset": on, "context": names[int(_u("bgc", case_id) * len(names))], "severity": sev("bg"),
                    "status": natural_status(on, bg["days"], T)})
    return out


def _illness_visit(case_id: str, diary: list[dict], lay: dict, T: int) -> dict | None:
    """A clinic visit for the diary's cold (`visit_share` of episodes), two days after its onset, between the
    chronic visits; the chronic prescription is not changed. None when the days do not fit."""
    from haenv.world.labworld.meds import meds
    cold = next((e for e in diary if e.get("cause") == "intercurrent"), None)
    if cold is None or h01("p3visit", case_id) >= float(meds()["confounders"]["intercurrent"]["visit_share"]):
        return None
    day = int(cold["onset"]) + 2
    taken = [int(x) for x in lay["visits"]] + [int(x) for x in lay["inc"]]
    if day <= int(lay["t0"]) or day >= T or any(abs(day - x) < 3 for x in taken):
        return None
    return {"day": day, "reason": "感冒"}


def _diary(case_id: str, plan: dict, lay: dict, T: int, j: int, ledger=None, sex: str = "", others=()) -> list[dict]:
    from haenv.world.labworld.meds import drug
    d = drug(plan["drug"])
    sub = plan["subtype"]
    out = []
    words = list(d["ae_symptoms"])
    word = words[int(_u("aew", case_id) * len(words))]          # the patient's own wording (A17)
    if sub == "ae_dose":
        out.append({"kind": "drug_symptom", "symptom": word, "onset": lay["onset"], "cause": "dose_related"})
    elif sub == "unattributed":
        out.append({"kind": "drug_symptom", "symptom": word, "onset": lay["onset"], "cause": "unattributed"})
    bg = background_on()
    if plan["decoys"]["predating"]:
        # the decoy is dealt in every decision, beside a true adverse effect too (the same wording library); an
        # episode that is over has its setting, one that never stopped says nothing about its cause and its onset
        # before the start is the visible evidence (the printed rule: not attributed to the drug)
        on = lay["t0"] - 14 - int(_u("pre", case_id, j) * 167)
        persists = bg and _u("prep", case_id) < PREDATING_PERSISTS
        pw = words[int(_u("pre2", case_id) * len(words))] if bg else word
        out.append({"kind": "drug_symptom", "symptom": pw, "onset": on, "cause": "predating",
                    "context": "无明显诱因" if persists else predating_context(d, pw, case_id), **({"persists": persists} if bg else {})})
    for e in out:
        # an adverse-effect entry is ongoing, mild (AE_MILD of them) or moderate, so the cause is not always the
        # most severe entry of the diary; the predating decoy is a resolved episode with a harmless cause
        e["status"] = "已缓解" if e["cause"] == "predating" and not e.get("persists") else "持续至今"
        from haenv.world.validation.difficulty_r2 import enabled
        if enabled(plan) and plan["drug"] == "gliclazide" and e["cause"] in ("dose_related", "unattributed"):
            e["status"] = "已缓解"
            e["context"] = "当次进食后症状缓解，处方尚未调整"
        e["severity"] = "轻" if bg and _u("aesev", case_id, e["cause"]) < AE_MILD else "中"
    # A20: the diary holds min(dealt count, the ledger's registered complaints) entries -- both
    # independent of the class -- and a drug symptom takes an ordinary entry's place; a ledger
    # without a registered complaint gives no diary (not emitted)
    ben = _benign(case_id, ledger, T, plan["drug"], 10 ** 6, sex)
    if ben is None:
        return None
    n = min(int(plan.get("n_diary", 1)), len(ben))
    if not bg:
        return out[:n] + ben[:n - len(out[:n])]
    return out + ben[:max(0, n - len(out))] + _confounders(case_id, plan, lay, T, list(others))


def k_digits(case_id: str) -> int:
    """The decimals the patient's laboratory reports potassium to (one laboratory per patient)."""
    return 2 if _u("kdig", case_id) < K_TWO_DIGITS else 1


def _fold(x: float, lo: float, hi: float) -> float:
    """x reflected into [lo, hi] (no pile-up at the bounds, as clipping gives)."""
    w = hi - lo
    y = (x - lo) % (2.0 * w)
    return lo + (y if y <= w else 2.0 * w - y)


def _k(case_id: str, k0: float, v: int, j: int, lo: float, hi: float) -> float:
    from haenv.world.labworld import z
    return round(_fold(k0 * (1.0 + K_SD * z("p3k", case_id, v, j)), lo, hi), k_digits(case_id))


def _renal(case_id: str, visits: list[int], j: int) -> list[list]:
    """[day, creatinine umol/L, potassium mmol/L] at every visit: a stable normal patient level
    with visit-to-visit variation (creatinine CV about 5%, 自拟; potassium `K_SD`), potassium in
    3.5-5.0 mmol/L at the laboratory's resolution (`k_digits`)."""
    from haenv.world.labworld import z
    cr0 = 55.0 + 40.0 * _u("cr0", case_id, j)
    k0 = 3.8 + 0.7 * _u("k0", case_id, j)
    return [[int(v), int(round(cr0 * (1.0 + 0.05 * z("p3cr", case_id, v, j)))),
             _k(case_id, k0, v, j, 3.5, 5.0)] for v in visits]


def _age(raw, case_id: str) -> int:
    a = str((getattr(raw, "user_profile", None) or {}).get("age_range") or "50-54")
    lo, hi = (int(x) for x in a.split("-")) if a[:2].isdigit() and "-" in a else (50, 54)
    return lo + int(h01("p3age", case_id) * (hi - lo + 1))


def comorbidity_record(raw, case_id: str, code: str) -> dict:
    """The patient's comorbidity as the clinic knows it (A24): CKD stage from the dealt eGFR."""
    from haenv.world.labworld.meds import comorbidity
    from haenv.world.labworld.truth import cr_for_egfr
    c = comorbidity(code)
    sex = "M" if str((getattr(raw, "user_profile", None) or {}).get("sex")) == "M" else "F"
    age = _age(raw, case_id)
    lo, hi = (float(x) for x in c["egfr"])
    egfr = lo + (hi - lo) * _u("egfr", case_id)
    u_lo, u_hi = (float(x) for x in c["uacr_mg_g"])
    r_lo, r_hi = (float(x) for x in c["cr_rise_on_start"])
    return {"code": code, "display": c["display"], "age": age, "sex": sex, "egfr0": round(egfr, 1),
            "stage": "G3a" if egfr >= 45 else "G3b", "cr0": round(cr_for_egfr(egfr, age, sex), 2),
            "uacr": [int(round(u_lo + (u_hi - u_lo) * _u("uacr", case_id, 0)))] * 2,   # today's is set by `_uacr_today` once the response is known
            "rise": round(r_lo + (r_hi - r_lo) * _u("crrise", case_id), 4), "co_medication": c.get("co_medication")}


def _uacr_today(cm: dict, d: dict, X_T: float) -> int:
    """Today's urine ACR: the first visit's value times a ratio that falls with the
    response X(T) of the RAS blocker, from `uacr_ratio[0]` (none) to `uacr_ratio[1]` (the top
    dose's full effect), so the albuminuria follows the titration instead of an independent draw."""
    from haenv.world.labworld.meds import comorbidity
    hi, lo = (float(x) for x in comorbidity(cm["code"])["uacr_ratio"])
    top = float(d["effect"][-1])
    frac = min(1.0, max(0.0, float(X_T) / top))
    return int(round(float(cm["uacr"][0]) * (hi + (lo - hi) * frac)))


def _renal_ckd(case_id: str, visits: list[int], start: int, cm: dict, j: int) -> list[list]:
    """[day, creatinine, potassium, eGFR] at every visit of a CKD patient (A24): the stage's
    creatinine, `rise` higher once the ACEI has started, visit-to-visit variation as `_renal`."""
    from haenv.world.labworld import z
    from haenv.world.labworld.meds import comorbidity
    from haenv.world.labworld.truth import egfr_ckd_epi
    k_lo, k_hi = (float(x) for x in comorbidity(cm["code"])["potassium"])
    k0 = k_lo + 0.1 + (k_hi - k_lo - 0.2) * _u("k0", case_id, j)
    out = []
    for v in visits:
        cr = float(cm["cr0"]) * (1.0 + (float(cm["rise"]) if int(v) > int(start) else 0.0))
        cr = int(round(cr * (1.0 + 0.05 * z("p3cr", case_id, v, j))))
        k = _k(case_id, k0, v, j, k_lo, k_hi)
        out.append([int(v), cr, k, int(round(egfr_ckd_epi(cr, int(cm["age"]), str(cm["sex"]))))])
    return out


def _renewal_acted(ax_name: str, rec: dict, upper: float, lay: dict, early_days: float) -> bool:
    """Whether the printed rules would change the dose on a renewal visit's reading: under the range
    anywhere; off target once the early window after the last change has passed (a recheck renewal)."""
    st = control_status(ax_name, float(rec["printed"]), upper)[0]
    day = int(rec["day"])
    last = max(x for x in [lay["t0"], *lay["inc"]] if x < day)
    return st == "over" or (st != "on" and day - last > early_days)


def realize(raw, case_id: str, plan: dict) -> dict:
    from haenv.world.validation.difficulty_r2 import enabled
    if not enabled(plan) or plan["drug"] != "levothyroxine":
        return _realize_course(raw, case_id, plan)
    from haenv.resources.regpath import load_registry
    counts = load_registry("p3_clinical_sources.yaml")["levothyroxine_followup"]["history_counts"]
    preferred = int(plan.get("n_inc", 0))
    # A proposed count is not a patient's fixed prescription history. Try
    # the same sourced clinical courses in a patient-keyed order, preserving
    # the preferred course when it is valid. Every candidate still checks
    # initial dosing, prior TSH, physical response and the unmodified G3.
    choices = [preferred, *sorted((int(c) for c in counts if int(c) != preferred),
                                  key=lambda c: _u("thyroid-course-count", case_id, c))]
    errors = []
    for count in choices:
        try:
            item = _realize_course(raw, case_id, {**plan, "n_inc": count})
        except Unrealizable as exc:
            errors.append(str(exc))
            continue
        item["course_history_count"] = count
        return item
    raise Unrealizable("no admissible thyroid course: " + "; ".join(errors))


def _realize_course(raw, case_id: str, plan: dict) -> dict:
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        from haenv.world.validation.shared_audit import side_effect_complaint
        if side_effect_complaint({"sp": {"managed_drug": plan["drug"]}, "case": raw}):
            raise Unrealizable("benign background complaint overlaps the managed drug's registered side-effect topic")
        if plan["drug"] == "levothyroxine":
            from haenv.world.validation.difficulty_r2 import config
            from haenv.world.labworld.meds import drug
            cfg = config("p3")["replacement_start"]
            up = raw.user_profile or {}
            age = int(str(up.get("age_range") or "65-69").split("-")[0])
            known = set(up.get("known_conditions") or ())
            pb = (raw.latent_premise or {}).get("patient_basics") or {}
            known.update(pb.get("comorbidities") or ())
            known.add(pb.get("disease"))
            appropriate = age < cfg["age_below"] and not known.intersection(cfg["exclude"])
            if appropriate:
                weights = (raw.longitudinal_data or {}).get("weight") or []
                if not weights:
                    raise Unrealizable("replacement start requires a recorded body weight")
                dose = float(weights[0]["value"]) * float(cfg["dose_per_kg"])
                d0 = drug("levothyroxine")
                step = float(cfg["microgram_step_max"])
                dose = step * math.floor(dose / step + 0.5)
                rungs = [k for k, value in enumerate(d0["ladder"]) if float(value) == dose]
                if not rungs:
                    raise Unrealizable("weight-based initial replacement lies outside the registered dose ladder")
                rung = rungs[0]
                # The initial prescription follows patient eligibility, not
                # today's response class. Subsequent adjustment follows TSH.
                plan = {**plan, "replacement_start_rung": rung}
            else:
                plan = {**plan, "initial_low_start": True}
    from haenv.world.labworld import z
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import band
    from haenv.world.labworld.meds import coverage_segments
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import input_steps
    from haenv.world.labworld.meds import response
    from haenv.world.labworld.meds import rule
    from haenv.world.labworld.meds import solve_untreated
    from haenv.world.labworld.meds import truth
    from haenv.world.labworld.observe import make_record
    from haenv.p3.plan import known_of
    T = int(raw.prediction_context["prediction_time_T"])
    d = drug(plan["drug"], known_of(plan))
    cm = comorbidity_record(raw, case_id, plan["comorbidity"]) if plan.get("comorbidity") else None
    ax_name = d["axis"]
    ax = axis(ax_name)
    R = band(ax_name)       # one reading's reference change, 1.96 x CV
    nd, rng = int(ax["ndigits"]), tuple(float(x) for x in ax["range"])
    sub = plan["subtype"]
    upper = float(plan["upper"])
    zone = "off" if sub in OFF else "over" if sub == "over" else "on"
    floor = float(d.get("titrate_on_target_min") or 0)
    # p3-fixes c: an on-target item below the top of a drug raised on target (CKD) is a maintain item
    # under the clinic's floor; its reading is placed 1.1-2 bands under the floor, as others under the target
    place = min(upper, floor) if (floor and zone == "on" and plan["position"] != "top") else upper
    r_lo, r_hi = (float(x) for x in d["response"])
    u_lo, u_hi = (float(x) for x in d["untreated"])
    early_days = float(rule()["early_max_frac"]) * float(d["tau_rule_days"])
    bg = background_on()
    contract16 = bg or int(plan.get("version") or 0) >= 16
    others = other_drugs(raw, plan, T) if bg else []
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
                       str((getattr(raw, "user_profile", None) or {}).get("sex") or ""), others)
        if diary is None:
            why["diary"] = why.get("diary", 0) + 1
            continue
        visits = _visits(case_id, lay, T, sub, j)
        seg = coverage_segments(rf["refills"], rf["a_before"], rf["a_last"], rf["supply"], lay["inc"])
        steps = input_steps(plan["drug"], lay["line"], seg)
        X_T = response(steps, T, float(d["tau_truth_days"]))
        cm_j = cm and {**cm, "uacr": [cm["uacr"][0], _uacr_today(cm, d, X_T)]}
        for k in range(N_LEVEL):
            rho = RHO[0] + (RHO[1] - RHO[0]) * _u("rho", case_id, j, k)
            if zone == "over":          # A22: below the range, 1.1-2 RCV under its lower bound
                y_T = float(ax["over_below"]) * math.exp(-rho * R)
            else:
                y_T = (upper if zone == "off" else place) * math.exp((rho if zone == "off" else -rho) * R)
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
                   "pill_count": rf["pill_count"], "diary": diary,
                   **({"other_drugs": [k for k, _ in others], "illness_visit": _illness_visit(case_id, diary, lay, T),
                       "background": True} if bg else {}), "subtype": sub, "class": plan["class"], "upper": upper, "supply": rf["supply"],
                   **({"comorbidity": cm_j} if cm else {}),
                   "params": {"r": round(r, 6), "y_u": round(y_u, 4), "y_T": round(y_T, 4), "zone": zone}}
            if enabled(plan):
                blk["documented_course_only"] = True
                blk["course_documentation_key"] = str(case_id)
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
            if floor and any(float(r["printed"]) < floor for r in recs
                             if int(r["day"]) == lay["t0"] or int(r["day"]) in lay["inc"]):
                why["increase_below_floor"] = why.get("increase_below_floor", 0) + 1
                continue                # the clinic raised the dose only on readings at or above its floor
            if any(_renewal_acted(ax_name, r, upper, lay, early_days) for r in recs
                   if lay["t0"] < int(r["day"]) < T and int(r["day"]) not in lay["inc"]):
                why["renewal_reading"] = why.get("renewal_reading", 0) + 1
                continue                # a renewal is never made on a reading the printed rules act on
            if plan.get("history_record"):
                blk["adverse_event_record"] = [dict(plan["history_record"])]
            g = g3(blk)
            if g["class"] != plan["class"] or set(g["states"]) != expected_states(sub):
                why["g3:" + str(g["reason"])[:40]] = why.get("g3:" + str(g["reason"])[:40], 0) + 1
                continue
            edge = contract16 and edge_margin_why(blk)
            if edge:
                why["edge"] = why.get("edge", 0) + 1
                continue                # no item sits on a rule line the clinic itself chose (w14-bg-p3)
            blk.update(**({"accept": accepted(blk, g)} if contract16 else {}), dip=rf["dip"], position=plan["position"], band=round(R, 6),
                       tau_rule=lay["tau"], s=lay["s"], start=lay["t0"], g3=g, layout_attempt=j, level_attempt=k)
            if d.get("renal_labs"):
                blk["renal"] = _renal_ckd(case_id, visits, lay["t0"], cm, j) if cm else _renal(case_id, visits, j)
                blk["k_digits"] = k_digits(case_id)
            return blk
    raise Unrealizable(f"no admissible realization: {why}")
