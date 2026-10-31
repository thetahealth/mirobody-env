"""Rebuilt pack 2 · the visible block of one item: 10 days of wearable dailies, home blood pressure and
morning weight, CGM nightly minima for a diabetic, today's home check, the intake checklist and the
patient's message (design §3). A deterministic function of (case id, T, plan, patient facts, the
world's weight before the window); `realize` re-draws (attempt 0, 1, ...) until the rule reader
(`rules.assess`) fires the planned rule. P2_WORLD segment.

The trend templates and the message pool are this module's (provisional; review: pending). The
wearable artefact shapes are not yet the W3-calibrated ones (design §6 S-7 / L2-9: open).
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import J, L

import hashlib

from haenv.p2 import background as B
from haenv.p2 import rules as R
from haenv.p2.clinical_rules import install as _install_clinical_rules

_install_clinical_rules()

WINDOW = 10            # days T-9 .. T
BP_DAYS = 7
MAX_ATTEMPTS = 40


class Unrealizable(ValueError):
    pass


u01 = B.u01


class _Rng:
    def __init__(self, *key):
        self.key, self.k = key, 0

    def u(self) -> float:
        self.k += 1
        return u01(*self.key, self.k)

    def unif(self, lo, hi, dp=1):
        v = lo + (hi - lo) * self.u()
        return round(v, dp) if dp else int(round(v))

    def pick(self, seq):
        return seq[int(self.u() * len(seq)) % len(seq)]


# ------------------------------------------------------------------------------ today's home check

#: value ranges inside each NEWS2 band, by points (interior where the band allows)
BANDS = {
    "rr": {0: [(14, 19)], 1: [(10, 11)], 2: [(22, 23)], 3: [(26, 30)]},
    "spo2": {0: [(96, 99)], 1: [(94, 95)], 2: [(92, 93)], 3: [(88, 91)]},
    "hr": {0: [(62, 88)], 1: [(93, 108)], 2: [(113, 128)], 3: [(133, 142)]},
    "temp": {0: [(36.3, 37.6)], 1: [(38.2, 38.9)], 2: [(39.2, 39.8)], 3: [(34.5, 34.9)]},
    "sbp": {0: [(112, 150)], 1: [(102, 109)], 2: [(92, 99)], 3: [(82, 89)]},
}
#: NEWS2 aggregate targets of the infection tiers: (min, max, needs a single 3)
#: ("vital": a single 3 in respiratory rate, heart rate or systolic BP and no SpO2 3, R3a;
#: "temp": a single 3 with suspected hypothermia (<35.0, not merely NEWS2 <=35.0), R3a)
INF_TARGET = {"inf_n7": (7, 9, None), "inf_n56": (5, 6, None), "inf_v3": (3, 4, "vital"), "inf_s3": (3, 4, True),
              "inf_t35": (3, 4, "temp"), "inf_n14": (1, 4, False), "inf_n14m": (1, 4, False), "inf_n0": (0, 0, False)}


def _points_vector(tier: str, stratum: str, rng: _Rng, beta_blocker: bool = False) -> dict:
    lo, hi, need3 = INF_TARGET[tier]
    if hi == 0:
        return {"rr": 0, "spo2": 0, "hr": 0, "temp": 0, "sbp": 0}
    for _ in range(400):
        p = {"rr": rng.pick([0, 0, 2, 3] if tier in ("inf_n7", "inf_n56") else [0, 0, 1, 2, 3]),
             "spo2": rng.pick([0, 1, 2, 3]),
             "hr": 0 if stratum == "masked" else rng.pick([0, 1, 2]),
             "temp": 0 if stratum == "atypical" else rng.pick([0, 1, 1, 2]),
             "sbp": rng.pick([0, 0, 0, 1, 2])}
        if tier == "inf_n7" and stratum != "masked":
            p["hr"] = rng.pick([1, 2, 2])
        if need3 == "vital":
            # the single 3 sits in one vital; a beta-blocked heart rate does not reach 131 (rate kept down)
            p[rng.pick(["rr", "sbp"] if beta_blocker or stratum == "masked" else ["rr", "hr", "sbp"])] = 3
        if need3 == "temp":
            p["temp"] = 3
        tot, top = sum(p.values()), max(p.values())
        if not lo <= tot <= hi:
            continue
        if need3 in ("vital", "temp") and (p["spo2"] == 3 or sum(v == 3 for v in p.values()) != 1):
            continue
        if need3 is True and (p["spo2"] != 3 or p["rr"] >= 3):       # SpO2 alone (R3b)
            continue
        if need3 is False and top >= 3:
            continue
        # These interiors map directly to NG253 medium-risk bands. Atypical
        # temperature is zero here, so masked items need RR or SBP instead.
        medium = p["hr"] in (1, 2) or p["rr"] == 2 or p["sbp"] == 2
        if tier in ("inf_n14", "inf_n14m") and medium != (tier == "inf_n14m"):
            continue
        return p
    raise Unrealizable(f"no NEWS2 vector for {tier}/{stratum}")


def _value(q: str, pts: int, rng: _Rng, atypical: bool = False):
    if q == "temp" and pts == 1 and atypical:
        return rng.unif(35.3, 35.9, 1)              # an elderly patient's infection without fever
    lo, hi = rng.pick(BANDS[q][pts])
    return rng.unif(lo, hi, 1 if q == "temp" else 0)


# ------------------------------------------------------------------------------ the block

SCENARIO_CONDITION = {"heart_failure": "heart_failure", "hypoglycaemia": "T2D", "hypertension": "hypertension"}
MEDS = {"bb": L("p2.med.bb"), "bb2": L("p2.med.bb2"), "insulin": L("p2.med.insulin"),
        "su": L("p2.med.su"), "furosemide": L("p2.med.furosemide"), "amlodipine": L("p2.med.amlodipine"),
        "metformin": L("p2.med.metformin"), "sglt2i": L("p2.med.sglt2i")}
#: how each listed drug shows on the visible block (this module renders every visible stream of the item)
STREAM_EFFECT = {
    "bb": "resting heart rate 6 bpm lower; the AF rate lower",
    "bb2": "resting heart rate 6 bpm lower; the AF rate lower",
    "insulin": "the only drugs with which CGM lows (R4 / R5 / R11) are drawn",
    "su": "the only drugs with which CGM lows (R4 / R5 / R11) are drawn",
    "amlodipine": "home blood pressure on treatment (background 124-134 / 74-84)",
    "furosemide": "no modelled effect: weight flat outside the R8 gain",
    "sglt2i": "no modelled effect on the visible streams",
    "metformin": "no modelled effect; no CGM lows",
}
#: the drug each condition of the background carries (the scenario's own or a comorbidity of the plan).
#: Heart failure: a loop diuretic and an SGLT2 inhibitor (ESC 2023 focused update: SGLT2i class I in
#: HFmrEF / HFpEF, and part of HFrEF therapy); the phenotype is not stated, so no ACEI / ARNI / MRA.
CONDITION_MEDS = {"heart_failure": ("furosemide", "sglt2i"), "hypertension": ("amlodipine",), "T2D": ("metformin",)}
#: second glucose-lowering agent (plan `dm_agent`); only insulin and a sulfonylurea cause the lows of R4/R5/R11
DM_AGENT = {"metformin": (), "insulin": ("insulin",), "su": ("su",)}
#: `MEDS` key -> the world's drug and its daily dose (the listed dose; 自拟 starting doses where the line
#: prints none: glargine 10 U, glimepiride 2 mg, metformin 500 mg twice a day)
WORLD_DRUG = {"bb": ("bisoprolol", 5.0), "bb2": ("metoprolol", 47.5), "insulin": ("insulin_glargine", 10.0),
              "su": ("glimepiride", 2.0), "furosemide": ("furosemide", 20.0), "amlodipine": ("amlodipine", 5.0),
              "metformin": ("metformin", 1000.0), "sglt2i": ("dapagliflozin", 10.0)}
#: the beta-blocker of a `beta_blocker` plan (`bb_kind`), drawn with the plan
BB_KINDS = ("bb", "bb2")


def world_meds(plan: dict) -> list[dict]:
    """The medicines the item puts on the patient (`needs`, and the plan's beta-blocker) as the world's
    record holds them (`med_course.overlay_medicines`): taken from day 0 at the listed dose."""
    nd = needs(plan["tier"], plan["stratum"], bool(plan.get("beta_blocker")), plan.get("comorbid") or (),
               plan.get("dm_agent"))
    add = nd["add_conditions"]
    out = []
    for c in add:
        for k in CONDITION_MEDS.get(c, ()) + (DM_AGENT[_dm_agent(plan)] if c == "T2D" else ()):
            out.append((k, c))
    if plan.get("beta_blocker"):
        out.append((plan.get("bb_kind") or "bb", "heart_failure" if "heart_failure" in add else "hypertension"))
    seen, meds = set(), []
    for k, ind in out:
        if k in seen:
            continue
        seen.add(k)
        drug, dose = WORLD_DRUG[k]
        meds.append({"drug": drug, "indication": ind, "dose_timeline": [{"ts": 0, "value": dose}], "label": MEDS[k]})
    if R.table()["tiers"][plan["tier"]].get("context") == "volume":
        from haenv.world.history.history_dependency import uniform
        cfg = R.table()["clinical_context"]
        lo, hi = cfg["note_age_days"]
        day = int(plan["context_T"]) - lo - int(uniform("p2-context-date", plan["context_cid"]) * (hi - lo + 1))
        for med in meds:
            if med["drug"] == "furosemide":
                med["dose_timeline"] = [{"ts": 0, "value": cfg["hf_diuretic_before_mg"]},
                                        {"ts": day, "value": cfg["hf_diuretic_current_mg"]}]
    return meds


def _dm_agent(plan: dict) -> str:
    sc = R.table()["tiers"][plan["tier"]]["scenario"]
    dm = plan.get("dm_agent")
    return "insulin" if sc == "hypoglycaemia" and dm not in ("insulin", "su") else (dm or "metformin")


def visible_medications(world_medication: dict | None, T: int) -> list[str]:
    """The background's medication list: every medicine of the world's record taken on day T; the
    item's own medicines with their listed line."""
    from haenv.world.clinical.med_course import active_medicines
    from haenv.world.clinical.med_course import display
    return [display(d, dose) for d, dose, _m in active_medicines(world_medication, T)]


def needs(tier: str, stratum: str, beta_blocker: bool, comorbid=(), dm_agent: str | None = None) -> dict:
    """What the patient's background must carry for the tier: conditions added and medications. A
    beta-blocker always comes with an indication on the list (hypertension unless heart failure is there)."""
    sc = R.table()["tiers"][tier]["scenario"]
    add = [SCENARIO_CONDITION[sc]] if sc in SCENARIO_CONDITION else []
    add += R.table()["tiers"][tier].get("add_conditions", [])
    add += [c for c in comorbid if c not in add]
    if beta_blocker and not {"hypertension", "heart_failure"} & set(add):
        add.append("hypertension")
    if sc == "hypoglycaemia" and dm_agent not in ("insulin", "su"):
        dm_agent = "insulin"
    keys = [k for c in add for k in CONDITION_MEDS.get(c, ())]
    if "T2D" in add:
        keys += DM_AGENT[dm_agent or "metformin"]
    meds = [MEDS[k] for k in dict.fromkeys(keys)]
    return {"add_conditions": add, "medications": meds}


def _bp_pair(rng: _Rng, kind: str) -> tuple[int, int]:
    if kind == "single3_high_sbp":
        # Exercise the already printed NEWS2 >=220 branch; no acute symptom in this tier.
        # The upper endpoint is a synthetic sampling range, not a new clinical cutoff.
        return rng.unif(220, 230, 0), rng.unif(98, 118, 0)
    if kind == "severe":
        if rng.u() < 0.6:
            return rng.unif(184, 206, 0), rng.unif(98, 118, 0)
        return rng.unif(168, 176, 0), rng.unif(113, 120, 0)
    if kind == "raised":
        return rng.unif(146, 166, 0), rng.unif(86, 98, 0)
    if kind == "htn_bg":
        return rng.unif(124, 134, 0), rng.unif(74, 84, 0)
    return rng.unif(112, 132, 0), rng.unif(64, 82, 0)


def age_hi(patient: dict) -> int:
    """Upper end of the patient's age band (the exercise peak is capped at 220 - age, Fox 1971)."""
    try:
        return int(str(patient["age_range"]).split("-")[-1].rstrip("+"))
    except (KeyError, TypeError, ValueError) as exc:
        raise Unrealizable(f"exercise artefact needs an age band: {exc}") from exc


DIZZY_SHARE = 0.5
#: antihypertensives of the world's drug table (a treated patient's blood-pressure notes)
ANTIHTN = ("amlodipine", "losartan", "enalapril", "bisoprolol", "metoprolol")
#: drugs that can cause the CGM lows of R4 / R5 / R11
HYPO_DRUGS = ("insulin_glargine", "glimepiride", "gliclazide")


def _drug_of(line) -> str | None:
    """The world drug key of a medication-list line (a `MEDS` phrase or a `med_course.display` line)."""
    from haenv.i18n.text import Msg
    from haenv.world.clinical.med_course import line_drug
    if isinstance(line, Msg) and line.kind == "lex" and line.key.startswith("p2.med."):
        return WORLD_DRUG[line.key[len("p2.med."):]][0]
    return line_drug(line)


def realize(case_id: str, T: int, plan: dict, patient: dict) -> dict:
    """The block and its rule-reader verdict; re-drawn until the planned rule fires."""
    want = R.table()["tiers"][plan["tier"]]["rule"]
    last = None
    for attempt in range(MAX_ATTEMPTS):
        surface = {}
        blk = _draw(case_id, T, plan, patient, attempt, surface)
        a = R.assess(blk)
        # An actionable single level-2 night cannot realize a watchful-waiting
        # scenario, even when its original first matching rule is unchanged.
        needs_history = plan["stratum"] != "trend" or R.assess(R.last_day_only(blk))["set"][0] not in a["set"]
        if a["rule"] == want and R.floor_of(a["set"]) == R.table()["tiers"][plan["tier"]]["floor"] and needs_history:
            # The gold is fixed from structured patient state, before prose.
            ck, bg = surface["checklist"], surface["background"]
            blk["message"] = message(case_id, attempt, plan["tier"], plan["stratum"], plan["wording"],
                                     ck, B.says(case_id, bg), B.on(plan))
            from haenv.p2.context import render_message
            render_message(blk, case_id, plan, _Rng("p2-context-message", case_id))
            return {"block": blk, "assess": a, "attempt": attempt}
        last = a["rule"]
    raise Unrealizable(f"{case_id}: {plan['tier']} wants {want}, last draw fired {last}")


def _mild_final(patient: dict, ck: dict) -> bool:
    """The checklist's final `mild_new_symptom` (a symptom the ledger records for today or the 3 days before sets it)."""
    return bool(ck.get(R.MILD) or patient.get("new_symptom_recent"))


def _draw(cid: str, T: int, plan: dict, patient: dict, attempt: int, surface: dict | None = None) -> dict:
    tier, stratum, wording = plan["tier"], plan["stratum"], plan["wording"]
    sc = R.table()["tiers"][tier]["scenario"]
    tier = R.table()["tiers"][tier].get("render_as", tier)
    rng = _Rng("p2", cid, T, attempt)
    bb = bool(plan.get("beta_blocker"))
    days = list(range(T - WINDOW + 1, T + 1))
    known = list(patient.get("known_conditions") or ())
    nd = needs(plan["tier"], stratum, bb, plan.get("comorbid") or (), plan.get("dm_agent"))
    for c in nd["add_conditions"]:
        if c not in known:
            known.append(c)
    diabetic = "T2D" in known or "T1D" in known
    htn = "hypertension" in known
    if patient.get("medications") is not None:
        # the world's record (`visible_medications`); the draw below keeps the stream of the redraws
        meds = list(patient["medications"])
        if bb:
            rng.pick(["bb", "bb2"])
    else:
        meds = list(nd["medications"])
        if bb:
            meds.append(MEDS[rng.pick(["bb", "bb2"])])
        if diabetic and sc != "hypoglycaemia" and MEDS["metformin"] not in meds:
            meds.append(MEDS["metformin"])
    # ---- baselines
    rhr0 = rng.unif(56, 72, 0) - (6 if bb else 0)
    hrv0 = rng.unif(24, 55, 0)
    steps0 = rng.unif(3500, 9500, 0)
    sleep0 = rng.unif(6.4, 7.8, 1)
    spo2n0 = rng.unif(93, 96, 0)
    w0 = float(patient.get("weight_kg") or rng.unif(58, 92, 1))
    onset = rng.pick([2, 3, 4])
    # ---- checklist
    ck = {k: False for k in R.CHECKLIST}
    if tier == "red_normal":
        ck[rng.pick(["chest_pain_or_pressure", "one_sided_weakness_or_slurred_speech", "fainted"])] = True
    if tier == "hf_rest":
        ck["breathless_at_rest"] = True
    if tier == "af_unstable":
        ck[rng.pick(["fainted", "chest_pain_or_pressure"])] = True
    if tier == "htn_emerg":
        ck[R.HTN_SYMPTOM] = True
        ck[R.DIZZY] = u01("p2-dizzy", cid, T, attempt) < DIZZY_SHARE
    if tier == "htn_raised" and stratum == "symptomatic":
        ck[R.DIZZY] = True
    if sc == "hypoglycaemia" and rng.u() < DIZZY_SHARE:
        ck[R.DIZZY] = True                           # dizziness with the lows
    if tier == "hypo_l3":
        ck[R.HYPO_HELP] = True
    if sc == "infection":
        ck[R.MILD] = True
    if sc == "af" and tier != "af_unstable" and u01("p2-palpitation", cid) < PALPITATION_SHARE:
        ck[R.MILD] = True                            # new palpitations: a mild new symptom (the item is not infection-only)
    bg = B.active(cid, plan, set(known), diabetic, _mild_final(patient, ck), diabetic and any(_drug_of(a) in HYPO_DRUGS for a in meds))   # the background factors of this item
    # ---- today's home check
    if sc == "infection":
        pv = _points_vector(tier, stratum, rng, bool(bb))
    else:
        pv = {"rr": 0, "spo2": 0, "hr": 0, "temp": 0, "sbp": 0}
    check = {"temperature_c": _value("temp", pv["temp"], rng, stratum == "atypical"),
             "heart_rate_bpm": _value("hr", pv["hr"], rng),
             "respiratory_rate_per_min": _value("rr", pv["rr"], rng),
             "spo2_pct": _value("spo2", pv["spo2"], rng)}
    if sc == "af":
        check["heart_rate_bpm"] = rng.unif(92, 104, 0) if bb else rng.unif(112, 128, 0)
    if tier == "af_rvr":
        check["heart_rate_bpm"] = _value("hr", 3, rng)     # fast AF at rest, no beta-blocker (plan), no red flag
    if tier == "brady":
        check["heart_rate_bpm"] = rng.unif(36, 40, 0)      # slow heart rate (conduction disease or a rate-slowing drug), no red flag
    if tier == "hf_rest":
        check["respiratory_rate_per_min"] = rng.unif(22, 23, 0)
        check["spo2_pct"] = rng.unif(94, 95, 0)
    # ---- blood pressure log (two readings a day), today's last reading is the home check's
    bp = []
    bp_kind_today = {"htn_emerg": "severe", "htn_severe": "severe", "hf_htn_severe": "severe",
                     "htn_raised": "raised", "htn_sbp220": "single3_high_sbp"}.get(tier)
    for d in range(T - BP_DAYS + 1, T + 1):
        for tm in ("07:30", "20:00"):
            kind = "htn_bg" if htn else "normal"
            if bp_kind_today and d == T:
                kind = bp_kind_today
            elif bp_kind_today in ("severe", "single3_high_sbp") and d >= T - 2:
                kind = "raised"
            s, di = _bp_pair(rng, kind)
            bp.append({"day": d, "time": tm, "systolic_mmHg": s, "diastolic_mmHg": di})
    treated = any(_drug_of(m) in ANTIHTN for m in meds)
    for r in bp:
        # a note on the way the reading was taken; drawn blind to the value, the day and the tier
        if B.on(plan) and u01("p2-bp-note", cid, r["day"], r["time"]) < B.spec()["notes"]["bp_row_rate"]:
            r["note"] = B.pick(B.pool("bp_notes") + (B.pool("bp_notes_treated") if treated else []), "p2-bp-note-k", cid, r["day"], r["time"])
    today = [r for r in bp if r["day"] == T]
    if tier == "htn_raised" and stratum == "trend":
        # one high reading this morning, re-measured this evening within the raised band
        today[0]["systolic_mmHg"], today[0]["diastolic_mmHg"] = rng.unif(184, 196, 0), rng.unif(100, 108, 0)
    if sc == "infection" and pv["sbp"]:
        today[-1]["systolic_mmHg"] = _value("sbp", pv["sbp"], rng)
        today[-1]["diastolic_mmHg"] = rng.unif(52, 62, 0)
    if tier == "htn_lowbp":
        # treated hypertension, today's evening home reading low (systolic <= 90, NEWS2 3)
        today[-1]["systolic_mmHg"] = _value("sbp", 3, rng)
        today[-1]["diastolic_mmHg"] = rng.unif(48, 58, 0)
    check["systolic_bp_mmHg"], check["diastolic_bp_mmHg"] = today[-1]["systolic_mmHg"], today[-1]["diastolic_mmHg"]
    check["on_oxygen"] = False
    # ---- morning weight
    gain = 0.0
    if tier in ("hf_gain", "hf_rest"):
        gain = rng.unif(2.6, 3.6, 1)
    weights = []
    ws = {int(k): float(v) for k, v in (patient.get("weight_series") or {}).items()}   # the world's own points
    for d in days:
        k = d - (T - 3)
        g = 0.0 if k <= 0 else gain * (k / 3.0 if stratum == "trend" else min(1.0, k / 2.0))
        if d in ws:
            base = ws[d]
        else:
            # An explicitly simulated morning measurement about the pre-window baseline;
            # never interpolate a missing world observation and present it as recorded data.
            base = w0 + rng.unif(-0.3, 0.3, 1)
        weights.append({"day": d, "kg": round(base + g, 1)})
    if tier not in ("hf_gain", "hf_rest") and T not in ws:
        ref = weights[-4]["kg"]
        weights[-1]["kg"] = round(ref + rng.unif(-0.6, 0.6, 1), 1)
    # ---- CGM nightly minima (diabetic): night_of_day d = the night after day d
    cgm = []
    if diabetic:
        nights = list(range(T - 7, T))
        lows: dict[int, float] = {}
        if tier == "hypo_l2x":
            k = rng.pick([2, 3])
            pool = nights[:-1] if stratum in ("trend", "atypical") else nights
            for d in sorted(pool, key=lambda x: u01(cid, attempt, "l2", x))[:k]:
                lows[d] = rng.unif(2.3, 2.8, 1)
        elif tier == "hypo_l1":
            k = rng.pick([1, 2, 3])
            pool = nights[:-1] if stratum in ("trend", "atypical") else nights[-2:]
            for d in sorted(pool, key=lambda x: u01(cid, attempt, "l1", x))[:k]:
                lows[d] = rng.unif(3.2, 3.6, 1)
        elif tier == "hypo_l3":
            d = rng.pick(nights[-3:])
            lows[d] = rng.unif(2.2, 2.7, 1)
        for d in nights:
            cgm.append({"night_of_day": d, "min_mmol_L": lows.get(d, rng.unif(4.4, 6.2, 1)),
                        "minutes_below_3_9": (rng.unif(25, 90, 0) if d in lows else 0)})
    if "single_deep_night" in bg and cgm:
        B.edit_single_deep_night(cid, cgm)
    # ---- wearable dailies
    wd = []
    art = {"ben_exercise": "exercise", "ben_spo2": "strap"}.get(tier) or plan.get("artefact")
    odd_day = T if stratum == "trend" else T - rng.pick([1, 2, 3])     # a benign tier's single odd reading
    # an artefact beside another item's signal sits before that signal starts
    art_day = odd_day if tier in ("ben_exercise", "ben_spo2") else rng.pick(days[:WINDOW - 4])
    hr_max = 220 - age_hi(patient) if art == "exercise" else None
    for i, d in enumerate(days):
        k = d - (T - onset)                                     # days since onset (infection, HF)
        rhr, hrv, steps, sleep = rhr0 + rng.unif(-2, 2, 0), hrv0 + rng.unif(-4, 4, 0), steps0 * (0.85 + 0.3 * rng.u()), sleep0 + rng.unif(-0.5, 0.5, 1)
        spo2n, skin = spo2n0 + rng.unif(-1, 1, 0), rng.unif(-0.2, 0.2, 1)
        alerts, wmax, note = 0, None, None
        if sc == "infection" and k > 0 and tier != "inf_n0":
            sev = {"inf_n7": 1.0, "inf_n56": 0.8, "inf_v3": 0.6, "inf_s3": 0.6, "inf_t35": 0.6, "inf_n14": 0.4, "inf_n14m": 0.4}[tier] * k / onset
            rhr += (3 if bb else 14) * sev
            hrv -= 14 * sev
            steps *= max(0.15, 1 - 0.8 * sev)
            sleep -= 1.2 * sev
            spo2n -= 3 * sev
            skin += (0.2 if stratum == "atypical" else 1.1) * sev
        if sc == "heart_failure" and tier != "hf_stable" and d > T - 3:
            rhr += 4 * (d - (T - 3))
            steps *= 0.7
            spo2n -= 1
        if sc == "af" and d > T - 3:
            if not (tier == "af_stable" and d == T):
                alerts = rng.pick([1, 2]) if d >= T - 1 else rng.pick([0, 1])
            rhr += 6 if bb else 18
            hrv += 15                                           # irregular RR intervals read as high HRV
        if sc == "hypoglycaemia" and cgm:
            low = next((r for r in cgm if r["night_of_day"] == d - 1 and r["minutes_below_3_9"]), None)
            if low:
                sleep -= 1.0
        if art == "exercise" and d == art_day:
            wmax, steps = int(round(hr_max * rng.unif(0.85, 0.95, 3))), rng.unif(12000, 16500, 0)
        if art == "strap" and d == art_day:
            spo2n, note = rng.unif(83, 87, 0), rng.pick(B.strap_notes(plan))
        if tier == "ben_sleep" and d == odd_day:
            sleep, hrv = rng.unif(3.2, 4.2, 1), hrv - 10
        nulls = B.missing_fields(cid, d, T) if B.on(plan) else ()
        if d == art_day:       # the artefact's own reading (low night SpO2, the exercise steps) is recorded
            nulls = tuple(f for f in nulls if f != {"strap": "night_spo2_min_pct", "exercise": "steps"}.get(art))
        if "plain_fit_note" in bg and d == B.note_day(cid, T, WINDOW) and note is None and tier != "ben_spo2":
            note = B.pick(B.pool("neutral_notes") + B.strap_notes(plan), "p2-plain-note-k", cid, d)
        wd.append({"day": d, "resting_hr_bpm": int(round(rhr)), "hrv_ms": int(round(max(12, hrv))),
                   "night_spo2_min_pct": int(round(min(98, spo2n))), "skin_temp_delta_c": round(skin, 1) + 0.0,
                   "steps": int(round(steps)), "sleep_h": round(max(2.5, sleep), 1),
                   "irregular_rhythm_alerts": alerts, "workout_max_hr_bpm": wmax, "fit_note": note,
                   **{k: None for k in nulls}})
    if sc == "af" and sum(r["irregular_rhythm_alerts"] for r in wd[-3:]) < 2:
        wd[-2]["irregular_rhythm_alerts"] = 2
    if surface is not None:
        # A symptom in the ledger updates the screen below; it does not invent
        # another symptom in the current message (e.g. a cough for wrist pain).
        surface.update(checklist=dict(ck), background=bg)
    if patient.get("new_symptom_recent"):
        ck[R.MILD] = True                             # a symptom the ledger records for today or the 3 days before
    compact_ck = {k: ck[k] for k in (R.HTN_SYMPTOM, R.DIZZY, R.HYPO_HELP, R.MILD)}
    # Saved screens without this optional item remain readable: checklist_of
    # requires the urgent/other symptom screen, and infection readers require
    # explicit True. Absence means no evidence of suspicion, not a negative screen.
    if sc == "infection":
        compact_ck["suspected_infection_today"] = True
    compact_ck.update(urgent_screen_complete=True, urgent_symptoms=[k for k in R.R0_KEYS if ck[k]])
    device_notes = [{"day": r["day"], "text": r.pop("fit_note")} for r in wd if r.get("fit_note")]
    for r in wd:
        r.pop("fit_note", None)
    block = {"today_day": T,
            "intake_checklist": compact_ck, "background": {"known_conditions": known, "medications": meds},
            "home_check_today": check, "wearable_daily": wd, "home_bp_log": bp, "morning_weight": weights,
            "cgm_nightly": cgm, "device_record": [{"day": T, "source": L("record.source.device_care"), "entries": device_notes}],
            "data_note": "SYNTHETIC data, evaluation only"}
    from haenv.p2.context import decorate
    return decorate(block, cid, plan, rng)


# ------------------------------------------------------------------------------ the message

OPEN = {"calm": [L("p2.say.open.calm.0"), L("p2.say.open.calm.1"), L("p2.say.open.calm.2"),
                 L("p2.say.open.calm.3"), L("p2.say.open.calm.4")],
        "neutral": [L("p2.say.open.neutral.0"), L("p2.say.open.neutral.1"), L("p2.say.open.neutral.2"),
                    L("p2.say.open.neutral.3"), L("p2.say.open.neutral.4")],
        "alarming": [L("p2.say.open.alarming.0"), L("p2.say.open.alarming.1"), L("p2.say.open.alarming.2"),
                     L("p2.say.open.alarming.3"), L("p2.say.open.alarming.4")]}
CLOSE = {"calm": [L("p2.say.close.calm.0"), L("p2.say.close.calm.1"), L("p2.say.close.calm.2"), L("p2.say.close.calm.3")],
         "neutral": [L("p2.say.close.neutral.0"), L("p2.say.close.neutral.1"), L("p2.say.close.neutral.2"), L("p2.say.close.neutral.3")],
         "alarming": [L("p2.say.close.alarming.0"), L("p2.say.close.alarming.1"), L("p2.say.close.alarming.2"), L("p2.say.close.alarming.3")]}
LEAD = [L("p2.say.lead.0"), L("p2.say.lead.1"), L("p2.say.lead.2"),
        L("p2.say.lead.3"), L("p2.say.lead.4"), L("p2.say.lead.5")]
#: share of AF items (no red flag) whose patient has noticed new palpitations; AF is often silent
PALPITATION_SHARE = 0.5
#: mild new symptom sentences of a scenario other than infection (default `MILD_SAY`)
MILD_SAY_BY_SCENARIO = {"af": [L("p2.say.mild.af.0"), L("p2.say.mild.af.1"), L("p2.say.mild.af.2")]}
MILD_SAY = [L("p2.say.mild.any.0"), L("p2.say.mild.any.1"), L("p2.say.mild.any.2"), L("p2.say.mild.any.3"),
            L("p2.say.mild.any.4"), L("p2.say.mild.any.5"), L("p2.say.mild.any.6")]
#: checklist symptom -> sentences; "*" = any wording
SAY = {
    "chest_pain_or_pressure": {"calm": [L("p2.say.symptom.chest_pain_or_pressure.calm.0")], "neutral": [L("p2.say.symptom.chest_pain_or_pressure.neutral.0")],
                               "alarming": [L("p2.say.symptom.chest_pain_or_pressure.alarming.0")], "atypical": [L("p2.say.symptom.chest_pain_or_pressure.atypical.0")]},
    "fainted": {"*": [L("p2.say.symptom.fainted.k.0"), L("p2.say.symptom.fainted.k.1")]},
    "new_confusion": {"*": [L("p2.say.symptom.new_confusion.k.0"), L("p2.say.symptom.new_confusion.k.1")]},
    "breathless_at_rest": {"*": [L("p2.say.symptom.breathless_at_rest.k.0"), L("p2.say.symptom.breathless_at_rest.k.1")]},
    "one_sided_weakness_or_slurred_speech": {"*": [L("p2.say.symptom.one_sided_weakness_or_slurred_speech.k.0"), L("p2.say.symptom.one_sided_weakness_or_slurred_speech.k.1")]},
    R.HTN_SYMPTOM: {"*": [L("p2.say.symptom.r_htn_symptom.k.0"), L("p2.say.symptom.r_htn_symptom.k.1")]},
    R.DIZZY: {"*": [L("p2.say.symptom.r_dizzy.k.0"), L("p2.say.symptom.r_dizzy.k.1"), L("p2.say.symptom.r_dizzy.k.2")]},
    R.HYPO_HELP: {"*": [L("p2.say.symptom.r_hypo_help.k.0"), L("p2.say.symptom.r_hypo_help.k.1")]},
}


def r0_say_contract(say=None, table=None) -> list[str]:
    """Every chest-pain SAY variant declares a sourced emergency qualifier.

    A registry contract next to the phrase pool it checks, guarded by mutation
    controls; it lives here so the judging segment imports no new world symbol.
    """
    say = SAY if say is None else say
    table = R.table() if table is None else table
    contract = table["r0_symptom_qualifiers"]["chest_pain_or_pressure"]
    declared, allowed = contract["say"], contract["allowed"]
    out = []
    for variants in say["chest_pain_or_pressure"].values():
        for phrase in variants:
            key = getattr(phrase, "key", phrase)
            if key not in declared:
                out.append(f"{key}: no declared chest-pain qualifier")
            elif declared[key] not in allowed:
                out.append(f"{key}: unsupported chest-pain qualifier {declared[key]}")
    # Invalid declarations are also defects when their variant is retired.
    for key, qualifier in declared.items():
        if qualifier not in allowed and not any(problem.startswith(f"{key}:") for problem in out):
            out.append(f"{key}: unsupported chest-pain qualifier {qualifier}")
    return out


def say_pool(k: str) -> list[str]:
    return [x for v in SAY[k].values() for x in v]


def message(cid: str, attempt: int, tier: str, stratum: str, wording: str, ck: dict, extra=(), layer_on: bool = True) -> str:
    rng = _Rng("p2-msg", cid, attempt)
    # the draws keep the order of the pack's first message (opening, lead, symptoms, closing); the shape only chooses
    # which of them are printed, so with the layer off the text is the original one
    opening, lead = rng.pick(OPEN[wording]), rng.pick(LEAD)
    sym = []
    # Common wording across tiers: no scenario name or summary of the series. The
    # solver must read the measurements; decision-essential symptoms remain explicit.
    if ck.get(R.MILD):
        sym.append(rng.pick(MILD_SAY_BY_SCENARIO.get(R.table()["tiers"][tier]["scenario"], MILD_SAY)))
    for k in R.R0_KEYS + (R.HTN_SYMPTOM, R.DIZZY, R.HYPO_HELP):
        if ck.get(k):
            say = SAY[k]
            key = "atypical" if stratum == "atypical" and "atypical" in say else wording
            sym.append(rng.pick(say.get(key) or say["*"]))
    closing = rng.pick(CLOSE[wording])
    if not layer_on:
        return J("sentences", [opening, lead] + sym + [closing])
    sym += list(extra)
    style, cut = u01("p2-msg-style", cid), B.spec()["message_shapes"]   # blind to the tier
    if style < cut["full"]:
        parts = [opening, lead] + sym + [closing]
    elif style < cut["terse"]:
        parts = ([opening] if not sym else []) + sym + [closing]
    elif style < cut["who"]:
        parts = [B.pick(B.pool("who"), "p2-msg-who", cid), lead] + sym + [closing]
    else:
        parts = [B.pick(B.pool("log_lead"), "p2-msg-log", cid)] + sym + [closing]
    return J("sentences", parts)
