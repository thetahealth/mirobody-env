"""Pack 3 allocation: which case carries which medication-adjustment item (PREREG section 4).

Every decision is a function of `sha256(tag, case_id)` and of the case's eligibility, never of
a feature the solver sees:

* class: inside each stratum of the hidden-disease flag H, cases in `h01("pack3", id)` order are
  dealt the five decisions round-robin from a stratum offset;
* drug: inside each class, in `h01("p3drug", id)` order, the eligible drug dealt least so far
  (a drug with `switch: false` never takes the switch class; only a drug with `uptitrate: true`
  takes the uptitrate class, A12);
* subtype, dose position and the decoys: per-case hashes with the PREREG shares;
* comorbidity (A24): inside each (H, drug) whose next step a registered comorbidity changes, every
  other case in `h01("p3comorb", id)` order carries it, and each half is dealt the classes it can
  take, so the comorbidity sits in every class the drug takes and its presence does not tell one.

Eligibility (PREREG 3, amendments A3/A5/A12): no world stream that bears on the drug's decision, in
the world as drawn or in the world of the patient the record describes (`record_case`: the drug's
condition among the comorbidities, the age band drawn for it),
no glycaemic axis when the world's primary drug acts on glucose, the drug's condition not already
treated in the world (its disease or a comorbidity: no ACEI on top of the world's ARB, no second
statin; L3-9), and a hidden line whose diagnosis names none of the diseases that change this
drug's decision (L3-7).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import hashlib

CLASSES = ("uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect")
DRUGS = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
GLUCOSE_PRIMARY = ("metformin", "semaglutide", "liraglutide", "dulaglutide", "tirzepatide")
#: subtypes and their share inside each class (PREREG 4, A5)
SUBTYPES = {
    "uptitrate": (("off", 1.0),),
    "downtitrate": (("ae_dose", 1.0),),                   # levothyroxine: over only
    "maintain": (("early", 2 / 3), ("on", 1 / 3)),                # A23: more of the hard type (recheck before steady state)
    "switch": (("top_off", 0.5), ("off", 0.5)),            # off below the top: the next rung does not reach the target (A12)
    "check_adherence_or_adverse_effect": (("adh_low", 2 / 3), ("unattributed", 1 / 3)),   # A23: more adherence items  # exposure: adh_low only
}
#: dose position shares (min / mid / top) per subtype
#: A18: an `early` item sits mid-ladder (off target right after a step to the top rung needs an untreated
#: level above the registered range)
POSITION = {"off": (0.5, 0.5, 0.0), "ae_dose": (0.0, 0.6, 0.4), "over": (0.0, 1.0, 0.0),
            "early": (0.0, 1.0, 0.0), "on": (0.45, 0.25, 0.3), "top_off": (0.0, 0.0, 1.0),
            "adh_low": (0.3, 0.3, 0.4), "unattributed": (0.3, 0.3, 0.4)}
AE_SUBTYPES = ("ae_dose", "unattributed")
DIP_Q, PREDATING_Q = 0.45, 0.45   # A17: decoys at a natural frequency (自拟), the adherence decoy withdrawn
#: Pack share of each decision: the five decisions alike (balanced accuracy weighs them alike), 20 per
#: 100 items; a remainder goes to the earlier decision here.
#: Inside a decision the drugs that can take it share alike (`planned_cells`).
CLASS_SHARE = {"check_adherence_or_adverse_effect": 1, "maintain": 1, "downtitrate": 1, "uptitrate": 1, "switch": 1}
WEIGHT = 3
#: Pool dealing only (`make_plans`; the published pack follows `planned_scenarios`).
#: p3-fixes e: (drug, class) cells whose items every model answered in every cell of the 1.3.0 board
#: (37 such items of 100 in the item-level response audit): levothyroxine downtitrate (TSH
#: suppressed, 5/5), gliclazide downtitrate (hypoglycaemia, 4/5), on-target maintain of the drugs
#: without an `early` item (10/11). They weigh `LOW_SIGNAL_WEIGHT` instead of `WEIGHT` inside their
#: class, so more of the class's items go to the cells that separated models (statin / ACEI adverse
#: effects, enalapril's just-raised and CKD items). 2, not 1: at 1 the drug alone tells the decision
#: (majority class of the drug, cc 0.225 on the planned N = 100 cells, over the 0.20 shortcut line);
#: at 2 it reads 0.150 (0.125 before). Every cell keeps at least one item at N = 100.
LOW_SIGNAL_WEIGHT = 2
LOW_SIGNAL_CELLS = {("levothyroxine", "downtitrate"), ("gliclazide", "downtitrate"),
                    ("gliclazide", "maintain"), ("atorvastatin", "maintain"), ("levothyroxine", "maintain")}
WEIGHT_COMORB = (2, 1)         # per class, with / without the comorbidity
CAP_PER_DRUG = 40        # cases per drug per H stratum at N = 50 (the generator scales it with N)
SUPPLY = (30, 60, 90)       # days per dispensing, a per-patient habit (A14, A15)
N_VIS = (3, 4, 5)           # visits on record including today, one range for every class (A19)
N_VIS_BACKGROUND = (3, 4, 5, 6, 7)   # with the background layer: longer histories (real EMR median 21 lab dates per patient)
N_DIARY = (1, 2, 3)         # diary entries, one range for every class; a drug symptom takes an ordinary entry's place (A20)
PLAN_VERSION = 15
PLAN_VERSION_BACKGROUND = 16


@functools.lru_cache(maxsize=1)
def background_on() -> bool:
    """`config.synth.background`: the pack-3 background layer (`gold.confounder_why`, `realize._confounders`); off builds
    the pack as before it, byte for byte."""
    from haenv.world.generation.world_knobs import background_enabled
    return background_enabled()


def h01(*parts) -> float:
    s = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(s[:12], 16) / float(16 ** 12)


def hidden(latent: dict) -> int:
    return int(str((latent or {}).get("ddx_join_gold")) in ("unified", "comorbidity"))


def dx_allowed(drug_name: str, case: dict) -> bool:
    """A16 whitelist: every component of the case's condition line is on the drug's `allowed_dx`."""
    from haenv.world.labworld.meds import drug
    dx = str((case.get("latent") or {}).get("ddx_diagnosis") or "")
    ok = drug(drug_name).get("allowed_dx") or ()
    return not dx or all(any(k in part for k in ok) for part in dx.split(" + "))


def _age(case: dict) -> int:
    a = str((case.get("raw") or {}).get("age_range") or "")
    return int(a.split("-")[0]) if a[:2].isdigit() else 0


def background_ok(drug_name: str, *texts, sex: str | None = None) -> bool:
    """No word of the drug's `background_deny` list, and no word of the other sex (A17,
    `events.sex_words`)."""
    from haenv.world.observations.events import sex_words
    from haenv.world.labworld.meds import drug
    words = list(drug(drug_name).get("background_deny") or ()) + list(sex_words(sex) if sex else ())
    return not any(w in str(t or "") for t in texts for w in words)


def record_conditions(drug_name: str, comorbidity: str | None = None) -> list[str]:
    """The conditions a pack-3 item puts on the patient's record (`hooks.observe_meds`): the drug's
    condition and the planned comorbidity."""
    from haenv.world.labworld.meds import drug
    return [drug(drug_name)["known_condition"]] + ([comorbidity] if comorbidity else [])


def record_case(case: dict, drug_name: str, comorbidity: str | None = None, *, condition_specs: dict | None = None) -> dict:
    """The case whose world is the patient the record describes: the item's conditions among the world's
    comorbidities, and the age band the build draws for them (`build.pack_age_of`)."""
    from haenv.world.generation.build import pack_age_of
    raw = dict(case.get("raw") or {})
    world = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    raw["comorbidities"] = list(raw.get("comorbidities") or ()) + [
        k for k in record_conditions(drug_name, comorbidity) if k not in world]
    raw["age_range"] = pack_age_of(str(case.get("case_id")), raw, case.get("latent") or {},
                                  **({"condition_specs": condition_specs} if condition_specs is not None else {}))
    return {**case, "raw": raw}


def world_meds(plan: dict) -> list[dict]:
    """The medicines an item prescribes, for the world's record (`med_course.overlay_medicines`): the
    managed drug for its condition (dose line set from the realized item, `hooks.observe_meds`) and a
    comorbidity's co-medication, taken from day 0."""
    from haenv.world.labworld.meds import comorbidity
    from haenv.world.labworld.meds import drug
    out = [{"drug": plan["drug"], "indication": drug(plan["drug"])["known_condition"]}]
    cm = plan.get("comorbidity")
    co = (comorbidity(cm).get("co_medication_world") or {}) if cm else {}
    if co:
        out.append({"drug": co["drug"], "indication": cm, "dose_timeline": [{"ts": 0, "value": float(co["dose"])}]})
    return out


def record_streams(case: dict) -> set[str]:
    """The clinical streams the world of `case` draws (`build.clinical_plan`)."""
    from haenv.world.generation.build import clinical_plan
    raw = case.get("raw") or {}
    return set(clinical_plan(str(raw.get("disease") or ""), list(raw.get("devices") or ()),
                             list(raw.get("comorbidities") or ()), str(case.get("case_id"))))


def eligible_drugs(case: dict, world_streams: set[str], *, condition_specs: dict | None = None) -> list[str]:
    """`world_streams` are the streams of the case as drawn; the streams and the age rules are also read
    on the world of the patient the item's record describes (`record_case`), so no control reading has
    a second source in a world stream."""
    from haenv.world.generation.demographics import avoid_for
    from haenv.world.labworld.meds import drug
    raw0 = case.get("raw") or {}
    primary = str(raw0.get("drug") or "")
    treated = {str(raw0.get("disease") or "")} | {str(x) for x in raw0.get("comorbidities") or ()}
    out = []
    for d in DRUGS:
        if drug(d)["known_condition"] in treated:
            continue
        rc = record_case(case, d, condition_specs=condition_specs)
        raw = rc["raw"]
        if set(drug(d).get("bearing_streams") or ()) & (set(world_streams) | record_streams(rc)):
            continue
        if drug(d)["axis"] == "HbA1c" and primary in GLUCOSE_PRIMARY:
            continue
        if not dx_allowed(d, case):
            continue
        lo_hi = drug(d).get("age")
        if lo_hi and not lo_hi[0] <= _age(rc) <= lo_hi[1]:
            continue
        # Thyroid treatment includes both full replacement and a low start;
        # the patient's age/cardiovascular history determines that regimen.
        # a woman who can become pregnant is not dealt an ACEI or a statin item (`demographics.avoid_for`);
        # the item is dealt to another patient, so no spec loses its supply to this rule
        if avoid_for(d, str(raw.get("sex") or ""), str(raw.get("age_range") or "")):
            continue
        out.append(d)
    return out


def _share(options, u: float):
    acc = 0.0
    for name, p in options:
        acc += p
        if u < acc:
            return name
    return options[-1][0]


def cell_weights() -> dict:
    """{(drug, class, comorbidity or None): weight} of the cells a class is split over: 3 per drug; a
    drug whose next step a comorbidity changes (enalapril, CKD) weighs 2 with the comorbidity and 1
    without per class, so every class holds the same number of comorbid items; a cell of
    `LOW_SIGNAL_CELLS` weighs `LOW_SIGNAL_WEIGHT`."""
    cm = comorbid_drugs()
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config
    shares = config("p3")["class_weights"]["value"] if enabled() else None
    out = {}
    for d in DRUGS:
        for code in (None,) + ((cm[d],) if d in cm else ()):
            known = (code,) if code else ()
            for c in CLASSES:
                if takes(d, c, known):
                    w = WEIGHT if d not in cm else WEIGHT_COMORB[0 if code else 1]
                    if (d, c) in LOW_SIGNAL_CELLS and not (shares and c == "switch"):
                        w = LOW_SIGNAL_WEIGHT   # p3-fixes e
                    if shares:
                        w *= int(shares[c])
                    out[(d, c, code)] = w
    return out


def planned_cells(n_items: int = 50) -> dict:
    """{(drug, class, comorbidity or None): n} of a published pack of `n_items` items, by largest
    remainder at every step: N over the decisions by `CLASS_SHARE`; a decision over the drugs that can
    take it, alike (`takes`; a drug that cannot take a decision gets no item); a drug with a registered
    comorbidity over with / without by `WEIGHT_COMORB`. Ties go to the earlier entry of `DRUGS`."""
    import haenv.world.validation.pack_size as PS
    cm = comorbid_drugs()
    out = {}
    from haenv.world.validation.difficulty_r2 import enabled
    from haenv.world.validation.difficulty_r2 import config
    shares = config("p3")["class_weights"]["value"] if enabled() else CLASS_SHARE
    for c, n in PS.apportion(n_items, shares).items():
        carry = {d: [code for code in (None,) + ((cm[d],) if d in cm else ()) if takes(d, c, (code,) if code else ())]
                 for d in DRUGS}
        carry = {d: codes for d, codes in carry.items() if codes}
        weights = dict.fromkeys(carry, 1)
        from haenv.world.validation.difficulty_r2 import enabled
        if c == "switch" and "atorvastatin" in carry and not enabled():
            from haenv.world.history.history_dependency import rate
            share = rate("p3", {})
            weights = {"atorvastatin": 1} if share == 1 else {**weights, "atorvastatin": share * (len(carry) - 1) / (1 - share)}
        for d, k in PS.apportion(n, weights).items():
            codes = carry[d]
            w = {code: WEIGHT_COMORB[0 if code else 1] for code in codes} if len(codes) > 1 else {codes[0]: 1}
            out.update({(d, c, code): j for code, j in PS.apportion(k, w).items() if j})
    return out


def planned_scenarios(n_items: int = 50) -> dict:
    """{(drug, class, comorbidity or None, subtype): n}: each cell of `planned_cells` over the subtypes
    its drug policy admits, by their `subtype_weights` shares (largest remainder). Subtypes get no quota
    of their own: a subtype only one drug can carry would make the drug tell the decision."""
    import haenv.world.validation.pack_size as PS
    return {(d, c, code, sub): k for (d, c, code), n in planned_cells(n_items).items()
            for sub, k in PS.apportion(n, subtype_weights(c, d, (code,) if code else ())).items() if k}


def comorbid_drugs() -> dict[str, str]:
    """{drug: comorbidity code} for every registered comorbidity that changes a drug's next step (A24)."""
    from haenv.world.labworld.meds import meds
    return {d: code for code, c in (meds().get("comorbidities") or {}).items() for d in c.get("drugs") or ()}


def known_of(plan: dict) -> tuple[str, ...]:
    return (plan["comorbidity"],) if plan.get("comorbidity") else ()


def has_early(drug_name: str, known=()) -> bool:
    """Whether the drug can carry an `early` item (a miss right after an increase, read as "not yet
    arrived"): it is increased at all, its recheck is a visit in practice (not TSH 10-20
    days or HbA1c 5-6 weeks after a change) and a normal responder can miss above the lowest rung."""
    from haenv.world.labworld.meds import drug as _drug
    x = _drug(drug_name, known)
    return x.get("early_items") is not False and x.get("when_off") != "add" and x.get("max_off_rung", 99) >= 1


def subtype_weights(cls: str, drug_name: str, known=()) -> dict[str, float]:
    """Existing admissible scenario families and shares, used by both planning and selection.

    Restrictions come from the same registered drug policy as the gold. Allocation must not
    invent a subtype to fill a quota (e.g. early HbA1c or symptom-only levothyroxine reduction).
    """
    from haenv.world.labworld.meds import drug as _drug
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled() and cls == "switch" and drug_name in ("atorvastatin", "enalapril", "gliclazide"):
        # The statin's registered response cannot reliably miss at its top
        # rung. The other medicines retain the existing maximum-dose branch.
        return {"off": 1.0} if drug_name == "atorvastatin" else {"off": 0.9, "top_off": 0.1}

    def drug(n):
        return _drug(n, known)
    if drug(drug_name).get("on_target") == "titrate":
        # A24: raised at every visit below the top, so below the top only an item to raise (on target
        # when a reading can sit clear of both the target and the floor, else off target; p3-fixes c)
        # or a just-raised one; every other decision sits on the top dose
        up = "on" if on_target_raise_possible(drug(drug_name)) else "off"
        fixed = {"uptitrate": up, "downtitrate": "ae_dose", "switch": "top_off",
                 "check_adherence_or_adverse_effect": "adh_low"}.get(cls)
        return {fixed: 1.0} if fixed else dict(SUBTYPES["maintain"])
    timing = drug(drug_name)["attribution"] == "timing"
    if cls == "downtitrate" and drug(drug_name).get("ae_dose_decision") is False:
        return {"over": 1.0}       # dosed by the axis; a symptom alone lowers nothing
    if cls == "check_adherence_or_adverse_effect" and (not timing or "unattributed" not in drug(drug_name).get(
            "check_subtypes", ("adh_low", "unattributed"))):
        return {"adh_low": 1.0}
    if cls == "maintain" and not has_early(drug_name, known):
        return {"on": 1.0}         # no off reading above the lowest rung
    if cls == "switch" and drug_name == "atorvastatin":
        from haenv.world.history.history_dependency import rate
        if rate("p3", {}) > 0:
            return {"off": 1.0}
    if cls == "switch" and drug(drug_name)["when_off"] == "titrate":
        return {"top_off": 1.0}    # a titrated drug switches only at the top (A15)
    if drug(drug_name)["when_off"] == "add" and cls in ("switch", "maintain"):
        return {"off" if cls == "switch" else "on": 1.0}
    return dict(SUBTYPES[cls])


def subtype(cls: str, drug_name: str, cid: str, known=()) -> str:
    return _share(subtype_weights(cls, drug_name, known).items(), h01("p3sub", cid))


def on_target_raise_possible(d: dict) -> bool:
    """p3-fixes c: whether a reading can be on target (one band under the target) and above the
    clinic's floor by one band at once -- the only on-target readings G3 raises. With home SBP (CV 3%,
    band 1.96 x CV = 0.059 on the log scale) the floor 120 and the target 130 are 0.080 apart, under
    two bands, so no on-target item is raised: the CKD uptitrate items are off target below the top."""
    import math
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import band
    fl = d.get("titrate_on_target_min")
    if not fl:
        return True
    R = band(d["axis"])
    return math.log(float(axis(d["axis"])["upper"])) - math.log(float(fl)) > 2 * R


ON_TARGET = ("ae_dose", "over", "on", "unattributed")


def position(sub: str, cid: str, drug_name: str | None = None) -> str:
    from haenv.world.labworld.meds import drug
    p = POSITION[sub]
    pos = _share((("min", p[0]), ("mid", p[1]), ("top", p[2])), h01("p3pos", cid))
    if sub == "off" and drug_name and drug(drug_name).get("max_off_rung") == 0:
        pos = "min"         # A18: a normal responder misses the target only on the lowest rung
    if pos == "top" and sub in ON_TARGET and drug_name and drug(drug_name).get("flat"):
        pos = "mid"         # a flat-response drug at target started at its rung; the top rung is never a start
    if pos == "min" and sub in ON_TARGET and drug_name and drug(drug_name).get("min_reaches_target") is False:
        pos = "mid"         # the lowest rung cannot bring this drug 1 RCV under the target (A9)
    return pos


def make_plans(cases: list[dict], world_streams: dict[str, set[str]], cap: int = CAP_PER_DRUG,
               *, condition_specs: dict | None = None,
               eligibility: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """A14: each drug appears in every class it can take, in equal shares. Inside each H stratum,
    cases (fewest eligible drugs first, then hash order) take their eligible drug dealt least so
    far, each drug taking at most `cap` cases per stratum; inside each (H, drug),
    the classes the drug can take are dealt round-robin in hash order."""
    from haenv.world.labworld.meds import drug
    ok = ({str(c["case_id"]): eligibility[str(c["case_id"])] for c in cases}
          if eligibility is not None else {str(c["case_id"]): eligible_drugs(c, world_streams.get(str(c["case_id"]), set()),
                                         **({"condition_specs": condition_specs} if condition_specs is not None else {}))
          for c in cases})
    Hs = {str(c["case_id"]): hidden(c.get("latent") or {}) for c in cases}
    cases_by = {str(c["case_id"]): c for c in cases}
    out = {}
    for h in (0, 1):
        ids = [i for i in ok if ok[i] and Hs[i] == h]
        supply = [sum(1 for i in ids if d in ok[i]) for d in DRUGS]
        n = dict.fromkeys(DRUGS, 0)
        drug_of = {}
        for cid in sorted(ids, key=lambda i: (len(ok[i]), h01("pack3", i))):
            cand = [d for d in ok[cid] if n[d] < cap]
            if not cand:
                continue
            d = min(cand, key=lambda x: (n[x], h01("p3drug-tie", cid, x)))
            n[d] += 1
            drug_of[cid] = d
        cm = comorbid_drugs()
        for d in DRUGS:
            mine = sorted([i for i in drug_of if drug_of[i] == d], key=lambda i: h01("p3pick", i))
            groups = [(None, mine)]
            if d in cm:                     # A24: every other case carries the comorbidity
                with_c = set(sorted(mine, key=lambda i: h01("p3comorb", i))[0::2])
                groups = [(cm[d], [i for i in mine if i in with_c]), (None, [i for i in mine if i not in with_c])]
            for code, ids_g in groups:
                known = (code,) if code else ()
                w = {k: cell_weights()[(d, k, code)] for k in CLASSES if takes(d, k, known)}
                # the classes dealt in the pack's cell weights, a class's turns spread over the ring
                can = [k for r in range(max(w.values())) for k in w if r < w[k]]
                off = int(h01("pack3", "offset", h, d, *known) * len(can))
                for k, cid in enumerate(ids_g):
                    out[cid] = _plan(can[(k + off) % len(can)], d, cid, ok[cid], h, target_upper(d, cases_by[cid]), code)
        # the dispensing size, the number of past increases, the number of visits and the number of
        # diary entries are dealt in turn inside each (H, decision), so none follows the decision
        # (A15, A19, A20)
        for c in CLASSES:
            mine = [i for i in out if out[i]["H"] == h and out[i]["class"] == c]
            for key, vals in (("supply", SUPPLY), ("n_inc", (0, 1, 2)), ("n_extra", (0, 1)), ("n_vis", N_VIS_BACKGROUND if background_on() else N_VIS), ("n_diary", N_DIARY)):
                for k, cid in enumerate(sorted(mine, key=lambda i: h01("p3" + key, i))):
                    # A21: a drug with a registered titration history deals it in every class
                    vv = (key == "n_inc" and drug(out[cid]["drug"]).get("titration_history")) or vals
                    from haenv.world.validation.difficulty_r2 import enabled
                    if key == "n_inc" and out[cid]["drug"] == "levothyroxine" and enabled():
                        # Include first reviews and later small-step courses
                        # in every response class, before clinical realization.
                        # These are synthetic history counts, not prevalence.
                        vv = vals
                    out[cid][key] = vv[k % len(vv)]
            # Overtreatment keeps the same optional-renewal allocation as other decisions.
            # Its realizer places that renewal before the final increase, so no earlier visit
            # renews a dose on a suppressed reading (A18) and no class flag zeroes the count.
    return out


def takes(d: str, cls: str, known=()) -> bool:
    from haenv.world.labworld.meds import drug as _drug

    def drug(n):
        return _drug(n, known)
    if cls == "switch":
        from haenv.world.history.history_dependency import rate
        if d == "atorvastatin" and rate("p3", {}) > 0:
            return True
        x = drug(d)
        top = len(x["ladder"]) - 1
        if x["when_off"] == "titrate" and x.get("max_off_rung", top) < top:
            return False    # A18: a titrated drug is switched only at the top, where it reaches the target
        return bool(x.get("switch", True))
    if cls == "uptitrate":
        x = drug(d)                                  # the guideline's next step below the top (A15, A23)
        return x["when_off"] == "titrate" and x.get("uptitrate_items", True)
    return True


def target_upper(d: str, case: dict) -> float:
    """The clinic target's upper bound for this patient (`upper_if_known`; none registered since
    A15: the 2026 ACC/AHA LDL-C < 70 mg/dL needs risk factors the world does not show)."""
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import drug
    ax = axis(drug(d)["axis"])
    raw = case.get("raw") or {}
    known = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    for k, v in (ax.get("upper_if_known") or {}).items():
        if k in known:
            return float(v)
    return float(ax["upper"])


def _plan(c: str, d: str, cid: str, ok: list[str], H: int, upper: float, comorb: str | None = None) -> dict:
    from haenv.world.labworld.meds import drug
    known = (comorb,) if comorb else ()
    sub = subtype(c, d, cid, known)
    pos = position(sub, cid, d)
    if drug(d, known).get("on_target") == "titrate":
        pos = ("mid" if c == "uptitrate" or sub == "early" or (sub == "on" and c == "maintain" and h01("p3ckdpos", cid) < 0.5)
               else "top")     # (see `subtype`): a CKD item on target below the top is kept only under the SBP floor (placed one band or more under it, p3-fixes c)
    bg = background_on()
    return {"version": PLAN_VERSION_BACKGROUND if bg else PLAN_VERSION, "class": c, "subtype": sub, "drug": d, "position": pos,
            **({"comorbidity": comorb} if comorb else {}),
            "decoys": {"dip": sub != "adh_low" and h01("p3dip", cid) < DIP_Q,
                       "predating": ((bg or sub not in AE_SUBTYPES) and bool(drug(d).get("predating_contexts"))
                                     and h01("p3pre", cid) < PREDATING_Q)},
            "drugs_ok": list(ok), "H": H, "upper": upper}
