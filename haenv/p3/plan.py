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

Eligibility (PREREG 3, amendments A3/A5/A12): no world stream that bears on the drug's decision,
no glycaemic axis when the world's primary drug acts on glucose, the drug's condition not already
treated in the world (its disease or a comorbidity: no ACEI on top of the world's ARB, no second
statin; L3-9), and a hidden line whose diagnosis names none of the diseases that change this
drug's decision (L3-7).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib

CLASSES = ("uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect")
DRUGS = ("gliclazide", "atorvastatin", "enalapril", "levothyroxine")
GLUCOSE_PRIMARY = ("metformin", "semaglutide", "liraglutide", "dulaglutide", "tirzepatide")
#: subtypes and their share inside each class (PREREG 4, A5)
SUBTYPES = {
    "uptitrate": (("off", 1.0),),
    "downtitrate": (("ae_dose", 1.0),),                   # levothyroxine: ae_dose / over, half each
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
CAP_PER_DRUG = 40        # cases per drug per H stratum at N = 50 (the generator scales it with N)
SUPPLY = (30, 60, 90)       # days per dispensing, a per-patient habit (A14, A15)
N_VIS = (3, 4, 5)           # visits on record including today, one range for every class (A19)
N_DIARY = (1, 2, 3)         # diary entries, one range for every class; a drug symptom takes an ordinary entry's place (A20)
PLAN_VERSION = 12


def h01(*parts) -> float:
    s = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(s[:12], 16) / float(16 ** 12)


def hidden(latent: dict) -> int:
    return int(str((latent or {}).get("ddx_join_gold")) in ("unified", "comorbidity"))


def dx_allowed(drug_name: str, case: dict) -> bool:
    """A16 whitelist: every component of the case's condition line is on the drug's `allowed_dx`."""
    from ..labworld.meds import drug
    dx = str((case.get("latent") or {}).get("ddx_diagnosis") or "")
    ok = drug(drug_name).get("allowed_dx") or ()
    return not dx or all(any(k in part for k in ok) for part in dx.split(" + "))


def _age(case: dict) -> int:
    a = str((case.get("raw") or {}).get("age_range") or "")
    return int(a.split("-")[0]) if a[:2].isdigit() else 0


def background_ok(drug_name: str, *texts, sex: str | None = None) -> bool:
    """No word of the drug's `background_deny` list, and no word of the other sex (A17,
    `events.sex_words`)."""
    from ..events import sex_words
    from ..labworld.meds import drug
    words = list(drug(drug_name).get("background_deny") or ()) + list(sex_words(sex) if sex else ())
    return not any(w in str(t or "") for t in texts for w in words)


def eligible_drugs(case: dict, world_streams: set[str]) -> list[str]:
    from ..labworld.meds import drug
    raw = case.get("raw") or {}
    primary = str(raw.get("drug") or "")
    treated = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    out = []
    for d in DRUGS:
        if drug(d)["known_condition"] in treated:
            continue
        if set(drug(d).get("bearing_streams") or ()) & set(world_streams):
            continue
        if drug(d)["axis"] == "HbA1c" and primary in GLUCOSE_PRIMARY:
            continue
        if not dx_allowed(d, case):
            continue
        lo_hi = drug(d).get("age")
        if lo_hi and not lo_hi[0] <= _age(case) <= lo_hi[1]:
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


def comorbid_drugs() -> dict[str, str]:
    """{drug: comorbidity code} for every registered comorbidity that changes a drug's next step (A24)."""
    from ..labworld.meds import meds
    return {d: code for code, c in (meds().get("comorbidities") or {}).items() for d in c.get("drugs") or ()}


def known_of(plan: dict) -> tuple[str, ...]:
    return (plan["comorbidity"],) if plan.get("comorbidity") else ()


def subtype(cls: str, drug_name: str, cid: str, known=()) -> str:
    from ..labworld.meds import drug as _drug

    def drug(n):
        return _drug(n, known)
    if drug(drug_name).get("on_target") == "titrate":
        # A24: raised at every visit below the top, so below the top only an on-target item to raise
        # or a just-raised one; every other decision sits on the top dose
        return {"uptitrate": "on", "downtitrate": "ae_dose", "switch": "top_off",
                "check_adherence_or_adverse_effect": "adh_low"}.get(cls) or (
            "early" if h01("p3sub", cid) < 2 / 3 else "on")
    timing = drug(drug_name)["attribution"] == "timing"
    if cls == "downtitrate" and drug_name == "levothyroxine":
        return "over" if h01("p3over", cid) < 0.5 else "ae_dose"
    if cls == "check_adherence_or_adverse_effect" and (not timing or "unattributed" not in drug(drug_name).get(
            "check_subtypes", ("adh_low", "unattributed"))):
        return "adh_low"
    if cls == "maintain" and drug(drug_name).get("max_off_rung", 99) < 1:
        return "on"         # A18: no off reading above the lowest rung, so no just-increased item
    if cls == "switch" and drug(drug_name)["when_off"] == "titrate":
        return "top_off"    # a titrated drug is switched or added to only at the top (A15)
    if drug(drug_name)["when_off"] == "add" and cls in ("switch", "maintain"):
        return "off" if cls == "switch" else "on"    # never increased: no top, no recent increase (A16)
    return _share(SUBTYPES[cls], h01("p3sub", cid))


ON_TARGET = ("ae_dose", "over", "on", "unattributed")


def position(sub: str, cid: str, drug_name: str | None = None) -> str:
    from ..labworld.meds import drug
    p = POSITION[sub]
    pos = _share((("min", p[0]), ("mid", p[1]), ("top", p[2])), h01("p3pos", cid))
    if sub == "off" and drug_name and drug(drug_name).get("max_off_rung") == 0:
        pos = "min"         # A18: a normal responder misses the target only on the lowest rung
    if pos == "top" and sub in ON_TARGET and drug_name and drug(drug_name).get("flat"):
        pos = "mid"         # a flat-response drug at target started at its rung; the top rung is never a start
    if pos == "min" and sub in ON_TARGET and drug_name and drug(drug_name).get("min_reaches_target") is False:
        pos = "mid"         # the lowest rung cannot bring this drug 1 RCV under the target (A9)
    return pos


def make_plans(cases: list[dict], world_streams: dict[str, set[str]], cap: int = CAP_PER_DRUG) -> dict[str, dict]:
    """A14: each drug appears in every class it can take, in equal shares. Inside each H stratum,
    cases (fewest eligible drugs first, then hash order) take their eligible drug dealt least so
    far, each drug taking at most `cap` cases per stratum; inside each (H, drug),
    the classes the drug can take are dealt round-robin in hash order."""
    from ..labworld.meds import drug
    ok = {str(c["case_id"]): eligible_drugs(c, world_streams.get(str(c["case_id"]), set())) for c in cases}
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
                can = [k for k in CLASSES if takes(d, k, known)]
                off = int(h01("pack3", "offset", h, d, *known) * len(can))
                for k, cid in enumerate(ids_g):
                    out[cid] = _plan(can[(k + off) % len(can)], d, cid, ok[cid], h, target_upper(d, cases_by[cid]), code)
        # the dispensing size, the number of past increases, the number of visits and the number of
        # diary entries are dealt in turn inside each (H, decision), so none follows the decision
        # (A15, A19, A20)
        for c in CLASSES:
            mine = [i for i in out if out[i]["H"] == h and out[i]["class"] == c]
            for key, vals in (("supply", SUPPLY), ("n_inc", (0, 1, 2)), ("n_extra", (0, 1)), ("n_vis", N_VIS), ("n_diary", N_DIARY)):
                for k, cid in enumerate(sorted(mine, key=lambda i: h01("p3" + key, i))):
                    # A21: a drug with a registered titration history deals it in every class
                    vv = (key == "n_inc" and drug(out[cid]["drug"]).get("titration_history")) or vals
                    out[cid][key] = vv[k % len(vv)]
            for cid in mine:
                if out[cid]["subtype"] == "over":
                    out[cid]["n_extra"] = 0     # A18: no visit renews the dose after a suppressed reading
    return out


def takes(d: str, cls: str, known=()) -> bool:
    from ..labworld.meds import drug as _drug

    def drug(n):
        return _drug(n, known)
    if cls == "switch":
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
    from ..labworld.meds import axis, drug
    ax = axis(drug(d)["axis"])
    raw = case.get("raw") or {}
    known = {str(raw.get("disease") or "")} | {str(x) for x in raw.get("comorbidities") or ()}
    for k, v in (ax.get("upper_if_known") or {}).items():
        if k in known:
            return float(v)
    return float(ax["upper"])


def _plan(c: str, d: str, cid: str, ok: list[str], H: int, upper: float, comorb: str | None = None) -> dict:
    from ..labworld.meds import drug
    known = (comorb,) if comorb else ()
    sub = subtype(c, d, cid, known)
    pos = position(sub, cid, d)
    if drug(d, known).get("on_target") == "titrate":
        pos = "mid" if c == "uptitrate" or sub == "early" else "top"     # A24 (see `subtype`)
    return {"version": PLAN_VERSION, "class": c, "subtype": sub, "drug": d, "position": pos,
            **({"comorbidity": comorb} if comorb else {}),
            "decoys": {"dip": sub != "adh_low" and h01("p3dip", cid) < DIP_Q,
                       "predating": (sub not in AE_SUBTYPES and bool(drug(d).get("predating_contexts"))
                                     and h01("p3pre", cid) < PREDATING_Q)},
            "drugs_ok": list(ok), "H": H, "upper": upper}
