"""Pack 4's declaration for the shared batch audit (`haenv/shared_audit.py`, `tools/pack_audit.py`).

A pack of N items: the four classes take equal shares of N, each halved over H (`quota`); per
(H, class) cell the first items by the seeded `sha256("p4pick", id)`, blind to analyte, sign and
magnitude. In each cell whose class a CKD patient can carry, floor(cell size x the registry's CKD
share) of those items have CKD. Every feature is read off the solver payload
(`features.py`); nothing here reads the gold block except the class.

Pack-specific gate P-10 (L4-9): a reader who only knows which factors appear on the two compared
draws or in the window between them cannot classify the source (GBM on the factor flags, cc <= 0.5).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import math
import re

from .. import pack_size as PS
from . import features as FT
from .plan import CLASSES, h01

#: content-free features of the rendered item (SA-6), each also a single-variable stub (SA-5)
SURFACE = ("ratio", "sign", "analyte", "H", "interval", "hour1", "hour0", "age", "sex", "n_known", "T2D",
           "HTN", "n_hist", "block_chars", "payload_chars", "n_med_rows", "n_notes", "drug", "T")
#: `ratio` is read by the gold (a noise change sits inside the RCV, the other classes beyond it): decisive
#: evidence: SA-6 leaves it out of the union (`decisive` below) and so do the single-variable stubs
DECISIVE = ("ratio",)
#: `comorbid`: the comorbidity-presence stub (R15)
UNIVARIATE = tuple(f for f in SURFACE if f not in DECISIVE) + ("grid", "comorbid")
#: conditions that indicate an ACE inhibitor (R13a)
ACEI_INDICATIONS = {"hypertension", "CKD", "T2D", "heart_failure", "CAD"}
#: clinical shorthand in a patient sentence: a lab abbreviation, a chart word, a template joiner,
#: a value with a unit (R13 doctor voice)
SHORTHAND = re.compile(r"[A-Za-z]{2,}|B超|新发|无家族史|持续高|显著| \+ |\d+(\.\d+)?\s*(mmol|mg|g/|kg|cm|mmHg|μmol|umol|U/L|%)")
HEAVY_MEAT = ("羊肉串", "牛腩", "酱牛肉")


def comorb_of(it) -> str | None:
    return (it["gold"].get("comorbidity") or {}).get("code")


def quota(n_items: int = PS.DEFAULT_N) -> dict:
    """{(H, class): n}: equal class shares of N, each halved over H."""
    return PS.split_h(PS.apportion(n_items, dict.fromkeys(CLASSES, 1)))


def planned_cells(n_items: int = PS.DEFAULT_N) -> dict:
    """{(H, class, comorbidity or None): n} of a pack of N items."""
    from ..labworld import tables
    from .plan import COMORB_CLASSES
    reg = (tables().get("comorbidities") or {}).get("CKD")
    share = float(reg["share"]) if reg else 0.0
    out = {}
    for (h, c), n in quota(n_items).items():
        k = int(n * share) if c in COMORB_CLASSES else 0
        out[(h, c, None)] = n - k
        if k:
            out[(h, c, "CKD")] = k
    return out


def assemble(items: list[dict], n_items: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED) -> list[dict]:
    by = collections.defaultdict(list)
    for it in items:
        by[(it["H"], it["class"], comorb_of(it))].append(it)
    need = planned_cells(n_items)
    PS.short(need, {k: len(v) for k, v in by.items()}, "p4 emitted items")
    tag = PS.seeded("p4pick", seed)
    out = []
    for cell, n in sorted(need.items(), key=str):
        out += sorted(by[cell], key=lambda it: h01(tag, it["case_id"]))[:n]
    return out


def rows(items: list[dict], ctx=None) -> list[dict]:
    from ..labworld import analyte as A
    out = []
    for it in items:
        s = FT.surface(it["sp"], it["H"])
        nd = int(A(it["gold"]["analyte"])["ndigits"])
        out.append({**s, **{"f_" + k: v for k, v in FT.factor_flags(it["sp"]).items()},
                    "grid": int(round((s["v1"] - s["v0"]) * 10 ** nd)) % 5,
                    "class": it["class"], "spec": it["spec"], "case_id": it["case_id"]})
    return out


def move_obs_beyond_rcv(g: dict) -> dict:
    """The record `ratio` counts, changed: today's printed value moved to three RCV above the previous draw's."""
    recs = {int(r["day"]): r for r in g["records"]}
    r0, r1 = recs[int(g["t0"])], recs[int(g["t1"])]
    r1["printed"] = float(r0["printed"]) * math.exp(3.0 * float(g["rcv"]))
    return g


def regold(g: dict):
    from .gold import g4
    recs = {int(r["day"]): r for r in g["records"]}
    return g4(recs[int(g["t0"])], recs[int(g["t1"])], float(g["rcv"]), tuple(g["band"]))["class"]


def _with_h(items):
    for it in items:
        it["H"] = int(it["gold"]["plan"]["H"])
    return items


# ------------------------------------------------------------------------------ SA-7 rules (R13)

def _blk(it) -> dict:
    return ((it["sp"] or {}).get("prediction_context") or {}).get("followup_p4") or {}


def acei_without_indication(it) -> list[str]:
    known = set((it["sp"].get("user_profile") or {}).get("known_conditions") or ())
    acei = any("依那普利" in m["drug"] for m in _blk(it).get("medication_record") or ())
    return ["ACEI without an indication"] if acei and not known & ACEI_INDICATIONS else []


def fasting_after_noon(it) -> list[str]:
    return [x["time"] for x in _blk(it).get("report") or ()
            if "空腹" in str(x.get("fasting")) and int(x["time"].split(":")[0]) >= 12]


def dose_direction_flips(it) -> list[str]:
    seq = collections.defaultdict(list)
    for m in _blk(it).get("medication_record") or ():
        if "→" in m["event"]:
            a, b = [float(v) for v in m["event"].replace("剂量", "").split("→")]
            seq[m["drug"]].append(1 if b > a else -1)
    return [d for d, q in seq.items() if sum(1 for i in range(1, len(q)) if q[i] != q[i - 1]) >= 3]


def heavy_meat_breakfast(it) -> list[str]:
    return [e["food"] for d in _blk(it).get("diet_record") or () for e in d["entries"]
            if e["meal"] == "早餐" and any(w in e["food"] for w in HEAVY_MEAT)]


def noise_with_method_offset(it) -> list[str]:
    """A noise item's judged analyte is measured by the same method at both compared draws."""
    g = it["gold"]
    if g.get("class") != "analytic_biological_noise":
        return []
    return [p["factor"] for r in g["records"] if r["day"] in (g["t0"], g["t1"])
            for p in r["perturbations"] if p.get("acts") and p.get("kind") == "method"]


def doctor_voice(it) -> list[str]:
    return [str(e.get("symptom")) for e in it["sp"].get("evidence_ledger") or ()
            if e.get("source_type") == "patient_reported_symptom" and SHORTHAND.search(str(e.get("symptom") or ""))]


# ------------------------------------------------------------------------------ P-10

def factor_only(rows_: list[dict], seed: int = 0) -> float:
    """cc of a GBM (60 trees, depth 2, GroupKFold(5) by condition spec) that sees only the factor flags."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from ..shared_audit import balanced_acc
    X = np.asarray([[r["f_" + k] for k in FT.FACTORS] for r in rows_], float)
    y = np.asarray([r["class"] for r in rows_])
    g = np.asarray([r["spec"] for r in rows_])
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=min(5, len(set(g.tolist())))).split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr][0]
            continue
        pred[te] = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed).fit(
            X[tr], y[tr]).predict(X[te])
    m = len(set(y.tolist()))
    return (m * balanced_acc(y.tolist(), pred.tolist()) - 1) / (m - 1)


def p10(items, rows_, ctx=None) -> dict:
    cc = factor_only(rows_)
    return {"id": "P-10", "factor_only_cc": round(cc, 4), "pass": bool(cc <= 0.5)}


def audit(n_items: int = 50, seed: str = ""):
    from ..shared_audit import PackAudit
    from .score import cc
    return PackAudit(
        name="p4", block="followup_p4", classes=CLASSES,
        class_of=lambda g: g.get("class") if g.get("class") in CLASSES else None,
        rows=lambda items, ctx: rows(_with_h(items), ctx), univariate=UNIVARIATE, surface=SURFACE, cc=cc,
        cell_of=lambda it, ctx: (int(it["gold"]["plan"]["H"]), it["class"], comorb_of(it)),
        planned_cells=lambda items, ctx: planned_cells(n_items),
        select=lambda items: assemble(_with_h(items), n_items, seed),
        decisive={"ratio": move_obs_beyond_rcv}, regold=regold,
        realism={"acei_without_indication": acei_without_indication, "fasting_after_noon": fasting_after_noon,
                 "dose_direction_flips": dose_direction_flips, "heavy_meat_breakfast": heavy_meat_breakfast,
                 "noise_with_method_offset": noise_with_method_offset,
                 "doctor_voice": doctor_voice},
        pack_gates={"P-10": p10})
