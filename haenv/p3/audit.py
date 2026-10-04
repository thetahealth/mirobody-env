"""Pack 3's declaration for the shared batch audit (`haenv/shared_audit.py`, `tools/pack_audit.py`).

The published pack of N items is apportioned over the (drug, class the drug can take) cells by
weight: 3 per cell; a drug whose next step a comorbidity changes (enalapril, CKD) weighs 2 per
class with the comorbidity and 1 per class without, so every class holds the same number of
comorbid items (N = 50 takes the weights as counts). Inside a cell, items are picked in
sha256(seeded "p3pick", id) order, each from the H stratum with fewer items of that class so far
(SA-3 asserts the cells; a cell the pool cannot fill is an error). Every feature is read off the solver payload (`features.py`); nothing here
reads the gold block except the class and, for the decisive-evidence check, G3.

* SA-5: the scenario variables, permuted inside each drug (A20: the drug fixes which classes are
  possible, so the question is what a variable says beyond the drug);
* SA-6: one classifier over the union of the surface and length features, permuted inside each
  drug. The refill-row count is the adherence evidence G3 reads (`decisive`): it leaves the union,
  and the shared audit checks that G3 does read it;
* SA-7: every complaint on the record and in the diary passes the drug's whitelist (`background_ok`);
* profile (reported, not judged): each decision input as a one-variable stub, and the rule readers.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections

from .. import pack_size as PS
from . import features as FT
from .gold import CLASSES
from .plan import DRUGS, comorbid_drugs, h01, takes

WEIGHT = 3
WEIGHT_COMORB = (2, 1)         # per class, with / without the comorbidity
SURFACE = ("drug", "H", "age", "sex", "n_known", "T", "D", "n_visits", "n_streams", "block_chars",
           "payload_chars", "n_diary", "n_refills", "n_increase_rows")
#: `comorbid`: the comorbidity-presence stub (A24)
SCENARIO = ("H", "age", "sex", "T", "D", "n_visits", "n_known", "comorbid")
DECISION = ("position_i", "status_i", "since_increase", "coverage90", "drug_symptom", "n_increase_rows", "n_refills")
POS_I = {"min": 0, "mid": 1, "top": 2}
STATUS_I = {"on": 0, "off": 1, "over": 2}


def _with_h(items):
    for it in items:
        it["H"] = int(it["gold"]["plan"]["H"])
    return items


def comorb_of(it) -> str | None:
    return (it["gold"].get("comorbidity") or {}).get("code")


def cell_weights() -> dict:
    """{(drug, class, comorbidity or None): weight}; the weights are the counts at N = 50."""
    cm = comorbid_drugs()
    out = {}
    for d in DRUGS:
        for code in (None,) + ((cm[d],) if d in cm else ()):
            for c in CLASSES:
                if takes(d, c, (code,) if code else ()):
                    out[(d, c, code)] = WEIGHT if d not in cm else WEIGHT_COMORB[0 if code else 1]
    return out


def planned_cells(n_items: int = PS.DEFAULT_N) -> dict:
    """{(drug, class, comorbidity or None): n} of the published pack of `n_items` items."""
    return PS.apportion(n_items, cell_weights())


def assemble(items: list[dict], n_items: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED) -> list[dict]:
    tag = PS.seeded("p3pick", seed)

    def key(it):
        return h01(tag, it["case_id"])
    out = []
    cells = planned_cells(n_items)
    PS.short(cells, collections.Counter((it["gold"]["drug"], it["class"], comorb_of(it)) for it in items),
             "p3 pack")
    for c in CLASSES:
        nh = collections.Counter()
        for (d, c_, code), n in cells.items():
            if c_ != c:
                continue
            left = sorted([it for it in items if it["gold"]["drug"] == d and it["class"] == c
                           and comorb_of(it) == code], key=key)
            for _ in range(n):
                it = min(left, key=lambda x: (nh[x["H"]], key(x)))
                left.remove(it)
                nh[it["H"]] += 1
                out.append(it)
    return out


def rows(items: list[dict], ctx=None) -> list[dict]:
    out = []
    for it in items:
        f = FT.inputs(it["sp"])
        out.append({**FT.surface(it["sp"], it["H"]), "position_i": POS_I[f["position"]],
                    "status_i": STATUS_I[f["status"]], "since_increase": f["since_increase"],
                    "coverage90": round(f["coverage90"], 4), "drug_symptom": f["drug_symptom_any"],
                    "class": it["class"], "spec": it["spec"], "case_id": it["case_id"]})
    return out


def drop_newest_refill(g: dict) -> dict:
    """The record the refill-row count counts, one row fewer (the newest pickup)."""
    return {**g, "refills": list(g["refills"])[:-1]}


def _regold(g: dict):
    from .gold import g3
    return g3(g)


def whitelist(it) -> list[str]:
    """L3-10 (A17): every complaint on the emitted record and in the diary passes the drug's whitelist."""
    from ..labworld.meds import drug
    from .plan import background_ok
    d = it["gold"]["drug"]
    sex = str((it["sp"].get("user_profile") or {}).get("sex") or "")
    own = set(drug(d)["ae_symptoms"])
    texts = [(e.get("symptom"), e.get("context")) for e in it["sp"].get("evidence_ledger") or ()
             if e.get("source_type") == "patient_reported_symptom"]
    texts += [(e["symptom"], e.get("context")) for e in it["sp"]["prediction_context"]["meds_p3"]["symptom_diary"]
              if e["symptom"] not in own]
    return [t for t, c in texts if not background_ok(d, t, c, sex=sex)]


def profile(pool, rows_, ctx=None) -> dict:
    """Reported, not judged (D2): each decision input as a one-variable lookup stub (cc), and the
    partial-rule readers (cc on the published pack's rule)."""
    from ..shared_audit import _bins, _lookup_ba
    from .score import cc
    y = [r["class"] for r in rows_]
    m = len(set(y))
    stubs = {v: round((m * _lookup_ba(_bins([r[v] for r in rows_]), y) - 1) / (m - 1), 4) for v in DECISION}
    readers = {"full_knowledge": FT.rule_reader, "anomaly_then_check": FT.anomaly_stub}
    readers.update({f"without_{k}": (lambda kk: (lambda sp: FT.rule_reader(sp, (kk,))))(k) for k in FT.KNOWLEDGE})
    gold = [it["class"] for it in pool]
    return {"id": "profile", "decision_input_cc": stubs,
            "rule_readers": {k: round(cc(gold, [fn(it["sp"]) for it in pool]), 4) for k, fn in readers.items()},
            "pass": True}


def audit(n_items: int = 50, seed: str = ""):
    from ..shared_audit import PackAudit
    from .score import cc
    return PackAudit(
        name="p3", block="meds_p3", classes=CLASSES,
        class_of=lambda g: g.get("class") if g.get("class") in CLASSES else None,
        rows=rows, univariate=SCENARIO, surface=SURFACE, cc=cc, strata="drug",
        cell_of=lambda it, ctx: (it["gold"]["drug"], it["class"], comorb_of(it)),
        planned_cells=lambda items, ctx: planned_cells(n_items),
        select=lambda items: assemble(items, n_items, seed), prepare=lambda items, ctx: _with_h(items),
        decisive={"n_refills": drop_newest_refill}, regold=_regold,
        realism={"whitelist": whitelist}, pack_gates={"profile": profile})
