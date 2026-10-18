"""Pack 3's declaration for the shared batch audit (`haenv/world/validation/shared_audit.py`, `tools/validation/pack_audit.py`).

The published pack of N items follows `plan.planned_scenarios`: the five decisions alike, a decision's
subtypes alike, the drugs that carry a (decision, subtype) alike, enalapril's share 2 with CKD / 1
without. Inside a cell, items are picked in
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
import statistics

import haenv.world.validation.pack_size as PS
from haenv.p3 import features as FT
from haenv.p3.gold import CLASSES
from haenv.p3.plan import h01
from haenv.p3.plan import planned_cells
from haenv.p3.plan import planned_scenarios

# The selected drug is now an answer component inferred from clinical source
# contents, not an immediate header fact. Keep it as the permutation stratum
# and a clinical profile, without supplying it to the content-free classifier.
SURFACE = ("H", "age", "sex", "n_known", "T", "D", "n_visits", "n_streams", "block_chars",
           "payload_chars", "n_diary", "n_refills", "n_increase_rows")
#: `comorbid`: the comorbidity-presence stub (A24)
SCENARIO = ("H", "age", "sex", "T", "D", "n_visits", "n_known", "comorbid")
DECISION = ("position_i", "status_i", "since_increase", "coverage90", "drug_symptom", "n_increase_rows", "n_refills")
POS_I = {"min": 0, "mid": 1, "top": 2}
STATUS_I = {"on": 0, "off": 1, "over": 2}


def symptom_status_readout(items):
    """Predict only from recorded symptom status counts, without any clinical content."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from haenv.world.validation.shared_audit import balanced_acc
    from haenv.world.validation.shared_audit import chance_corrected
    data, labels, groups = [], [], []
    for it in items:
        b = it["sp"]["prediction_context"]["meds_p3"]
        count = collections.Counter(e.get("status") for e in b["symptom_diary"])
        data.append([count["已缓解"], count["持续至今"], count[None]])
        labels.append(it["case"]["adjudication"]["meds_p3"]["class"])
        groups.append(it.get("spec") or (it["case"].get("latent_premise") or {}).get("ddx_spec_id") or it["case_id"])
    classes = sorted(set(labels))
    if len(classes) < 2 or len(set(groups)) < 2:
        raise ValueError("symptom-status probe requires multiple classes and patient groups")
    X, y, group = np.asarray(data), np.asarray(labels), np.asarray(groups)
    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(min(5, len(set(groups)))).split(X, y, group):
        if len(set(y[train])) < 2:
            pred[test] = y[train][0]
        else:
            pred[test] = GradientBoostingClassifier(n_estimators=60, max_depth=2,
                random_state=20261012).fit(X[train], y[train]).predict(X[test])
    ba = balanced_acc(labels, pred.tolist())
    return {"n": len(items), "classes": len(classes), "balanced_accuracy": ba,
            "chance_corrected": chance_corrected(ba, len(classes))}


def _with_h(items):
    for it in items:
        it["H"] = int(it["gold"]["plan"]["H"])
    return items


def comorb_of(it) -> str | None:
    return (it["gold"].get("comorbidity") or {}).get("code")


def scenario_of(it) -> tuple:
    return (it["gold"]["drug"], it["class"], comorb_of(it), it["gold"]["subtype"])




def assemble(items: list[dict], n_items: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED, *,
             scenarios: bool = False) -> list[dict]:
    """The published pack of n_items: it extends the pack of each smaller nested size
    (`PS.nest_chain`). The audited pack opts into `scenarios=True`; the aggregate-cell API
    remains available for callers that provide only composition cells, without clinical records.
    This is an explicit allocation choice, never a fallback for missing subtype supply.
    """
    out: list[dict] = []
    for n in PS.nest_chain(n_items):
        out = _extend(items, n, seed, out, scenarios=scenarios)
    return out


def _extend(items: list[dict], n_items: int, seed: str, base: list[dict], *, scenarios: bool) -> list[dict]:
    tag = PS.seeded("p3pick", seed)

    def key(it):
        return h01(tag, it["case_id"])
    taken = {it["case_id"] for it in base}
    out = list(base)
    spec_counts = collections.Counter(it.get("spec") for it in base)
    from haenv.world.validation.shared_audit import take_patient
    cell_of = scenario_of if scenarios else lambda it: (it["gold"]["drug"], it["class"], comorb_of(it))
    cells = planned_scenarios(n_items) if scenarios else planned_cells(n_items)
    PS.short(cells, collections.Counter(cell_of(it) for it in items), "p3 subtype cells" if scenarios else "p3 pack")
    for c in CLASSES:
        nh = collections.Counter(it["H"] for it in base if it["class"] == c)
        for cell, n in cells.items():
            d, c_, code = cell[:3]
            if c_ != c:
                continue
            have = sum(cell_of(it) == cell for it in base)
            if have > n:
                raise SystemExit(f"[pack] p3 non-monotone nested quota: {cell} {have} > {n}")
            left = sorted([it for it in items if cell_of(it) == cell
                           and it["case_id"] not in taken], key=key)
            for _ in range(n - have):
                # Preserve the H balance, then distribute eligible diagnosis
                # specs uniformly inside the unchanged drug/subtype cell.
                minimum = min(nh[x["H"]] for x in left)
                eligible = [x for x in left if nh[x["H"]] == minimum]
                it = take_patient(eligible, spec_counts, key)
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
    from haenv.p3.gold import g3
    return g3(g)


def whitelist(it) -> list[str]:
    """L3-10 (A17): every complaint on the emitted record and in the diary passes the drug's whitelist."""
    from haenv.world.labworld.meds import drug
    from haenv.p3.plan import background_ok
    d = it["gold"]["drug"]
    sex = str((it["sp"].get("user_profile") or {}).get("sex") or "")
    from haenv.world.labworld.meds import meds
    cf = meds()["confounders"]      # conspicuous background entries are read by G3 on their own rules (`gold.confounder_why`)
    own = set(drug(d)["ae_symptoms"]) | {x for v in (cf["other_drug"] or {}).values() for x in v["symptoms"]} \
        | {x["symptom"] for x in cf["intercurrent"]["illnesses"]}
    texts = [(e.get("symptom"), e.get("context")) for e in it["sp"].get("evidence_ledger") or ()
             if e.get("source_type") == "patient_reported_symptom"]
    texts += [(e["symptom"], e.get("context")) for e in it["sp"]["prediction_context"]["meds_p3"]["symptom_diary"]
              if e["symptom"] not in own]
    return [t for t, c in texts if not background_ok(d, t, c, sex=sex)]


def profile(pool, rows_, ctx=None) -> dict:
    """Reported, not judged (D2): each decision input as a one-variable lookup stub (cc), and the
    partial-rule readers (cc on the published pack's rule)."""
    from haenv.world.validation.shared_audit import _bins
    from haenv.world.validation.shared_audit import _lookup_ba
    from haenv.p3.score import cc
    y = [r["class"] for r in rows_]
    m = len(set(y))
    stubs = {v: round((m * _lookup_ba(_bins([r[v] for r in rows_]), y) - 1) / (m - 1), 4) for v in DECISION}
    readers = {"full_knowledge": FT.rule_reader, "printed_rules_only": FT.printed_rule_reader,
               "anomaly_then_check": FT.anomaly_stub}
    readers.update({f"without_{k}": (lambda kk: (lambda sp: FT.rule_reader(sp, (kk,))))(k) for k in FT.KNOWLEDGE})
    gold = [it["class"] for it in pool]
    return {"id": "profile", "decision_input_cc": stubs,
            "rule_readers": {k: round(cc(gold, [fn(it["sp"]) for it in pool]), 4) for k, fn in readers.items()},
            "pass": True}


def allocation_readout(pool, n_items: int, seed: str) -> dict:
    """Coverage and rule readouts of the published subset, not of unused candidate items."""
    from haenv.world.validation.shared_audit import _bins
    from haenv.world.validation.shared_audit import _lookup_ba_loo
    from haenv.world.validation.shared_audit import chance_corrected
    from haenv.p3.gold import g3
    from haenv.p3.score import cc
    selected = assemble(pool, n_items, seed, scenarios=True)
    have = collections.Counter(scenario_of(it) for it in selected)
    want = planned_scenarios(n_items)
    mismatch = {str(k): {"need": n, "have": have[k]} for k, n in want.items() if have[k] != n}
    nested = assemble(pool, 50, seed, scenarios=True) == selected[:50] if n_items >= 50 else None
    gold = [it["class"] for it in selected]
    policy_bad = [it["case_id"] for it in selected if FT.printed_rule_reader(it["sp"]) != it["class"]]
    record_bad = [it["case_id"] for it in selected if g3(it["gold"])["class"] != it["class"]]
    rr = rows(selected)
    topology = {v: round(chance_corrected(_lookup_ba_loo(_bins([r[v] for r in rr]), gold), len(CLASSES)), 4)
                for v in SURFACE if v != "n_refills"}
    specs = collections.Counter(it["spec"] for it in selected)
    return {"id": "allocation", "n_items": len(selected), "nested_50_prefix": nested,
            "classes": dict(collections.Counter(gold)),
            "H_by_class": {str(k): n for k, n in sorted(collections.Counter((it["class"], it["H"])
                                                                            for it in selected).items())},
            "scenario_counts": {str(k): n for k, n in sorted(have.items(), key=str)},
            "scenario_quota": {str(k): n for k, n in want.items()}, "mismatch": mismatch,
            "n_condition_families": len(specs), "largest_condition_family": max(specs.values()),
            "payload_chars": {"median": statistics.median(r["payload_chars"] for r in rr),
                              "max": max(r["payload_chars"] for r in rr)},
            "block_chars": {"median": statistics.median(r["block_chars"] for r in rr),
                            "max": max(r["block_chars"] for r in rr)},
            "printed_policy_cc": round(cc(gold, [FT.printed_rule_reader(it["sp"]) for it in selected]), 4),
            "printed_policy_mismatch": policy_bad, "record_gold_mismatch": record_bad,
            "without_knowledge_cc": {k: round(cc(gold, [FT.rule_reader(it["sp"], (k,)) for it in selected]), 4)
                                     for k in FT.KNOWLEDGE},
            "anomaly_stub_cc": round(cc(gold, [FT.anomaly_stub(it["sp"]) for it in selected]), 4),
            "topology_loo_cc_report_only": topology,
            "pass": not mismatch and not policy_bad and not record_bad and nested is not False}


def audit(n_items: int = 50, seed: str = ""):
    from haenv.world.validation.shared_audit import PackAudit
    from haenv.p3.score import cc
    return PackAudit(
        name="p3", block="meds_p3", classes=CLASSES,
        class_of=lambda g: g.get("class") if g.get("class") in CLASSES else None,
        rows=rows, univariate=SCENARIO, surface=SURFACE, cc=cc, strata="drug",
        cell_of=lambda it, ctx: (it["gold"]["drug"], it["class"], comorb_of(it)),
        planned_cells=lambda items, ctx: planned_cells(n_items),
        select=lambda items: assemble(items, n_items, seed, scenarios=True), prepare=lambda items, ctx: _with_h(items),
        decisive={"n_refills": drop_newest_refill}, regold=_regold,
        realism={"whitelist": whitelist}, pack_gates={"profile": profile,
            "allocation": lambda pool, rows_, ctx: allocation_readout(pool, n_items, seed)})
