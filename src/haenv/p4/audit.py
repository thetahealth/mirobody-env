"""Pack 4's declaration for the shared batch audit (`haenv/world/validation/shared_audit.py`, `tools/validation/pack_audit.py`).

A pack of N items: the four classes take equal shares of N, each halved over H (`quota`).
CKD and direction quotas are retained in each cell. Diverse plans balance applicable analytes
and causes inside those quotas while extending the same nested selection; legacy plans retain
their seeded order. Neither path selects on observed amplitude or audit scores. Features come from the solver payload
(`features.py`); nothing here reads the gold block except the class.

Pack-specific gate P-10 (L4-9): a reader who only knows which factors appear on the two compared
draws or in the window between them cannot classify the source (GBM on the factor flags, cc <= 0.5).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections
import math
import re
import statistics

import haenv.world.validation.pack_size as PS
from haenv.p4 import features as FT
from haenv.p4.plan import CLASSES
from haenv.p4.plan import h01

#: content-free features of the rendered item (SA-6), each also a single-variable stub (SA-5)
SURFACE = ("ratio", "sign", "analyte", "H", "interval", "hour1", "hour0", "age", "sex", "n_known", "T2D",
           "HTN", "n_hist", "block_chars", "payload_chars", "n_med_rows", "n_notes", "drug", "T")
#: `ratio` is an observation input read by the gold; clinical-source classes may also be inside RCV: decisive
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
    from haenv.world.labworld import tables
    from haenv.p4.plan import comorbidity_classes
    reg = (tables().get("comorbidities") or {}).get("CKD")
    share = float(reg["share"]) if reg else 0.0
    out = {}
    for (h, c), n in quota(n_items).items():
        k = int(n * share) if c in comorbidity_classes() else 0
        out[(h, c, None)] = n - k
        if k:
            out[(h, c, "CKD")] = k
    return out


def assemble(items: list[dict], n_items: int = PS.DEFAULT_N, seed: str = PS.PUBLIC_SEED, *,
             directions: bool = False) -> list[dict]:
    """Extend nested sizes. The audited pack uses `directions=True` for equal direction quotas.

    The aggregate-cell API is preserved for callers supplying composition cells only. Missing
    direction supply in the audited path is an error, never a reason to use aggregate selection.
    """
    out = []
    for n in PS.nest_chain(n_items):
        out = _extend(items, n, seed, out, directions=directions)
    return out


def direction_of(it) -> tuple:
    return (it["H"], it["class"], comorb_of(it), int(it["gold"]["sign"]))


def planned_directions(n_items: int) -> dict:
    return {(*cell, sign): k for cell, n in planned_cells(n_items).items()
            for sign, k in PS.apportion(n, {1: 1, -1: 1}).items()}


def _extend(items, n_items, seed, base, *, directions):
    cell_of = direction_of if directions else lambda it: (it["H"], it["class"], comorb_of(it))
    by = collections.defaultdict(list)
    for it in items:
        by[cell_of(it)].append(it)
    need = planned_directions(n_items) if directions else planned_cells(n_items)
    PS.short(need, {k: len(v) for k, v in by.items()}, "p4 direction cells" if directions else "p4 emitted items")
    tag = PS.seeded("p4pick", seed)
    out = list(base)
    taken = {it["case_id"] for it in base}
    have = collections.Counter(cell_of(it) for it in base)
    branches = {(it["class"], it["gold"].get("next_step")) for it in base}
    spec_counts = collections.Counter(it.get("spec") for it in base)
    analyte_counts = collections.Counter((it["class"], it["gold"].get("analyte")) for it in base)
    cause_counts = collections.Counter((it["gold"].get("analyte"), it["gold"].get("cause")) for it in base)
    from haenv.world.validation.shared_audit import take_patient
    for cell, n in sorted(need.items(), key=str):
        if have[cell] > n:
            raise SystemExit(f"[pack] p4 non-monotone nested quota: {cell} {have[cell]} > {n}")
        available = sorted([it for it in by[cell] if it["case_id"] not in taken],
                           key=lambda it: h01(tag, it["case_id"]))
        for _ in range(n - have[cell]):
            # Cover real next-step branches within the unchanged composition /
            # direction quotas, then use the ordinary seeded order. No case names.
            eligible = [x for x in available if (x["class"], x["gold"].get("next_step")) not in branches] or list(available)
            if eligible and eligible[0]["gold"].get("diversity_version"):
                # Balanced mechanism challenge design, declared in the registry.
                # No amplitude, model response or shortcut score enters selection.
                rank = lambda x: (analyte_counts[(x["class"], x["gold"]["analyte"])],
                                  cause_counts[(x["gold"]["analyte"], x["gold"].get("cause"))])
                least = min(map(rank, eligible))
                eligible = [x for x in eligible if rank(x) == least]
            it = take_patient(eligible, spec_counts, lambda x: h01(tag, x["case_id"]))
            out.append(it)
            available.remove(it)
            branches.add((it["class"], it["gold"].get("next_step")))
            analyte_counts[(it["class"], it["gold"].get("analyte"))] += 1
            cause_counts[(it["gold"].get("analyte"), it["gold"].get("cause"))] += 1
    return out


def rows(items: list[dict], ctx=None) -> list[dict]:
    from haenv.world.labworld import analyte as A
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
    from haenv.p4.gold import g4
    recs = {int(r["day"]): r for r in g["records"]}
    return g4(recs[int(g["t0"])], recs[int(g["t1"])], float(g["rcv"]), tuple(g["band"]), source_policy=g.get("source_policy"))["class"]


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

    from haenv.world.validation.shared_audit import balanced_acc
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


def decoy_counts(items) -> dict:
    """Report only: decoys the plan named, and those the realization could not place on a free draw
    (the factor-only reader of P-10 sees only the placed ones)."""
    det = [[x for x in (it["gold"].get("decoy_detail") or ())] for it in items or ()]
    drop = [sum(1 for x in d if x.get("dropped")) for d in det]
    return {"planned": sum(len(d) for d in det), "dropped": sum(drop), "items_with_dropped": sum(1 for k in drop if k)}


def p10(items, rows_, ctx=None) -> dict:
    cc = factor_only(rows_)
    return {"id": "P-10", "factor_only_cc": round(cc, 4), "decoys": decoy_counts(items), "pass": bool(cc <= 0.5)}


def _summary(values) -> dict:
    vals = sorted(float(x) for x in values)
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": round(vals[0], 6), "median": round(statistics.median(vals), 6),
            "max": round(vals[-1], 6)}


def lab_readout(items) -> dict:
    """Describe the archived paired observations; the 10% edge bin is not a clinical rule.

    CV/RCV and reference intervals are read from the artifacts. The current registry is reported
    separately, so a historic artifact is not silently relabelled with today's noise parameters.
    No actual medical data or real-vs-synthetic instrument is consulted.
    """
    from haenv.world.labworld import analyte
    from haenv.world.labworld import tables_sha16
    compared, positions, edge, ratios, changes, levels, methods = [], [], 0, [], [], [], []
    cvs, hashes, by = collections.Counter(), set(), collections.defaultdict(list)
    printed_bad = []
    for it in items:
        g, block = it["gold"], _blk(it)
        ref = re.fullmatch(r"\s*([0-9.]+)\s*[–−-]\s*([0-9.]+)\s*", block["reference_range"])
        if not ref or float(ref[1]) >= float(ref[2]):
            raise ValueError(f"invalid printed reference interval: {block['reference_range']!r}")
        lo, hi = float(ref[1]), float(ref[2])
        reports = {int(r["day"]): r for r in block["report"]}
        pair = [reports[int(g[k])] for k in ("t0", "t1")]
        records = {int(r["day"]): r for r in g["records"]}
        for r in pair:
            recorded = float(records[int(r["day"])]["printed"])
            if float(r["value"]) != recorded:
                printed_bad.append({"case_id": it["case_id"], "day": int(r["day"]),
                                    "printed": float(r["value"]), "record": recorded})
        v0, v1 = (float(r["value"]) for r in pair)
        compared += [v0, v1]
        methods += [r["method"] for r in pair]
        levels.append(v0)
        changes.append(abs(math.log(v1 / v0)))
        ratios.append(float(g["g4"]["ratio"]))
        cvs["|".join(str(float(v)) for v in g["cv_total"])] += 1
        hashes.add(g["labworld_tables_sha16"])
        for v in (v0, v1):
            if lo <= v <= hi:
                pos = (v - lo) / (hi - lo)
                positions.append(pos)
                edge += int(min(pos, 1 - pos) <= 0.10 + 1e-12)
        by[it["class"]].append((v0, v1, float(g["rcv"]), float(g["g4"]["ratio"])))
    return {"compared_values": len(compared), "in_reference": len(positions), "near_reference_edge": edge,
            "printed_record_mismatch": printed_bad,
            "near_edge_fraction_in_reference": round(edge / len(positions), 6) if positions else None,
            "edge_definition": "within 10% of the printed reference interval width; descriptive only",
            "reference_position": _summary(positions), "previous_level": _summary(levels),
            "abs_log_change": _summary(changes), "abs_change_over_rcv": _summary(ratios),
            "methods": dict(collections.Counter(methods)), "artifact_cv_pairs": dict(cvs),
            "artifact_tables_sha16": sorted(hashes),
            "current_registry": {"tables_sha16": tables_sha16(),
                                 "creatinine_cvi": analyte("creatinine")["cvi"],
                                 "creatinine_cva": dict(analyte("creatinine")["cva"]),
                                 "review": "pending: existing provisional source parameters; unchanged"},
            "by_class": {c: {"n_items": len(vals), "previous_level": _summary(x[0] for x in vals),
                             "rcv": _summary(x[2] for x in vals), "ratio": _summary(x[3] for x in vals)}
                         for c, vals in by.items()}}


def allocation_readout(pool, n_items: int, seed: str) -> dict:
    from haenv.world.validation.shared_audit import _bins
    from haenv.world.validation.shared_audit import _lookup_ba_loo
    from haenv.world.validation.shared_audit import chance_corrected
    from haenv.p4.gold import gold_block_consistent
    selected = assemble(pool, n_items, seed, directions=True)
    have = collections.Counter(direction_of(it) for it in selected)
    want = planned_directions(n_items)
    mismatch = {str(k): {"need": n, "have": have[k]} for k, n in want.items() if have[k] != n}
    nested = assemble(pool, 50, seed, directions=True) == selected[:50] if n_items >= 50 else None
    rr, gold = rows(selected), [it["class"] for it in selected]
    topology = {v: round(chance_corrected(_lookup_ba_loo(_bins([r[v] for r in rr]), gold), len(CLASSES)), 4)
                for v in UNIVARIATE}
    specs = collections.Counter(it["spec"] for it in selected)
    records_bad = {it["case_id"]: bad for it in selected if (bad := gold_block_consistent(it["gold"]))}
    lab = lab_readout(selected)
    return {"id": "allocation", "n_items": len(selected), "nested_50_prefix": nested,
            "classes": dict(collections.Counter(gold)),
            "direction_counts": {str(k): n for k, n in sorted(have.items(), key=str)},
            "direction_quota": {str(k): n for k, n in want.items()}, "mismatch": mismatch,
            "n_condition_families": len(specs), "largest_condition_family": max(specs.values()),
            "payload_chars": _summary(r["payload_chars"] for r in rr),
            "block_chars": _summary(r["block_chars"] for r in rr),
            "record_gold_mismatch": records_bad, "lab": lab,
            "topology_loo_cc_report_only": topology,
            "factor_only_cc_report_only": round(factor_only(rr), 4),
            "pass": not mismatch and not records_bad and not lab["printed_record_mismatch"] and nested is not False}


def audit(n_items: int = 50, seed: str = ""):
    from haenv.world.validation.shared_audit import PackAudit
    from haenv.p4.score import cc
    return PackAudit(
        name="p4", block="followup_p4", classes=CLASSES,
        class_of=lambda g: g.get("class") if g.get("class") in CLASSES else None,
        rows=lambda items, ctx: rows(_with_h(items), ctx), univariate=UNIVARIATE, surface=SURFACE, cc=cc,
        cell_of=lambda it, ctx: (int(it["gold"]["plan"]["H"]), it["class"], comorb_of(it)),
        planned_cells=lambda items, ctx: planned_cells(n_items),
        select=lambda items: assemble(_with_h(items), n_items, seed, directions=True),
        decisive={"ratio": move_obs_beyond_rcv}, regold=regold,
        realism={"acei_without_indication": acei_without_indication, "fasting_after_noon": fasting_after_noon,
                 "dose_direction_flips": dose_direction_flips, "heavy_meat_breakfast": heavy_meat_breakfast,
                 "noise_with_method_offset": noise_with_method_offset,
                 "doctor_voice": doctor_voice},
        pack_gates={"P-10": p10,
                    "allocation": lambda pool, rows_, ctx: allocation_readout(pool, n_items, seed)})
