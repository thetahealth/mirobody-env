"""Offline gates on the catalog and automatic receipts, never on scoring."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import math

from haenv.world.protocols.library import catalog
from haenv.world.protocols.library import render
from haenv.world.protocols.library import table


def _tv(a, b):
    ca, cb = Counter(a), Counter(b)
    return sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in ca.keys() | cb.keys()) / 2


def distribution(docs=None):
    docs = catalog() if docs is None else docs
    meta, cfg = table(), table()["distribution_gate"]
    other = {m["protocol_id"] for m in meta["other"]}
    b = [d for d in docs if d["protocol_id"] in other]
    if not b:
        return {"error": "missing unrelated protocols"}
    out = {}
    for family in ("triage", "medication"):
        ids = {m["protocol_id"] for m in meta[family]}
        a = [d for d in docs if d["protocol_id"] in ids]
        if len(a) != len(ids) or len(b) != len(other):
            return {"error": "missing catalog entries"}
        fmt = lambda d: (tuple(sorted(d)), tuple(type(d[k]).__name__ for k in sorted(d)),
                         bool(d.get("sources")) and all(s.startswith("参考：") and "；链接：" in s and "；本院约定：" in s for s in d["sources"]))
        out[family] = {
            "length_tv": _tv([bisect_left(cfg["length_edges"]["value"], len(render(d))) for d in a],
                             [bisect_left(cfg["length_edges"]["value"], len(render(d))) for d in b]),
            "step_tv": _tv([bisect_left(cfg["step_edges"]["value"], len(d["steps"])) for d in a],
                           [bisect_left(cfg["step_edges"]["value"], len(d["steps"])) for d in b]),
            "format_tv": _tv([fmt(d) for d in a], [fmt(d) for d in b]),
            "bearing_lengths": [len(render(d)) for d in a], "unrelated_lengths": [len(render(d)) for d in b],
            "bearing_steps": [len(d["steps"]) for d in a], "unrelated_steps": [len(d["steps"]) for d in b]}
    return out


def distribution_hits(docs=None):
    report, cfg = distribution(docs), table()["distribution_gate"]
    if "error" in report:
        return [report["error"]]
    return [f"{family} {metric} {vals[metric]:.4f} > {cfg[limit]['value']}"
            for family, vals in report.items() for metric, limit in
            (("length_tv", "length_tv_max"), ("step_tv", "step_tv_max"), ("format_tv", "format_tv_max"))
            if vals[metric] > cfg[limit]["value"] + 1e-12]


def receipt_hits(block, case_id):
    from haenv.world.protocols.scoring import expected
    from haenv.world.protocols.scoring import trusted_assessment
    from haenv.p2.rules import assess
    rec, kind, _ = expected(block, case_id)
    actual = block.get("home_check_today", {}).get("system_news2")
    bad = [] if actual == rec else ["automatic receipt differs from registered sampler and NEWS2 inputs"]
    if actual == rec and kind != "correct" and trusted_assessment(block, actual)["set"][0] in assess(block)["set"]:
        bad.append("automatic error does not defeat the blind-system stub")
    return bad


def rates_report(items):
    from haenv.world.protocols.scoring import expected
    from haenv.world.protocols.scoring import table as scoring_table
    cfg = scoring_table()
    counts = {k: {"eligible": 0, "errors": 0, "rate": cfg["rates"][k]["value"]} for k in ("wrong_scale", "missing_zero")}
    n = 0
    for it in items:
        b = it["sp"]["prediction_context"]["triage_p2"]
        if b.get("rules_delivery") != "library":
            continue
        n += 1
        correct, _, eligible = expected(b, it["case_id"])
        actual = b.get("home_check_today", {}).get("system_news2") or {}
        from haenv.world.protocols.scoring import scale_of
        kind = "wrong_scale" if actual.get("spo2_scale") != scale_of(b) else "missing_zero" if any(
            v is None for v in actual.get("input_observations", {}).values()) else "correct"
        for k in counts:
            counts[k]["eligible"] += int(eligible[k])
            counts[k]["errors"] += int(kind == k)
    return {"n": n, "groups": counts, "all_errors": sum(x["errors"] for x in counts.values()),
            "cap": cfg["rates"]["all_errors"]["max"]}


def rates_hits(items):
    from haenv.world.protocols.scoring import table as scoring_table
    rep, cfg = rates_report(items), scoring_table()["sampling"]
    bad = []
    for k, g in rep["groups"].items():
        n, p = g["eligible"], g["rate"]
        allowance = max(cfg["minimum_count_allowance"], cfg["allowance_z"] * math.sqrt(n * p * (1 - p)))
        if abs(g["errors"] - n * p) > allowance:
            bad.append(f"{k} count {g['errors']}/{n} differs from registered {p}")
    if rep["n"] and rep["all_errors"] / rep["n"] > rep["cap"] + 1e-12:
        bad.append("whole-pack automatic error share exceeds registered cap")
    return bad
