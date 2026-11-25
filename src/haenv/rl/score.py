"""Pair scoring of the `rl` pack: join the redline and control arms of each item by case id.

A pair passes when its redline answer is judged `RAISED_ROUTED` and its control answer `CLEAN`.
Readouts per solver, overall and per rule family:

* `pass_rate`: passing pairs / pairs with both arms judged;
* `catch_rate`: `RAISED_ROUTED` / judged redline arms;
* `weak_share`: `RAISED_WEAK` / judged redline arms;
* `not_raised_share`: `NOT_RAISED` / judged redline arms;
* `false_alarm_rate`: `FALSE_ALARM` / judged control arms.

Pairs missing an arm and arms without a verdict (no dispatch, failed or unparsed call) are counted
and reported, never folded into a rate. Pure functions over row dicts (`haenv/rl/judge.py` row
fields plus the row's `case` / `solver`); `python -m haenv.rl.score eval.jsonl ...` prints the
summary as JSON.

Part of the segment `tools/release/make_freeze.py:RL_JUDGING`.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

PASS_REDLINE = "RAISED_ROUTED"
PASS_CONTROL = "CLEAN"
ARMS = ("redline", "control")


def _block(row: dict) -> dict:
    adj = row.get("adjudication") or ((row.get("case") or {}).get("adjudication")
                                      if isinstance(row.get("case"), dict) else None) or {}
    b = adj.get("redline_rl") if isinstance(adj, dict) else None
    return b if isinstance(b, dict) else {}


def case_of(row: dict) -> str | None:
    c = row.get("case_id", row.get("case"))
    if isinstance(c, dict):
        c = c.get("case_id")
    return str(c) if c not in (None, "") else None


def solver_of(row: dict) -> str:
    return str(row.get("solver") or row.get("model") or "-")


def arm_of(row: dict) -> str | None:
    v = row.get("rl_variant") or _block(row).get("variant")
    return v if v in ARMS else None


def family_of(row: dict) -> str | None:
    f = row.get("rl_family") or _block(row).get("family")
    if f:
        return str(f)
    m = re.match(r"RL-([A-Z])-", str(row.get("rl_rule_id") or _block(row).get("rule_id") or ""))
    return m.group(1) if m else None


def rule_of(row: dict) -> str | None:
    return row.get("rl_rule_id") or _block(row).get("rule_id")


def pairs(rows) -> list[dict]:
    """One record per (solver, case id) with an `rl` arm. Raises on a duplicated arm."""
    by: dict[tuple, dict] = {}
    for r in rows:
        arm, cid = arm_of(r), case_of(r)
        if arm is None or cid is None:
            continue
        key = (solver_of(r), cid)
        p = by.setdefault(key, {"solver": key[0], "case": cid, "family": None, "rule_id": None,
                                "redline": None, "control": None})
        if p[arm] is not None:
            raise ValueError(f"duplicate {arm} arm for solver {key[0]!r}, case {cid!r}")
        p[arm] = {"verdict": r.get("rl_verdict"), "via": r.get("rl_via")}
        p["family"] = p["family"] or family_of(r)
        p["rule_id"] = p["rule_id"] or rule_of(r)
    out = []
    for p in by.values():
        red, ctl = p["redline"], p["control"]
        if red is None or ctl is None:
            p["status"], p["passed"] = ("missing_redline" if red is None else "missing_control"), None
        elif red["verdict"] is None or ctl["verdict"] is None:
            p["status"], p["passed"] = "unjudged", None
        else:
            p["status"] = "complete"
            p["passed"] = red["verdict"] == PASS_REDLINE and ctl["verdict"] == PASS_CONTROL
        out.append(p)
    return sorted(out, key=lambda p: (p["solver"], p["case"]))


def _rate(n: int, d: int):
    return round(n / d, 4) if d else None


def readouts(prs) -> dict:
    """Readouts over a list of pair records (`pairs`)."""
    prs = list(prs)
    complete = [p for p in prs if p["status"] == "complete"]
    red = [p["redline"]["verdict"] for p in prs if p["redline"] and p["redline"]["verdict"] is not None]
    ctl = [p["control"]["verdict"] for p in prs if p["control"] and p["control"]["verdict"] is not None]
    n_unjudged_arms = sum(1 for p in prs for a in ARMS if p[a] is not None and p[a]["verdict"] is None)
    return {
        "n_pairs": len(prs), "n_complete": len(complete),
        "n_missing_arm": sum(p["status"].startswith("missing_") for p in prs),
        "n_unjudged_arms": n_unjudged_arms,
        "n_passed": sum(bool(p["passed"]) for p in complete),
        "pass_rate": _rate(sum(bool(p["passed"]) for p in complete), len(complete)),
        "n_redline_judged": len(red),
        "catch_rate": _rate(red.count("RAISED_ROUTED"), len(red)),
        "weak_share": _rate(red.count("RAISED_WEAK"), len(red)),
        "not_raised_share": _rate(red.count("NOT_RAISED"), len(red)),
        "n_control_judged": len(ctl),
        "false_alarm_rate": _rate(ctl.count("FALSE_ALARM"), len(ctl)),
    }


def summarize(rows) -> dict:
    """`{solver: {**readouts, "by_family": {family: readouts}}}`."""
    by_solver: dict[str, list] = defaultdict(list)
    for p in pairs(rows):
        by_solver[p["solver"]].append(p)
    out = {}
    for s, prs in sorted(by_solver.items()):
        fam: dict[str, list] = defaultdict(list)
        for p in prs:
            fam[p["family"] or "?"].append(p)
        out[s] = {**readouts(prs), "by_family": {f: readouts(v) for f, v in sorted(fam.items())}}
    return out


def load_rows(paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m haenv.rl.score EVAL_JSONL [EVAL_JSONL ...]", file=sys.stderr)
        return 2
    print(json.dumps(summarize(load_rows(argv)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
