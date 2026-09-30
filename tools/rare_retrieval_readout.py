"""Rare retrieval readout: per model, question-weighted accuracy by kind, tool use, tokens and time,
against a majority-answer baseline (docs/decisions/2026-09-30-rare-检索题.md §4).

    uv run python tools/rare_retrieval_readout.py results/joint_dx/rare_retrieval-p6p7/<batch> [--out <json>]

One extraction path: the solver's raw reply in the batch's `responses.jsonl` (`retrieval_answers`),
graded with `retrieval.grade` against the gold in the batch's `cases.jsonl` — the same function the
judge uses. The baseline answers every question of a kind with the kind's most common gold answer
(for variants: the empty list, right only where the VCF is refused; for inheritance: every call
the most common origin at the gold positions — a guesser who already knows the positions). SYNTHETIC data, evaluation only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "haenv_rare"))
from haenv_rare import retrieval as RR  # noqa: E402


def load(batch: Path):
    gold = {}
    for line in open(batch / "cases.jsonl", encoding="utf-8"):
        c = json.loads(line)
        rr = (((c.get("case") or c).get("adjudication") or {}).get("rare") or {}).get("retrieval")
        if rr is None:
            rr = ((c.get("latent") or {}).get("rare_retrieval"))
        if rr:
            gold[c["case_id"]] = rr
    resp = {}
    for line in open(batch / "responses.jsonl", encoding="utf-8"):
        r = json.loads(line)
        try:
            ans = json.loads(r["raw"]).get("retrieval_answers") or {}
        except (ValueError, TypeError, AttributeError):
            ans = {}
        resp[(r["solver"], r["case"])] = ans           # the last line of a (model, case) wins: a resumed cell
    return gold, resp


def baseline(gold: dict) -> dict:
    by_kind = defaultdict(list)
    for rr in gold.values():
        for q in rr["questions"]:
            by_kind[q["kind"]].append(rr["gold"][q["qid"]])
    base = {}
    for kind, gs in by_kind.items():
        if kind == "inheritance":
            top = Counter(v["origin"] for g in gs for v in g["variants"]).most_common(1)[0][0]
            base[kind] = lambda g, top=top: {"variants": [dict(v, origin=top) for v in g["variants"]]}
        elif kind == "variants":
            base[kind] = lambda g: {"variants": []}
        elif kind == "sleep_stages":
            med = {k: statistics.median(g[k] for g in gs) for k in ("W", "N1", "N2", "N3", "R")}
            base[kind] = lambda g, med=med: med
        else:
            top = Counter(json.dumps(g, sort_keys=True) for g in gs).most_common(1)[0][0]
            base[kind] = lambda g, top=top: json.loads(top)
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    batch = Path(a.batch)
    gold, resp = load(batch)
    solvers = sorted({s for s, _ in resp})
    base = baseline(gold)
    rows = {}
    for s in solvers + ["baseline:majority"]:
        per = defaultdict(lambda: [0, 0])
        tool = parse = n = 0
        tin = tout = 0
        secs = []
        missing = 0
        for cid, rr in gold.items():
            ans = resp.get((s, cid)) if s != "baseline:majority" else {}
            if s != "baseline:majority" and ans is None:
                missing += len(rr["questions"])
                continue
            for q in rr["questions"]:
                g = rr["gold"][q["qid"]]
                if s == "baseline:majority":
                    got = {"answer": base[q["kind"]](g), "tools": []}
                else:
                    got = ans.get(q["qid"]) or {}
                ok = RR.grade(q["kind"], g, got.get("answer"))["correct"]
                per[q["kind"]][0] += ok
                per[q["kind"]][1] += 1
                n += 1
                parse += isinstance(got.get("answer"), dict)
                tool += bool(set(got.get("tools") or []) & RR.RETRIEVAL_TOOLS)
                u = got.get("usage") or {}
                tin += u.get("input_tokens") or 0
                tout += u.get("output_tokens") or 0
                if got.get("seconds") is not None:
                    secs.append(got["seconds"])
        tot = sum(v[0] for v in per.values())
        rows[s] = {"n": n, "missing": missing, "correct": tot, "acc": tot / n if n else None,
                   "by_kind": {k: {"ok": v[0], "n": v[1]} for k, v in sorted(per.items())},
                   "tool_use": tool / n if n else None, "parse_ok": parse / n if n else None,
                   "input_tokens": tin, "output_tokens": tout,
                   "median_seconds": statistics.median(secs) if secs else None}
    kinds = sorted({q["kind"] for rr in gold.values() for q in rr["questions"]})
    print(f"batch {batch.name}: {len(gold)} cases, {sum(len(rr['questions']) for rr in gold.values())} questions")
    print(f"{'solver':42s} {'acc':>7s} " + " ".join(f"{k[:12]:>12s}" for k in kinds) + f" {'tools':>6s} {'parse':>6s} {'in_tok':>10s} {'med_s':>6s}")
    for s, r in rows.items():
        cells = " ".join(f"{r['by_kind'].get(k, {}).get('ok', 0):>5d}/{r['by_kind'].get(k, {}).get('n', 0):<6d}" for k in kinds)
        print(f"{s:42s} {r['acc'] if r['acc'] is not None else float('nan'):7.1%} {cells} {r['tool_use'] or 0:6.0%} {r['parse_ok'] or 0:6.0%} "
              f"{r['input_tokens']:>10d} {r['median_seconds'] or 0:6.0f}" + (f"  (missing {r['missing']})" if r["missing"] else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"batch": batch.name, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
