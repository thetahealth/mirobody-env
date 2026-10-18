"""Read the 1.3.0 overall board from stored answers and management-plan scores.

    python -P board_score.py --work runs/main-n50 --tree <board tree> [--q q.jsonl] [--packs p2,p3,p4,m2] [--boot 2000] --out <dir>

Per item:  S = G * D * (1 + Q) / 2   on the first answer (pk1).
  D  stored decision result: p2 p2_in_set · p3 p3_class_ok · p4 p4_class_ok and p4_next_ok · m2 tools/m2_raw_correct.raw_correct
     M2's free-text diagnosis matching is judge-derived in the evaluated board.
  G  no hard gate (propensity gates `haenv.analytics.PROPENSITY_GATES` excluded); ABORT -> D = 0
  Q  management-plan rubric in [0,1], from --q (JSONL rows {pack, case, solver, pk, q}); missing -> S not computed
     unless G*D = 0, where S = 0 for every Q
Pack score = mean S over the pack's items; overall = equal-weight mean of pack scores.
Uncertainty: paired item-cluster bootstrap inside each pack (same draws for every model), combined across packs.
Rank interval (added 2026-10-14, additive): overall.<model>.rank / rank_ci, the 2026-10-06 candidate board's rule (tools/evaluation/board_readout.py) on the same draws.
Also reported: D accuracy and pass^k over k=3, hard-gate and propensity-gate rates, Q mean on D=1 items.
"""
from __future__ import annotations

# Direct file execution and imports select the same marked checkout.
import sys as _tool_sys
from pathlib import Path as _ToolPath
for _tool_root in _ToolPath(__file__).resolve().parents:
    if (_tool_root / "pyproject.toml").is_file() and (_tool_root / "tools").is_dir():
        _tool_sys.path.insert(0, str(_tool_root))
        break
else:
    raise FileNotFoundError(f"Cannot locate repository root from {__file__}")
from tools._bootstrap import bootstrap as _tool_bootstrap, repository_root as _repository_root
_tool_bootstrap(__file__)
import argparse, collections, glob, importlib.util, json, math, os, pathlib, random, sys

PACKS = ("p2", "p3", "p4", "m2")


def load_rows(work, pack):
    """{model: {pk: {case: row}}} merged over pools and over every work root in `work`
    (comma-separated). A cell whose latest row is ABORT(no_response) is replaced by its answer in
    the transport re-answer batch `<stamp>-rep<k>` when that batch holds one (10-14 rule: aborts
    caused on our side are re-answered in a fresh batch; documented in the master plan ledger)."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for root in [w for w in str(work).split(",") if w]:
        for f in glob.glob(f"{root}/*/{pack}/results/joint_dx/*/*-pk[0-9]/eval.jsonl"):
            pk = int(f.rsplit("-pk", 1)[1].split("/")[0])
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    out[r["solver"]][pk][r["case"]] = r
            rep = f.replace(f"-pk{pk}/eval.jsonl", f"-rep{pk}/eval.jsonl")
            if os.path.exists(rep):
                for line in open(rep):
                    if line.strip():
                        r = json.loads(line)
                        base = out[r["solver"]][pk].get(r["case"])
                        if base is not None and str(base.get("overall", "")).startswith("ABORT(no_response)") \
                                and not str(r.get("overall", "")).startswith("ABORT"):
                            r = {**r, "rerun_of_transport_abort": True}
                            out[r["solver"]][pk][r["case"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--tree", required=True)
    ap.add_argument("--q"); ap.add_argument("--packs", default=",".join(PACKS))
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregation", choices=("pk1", "k3"), default="pk1",
                    help="pk1 = preregistered (first answer); k3 = item S averaged over the k answers, reported beside pk1")
    a = ap.parse_args()
    tree = pathlib.Path(a.tree); sys.path[:0] = [str(tree), str(tree / "plugins")]
    from haenv.scoring.analysis.gates import PROPENSITY_GATES
    from tools._bootstrap import command_path
    spec = importlib.util.spec_from_file_location("m2raw", command_path('m2_raw_correct.py', root=tree))
    m2mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2mod)

    def abort(r):
        return str(r.get("overall") or "").startswith("ABORT")

    def D(pack, r):
        if abort(r):
            return 0
        if pack == "p2":
            return int(bool(r.get("p2_in_set")))
        if pack == "p3":
            return int(bool(r.get("p3_class_ok")))
        if pack == "p4":
            return int(bool(r.get("p4_class_ok")) and bool(r.get("p4_next_ok")))
        v = m2mod.raw_correct(r)
        return 0 if v is None else int(bool(v))

    def hard(r):
        return [g for g in (r.get("gates") or []) if str(g).split(":")[0] not in PROPENSITY_GATES]

    def prop(r):
        return [g for g in (r.get("gates") or []) if str(g).split(":")[0] in PROPENSITY_GATES]

    Q = {}
    if a.q:
        for line in open(a.q):
            if line.strip():
                x = json.loads(line)
                Q[(x["pack"], x["case"], x["solver"], int(x.get("pk", 1)))] = float(x["q"])

    req = {p for p in a.packs.split(",") if p}
    packs = [p for p in PACKS if p in req]   # canonical order: bootstrap draws do not depend on the --packs order (10-15)
    data = {p: load_rows(a.work, p) for p in packs}
    models = sorted({m for p in packs for m in data[p]})
    per = {}   # (pack, model) -> {case: S or None}
    prof = collections.defaultdict(dict)
    for p in packs:
        for m, ks in data[p].items():
            rows1 = ks.get(1, {})
            s_map = {}

            def s_of(r, c, k):
                g = 0 if (abort(r) or hard(r)) else 1
                d = D(p, r)
                q = Q.get((p, c, m, k))
                # G*D = 0 gives S = 0 for every Q, so such a cell needs no Q (exact, not an imputation; added 10-14 22:40)
                return 0.0 if g * d == 0 else (None if q is None else g * d * (1 + q) / 2)
            if a.aggregation == "pk1":
                for c, r in rows1.items():
                    s_map[c] = s_of(r, c, 1)
            else:   # k3: mean of the per-answer S over every round the item has; a missing Q leaves the item uncomputed
                for c in set.intersection(*[set(v) for v in ks.values()]) if ks else set():
                    vals = [s_of(ks[k][c], c, k) for k in sorted(ks)]
                    s_map[c] = None if any(v is None for v in vals) else sum(vals) / len(vals)
            per[(p, m)] = s_map
            allrows = [r for k in ks.values() for r in k.values()]
            cases = set.intersection(*[set(k) for k in ks.values()]) if ks else set()
            passk = sum(all(D(p, ks[k][c]) for k in ks) for c in cases) / len(cases) if cases else None
            qd = [Q[(p, c, m, 1)] for c, r in rows1.items() if (p, c, m, 1) in Q and D(p, r)]
            prof[p][m] = {
                "n_pk1": len(rows1), "k": len(ks),
                "D_acc_k": round(sum(D(p, r) for r in allrows) / len(allrows), 4) if allrows else None,
                "pass_k": round(passk, 4) if passk is not None else None,
                "hard_gate_rate": round(sum(bool(hard(r)) for r in allrows) / len(allrows), 4) if allrows else None,
                "propensity_gate_rate": round(sum(bool(prop(r)) for r in allrows) / len(allrows), 4) if allrows else None,
                "abort": sum(abort(r) for r in allrows),
                "Q_mean_on_D1": round(sum(qd) / len(qd), 4) if qd else None,
                "Q_coverage_pk1": f"{sum((p, c, m, 1) in Q for c in rows1)}/{len(rows1)}",
            }

    def pack_score(p, m, cases):
        v = [per[(p, m)].get(c) for c in cases]
        if not v or any(x is None for x in v):
            return None
        return sum(v) / len(v)

    case_lists = {p: sorted(set.intersection(*[set(per[(p, m)]) for m in models if (p, m) in per])) for p in packs}
    point = {}
    for m in models:
        ps = {p: pack_score(p, m, case_lists[p]) for p in packs if (p, m) in per}
        point[m] = {"packs": ps, "overall": (sum(ps.values()) / len(ps)) if ps and all(v is not None for v in ps.values()) and len(ps) == len(packs) else None}

    rng = random.Random(a.seed)
    boots = collections.defaultdict(list)
    draw_overall = []   # added 2026-10-14 for rank_ci only: each draw's overall per model (same draws as `boots`)
    # models without an overall (e.g. gpt-6-sol, M2 not run by design) are left out of the draws (10-14 22:45)
    if any(point[m]["overall"] is not None for m in models):
        for _ in range(a.boot):
            draws = {p: [rng.choice(case_lists[p]) for _ in case_lists[p]] for p in packs}
            dm = {}
            for m in models:
                if point[m]["overall"] is None:   # no point estimate -> no interval (a draw that misses the uncomputed items would bias it)
                    continue
                ps = [pack_score(p, m, draws[p]) for p in packs if (p, m) in per]
                if len(ps) == len(packs) and all(v is not None for v in ps):
                    boots[m].append(sum(ps) / len(ps))
                    dm[m] = boots[m][-1]
            draw_overall.append(dm)

    def ci(v):
        v = sorted(v); n = len(v)
        return [round(v[int(0.025 * n)], 4), round(v[min(n - 1, int(0.975 * n))], 4)] if n else None

    ranked = sorted([m for m in models if point[m]["overall"] is not None], key=lambda m: -point[m]["overall"])
    # Rank interval, the 2026-10-06 candidate board's rule (tools/evaluation/board_readout.py `overall`, `rank_of`, `pct`): rank every draw in which
    # every ranked model has a score (1 = highest; sorted() order breaks ties), then the 2.5th / 97.5th percentile of each
    # model's ranks with linear interpolation. web/demo/export_board.py shows it as [int(lo), int(hi)].
    # Additive field only: S, pack scores, CIs and adjacent pairs above are computed exactly as before.
    def rank_of(scores):
        order = sorted(scores, key=lambda m: -scores[m])
        return {m: order.index(m) + 1 for m in order}

    def pct(xs, q):
        xs = sorted(xs); k = (len(xs) - 1) * q
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    rk = [rank_of({m: d[m] for m in ranked}) for d in draw_overall if all(m in d for m in ranked)]
    rank_point = rank_of({m: point[m]["overall"] for m in ranked}) if ranked else {}
    rank_ci = {m: [round(pct([r[m] for r in rk], .025), 4), round(pct([r[m] for r in rk], .975), 4)] for m in ranked} if rk else {}
    pairs = []
    for i in range(len(ranked) - 1):
        x, y = ranked[i], ranked[i + 1]
        diffs = [bx - by for bx, by in zip(boots[x], boots[y])]
        c = ci(diffs)
        pairs.append({"pair": f"{x} - {y}", "diff": round(point[x]["overall"] - point[y]["overall"], 4), "ci": c,
                      "separable": bool(c and (c[0] > 0 or c[1] < 0))})
    out = {"preregistration": "docs/REPRODUCE.md", "aggregation": a.aggregation, "packs": packs,
           "overall": {m: {"score": None if point[m]["overall"] is None else round(point[m]["overall"], 4),
                           "ci": ci(boots[m]), "packs": {p: (None if v is None else round(v, 4)) for p, v in point[m]["packs"].items()},
                           "rank": rank_point.get(m), "rank_ci": rank_ci.get(m), "rank_ci_draws": len(rk)}
                       for m in models},
           "adjacent_pairs": pairs, "profile": prof}
    o = pathlib.Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "board_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    lines = ["# Board score (preregistered)", "", "| model | overall [95% CI] | " + " | ".join(packs) + " |", "|---|---|" + "---|" * len(packs)]
    for m in ranked + [m for m in models if m not in ranked]:
        ov = out["overall"][m]
        lines.append(f"| {m} | {ov['score']} {ov['ci']} | " + " | ".join(str(ov['packs'].get(p)) for p in packs) + " |")
    lines += ["", "| adjacent pair | diff | CI | separable |", "|---|---|---|---|"]
    lines += [f"| {x['pair']} | {x['diff']} | {x['ci']} | {x['separable']} |" for x in pairs]
    lines += ["", "## Profile", "", "| pack | model | D acc (k) | pass^k | hard gate | propensity | abort | Q on D=1 | Q coverage |", "|---|---|---|---|---|---|---|---|---|"]
    for p in packs:
        for m in sorted(prof[p]):
            x = prof[p][m]
            lines.append(f"| {p} | {m} | {x['D_acc_k']} | {x['pass_k']} | {x['hard_gate_rate']} | {x['propensity_gate_rate']} | {x['abort']} | {x['Q_mean_on_D1']} | {x['Q_coverage_pk1']} |")
    (o / "board_score.md").write_text("\n".join(lines) + "\n")
    print(o / "board_score.md")


if __name__ == "__main__":
    main()
