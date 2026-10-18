"""board_readout.py -- the four-pack board from stored batches: per-pack boards with a case x round
two-level bootstrap, every pack on a chance-corrected scale, the overall board under two
aggregation rules, separability and a rank-stability table. Read-only on batches; no model call.

    PYTHONPATH=<repo>:<repo>/plugins python tools/evaluation/board_readout.py --out <dir> \
        --pack m2=<batch prefix> --pack pack2=<prefix> --pack p3=<prefix> --pack p4=<prefix> \
        [--reps 1,2,3] [--boot 2000] [--workers 16] [--m2-stub-rows <rows.jsonl>] \
        [--items <batch prefix>=<job.yaml> ...]

A batch prefix `<dir>/<stamp>` reads the rounds `<stamp>-pk1..pkR`. A pack whose items were run in
several batches is given as `--pack <pack>=<prefix A>+<prefix B>`: the batches' item sets must be
disjoint, they must hold the same models and rounds, and their rows are read as one pack. When a batch
holds more items than it owns (a byte copy of an earlier batch), `--items <prefix>=<job.yaml>` limits it
to the items that job file lists.

Pack scores (production scorers): m2 `haenv.m2.score.composite_scores`, pack2
`haenv.pack2.score.composite`, p3 `haenv.p3.score.pack_score`, p4 `haenv.p4.score.pack_score`.
Rebuilt p2 may replace pack2: use `--pack p2=<prefix>` alongside m2/p3/p4.
Its headline is `haenv.p2.score.composite` (cost_skill relative to the best
constant, with the non-compensatory emergency gate); cost_cc, under and over
rates remain separate profile diagnostics. Legacy pack2 reads are unchanged.
Legacy pack2 and p3/p4 use chance-corrected balanced accuracies (a constant or random answer scores 0). The m2
composite is on the raw scale, so it is mapped with the random-menu stub measured on the same pack
(`haenv.m2.stubs.randoms`, run through the production row builder by `tools/testing/m2_stub_ladder.py`):
cc = (score - stub) / (1 - stub), stub = mean composite of the random stubs. The stub rows are
written next to the readout, so a judging change re-scores them instead of reusing a number.
M2 always uses the current scoring recipe; reproduce 1.3 with the archived release.
Supplied random-stub rows need `<rows.jsonl>.meta.json`, equal to
`m2_baseline_metadata(prefixes, cases, reps, items)`: the selected cases, source-case
hashes, repetitions and runtime M2 stamp. Every random stub must cover those cells.
The readout records scoring method, dimensions and fingerprint explicitly.

Overall board, models ranked in every pack:
  equal     mean of the four baseline-corrected pack scores
  sd_equal  mean of the baseline-corrected pack scores each divided by that pack's between-model SD
Each bootstrap draw combines independent per-pack draws (seed = 20261004 + crc32(pack)). The
rank table gives each model's rank and bootstrap 95% rank interval under both rules; a rank
"holds" when the two rules give the same rank. Only those ranks are claims.

No pack score is floored (one rule for the four packs). Unanswered cells of packs 2-4 enter as
a row with the case's gold and no answer (scores as wrong); m2 rows are taken as written (the m2
scorer handles ABORT rows). m2 rows carry the semantic view when the batch has one.
SYNTHETIC data, evaluation only, not medical advice.
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

import argparse
import collections
import functools
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import zlib

ROOT = _repository_root(__file__)
for _p in (str(ROOT), str(ROOT / "plugins"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PACKS = ("m2", "pack2", "p3", "p4")
P2_PACKS = ("m2", "p2", "p3", "p4")
SAID = {"pack2": "pack2_disposition_said", "p3": "p3_class_said", "p4": "p4_class_said", "p2": "p2_said"}
GOLD = {"pack2": "pack2_disposition_gold", "p3": "p3_class_gold", "p4": "p4_class_gold", "p2": "p2_set"}
OK = {"pack2": "pack2_disposition_ok", "p3": "p3_class_ok", "p4": "p4_class_ok", "p2": "p2_in_set"}
SEED = 20261004
N_RANDOM_STUBS = 20
_P4_BASE: dict = {}


@functools.lru_cache(maxsize=1)
def m2_score_contract() -> dict:
    """The runtime scorer's identity, cached once for the immutable bootstrap run.

    No proposed-test weight or fixed historical chance level is defined here.
    The coordinator's scored-dimension change is read from the production module.
    """
    from haenv.m2 import score
    from haenv.m2.hooks import m2_judging_fingerprint
    fingerprint = m2_judging_fingerprint()
    return {"scored_case_dims": list(score.SCORED_CASE_DIMS),
            "scored_solver_dims": list(score.SCORED_SOLVER_DIMS),
            "m2_judging_sha16": fingerprint,
            "method": "production composite_scores: applicable case dimensions and hard-gate multiplier",
            "normalization": "(composite - same-pack random-menu stub) / (1 - stub)"}


def board_packs(names) -> tuple:
    """Exactly four complementary packs: legacy pack2 or rebuilt p2, never both."""
    for packs in (PACKS, P2_PACKS):
        if set(names) == set(packs):
            return packs
    raise ValueError(f"--pack must name each of {PACKS} or {P2_PACKS}")


def pack_seed(pack: str) -> int:
    return SEED + zlib.crc32(pack.encode())


def is_model(solver: str) -> bool:
    from haenv.world.generation.baselines import BASELINE_NAMES
    return solver not in BASELINE_NAMES and not str(solver).startswith(("S_", "G_", "X_", "C_", "D_", "W_"))


def load(pack: str, prefix: pathlib.Path, reps) -> list[dict]:
    """Last row per (case, solver) of each round, real models only, tagged rep=i."""
    out = []
    for i in reps:
        d = prefix.parent / f"{prefix.name}-pk{i}"
        f = d / "eval.jsonl"
        if not f.is_file():
            continue
        last = {}
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if is_model(r.get("solver")):
                    last[(r["case"], r["solver"])] = r
        rows = list(last.values())
        if pack == "m2":
            rows = overlay_semantic(rows, d)
        out += [{**r, "rep": i} for r in rows]
    return out


def overlay_semantic(rows, d):
    from haenv.provenance.judging_provenance import view_for_batch
    from haenv.scoring.semantic.rubric import load_policy
    sd = d / "semantic" / load_policy()["version"]
    if (sd / "manifest.json").is_file():
        return view_for_batch(rows, d, sd)
    return rows


def cases_of(prefix: pathlib.Path, reps) -> list[str]:
    f = prefix.parent / f"{prefix.name}-pk{reps[0]}" / "cases.jsonl"
    return sorted(json.loads(x)["case_id"] for x in open(f, encoding="utf-8") if x.strip())


def pack_rows(pack: str, prefixes, items: dict, reps):
    """Rows and case ids of one pack from one or more batch prefixes. Every prefix contributes the items
    it owns (all its cases, or those listed for it in `items`); the parts must not overlap and must
    hold the same models and rounds."""
    rows, cases, parts = [], [], []
    for p in prefixes:
        part_rows, part_cases = load(pack, p, reps), cases_of(p, reps)
        keep = items.get(str(p))
        if keep is not None:
            missing = sorted(keep - set(part_cases))
            if missing:
                raise SystemExit(f"{p}: --items names cases the batch does not hold: {missing[:3]}")
            part_rows = [r for r in part_rows if r["case"] in keep]
            part_cases = [c for c in part_cases if c in keep]
        parts.append((p, part_rows, part_cases))
        rows += part_rows
        cases += part_cases
    if len(parts) > 1:
        if len(set(cases)) != len(cases):
            dup = sorted({c for c in cases if cases.count(c) > 1})
            raise SystemExit(f"the batches of pack {pack} share items: {dup[:3]}")
        for p, part_rows, _ in parts:
            if {r["solver"] for r in part_rows} != {r["solver"] for r in parts[0][1]} or \
               {r["rep"] for r in part_rows} != set(reps):
                raise SystemExit(f"{p}: the batches of pack {pack} do not hold the same models and rounds")
    return rows, sorted(cases)


def unanswered(row):
    from tools.evaluation.rank_rule import unanswered_reason
    return unanswered_reason(row)


def fill(pack, rows, cases, models, reps):
    """Packs 2-4: every (case, model, rep) cell present; a missing / answerless cell gets the case's
    gold and no answer."""
    if pack == "m2":
        return rows
    cell = {(r["case"], r["solver"], r["rep"]): r for r in rows}
    gold_of = {}
    for r in rows:
        if r.get(GOLD[pack]) is not None:
            gold_of.setdefault(r["case"], r)
    out = []
    for c in cases:
        peer = gold_of.get(c)
        for m in models:
            for i in reps:
                r = cell.get((c, m, i))
                if r is not None and r.get(GOLD[pack]) is not None and unanswered(r) is None:
                    out.append(r)
                    continue
                if peer is None:
                    if pack == "p2":
                        raise ValueError(f"p2 case {c}: no stored acceptable-set gold for unanswered cells")
                    continue
                if pack == "p2":
                    from haenv.p2.score import atoms
                    gold = {k: peer.get("p2_" + k) for k in
                            ("set", "floor", "rule", "tier", "stratum", "wording")}
                    out.append({"case": c, "solver": m, "rep": i, "overall": "NO_ANSWER",
                                **atoms({}, gold), "synthetic_no_answer": True})
                    continue
                keep = {k: v for k, v in peer.items() if k.startswith(pack + "_") and
                        any(t in k for t in ("gold", "drug", "analyte", "obs_pct", "focal", "hidden", "wording",
                                              "prior_tier"))}
                z = {"case": c, "solver": m, "rep": i, "overall": "NO_ANSWER", **keep, SAID[pack]: None,
                     OK[pack]: 0, "synthetic_no_answer": True}
                if pack == "p3":
                    z["p3_gate_med_change"] = 0
                if pack == "p4":
                    z["p4_pct_abs_err"] = abs(float(peer["p4_pct_gold"]))
                    z["p4_next_ok"] = 0
                if pack == "pack2" and peer["pack2_disposition_gold"] in ("ed_now", "within_24h"):
                    z["pack2_under_triage"] = 1
                out.append(z)
    return out


def by_solver(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["solver"]].append(r)
    return d


def sf_m2(rows):
    from haenv.m2.score import composite_scores
    return composite_scores(rows)


def sf_pack2(rows):
    from haenv.pack2.score import composite
    return {s: composite(rs)["composite"] for s, rs in by_solver(rows).items()}


def sf_p2(rows):
    from haenv.p2.score import composite
    return {s: composite(rs)["composite"] for s, rs in by_solver(rows).items()}


def sf_p3(rows):
    from haenv.p3.score import pack_score
    return {s: pack_score(rs)["composite"] for s, rs in by_solver(rows).items()}


def sf_p4(rows):
    from haenv.p4.score import pack_score
    return {s: pack_score(rs, _P4_BASE["mae"])["composite"] for s, rs in by_solver(rows).items()}


SF = {"m2": sf_m2, "pack2": sf_pack2, "p3": sf_p3, "p4": sf_p4, "p2": sf_p2}


def profile(pack, rows):
    """Per-model readings shown next to the board (not in the pack score)."""
    out = {}
    for s, rs in by_solver(rows).items():
        if pack == "m2":
            from haenv.m2.score import solver_scores
            v = solver_scores(rs)
            out[s] = {k: v.get(k) for k in ("tests_recall_f1", "dx_listed", "key_covered", "review_utility_cc",
                                            "review_f2_cc", "quant_ok", "key_acquired", "gate_mult")}
            # an empty column says so instead of printing a blank (no insufficient-data item in the pack)
            out[s]["abst_utility_cc"] = v.get("abst_utility_cc") if v.get("abst_utility_cc") is not None \
                else "not applicable"
        elif pack == "p2":
            from haenv.p2.score import composite
            v = composite(rs)
            out[s] = {k: v.get(k) for k in ("cost_skill", "hard_gate_rate", "cost_cc", "under_rate",
                                           "over_rate", "in_set_rate", "in_set_single_gold", "in_set_borderline")}
        elif pack == "pack2":
            from haenv.pack2.score import composite
            v = composite(rs)
            out[s] = {k: v.get(k) for k in ("disposition_cc", "hard_gate_rate", "under_triage_rate",
                                            "faster_than_gold_rate")}
        elif pack == "p3":
            from haenv.p3.score import pack_score
            v = pack_score(rs)
            out[s] = {"med_decision_cc": v["med_decision_cc"], "own_gate_rate": v["gate_rate"],
                      "new_dose_adjacent": v["profile"]["new_dose_adjacent"]}
        else:
            from haenv.p4.score import pack_score
            v = pack_score(rs, _P4_BASE["mae"])
            out[s] = {"change_source_cc": v["change_source_cc"], "true_change_skill": v["true_change_skill"],
                      "next_step_agree": v["profile"]["next_step_agree"]}
    return out


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ------------------------------------------------------------------ m2 random-menu stub
def m2_baseline_metadata(prefixes, cases, reps, items) -> dict:
    """Current-pack identity for an offline baseline, including source cases of every rep."""
    if not cases or not reps or len(set(reps)) != len(reps):
        raise ValueError("M2 baseline needs cases and unique repetitions")
    hashes = {}
    for p in prefixes:
        selected = set(cases)
        if str(p) in items:
            selected &= items[str(p)]
        for rep in reps:
            f = p.parent / f"{p.name}-pk{rep}" / "cases.jsonl"
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["case_id"] in selected:
                    key = f"{rep}|{r['case_id']}"
                    if key in hashes:
                        raise ValueError(f"M2 baseline source case overlaps: {key}")
                    hashes[key] = hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                                            separators=(",", ":")).encode()).hexdigest()
    if set(hashes) != {f"{rep}|{c}" for c in cases for rep in reps}:
        raise ValueError("M2 baseline source cases do not cover the selected pack/repetitions")
    return {"cases": sorted(cases), "reps": list(reps), "case_sha256": hashes,
            "m2_judging_sha16": m2_score_contract()["m2_judging_sha16"]}


def validate_m2_baseline(rows, metadata) -> None:
    """Every supplied random stub must cover the same case x rep cells exactly once."""
    randoms = [r for r in rows if isinstance(r, dict) and str(r.get("solver", "")).startswith("S_rand")]
    if not randoms:
        raise ValueError("M2 baseline needs scored random-stub rows, not a fixed numeric level")
    cells = {(c, rep) for c in metadata["cases"] for rep in metadata["reps"]}
    for solver, rs in by_solver(randoms).items():
        got = collections.Counter((r.get("case"), r.get("rep")) for r in rs)
        if set(got) != cells or any(n != 1 for n in got.values()):
            raise ValueError(f"M2 baseline {solver} must cover the SAME selected cases/reps exactly once")
        if any("m2_judging_sha16" in r and r["m2_judging_sha16"] != metadata["m2_judging_sha16"] for r in rs):
            raise ValueError(f"M2 baseline {solver} has a different M2 stamp")


def m2_stub_rows(prefixes, reps, out_dir: pathlib.Path, workers: int, given=None, *, cases, items) -> list[dict]:
    """Validate supplied baselines; otherwise generate every selected case/rep offline."""
    expected = m2_baseline_metadata(prefixes, cases, reps, items)
    if given:
        meta_path = pathlib.Path(str(given) + ".meta.json")
        if not meta_path.is_file():
            raise ValueError(f"M2 baseline needs {meta_path}; regenerate it for the SAME pack cases/reps/M2 stamp")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError("M2 baseline metadata differs from the SAME pack cases/reps/M2 stamp")
        rows = [json.loads(x) for x in pathlib.Path(given).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        if len(prefixes) != 1:
            raise ValueError("M2 baseline for a split pack must be supplied with --m2-stub-rows and its metadata")
        p = prefixes[0]
        rows = []
        for rep in reps:
            batch = p.parent / f"{p.name}-pk{rep}"
            job_id = json.loads((batch / "batch.json").read_text(encoding="utf-8"))["job_id"]
            os.environ.setdefault("HAENV_M2_JOB", str(ROOT / 'benchmarks' / 'inputs' / f'{job_id}.job.yaml'))
            from tools.testing import m2_stub_ladder as L
            names = [f"S_rand{s:02d}" for s in range(N_RANDOM_STUBS)]
            rows += [{**r, "rep": rep} for r in L.run_all(batch, workers, cases, names)]
    validate_m2_baseline(rows, expected)
    with (out_dir / "m2_stub_rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out_dir / "m2_stub_rows.jsonl.meta.json").write_text(json.dumps(expected, ensure_ascii=False, indent=1) + "\n")
    return rows


def m2_stub_level(stub_rows) -> dict:
    from haenv.m2.score import composite_scores
    sc = {s: v for s, v in composite_scores([r for r in stub_rows if str(r.get("solver", "")).startswith("S_rand")]).items()
          if v is not None}
    if not sc or not all(math.isfinite(v) for v in sc.values()):
        raise ValueError("M2 baseline has no finite random-stub scores under the current recipe")
    level = statistics.fmean(sc.values())
    if level >= 1.0:
        raise ValueError("M2 baseline cannot normalize scores: random-stub level is >= 1")
    return {"stub": level, "n_stubs": len(sc),
            "min": min(sc.values()) if sc else None, "max": max(sc.values()) if sc else None,
            "n_error_rows": sum(r.get("overall") == "ERROR" for r in stub_rows)}


# ------------------------------------------------------------------ boards
def board(pack, rows, ranked, cases, reps, boot, workers):
    from haenv.scoring.board_twolevel import bootstrap_scores_twolevel
    from haenv.scoring.board_twolevel import pooled_rows
    rows = [r for r in rows if r["solver"] in ranked]
    point = SF[pack](pooled_rows(rows))
    samples = bootstrap_scores_twolevel(rows, cases, list(reps), score_fn=SF[pack], boot=boot, seed=pack_seed(pack),
                                        mode="crossed", workers=workers)
    return point, samples


def table(point, samples, models):
    tab = []
    for m in models:
        xs = [s[m] for s in samples if s.get(m) is not None]
        tab.append({"model": m, "score": point.get(m), "ci95": [pct(xs, .025), pct(xs, .975)] if xs else None,
                    "se": statistics.stdev(xs) if len(xs) > 1 else None})
    tab.sort(key=lambda x: -(x["score"] if x["score"] is not None else -9))
    return tab


def sep(tab, samples, k=6):
    """Pairwise separability: |diff| > 2 SE of the paired bootstrap difference. `pairs` covers the top k
    models, `all_pairs` every model pair of the board (the matrix the demo page shows)."""
    ranked = [x["model"] for x in tab if x["score"] is not None]
    score = {x["model"]: x["score"] for x in tab}

    def pairs_of(models):
        out = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                d = [s[a] - s[b] for s in samples if s.get(a) is not None and s.get(b) is not None]
                se = statistics.stdev(d)
                diff = score[a] - score[b]
                out.append({"a": a, "b": b, "diff": diff, "se": se, "separable": abs(diff) > 2 * se})
        return out

    top = ranked[:k]
    pairs = pairs_of(top)
    return {"top": top, "pairs": pairs, "n_separable": sum(p["separable"] for p in pairs),
            "adjacent_separable": sum(p["separable"] for p in pairs if top.index(p["b"]) == top.index(p["a"]) + 1),
            "all_pairs": pairs_of(ranked)}


def rank_of(scores: dict) -> dict:
    order = sorted(scores, key=lambda m: -scores[m])
    return {m: order.index(m) + 1 for m in order}


def overall(cc_point: dict, cc_samples: dict, models: list[str], packs=PACKS) -> dict:
    """Overall boards under `equal` and `sd_equal`, with the rank-stability table."""
    nb = min(len(cc_samples[p]) for p in packs)
    sd = {p: statistics.stdev([cc_point[p][m] for m in models]) for p in packs}
    # a pack on which every model scores the same separates nothing: it gets weight 0, not 1 / 0
    w = {"equal": {p: 1.0 for p in packs}, "sd_equal": {p: 1.0 / sd[p] if sd[p] > 0 else 0.0 for p in packs}}
    res = {"pack_sd": sd, "models": models}
    ranks = {}
    for rule, wt in w.items():
        pts = {m: statistics.fmean(wt[p] * cc_point[p][m] for p in packs) for m in models}
        samp = [{m: statistics.fmean(wt[p] * cc_samples[p][b][m] for p in packs) for m in models
                 if all(cc_samples[p][b].get(m) is not None for p in packs)} for b in range(nb)]
        tab = table(pts, samp, models)
        rk = [rank_of(s) for s in samp if len(s) == len(models)]
        ranks[rule] = {m: {"rank": rank_of(pts)[m], "rank_ci95": [pct([r[m] for r in rk], .025),
                                                                    pct([r[m] for r in rk], .975)]} for m in models}
        res[rule] = {"board": tab, "top6": sep(tab, samp)}
    res["rank_table"] = [{"model": m, "rank_equal": ranks["equal"][m]["rank"],
                          "rank_ci95_equal": ranks["equal"][m]["rank_ci95"],
                          "rank_sd_equal": ranks["sd_equal"][m]["rank"],
                          "rank_ci95_sd_equal": ranks["sd_equal"][m]["rank_ci95"],
                          "holds": ranks["equal"][m]["rank"] == ranks["sd_equal"][m]["rank"]}
                         for m in sorted(models, key=lambda m: ranks["equal"][m]["rank"])]
    sep_e = {(p["a"], p["b"]) for p in res["equal"]["top6"]["pairs"] if p["separable"]}
    sep_s = {tuple(sorted((p["a"], p["b"]), key=lambda m: ranks["equal"][m]["rank"]))
             for p in res["sd_equal"]["top6"]["pairs"] if p["separable"]}
    res["separable_under_both"] = sorted(sep_e & sep_s)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory (readout.json, m2_stub_rows.jsonl)")
    ap.add_argument("--pack", action="append", required=True,
                    help="<pack>=<batch prefix>, or <pack>=<prefix A>+<prefix B> for a pack run in several batches")
    ap.add_argument("--items", action="append", default=[], metavar="PREFIX=JOB.yaml",
                    help="limit a batch to the items this job file lists (a byte copy holds the earlier items too)")
    ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-rate", type=float, default=0.95)
    ap.add_argument("--m2-stub-rows", help="same-pack/rep stub rows with matching <rows.jsonl>.meta.json")
    a = ap.parse_args(argv)
    parts = {k: [pathlib.Path(x) for x in v.split("+")] for k, v in (x.split("=", 1) for x in a.pack)}
    prefixes = {k: v[0] for k, v in parts.items()}
    import yaml
    items = {str(pathlib.Path(k)): {c["case_id"] for c in yaml.safe_load(pathlib.Path(v).read_text(encoding="utf-8"))["cases"]}
             for k, v in (x.split("=", 1) for x in a.items)}
    try:
        packs = board_packs(prefixes)
    except ValueError as exc:
        ap.error(str(exc))
    reps = [int(x) for x in a.reps.split(",")]
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"packs": {}, "boot": a.boot, "mode": "crossed (cases x rounds)",
           "seeds": {p: pack_seed(p) for p in packs}, "inputs": {p: "+".join(str(x) for x in v) for p, v in parts.items()}}
    cc_point, cc_samples, ranked_all = {}, {}, {}
    for pack in packs:
        raw, cases = pack_rows(pack, parts[pack], items, reps)
        models = sorted({r["solver"] for r in raw})
        n_cells = len(cases) * len(reps)
        cell = {(r["case"], r["solver"], r["rep"]): r for r in raw}
        ans, fmt = {}, {}
        for m in models:
            got = [cell.get((c, m, i)) for c in cases for i in reps]
            answered = [r for r in got if r is not None and unanswered(r) is None]
            ans[m] = {"answered": len(answered), "of": n_cells, "rate": round(len(answered) / n_cells, 4)}
            unp = sum(1 for r in got if r is not None and "unparseable" in str(r.get("overall")))
            fmt[m] = unp + (sum(1 for r in answered if r.get(GOLD[pack]) is not None and r.get(SAID[pack]) is None)
                            if pack in SAID else 0)
        rows = fill(pack, raw, cases, models, reps)
        if pack == "p4":
            from haenv.p4.score import naive_base
            g = {}
            for r in rows:
                if r.get("p4_obs_pct") is not None:
                    g.setdefault(r["case"], (float(r["p4_pct_gold"]), float(r["p4_obs_pct"])))
            nb = naive_base([v[0] for v in g.values()], [v[1] for v in g.values()])
            _P4_BASE["mae"] = nb["base_mae"]
            res["p4_naive_base"] = nb
        ranked = [m for m in models if ans[m]["rate"] >= a.min_rate]
        ranked_all[pack] = ranked
        stub = None
        if pack == "m2":
            stamp = m2_score_contract()["m2_judging_sha16"]
            if any(r.get("m2_judging_sha16", stamp) != stamp for r in raw):
                raise ValueError("M2 baseline/readout rows have a different M2 stamp; recompute with the final recipe")
            lvl = m2_stub_level(m2_stub_rows(parts[pack], reps, out_dir, a.workers, a.m2_stub_rows,
                                           cases=cases, items=items))
            res["m2_stub"] = lvl
            res["m2_scoring"] = m2_score_contract()
            res["m2_baseline"] = m2_baseline_metadata(parts[pack], cases, reps, items)
            stub = lvl["stub"]
        point, samples = board(pack, rows, ranked, cases, reps, a.boot, a.workers)
        from haenv.m2.score import chance_corrected
        cc = (lambda x: chance_corrected(x, stub)) if pack == "m2" else (lambda x: x)
        cc_point[pack] = {m: cc(point.get(m)) for m in ranked}
        cc_samples[pack] = [{m: cc(s.get(m)) for m in ranked if s.get(m) is not None} for s in samples]
        tab = table(point, samples, ranked)
        tab_cc = table(cc_point[pack], cc_samples[pack], ranked)
        per_round = {i: SF[pack]([r for r in rows if r["rep"] == i and r["solver"] in ranked]) for i in reps}
        allwrong = []
        if pack in OK:
            for c in cases:
                cr = [r for r in rows if r["case"] == c and r["solver"] in ranked]
                if cr and all(not r.get(OK[pack]) for r in cr):
                    allwrong.append(c)
        res["packs"][pack] = {"n_cases": len(cases), "answer": ans, "format_fail": fmt, "ranked": ranked,
                              "unranked": [m for m in models if m not in ranked], "board_raw": tab,
                              "board_cc": tab_cc, "chance_level": stub if pack == "m2" else 0.0,
                              "per_round": per_round, "all_wrong": allwrong,
                              "profile": profile(pack, [r for r in rows if r["solver"] in ranked]),
                              "top6": sep(tab_cc, cc_samples[pack])}
        if pack == "p2":
            res["packs"][pack]["score_metric"] = "cost_skill with non-compensatory emergency gate"
    common = [m for m in ranked_all["m2"] if all(m in ranked_all[p] and cc_point[p].get(m) is not None for p in packs)]
    res["overall"] = overall(cc_point, cc_samples, common, packs=packs)
    (out_dir / "readout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for p in packs:
        print(p, [(x["model"], round(x["score"], 3)) for x in res["packs"][p]["board_cc"]])
    for rule in ("equal", "sd_equal"):
        print("overall", rule, [(x["model"], round(x["score"], 3)) for x in res["overall"][rule]["board"]])
    print("ranks that hold under both:", [r["model"] for r in res["overall"]["rank_table"] if r["holds"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
