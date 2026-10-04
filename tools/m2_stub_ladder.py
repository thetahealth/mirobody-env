"""m2_stub_ladder.py -- zero-cost validation of the M2 judging with capability-known stubs
(spec section 8, gates Z-1..Z-4). No model is called.

Runs every stub of `haenv.m2.stubs` through the production gated row builder
(`evaluate._row_gated`, which now ends with the M2 round 2) on a deterministic M2 batch, writes the
rows (raw answers included, so any judging fix is a recompute), the M2 board and the gate verdicts.

  PYTHONPATH=<repo>:<repo>/plugins python tools/m2_stub_ladder.py --batch <dir> --out <dir> [--workers 32]
  python tools/m2_stub_ladder.py --rows <out>/rows.jsonl --out <dir>    # re-score stored rows only

Exit code: 0 all gates pass, 1 a gate fails, 2 usage / run error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: The M2 job the tool reads (revision r2: pack ① is `inputs/m2-pack1.job.yaml`); override with HAENV_M2_JOB.
M2_JOB = pathlib.Path(__import__("os").environ.get("HAENV_M2_JOB") or (ROOT / "inputs" / "m2-core.job.yaml"))
for _p in (str(ROOT), str(ROOT / "plugins")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GRADIENT = ("S_p0.0", "S_p0.2", "S_p0.4", "S_p0.6", "S_p0.8", "S_p1.0")
SPLIT = ("S_buy", "S_read")
Z3_CC_MAX = 0.05
Z3_DX_LISTED_MAX = 0.315
#: Revision r3-C (PREREG r3 C): on the raw scale a question-blind or prior stub must rank below
#: the p = 0.4 ladder stub on the composite.
Z3_RANK_REF = "S_p0.4"
#: Prior stubs gated on review_utility_cc <= 0.05 and the rank line: the reviewer's r2 frame stubs,
#: the leave-one-out priors and frame-blind buyers, and the r3 reviewer's leave-one-out cell priors;
#: the in-sample cell priors are reported only.
Z3_FRAME_GATED = ("G_frame_dx", "G_frame_tests", "G_frame_review", "G_frame_all", "G_frame_loo", "G_framesex_loo",
                  "X_fs_loo", "X_fkc_loo", "X_fskc_loo", "X_fsa_loo")
Z3_FRAME_REPORTED = ("G_framesex_in", "X_fs_in", "X_fkc_in", "X_fskc_in")
#: Revision r4: Z-3 is the ceiling for stubs that do not read case data. A stub that buys the frame's
#: candidate panels and reads the results does read it, so it is a rule baseline: printed next to the
#: model scores, never a gate.
Z3_RULE_BASELINE = ("G_frame_buyread", "X_fs_loo_buyread", "X_fkc_loo_buyread", "X_frame_loo_buyread", "X_fs_in_buyread")
#: Z-4 "reading what was bought" (revision r2-3, renamed r3-D1): the buy-and-read stub against the
#: same stub without buying; both buy (or not) by frame, so this gate does not test the choice.
Z4_PAIR = ("S_frame_buyread", "S_frame_nobuy")
Z4_DX_MARGIN = 0.10
Z4_MARGIN = 0.05
BOOT = 10000                      # production board rule (semantic_report.BOARD_RULE)
SEED = 20260929


# ------------------------------------------------------------------ gold bundle for oracle stubs
def gold_bundle(cid, raw, T, sp, vp, quant):
    from haenv.gated import DECOY_SIGNALS, discriminator_tests, test_catalogue
    from haenv.m2 import core
    from haenv.m2.round2 import alias_sets_of
    from haenv.tracks import alias_hit_asserted
    adj = vp.adjudication or {}
    ddx = adj.get("ddx") or {}
    plan = adj.get("revision") or {}
    threads = plan.get("threads") if plan else core.threads_of(ddx)
    sets = alias_sets_of(vp)
    th = []
    for i, t in enumerate(threads or []):
        cands = list(t.get("aliases") or []) + [ddx.get("diagnosis") or ""]
        s = sets[i] if i < len(sets) else cands
        say = next((c for c in cands if c and alias_hit_asserted(c, s)), cands[0])
        th.append({"say": say, "key_signals": list(t.get("key_signals") or []), "component": t.get("component")})
    wrong = []
    allsets = [x for s in sets for x in s]
    for r in core.rival_names(ddx):
        if not alias_hit_asserted(r["name"], allsets):
            wrong.append(r["name"])
    pt = ((plan.get("distractor") or {}).get("points_to") or {}).get("name")
    dname = pt if (pt and not alias_hit_asserted(pt, allsets)) else None
    if not wrong:
        wrong = ["功能性不适"]
    keys = [k for t in th for k in t["key_signals"]]
    mt = core.menu_target_of()
    key_targets = {k: mt[k] for k in dict.fromkeys(keys) if k in mt}
    decoy_for = {k: DECOY_SIGNALS[int(hashlib.sha256(f"{cid}|{k}".encode()).hexdigest(), 16) % len(DECOY_SIGNALS)]
                 for k in key_targets}
    tests = [str(t) for t in (ddx.get("tests") or [])]
    cat = [t for t in (core.m2_catalogue() if core.is_m2_framed(raw) else test_catalogue() + discriminator_tests())
           if t not in tests]
    test_wrong = [cat[int(hashlib.sha256(f"{cid}|tw|{k}".encode()).hexdigest(), 16) % len(cat)]
                  for k in range(len(tests))]
    fr = adj.get("m2_frame") or {}
    rcv = core.f3_truth(raw.longitudinal_data or {}, fr, int(T)) if fr.get("frame") == "F3" else None
    q = None
    if quant:
        t = quant.get("truth")
        if isinstance(t, str):
            w = "flat" if t != "flat" else "rising"
        else:
            try:
                w = int(t) + 7
            except (TypeError, ValueError):
                w = 0
        q = {"truth": t, "wrong": w, "kind": quant.get("kind")}
    join = ddx.get("join_gold") or "unified"
    sex = str(((getattr(raw, "user_profile", None) or {}).get("sex")) or "").upper()[:1] or None
    return {"cid": cid, "sex": sex, "threads": th, "wrong_names": wrong, "distractor_name": dname,
            "key_targets": key_targets,
            "decoy_for": decoy_for, "tests": tests, "test_wrong": test_wrong,
            "warranted": bool(adj.get("clinician_action_warranted")),
            "insufficient": bool(ddx.get("insufficient")), "join": join,
            "join_wrong": "unified" if join != "unified" else "comorbidity",
            "urgency": ddx.get("urgency"), "frame": fr.get("frame"),
            "med_gold": fr.get("med_direction"), "rcv_gold": (rcv or {}).get("exceeds_rcv"),
            "rcv_stream": fr.get("followed_stream"), "quant": q,
            "ev": [e["evidence_id"] for e in (sp.evidence_ledger or [])][:3],
            "push_items": list(plan.get("items") or [])}


def blind_constants(golds: dict) -> dict:
    """M1-style question-blind optimum: most frequent gold values over the batch."""
    from collections import Counter
    names = Counter(t["say"] for g in golds.values() for t in g["threads"])
    tests = Counter(t for g in golds.values() for t in g["tests"])
    joins = Counter(g["join"] for g in golds.values() if not g["insufficient"])
    urg = Counter(g["urgency"] for g in golds.values() if g["frame"] == "F1")
    med = Counter(g["med_gold"] for g in golds.values() if g["frame"] == "F2")
    rcv = Counter(g["rcv_gold"] for g in golds.values() if g["frame"] == "F3" and g["rcv_gold"] is not None)
    qk: dict = {}
    for g in golds.values():
        if g["quant"]:
            qk.setdefault(g["quant"]["kind"], Counter())[json.dumps(g["quant"]["truth"])] += 1
    # Five names: under the n + 1 cap only the first two or three count (the M1 blind reading);
    # the list is long enough that the cap mutation (n + 3) can show up.
    return {"dx_list": [n for n, _ in names.most_common(5)], "tests": [t for t, _ in tests.most_common(6)],
            "join": joins.most_common(1)[0][0], "urgency": urg.most_common(1)[0][0] if urg else "🔴",
            "med": med.most_common(1)[0][0] if med else "investigate_first",
            "rcv": rcv.most_common(1)[0][0] if rcv else False,
            "quant": {k: json.loads(c.most_common(1)[0][0]) for k, c in qk.items()}}


# ------------------------------------------------------------------ run
_W: dict = {}


def load_built(batch_dir: pathlib.Path) -> dict:
    """The batch's cases, with the Q-side ledger registered back (`store.load_cases`)."""
    from haenv.store import load_cases
    return load_cases(batch_dir / "cases.jsonl")


def run_case(args):
    cid, names = args
    from haenv import evaluate as ev
    from haenv.gold_kinds import derive
    from haenv_kernel.build import build_instance
    raw = _W["built"][cid]
    T = int(raw.prediction_context["prediction_time_T"])
    sp, vp = build_instance(raw, T)
    kind = derive(vp)
    out = []
    for sname in names:
        make = _W["factories"][sname]
        t0 = time.time()
        try:
            row = ev._row_gated(cid, sname, raw, T, make(), ctx=_W["ctx"])
        except Exception as e:                        # noqa: BLE001
            import traceback
            row = {"case": cid, "solver": sname, "overall": "ERROR", "error": repr(e),
                   "traceback": traceback.format_exc()[-3000:]}
        row["gold_kind"] = kind
        row["m2_frame"] = row.get("m2_frame") or ((vp.adjudication or {}).get("m2_frame") or {}).get("frame")
        row["m2_insufficient"] = kind == "ddx:insufficient"
        if row.get("review_warranted") is None:      # an aborted row: the review gold, for r3-A4'
            row["review_warranted"] = bool((vp.adjudication or {}).get("clinician_action_warranted"))
        row["cell_s"] = round(time.time() - t0, 3)
        out.append(row)
    return out


def all_factories():
    from haenv.m2 import stubs
    fac = dict(stubs.ladder() + stubs.split() + stubs.randoms() + stubs.consts()
               + stubs.frame_gate_stubs() + stubs.blind_buyer() + stubs.z4_pair() + stubs.z5_pairs())
    return fac


def setup_globals(batch_dir: pathlib.Path):
    import logging
    logging.disable(logging.WARNING)
    from haenv import evaluate as ev
    from haenv.job import load_job
    from haenv.m2 import stubs
    import haenv  # noqa: F401  (mounts the kernel on sys.path)
    from haenv_kernel.build import build_instance
    job = load_job(M2_JOB, root=ROOT)
    built = load_built(batch_dir)
    from haenv.run_state import RunContext
    ctx = RunContext(resp_path=None, probes=ev.load_probes(job.root / "probes"))
    ev.assign_premises(built, ctx=ctx)
    ev.assign_qside_probes(built, answer_t=ev.answer_windows(job, built, batch_dir), ctx=ctx)
    golds = {}
    for cid, raw in built.items():
        T = int(raw.prediction_context["prediction_time_T"])
        sp, vp = build_instance(raw, T)
        golds[cid] = gold_bundle(cid, raw, T, sp, vp, ctx.quant_for.get(cid))
    stubs.GOLD.clear()
    stubs.GOLD.update(golds)
    stubs.BLIND.clear()
    stubs.BLIND.update(blind_constants(golds))
    _W.update(built=built, factories=all_factories(), job=job, ctx=ctx)
    return job, built


def run_all(batch_dir: pathlib.Path, workers: int, cases=None, names=None) -> list[dict]:
    setup_globals(batch_dir)
    names = list(names or _W["factories"])
    cids = sorted(cases or _W["built"])
    tasks = [(c, names) for c in cids]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            parts = pool.map(run_case, tasks, chunksize=1)
    else:
        parts = [run_case(t) for t in tasks]
    return [r for p in parts for r in p]


# ------------------------------------------------------------------ gates
def _boot_sig(rows, models, alpha=0.05, boot=BOOT, seed=SEED, score_fn=None):
    from haenv.m2.score import composite_scores
    from haenv.semantic_report import _significant, bootstrap_scores
    score_fn = score_fn or composite_scores
    rs = [r for r in rows if r["solver"] in models]
    cases = sorted({r["case"] for r in rs})
    point = {m: s for m, s in score_fn(rs).items() if s is not None}
    samples = bootstrap_scores(rs, cases, score_fn=score_fn, boot=boot, seed=seed, workers=min(32, os.cpu_count() or 1))
    sig = _significant(point, samples, alpha)
    return point, sig


#: Ladder readings kept as profile, not gate (D3 of the four-pack consolidation, 2026-10-03): Z-2 could
#: not be met at B=1000 by construction, Z-4's margin had been redefined twice, Z-5 never turned red.
PROFILE_ONLY = ("Z2", "Z4", "Z5")


def gates(rows: list[dict], *, boot=BOOT, score_kw=None) -> dict:
    from haenv.m2 import score
    from haenv.m2.score import CC_DIMS, SCORED_CASE_DIMS, SCORED_DIMS
    kw = dict(score_kw or {})
    b = score.board(rows, **kw)
    out: dict = {"board": b}
    # Z-1
    grad = [s for s in GRADIENT if s in b]
    comp = [b[s]["composite_raw"] for s in grad]
    # revision r4: the composite is floored at 0, so two below-chance levels tie there; any other tie fails
    strict = all(a < b or a == b == 0.0 for a, b in zip(comp, comp[1:])) and len(comp) == len(GRADIENT)
    dim_mono = {}
    for d in SCORED_DIMS:
        v = [b[s].get(d) for s in grad]
        dim_mono[d] = all(v[i] is None or v[i + 1] is None or v[i] <= v[i + 1] + 1e-12 for i in range(len(v) - 1))
    out["Z1"] = {"composites": comp, "strict_increase": strict, "dims_nondecreasing": dim_mono,
                 "pass": strict and all(dim_mono.values())}
    # Z-2
    def _score_fn(rs):
        return {s: v["composite_raw"] for s, v in score.board(rs, **kw).items()}
    models = [m for m in GRADIENT + SPLIT if m in b]
    if boot:
        point, sig = _boot_sig(rows, models, boot=boot, score_fn=_score_fn)
    else:                                            # unit tests: bootstrap gates skipped
        point, sig = {}, {}
    need = [(a, c) for i, a in enumerate(grad) for c in grad[i + 1:]
            if float(c[3:]) - float(a[3:]) >= 0.4 - 1e-9]
    sep = {f"{c}>{a}": ((c, a) in sig) for a, c in need}
    out["Z2"] = {"pairs_needed": len(need), "separated": sum(sep.values()), "detail": sep,
                 "n_sig_pairs_total": len(sig), "pass": all(sep.values()) if boot else None}
    # Z-3 (PREREG r3 C): raw scale; chance-corrected dims <= 0.05, the rank line against S_p0.4
    ref = (b.get(Z3_RANK_REF) or {}).get("composite_raw")
    ok3 = True

    def _blind(v, *, consts: bool):
        ccs = {d: v.get(d) for d in CC_DIMS if v.get(d) is not None}
        fails = [d for d, x in ccs.items() if x > Z3_CC_MAX and (consts or d == "review_utility_cc")]
        if ref is not None and v["composite_raw"] is not None and v["composite_raw"] >= ref:
            fails.append(f"composite>={Z3_RANK_REF}")
        if consts and v.get("dx_listed") is not None and v["dx_listed"] > Z3_DX_LISTED_MAX:
            fails.append("dx_listed")
        return {"cc": ccs, "composite": v["composite_raw"], "rank_ref": ref,
                "dims": {d: v.get(d) for d in SCORED_CASE_DIMS}, "fails": fails}

    c_detail = {s: _blind(b[s], consts=True) for s in b if s.startswith("C_")}
    ok3 = not any(v["fails"] for v in c_detail.values())
    f_detail = {}
    for s in Z3_FRAME_GATED + Z3_FRAME_REPORTED:
        if s in b:
            f_detail[s] = {**_blind(b[s], consts=False), "gated": s in Z3_FRAME_GATED}
            if s in Z3_FRAME_GATED:
                ok3 = ok3 and not f_detail[s]["fails"]
    rnd = [s for s in b if s.startswith("S_rand")]
    r_detail = {}
    # Expected value of a uniform random answer: BA = 1/k over k legal values, so the corrected
    # value is (m/k - 1)/(m - 1); it is 0 only when the legal answer space equals the m gold
    # classes (review, abstention, RCV). revision_ba_cc judges a conjunction (decision, lines kept
    # or added, distractor not taken), so a uniform random answer has no closed-form chance level;
    # it is reported, not gated (profile dim either way).
    k_legal = {"review_utility_cc": 2, "abst_utility_cc": 2, "urgency_cc": 4, "med_direction_cc": 4,
               "rcv_cc": 2, "revision_ba_cc": None}
    for d in CC_DIMS:
        xs = [b[s][d] for s in rnd if b[s].get(d) is not None]
        if len(xs) < 2:
            continue
        mu = math.fsum(xs) / len(xs)
        sd = math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
        k = k_legal.get(d)
        exp = None if k is None else (2 / k - 1) / (2 - 1)
        tol = 3 * sd / math.sqrt(len(xs))
        ok = None if exp is None else abs(mu - exp) <= tol
        r_detail[d] = {"mean": mu, "sd": sd, "expected": exp, "tol": tol, "n": len(xs),
                       "pass": ok, "scored": d in SCORED_DIMS}
        # Spec 8 Z-3 reads "every chance-corrected dim", scored or profile (R-17).
        if ok is not None:
            ok3 = ok3 and bool(ok)
    out["Z3"] = {"constants": c_detail, "frame_stubs": f_detail, "random": r_detail,
                 "pass": ok3 and ref is not None}
    out["rule_baseline"] = {s: b[s]["composite_raw"] for s in Z3_RULE_BASELINE if s in b}
    # Z-4 (revision r2-3; r3-D1 "reading what was bought"): buy-and-read against the same stub without buying
    if all(s in b for s in Z4_PAIR):
        bb, bn = b[Z4_PAIR[0]], b[Z4_PAIR[1]]
        ddx = (bb.get("dx_listed") or 0) - (bn.get("dx_listed") or 0)
        diff = bb["composite_raw"] - bn["composite_raw"]
        s2 = _boot_sig(rows, list(Z4_PAIR), boot=boot, score_fn=_score_fn)[1] if boot else None
        sepable = None if s2 is None else (Z4_PAIR[0], Z4_PAIR[1]) in s2
        out["Z4"] = {"pair": Z4_PAIR, "dx_listed": (bb.get("dx_listed"), bn.get("dx_listed")),
                     "tests_recall_f1": (bb.get("tests_recall_f1"), bn.get("tests_recall_f1")),
                     "key_acquired": (bb.get("key_acquired"), bn.get("key_acquired")),
                     "dx_diff": ddx, "dx_threshold": Z4_DX_MARGIN, "diff": diff, "threshold": Z4_MARGIN,
                     "holm_separable": sepable,
                     "pass": ddx >= Z4_DX_MARGIN and diff >= Z4_MARGIN and sepable is not False}
    # Z-5 (revision r3-D3, "choosing what to buy"): at every ladder level the frame-wide buyer's
    # tests dim is at most the targeted buyer's
    lv = [p[3:] for p in GRADIENT if f"D_p{p[3:]}" in b and f"W_p{p[3:]}" in b]
    if lv:
        det = {}
        for p in lv:
            d, w, n = b[f"D_p{p}"], b[f"W_p{p}"], b.get(f"S_p{p}") or {}
            det[p] = {"tests_D": d.get("tests_recall_f1"), "tests_W": w.get("tests_recall_f1"),
                      "tests_nobuy": n.get("tests_recall_f1"), "composite_D": d["composite_raw"],
                      "composite_W": w["composite_raw"],
                      "pass": (w.get("tests_recall_f1") or 0) <= (d.get("tests_recall_f1") or 0) + 1e-12}
        out["Z5"] = {"levels": det, "pass": all(v["pass"] for v in det.values())}
    if all(s in b for s in SPLIT):                   # oracle split: an implementation self-check only
        bb, br = b["S_buy"], b["S_read"]
        out["split_selfcheck"] = {"composite": (bb["composite_raw"], br["composite_raw"]),
                                  "dx_listed": (bb.get("dx_listed"), br.get("dx_listed")),
                                  "key_acquired": (bb.get("key_acquired"), br.get("key_acquired")),
                                  "tests_recall_f1": (bb.get("tests_recall_f1"), br.get("tests_recall_f1")),
                                  "note": "S_read is built to list one line of two; the gap is set by construction"}
    # Z-2, Z-4 and Z-5 are profile readings (decision D3, 2026-10-03): reported, never refusing
    for z in PROFILE_ONLY:
        if z in out:
            out[z]["profile"] = True
    out["pass"] = all(out[z]["pass"] is not False for z in ("Z1", "Z3") if z in out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch")
    ap.add_argument("--rows", nargs="+", help="stored rows (several files are concatenated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--stubs", nargs="*")
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.rows:
        rows = [json.loads(x) for f in a.rows for x in pathlib.Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    elif a.batch:
        rows = run_all(pathlib.Path(a.batch), a.workers, a.cases, a.stubs)
        with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    else:
        ap.error("--batch or --rows")
        return 2
    errs = [r for r in rows if r.get("overall") == "ERROR"]
    g = gates(rows, boot=a.boot)
    g["n_rows"] = len(rows)
    g["n_error_rows"] = len(errs)
    g["wall_s"] = round(time.time() - t0, 1)
    (out / "gates.json").write_text(json.dumps(g, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for z in ("Z1", "Z2", "Z3", "Z4", "Z5"):
        if z in g:
            print(f"[m2_stub_ladder] {z}: {'PASS' if g[z]['pass'] else 'FAIL'}"
                  + (" (profile, not a gate)" if z in PROFILE_ONLY else ""))
    print("[m2_stub_ladder] rule baseline (reads case data, not a gate): "
          + ", ".join(f"{k} {v:.3f}" for k, v in g["rule_baseline"].items() if v is not None))
    print(f"[m2_stub_ladder] rows {len(rows)} · errors {len(errs)} · wall {g['wall_s']}s -> {out / 'gates.json'}")
    if errs:
        return 2
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
