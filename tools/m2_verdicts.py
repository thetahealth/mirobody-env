"""m2_verdicts.py -- the M2 acceptance verdicts V-1 .. V-5 (spec section 9), mechanically, from rows.

Pre-registered decision script: its sha256 is recorded before the formal scores are looked at
(spec section 12 step 11). Every verdict is a pure function of the rows and the job; no model is
called here.

  PYTHONPATH=<repo>:<repo>/plugins python tools/m2_verdicts.py \\
      --rows <formal eval rows .jsonl, one round>  --anchor <anchored subset rows .jsonl, rep 1..3> \\
      [--batch <the deterministic M2 batch, for the V-3 leak channels>] \\
      [--ladder-rows <stub rows on that batch, for the V-3 rerun of Z-3>] \\
      [--receipts <json {model: usd} or {model: {"solver_usd": .., "judge_usd": ..}}>] \\
      [--top6 m1 m2 ...] [--boot 10000] --out <dir>
  python tools/m2_verdicts.py --draw-anchor          # the 40-case anchored subset (spec 10.1)

  V-1   two steps. Noise anchor: sigma2_mr from the anchored subset (`board_twolevel.drift_variance`);
        its 95 % upper bound's SD <= 0.03 -> intervals use the upper bound, else the point estimate
        and the board is stamped "noise not fully characterised". P6: separable pairs among the fixed
        top six (Holm alpha 0.05 over their 15 pairs, `anchored_drift` board). Fail: P6 < 2. Both
        sigma sets are reported; pairs on which they disagree are marked "pending".
  V-1b  a pair separable on the formal board whose order reverses on the anchored three-round mean by
        more than 2 paired SE. Fail: one such pair.
  V-2   the strongest ranked model's composite on every prior-hard case. Fail: point > 0.50.
  V-3   Z-3 rerun on the formal case set (stub rows), and two leak channels against permutation
        nulls, each as an AUC of gold vs non-gold candidate lines: the frequency prior, read under three
        priors (the v1.0.1 hidden-diagnosis counts, leave-one-case-out counts in this pack, and
        leave-one-case-out counts within the case's frame), and the maximum out-of-band distance of a
        candidate's spectrum findings. Fail: Z-3 fails or any prior's / the distance channel's AUC > its
        null 95 % quantile.
  V-4   receipts: every ranked model's full run (solver + judge). Fail: any model > $20.
  V-5   review_utility_cc: the specificity half's share of the dim's variance across models <= 0.60
        and the single-case G (sigma2_model / (sigma2_model + sigma2_residual)) of "both right" on
        paired cases >= 0.2. Fail on either: the dim is demoted to the profile.

Exit: 0 every evaluated verdict passes, 1 a verdict fails, 2 usage error. A verdict whose input is
missing is reported "not evaluated" (never passed by default).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: The M2 job the tool reads (revision r2: pack ① is `inputs/m2-pack1.job.yaml`); override with HAENV_M2_JOB.
M2_JOB = pathlib.Path(__import__("os").environ.get("HAENV_M2_JOB") or (ROOT / "inputs" / "m2-core.job.yaml"))
for _p in (str(ROOT), str(ROOT / "plugins"), str(ROOT / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Fixed top six (spec 9 V-1): M1 section 12 production board, workup, ability dims, point order.
TOP6 = ("gpt-6-sol", "kimi-k3", "deepseek-v4-pro", "gemini-3.1-pro", "gemini-3.7-flash", "gpt-6-luna")
SIGMA_UPPER_LINE = 0.03
P6_MIN = 2
V2_MAX = 0.50
V4_MAX_USD = 20.0
V5_SPEC_SHARE_MAX = 0.60
V5_G_MIN = 0.2
ANCHOR_N = 40
ANCHOR_SEED = 20261001
PERM = 1000
BOOT = 10000
SEED = 20260929


def _load_rows(p) -> list[dict]:
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _job() -> dict:
    import yaml
    return yaml.safe_load(M2_JOB.read_text(encoding="utf-8"))


def with_completeness(rows: list[dict]) -> list[dict]:
    """The board rule's common-complete filter reads `row["semantic"]`. An M2 row judged by code
    only carries none: its completeness is "the cell has a scored answer" (`overall` not ABORT /
    ERROR). Rows that do carry a semantic block are left as they are."""
    out = []
    for r in rows:
        if "semantic" in r:
            out.append(r)
            continue
        bad = str(r.get("overall") or "").startswith(("ABORT", "ERROR"))
        out.append({**r, "semantic": {"status": "missing_response" if bad else "resolved", "metric_states": {}}})
    return out


def _score_fn(rows):
    from haenv.m2.score import composite_scores
    return composite_scores(rows)


# ------------------------------------------------------------------ anchored subset (spec 10.1)
def anchor_subset(job: dict, n: int = ANCHOR_N, seed: int = ANCHOR_SEED) -> list[str]:
    """40 cases stratified by frame x prior tier, proportional allocation (largest remainder),
    a seeded draw within each stratum. Fixed before any score is seen."""
    pr = job["_provenance"]["m2_prior"]
    strata = defaultdict(list)
    for cid, v in sorted(pr.items()):
        strata[(v["frame"], v["tier"])].append(cid)
    tot = sum(len(v) for v in strata.values())
    quota = {k: n * len(v) / tot for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), k))[: n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    out = []
    for k in sorted(strata):
        out += sorted(rng.sample(sorted(strata[k]), take[k]))
    return sorted(out)


# ------------------------------------------------------------------ V-1 / V-1b
def v1(rows, anchor_rows, top6, *, boot=BOOT, workers=8) -> dict:
    from haenv.board_twolevel import drift_variance, track_board_twolevel
    t6 = [m for m in top6 if any(r["solver"] == m for r in rows)]
    if len(t6) < 2:
        return {"evaluated": False, "why": f"top-six models present: {t6}"}
    main = with_completeness([r for r in rows if r["solver"] in t6])
    anc = with_completeness([r for r in anchor_rows if r["solver"] in t6])
    est = drift_variance(anc, score_fn=_score_fn)
    sd_up = math.sqrt(est["drift_var_ci"][1])
    use = "upper" if sd_up <= SIGMA_UPPER_LINE else "point"
    boards = {d: track_board_twolevel(main, score_fn=_score_fn, boot=boot, seed=SEED, workers=workers,
                                      mode="anchored_drift", anchor_rows=anc, drift=d)
              for d in ("point", "upper")}
    sig = {d: {(p["better"], p["worse"]) for p in b.get("significant_pairs") or []} for d, b in boards.items()}
    p6 = len(sig[use])
    pending = sorted(f"{a}>{b}" for a, b in (sig["point"] ^ sig["upper"]))
    return {"evaluated": True, "drift": est, "sigma_upper_sd": sd_up, "sigma_used": use,
            "stamp": None if use == "upper" else "噪声未充分刻画 (noise not fully characterised)",
            "P6": p6, "P6_point": len(sig["point"]), "P6_upper": len(sig["upper"]), "pending_pairs": pending,
            "pairs": sorted(f"{a}>{b}" for a, b in sig[use]), "top6": t6,
            "boards": {d: b.get("models") for d, b in boards.items()},
            "pass": p6 >= P6_MIN}


def v1b(anchor_rows, pairs: list[str], *, boot=2000) -> dict:
    """Separable formal pairs whose order reverses on the anchored three-round mean by > 2 paired SE."""
    from haenv.semantic_report import bootstrap_scores
    if not pairs:
        return {"evaluated": True, "reversed": [], "pass": True, "note": "no separable pair on the formal board"}
    models = sorted({m for p in pairs for m in p.split(">")})
    rs = [r for r in anchor_rows if r["solver"] in models]
    cases = sorted({r["case"] for r in rs})
    point = _score_fn(rs)
    samples = bootstrap_scores(rs, cases, score_fn=_score_fn, boot=boot, seed=SEED)
    out = []
    for p in pairs:
        a, b = p.split(">")
        d = (point.get(a) or 0) - (point.get(b) or 0)
        ds = [s[a] - s[b] for s in samples if s.get(a) is not None and s.get(b) is not None]
        mu = sum(ds) / len(ds)
        se = math.sqrt(sum((x - mu) ** 2 for x in ds) / (len(ds) - 1)) if len(ds) > 1 else 0.0
        if d < 0 and abs(d) > 2 * se:
            out.append({"pair": p, "anchor_diff": d, "paired_se": se})
    return {"evaluated": True, "reversed": out, "pass": not out}


# ------------------------------------------------------------------ V-2
def v2(rows, job, ranked: list[str]) -> dict:
    pr = job["_provenance"]["m2_prior"]
    hard = {c for c, v in pr.items() if v["tier"] == "hard"}
    rs = [r for r in rows if r["case"] in hard and r["solver"] in ranked]
    if not rs:
        return {"evaluated": False, "why": "no rows on prior-hard cases"}
    from haenv.m2.score import board
    sc = {m: v["composite_raw"] for m, v in board(rs).items() if v["composite_raw"] is not None}
    best = max(sc, key=sc.get)
    return {"evaluated": True, "n_hard_cases": len(hard), "scores": sc, "best": best, "best_score": sc[best],
            "pass": sc[best] <= V2_MAX}


# ------------------------------------------------------------------ V-3
def _auc(pos, neg) -> float:
    """P(pos > neg) + 0.5 P(pos == neg), by midranks (equal to the pairwise count)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        mid = (i + j + 1) / 2.0
        rank_sum += mid * sum(y for _, y in allv[i:j])
        i = j
    n_p, n_n = len(pos), len(neg)
    return (rank_sum - n_p * (n_p + 1) / 2.0) / n_p / n_n


def _q95(xs) -> float:
    return sorted(xs)[int(0.95 * len(xs)) - 1]


def _q05(xs) -> float:
    return sorted(xs)[int(0.05 * len(xs)) - 1]


#: The three frequency priors of the V-3 prior channel (all must stay within their nulls):
#:  v101       the v1.0.1 workup pack's hidden-component counts (the prior the v3 recomposition was tuned to);
#:  loo_pack   leave-one-case-out counts in this pack: for case i, candidate c scores the number of OTHER
#:             named cases whose gold carries c;
#:  loo_frame  the same, counted only over the other named cases of case i's frame (F0-F3).
PRIORS = ("v101", "loo_pack", "loo_frame")
#: Revision r2-5 (PREREG r2 F5): the held-out real-prevalence prior -- candidate c scores its
#: `registry/disease_prevalence.yaml` value (the same for every case, null as for v101). The pack-1
#: selection rule never reads it.
PREVALENCE = "prevalence"
#: The priors V-3 gates on (revision r3-G1, the owner's option D): the v1.0.1 frequency prior only.
#: The two leave-one-out priors and the prevalence prior are computed, stored and shown (profile):
#: the prevalence table entered the v3 recomposition (admission, pair weights, greedy order), so it
#: is not held out from the composition.
GATED_DEFAULT = ("v101",)


def load_prevalence() -> dict[str, float]:
    import yaml
    d = yaml.safe_load((ROOT / "registry" / "disease_prevalence.yaml").read_text(encoding="utf-8"))
    return {k: float(v["value"]) for k, v in (d.get("prevalence") or {}).items() if v.get("value") is not None}


def prior_channels(rows, freq, *, perm=PERM, seed=SEED, rng=None, prevalence=None, gated=None) -> dict:
    """Frequency-prior channels over (case, candidate line) pairs. `rows` = [(cid, gold set, frame)],
    named cases only; candidates = the v1.0.1 components plus every gold component. Null: the prior's
    frequency vector is shuffled across candidates (one permutation per frame for loo_frame), 1000
    times, and the case's leave-one-out vector is read through the same relabelling. The leave-one-out
    observed AUC sits below 0.5 when lines are uniform (a case's own gold is one count short); the null
    is centred at 0.5, so that offset never helps a channel fail and never hides a skew above it."""
    cands = sorted(set(freq) | {c for _, g, _ in rows for c in g})
    idx = {c: k for k, c in enumerate(cands)}
    gold = [[c in g for c in cands] for _, g, _ in rows]
    tot = Counter(c for _, g, _ in rows for c in g)
    byf: dict[str, Counter] = defaultdict(Counter)
    for _, g, f in rows:
        byf[f].update(g)
    frames = sorted(byf)

    def base(name, i):
        _, g, f = rows[i]
        if name == "v101":
            return [freq.get(c, 0) for c in cands]
        if name == PREVALENCE:
            return [float((prevalence or {}).get(c, 0.0)) for c in cands]
        src = tot if name == "loo_pack" else byf[f]
        return [src[c] - (c in g) for c in cands]

    def auc_of(score_rows):
        pos, neg = [], []
        for sr, gr in zip(score_rows, gold):
            for s, y in zip(sr, gr):
                (pos if y else neg).append(s)
        return _auc(pos, neg)

    out = {}
    names = PRIORS + ((PREVALENCE,) if prevalence is not None else ())
    for k, name in enumerate(names):
        b = [base(name, i) for i in range(len(rows))]
        obs = auc_of(b)
        r = rng if (name == "v101" and rng is not None) else random.Random(seed + k)
        null = []
        for _ in range(perm):
            if name == "loo_frame":
                pis = {}
                for f in frames:
                    p = list(range(len(cands)))
                    r.shuffle(p)
                    pis[f] = p
                sc = [[bi[pis[rows[i][2]][j]] for j in range(len(cands))] for i, bi in enumerate(b)]
            else:
                p = list(range(len(cands)))
                r.shuffle(p)
                sc = [[bi[p[j]] for j in range(len(cands))] for bi in b]
            null.append(auc_of(sc))
        out[name] = {"auc": obs, "null_q95": _q95(null), "null_q05": _q05(null),
                     "null_median": sorted(null)[len(null) // 2], "pass": obs <= _q95(null)}
    out["n_cases"], out["n_candidates"] = len(rows), len(cands)
    out["lines_by_frame"] = {f: dict(byf[f].most_common()) for f in frames}
    gated = tuple(gated) if gated is not None else names
    out["gated"] = [n for n in gated if n in out]
    out["profile"] = [n for n in names if n not in out["gated"]]
    out["pass"] = all(out[n]["pass"] for n in out["gated"])
    return out


def leak_channels(batch: pathlib.Path, *, perm=PERM, seed=SEED, gated=None) -> dict:
    """Diagnosis-level channels over (case, candidate line) pairs of the named cases: candidates
    are every hidden component of the pack; gold = the case's components. The prior channel reads
    three frequency priors (`PRIORS`); the second channel is the maximum out-of-band distance."""
    import logging
    logging.disable(logging.WARNING)
    import yaml
    from haenv_kernel.build import build_instance
    from haenv.indicators import of
    from haenv.job import load_job
    from haenv.m2 import core
    from haenv.store import load_cases
    load_job(M2_JOB, root=ROOT)
    built = load_cases(batch / "cases.jsonl")
    src = yaml.safe_load((ROOT / "inputs" / "ddx-workup.job.yaml").read_text(encoding="utf-8"))
    freq = Counter(c for cs in src["cases"] for c in core.components(cs["latent"].get("ddx_spec_id", ""))
                   if cs["latent"].get("ddx_join_gold") in ("unified", "comorbidity"))
    sfid = {"HbA1c": "HbA1c", "fasting_glucose": "FBG", "LDL": "LDL", "triglycerides": "TG", "ALT": "ALT", "AST": "AST"}
    rows = []
    for cid, raw in sorted(built.items()):
        T = int(raw.prediction_context["prediction_time_T"])
        sp, vp = build_instance(raw, T)
        adj = vp.adjudication or {}
        ddx = adj.get("ddx") or {}
        if ddx.get("join_gold") not in ("unified", "comorbidity") or ddx.get("insufficient"):
            continue
        ex = {}
        for st, fid in sfid.items():
            rr = of(st).get("reference_range") or {}
            lo, hi = rr.get("low"), rr.get("high")
            vals = [p["value"] for p in (sp.longitudinal_data or {}).get(st) or []
                    if isinstance(p.get("value"), (int, float))]
            if vals and hi is not None and lo is not None and hi > lo:
                ex[fid] = max(max(0.0, (v - hi) / (hi - lo)) for v in vals)
        frame = str((adj.get("m2_frame") or {}).get("frame") or "F0")
        rows.append((cid, set(core.components(str(ddx.get("spec_id") or ""))), ex, frame))
    cands = sorted(set(freq) | {c for _, g, _, _ in rows for c in g})
    spec_of = {c: core.spectrum_ids(c) for c in cands}
    rng = random.Random(seed)
    pri = prior_channels([(cid, g, f) for cid, g, _, f in rows], freq, perm=perm, seed=seed, rng=rng,
                         prevalence=load_prevalence(), gated=gated or GATED_DEFAULT)

    obs_e = [(max([e.get(f, 0.0) for f in spec_of[c]] or [0.0]), c in g, cid) for cid, g, e, _ in rows for c in cands]
    auc_e = _auc([s for s, y, _ in obs_e if y], [s for s, y, _ in obs_e if not y])
    by_case = defaultdict(list)
    for s, y, cid in obs_e:
        by_case[cid].append((s, y))
    null_e = []
    for _ in range(perm):
        pos, neg = [], []
        for cid, lst in by_case.items():
            sc = [s for s, _ in lst]
            rng.shuffle(sc)
            for s, (_, y) in zip(sc, lst):
                (pos if y else neg).append(s)
        null_e.append(_auc(pos, neg))
    res = {"n_cases": len(rows), "n_candidates": len(cands),
           "prior": pri["v101"],
           "prior_loo_pack": pri["loo_pack"], "prior_loo_frame": pri["loo_frame"],
           "prior_prevalence": pri.get(PREVALENCE), "priors_gated": pri["gated"], "priors_profile": pri["profile"],
           "lines_by_frame": pri["lines_by_frame"],
           "max_out_of_band": {"auc": auc_e, "null_q95": _q95(null_e), "pass": auc_e <= _q95(null_e)}}
    res["pass"] = pri["pass"] and res["max_out_of_band"]["pass"]
    return res


def v3(ladder_rows, batch, *, gated=None) -> dict:
    out: dict = {"evaluated": True}
    if ladder_rows is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import m2_stub_ladder as L
        out["Z3"] = L.gates(ladder_rows, boot=0)["Z3"]
    else:
        out["Z3"] = {"pass": None, "why": "no stub rows on the formal case set"}
    out["leak"] = leak_channels(pathlib.Path(batch), gated=gated) if batch else {"pass": None, "why": "no batch"}
    vals = [out["Z3"].get("pass"), out["leak"].get("pass")]
    out["evaluated"] = all(v is not None for v in vals)
    out["pass"] = all(v is not False for v in vals) if out["evaluated"] else None
    return out


# ------------------------------------------------------------------ V-4
def v4(receipts: dict | None, ranked: list[str]) -> dict:
    if receipts is None:
        return {"evaluated": False, "why": "no receipts (paid stage)"}
    tot = {}
    for m in ranked:
        r = receipts.get(m)
        if r is None:
            return {"evaluated": False, "why": f"no receipt for {m}"}
        tot[m] = float(r) if isinstance(r, (int, float)) else float(r.get("solver_usd", 0)) + float(r.get("judge_usd", 0))
    over = {m: v for m, v in tot.items() if v > V4_MAX_USD}
    return {"evaluated": True, "usd": tot, "over_20": over, "pass": not over}


# ------------------------------------------------------------------ V-5
def _g_single(table: dict[str, dict[str, float]]) -> float | None:
    """sigma2_model / (sigma2_model + sigma2_residual) of a model x case table without replication."""
    models = sorted(table)
    cases = sorted(set.intersection(*(set(v) for v in table.values()))) if table else []
    M, C = len(models), len(cases)
    if M < 2 or C < 2:
        return None
    Y = [[table[m][c] for c in cases] for m in models]
    g = sum(map(sum, Y)) / (M * C)
    ym = [sum(r) / C for r in Y]
    yc = [sum(Y[i][j] for i in range(M)) / M for j in range(C)]
    ms_m = C * sum((x - g) ** 2 for x in ym) / (M - 1)
    ms_e = sum((Y[i][j] - ym[i] - yc[j] + g) ** 2 for i in range(M) for j in range(C)) / ((M - 1) * (C - 1))
    s2m = max(0.0, (ms_m - ms_e) / C)
    return s2m / (s2m + ms_e) if (s2m + ms_e) > 0 else 0.0


def review_pairs(job: dict) -> list[tuple[str, str]]:
    """Paired cases for V-5: within each frame F1-F3, each negative-arm case is paired with one
    positive-arm case (case-keyed order). M2's negative arm comes from independent specs, so
    same-spec pairs do not exist; the frame is the pairing stratum."""
    by = defaultdict(lambda: {"positive": [], "negative": []})
    for c in job["cases"]:
        m = c["latent"].get("m2")
        if m:
            by[m["frame"]][m["arm"]].append(c["case_id"])
    h = lambda x: hashlib.sha256(f"v5|{x}".encode()).hexdigest()
    out = []
    for fr in sorted(by):
        neg, pos = sorted(by[fr]["negative"], key=h), sorted(by[fr]["positive"], key=h)
        out += list(zip(neg, pos))
    return out


def v5(rows, job, ranked: list[str]) -> dict:
    from haenv.m2.score import review_cc
    by = defaultdict(list)
    for r in rows:
        if r["solver"] in ranked:
            by[r["solver"]].append(r)
    sens, spec, cc = {}, {}, {}
    for m, rs in by.items():
        pos = [r["review_flag_ok"] for r in rs if r.get("review_warranted") and isinstance(r.get("review_flag_ok"), (int, float))]
        neg = [r["review_flag_ok"] for r in rs if not r.get("review_warranted") and isinstance(r.get("review_flag_ok"), (int, float))]
        if pos and neg:
            sens[m], spec[m] = sum(pos) / len(pos), sum(neg) / len(neg)
            cc[m] = review_cc(rs)[0]
    ms = sorted(cc)
    if len(ms) < 2:
        return {"evaluated": False, "why": "fewer than two ranked models with both classes"}
    mean = lambda xs: sum(xs) / len(xs)
    xc, xs_ = [cc[m] for m in ms], [spec[m] for m in ms]
    var_cc = sum((x - mean(xc)) ** 2 for x in xc) / (len(ms) - 1)
    cov = sum((a - mean(xs_)) * (b - mean(xc)) for a, b in zip(xs_, xc)) / (len(ms) - 1)
    share = (cov / var_cc) if var_cc > 0 else None
    ok = {m: {r["case"]: r.get("review_flag_ok") for r in by[m]} for m in ms}
    table = {m: {} for m in ms}
    for a, b in review_pairs(job):
        for m in ms:
            if isinstance(ok[m].get(a), (int, float)) and isinstance(ok[m].get(b), (int, float)):
                table[m][f"{a}|{b}"] = float(ok[m][a] and ok[m][b])
    g = _g_single(table)
    passed = share is not None and share <= V5_SPEC_SHARE_MAX and g is not None and g >= V5_G_MIN
    return {"evaluated": True, "spec_share": share, "G_pairs": g, "n_pairs": len(review_pairs(job)),
            "sens": sens, "spec": spec, "review_utility_cc": cc, "pass": passed,
            "consequence": None if passed else "review_utility_cc demoted to the profile"}


# ------------------------------------------------------------------ main
def verdicts(rows, anchor_rows, *, top6=TOP6, batch=None, ladder_rows=None, receipts=None, boot=BOOT,
             workers=8, v3_gated=None) -> dict:
    job = _job()
    ranked = sorted({r["solver"] for r in rows})
    res = {"V-1": v1(rows, anchor_rows, top6, boot=boot, workers=workers)}
    res["V-1b"] = (v1b(anchor_rows, res["V-1"]["pairs"]) if res["V-1"].get("evaluated")
                   else {"evaluated": False, "why": "V-1 not evaluated"})
    res["V-2"] = v2(rows, job, ranked)
    res["V-3"] = v3(ladder_rows, batch, gated=v3_gated)
    res["V-4"] = v4(receipts, ranked)
    res["V-5"] = v5(rows, job, ranked)
    ev = {k: v for k, v in res.items() if v.get("evaluated")}
    res["evaluated"] = sorted(ev)
    res["not_evaluated"] = sorted(set(res) - set(ev) - {"evaluated"})
    res["pass"] = all(v.get("pass") is not False for v in ev.values())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows")
    ap.add_argument("--anchor")
    ap.add_argument("--batch")
    ap.add_argument("--ladder-rows")
    ap.add_argument("--receipts")
    ap.add_argument("--top6", nargs="*")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--draw-anchor", action="store_true")
    a = ap.parse_args(argv)
    if a.draw_anchor:
        print(json.dumps(anchor_subset(_job()), ensure_ascii=False))
        return 0
    if not (a.rows and a.anchor and a.out):
        ap.error("--rows, --anchor and --out are required")
        return 2
    res = verdicts(_load_rows(a.rows), _load_rows(a.anchor), top6=tuple(a.top6 or TOP6), batch=a.batch,
                   ladder_rows=_load_rows(a.ladder_rows) if a.ladder_rows else None,
                   receipts=json.loads(pathlib.Path(a.receipts).read_text(encoding="utf-8")) if a.receipts else None,
                   boot=a.boot, workers=a.workers)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in ("V-1", "V-1b", "V-2", "V-3", "V-4", "V-5"):
        v = res[k]
        print(f"[m2_verdicts] {k}: {'not evaluated' if not v.get('evaluated') else ('PASS' if v['pass'] else 'FAIL')}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
