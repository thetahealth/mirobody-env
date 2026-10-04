"""Board with a case x round two-level bootstrap.

`semantic_report.track_board` resamples cases only, so an interval built from it carries the
case-sampling noise of ONE run. When the same cases were answered in several rounds, the answers
also carry re-answer noise and a shared whole-round drift; `track_board_twolevel` resamples both.

Interface: the same call and the same result dict as `track_board`. Each row belongs to a round
(`row[round_key]`, default `"rep"`; a row without it is round 1). Every (case, model) has one row
per round. The result adds `n_rounds` and `bootstrap`.

Modes:
  crossed      draw n cases and R rounds independently, take every drawn (case, round) cell.
               A whole-round drift is shared by all cases, so only this scheme sees it.
  nested       draw n cases; inside each drawn case draw R rounds independently. Re-answer noise
               without the shared drift.
  cases_only   draw n cases, take all of that case's rounds. With one round it is `track_board`'s
               own bootstrap, draw for draw (same seed, same draws).

  anchored_drift
               M2 spec section 10. The board is the main batch (one round, every case): draw n cases as
               `track_board` does (same seed, same draws), then add to every model an independent
               N(0, sigma2_mr) offset per draw. sigma2_mr is the model x round variance component of
               the composite, estimated on a smaller anchored subset answered in several rounds
               (`anchor_rows`, `drift_variance`); the offset carries the whole-round drift that does not
               shrink with the number of cases. `drift` picks the point estimate ("point"), the case-
               bootstrap 95% upper bound ("upper"), or a given variance (a float).

Out of scope: the `R1` branch of `track_board` (fewer than `min_common_cases` complete cases, fill
intervals). With fewer cases the function returns `ranked: False` and says why.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import random
from collections import defaultdict

from . import semantic_report as sr

MODES = ("crossed", "nested", "cases_only")
#: Modes of `track_board_twolevel`; `anchored_drift` is not a draw scheme of `make_draws`.
BOARD_MODES = MODES + ("anchored_drift",)
#: Seed of the case bootstrap for the sigma2_mr upper bound (drift-check used 20261002).
DRIFT_SEED = 20261002
_STATE: dict = {}


def _round_of(row: dict, key: str):
    return row.get(key, 1)


def split_rounds(rows: list[dict], round_key: str = "rep") -> dict:
    out = defaultdict(list)
    for r in rows:
        out[_round_of(r, round_key)].append(r)
    return dict(out)


def complete_cases(rows: list[dict], round_key: str = "rep") -> tuple[list[str], dict, list]:
    """Cases every model completed in EVERY round (`common_complete_cases` per round, intersected)."""
    by_round = split_rounds(rows, round_key)
    rounds = sorted(by_round)
    keep = None
    excluded: dict = {}
    for r in rounds:
        k, ex = sr.common_complete_cases(by_round[r])
        for c, why in ex.items():
            excluded.setdefault(c, []).extend(f"round{r}:{w}" for w in why)
        keep = set(k) if keep is None else keep & set(k)
    for r in rounds:
        for c in {x["case"] for x in by_round[r]}:
            if c not in keep and c not in excluded:
                excluded[c] = [f"round{r}:absent_in_other_round"]
    return sorted(keep or []), excluded, rounds


def pooled_rows(rows: list[dict], round_key: str = "rep") -> list[dict]:
    """Every (case, round) cell as its own unit, for the point estimate over all rounds."""
    return [{**r, "case": f"{r['case']}@{_round_of(r, round_key)}"} for r in rows]


def make_draws(cases: list[str], rounds: list, boot: int, seed: int, mode: str) -> list[list[tuple]]:
    """Each draw is a list of (source case, source round, slot) cells."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    rng = random.Random(seed)
    n, R = len(cases), len(rounds)
    draws = []
    for _ in range(boot):
        cs = [cases[rng.randrange(n)] for _ in range(n)]          # same call order as track_board
        if mode == "cases_only":
            draws.append([(c, r, f"{i}") for i, c in enumerate(cs) for r in rounds])
        elif mode == "crossed":
            rs = [rounds[rng.randrange(R)] for _ in range(R)]
            draws.append([(c, r, f"{i}.{j}") for i, c in enumerate(cs) for j, r in enumerate(rs)])
        else:
            cells = []
            for i, c in enumerate(cs):
                for j in range(R):
                    cells.append((c, rounds[rng.randrange(R)], f"{i}.{j}"))
            draws.append(cells)
    return draws


def _one(draw):
    rows = []
    for c, r, slot in draw:
        for row in _STATE["cells"][(c, r)]:
            rows.append({**row, "case": f"{c}#{slot}"})
    return _STATE["score_fn"](rows)


def bootstrap_scores_twolevel(rows: list[dict], cases: list[str], rounds: list, *, score_fn, boot: int, seed: int,
                              mode: str = "crossed", workers: int = 1, round_key: str = "rep") -> list[dict]:
    """`boot` resamples; the model scores of every resample (same draws for every model)."""
    keep = set(cases)
    cells = defaultdict(list)
    for row in rows:
        if row["case"] in keep:
            cells[(row["case"], _round_of(row, round_key))].append(row)
    draws = make_draws(cases, rounds, boot, seed, mode)
    _STATE.update(cells=cells, score_fn=score_fn)
    try:
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                return pool.map(_one, draws, chunksize=max(1, boot // (workers * 8)))
        return [_one(d) for d in draws]
    finally:
        _STATE.clear()


def _ems(Y):
    """Variance components of a balanced model x case x round cube (crossed random effects, expected
    mean squares; the drift-check estimator). Returns (components, mean squares)."""
    import numpy as np
    M, C, R = Y.shape
    g = Y.mean()
    ym, yc, yr = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean((0, 1))
    ymc, ymr, ycr = Y.mean(2), Y.mean(1), Y.mean(0)
    MS = {"mr": C * ((ymr - ym[:, None] - yr[None, :] + g) ** 2).sum() / ((M - 1) * (R - 1)),
          "mc": R * ((ymc - ym[:, None] - yc[None, :] + g) ** 2).sum() / ((M - 1) * (C - 1))}
    res = (Y - ymc[:, :, None] - ymr[:, None, :] - ycr[None, :, :] + ym[:, None, None] + yc[None, :, None]
           + yr[None, None, :] - g)
    MS["mcr"] = (res ** 2).sum() / ((M - 1) * (C - 1) * (R - 1))
    return {"mr": (MS["mr"] - MS["mcr"]) / C, "mc": (MS["mc"] - MS["mcr"]) / R, "mcr_e": MS["mcr"]}, MS


def pseudo_values(rows: list[dict], *, score_fn, round_key: str = "rep") -> tuple[list[str], list[str], list, "object"]:
    """Jackknife pseudo-values psi[m, c, r] = C S_r[m] - (C - 1) S_r^(-c)[m] of the model-level composite,
    per round, on the cases complete in every round. Their case mean is the composite (exactly for a
    linear score); the cube feeds the variance decomposition. Returns (models, cases, rounds, cube)."""
    import numpy as np
    keep, _, rounds = complete_cases(rows, round_key)
    by = split_rounds([r for r in rows if r["case"] in set(keep)], round_key)
    full = {rd: score_fn(by[rd]) for rd in rounds}
    models = sorted(m for m in set.intersection(*(set(full[rd]) for rd in rounds))
                    if all(full[rd][m] is not None for rd in rounds))
    C = len(keep)
    Y = np.zeros((len(models), C, len(rounds)))
    for k, rd in enumerate(rounds):
        for j, c in enumerate(keep):
            loo = score_fn([r for r in by[rd] if r["case"] != c])
            for i, m in enumerate(models):
                Y[i, j, k] = C * full[rd][m] - (C - 1) * loo[m]
    return models, keep, rounds, Y


def drift_variance(anchor_rows: list[dict], *, score_fn=None, round_key: str = "rep", boot: int = 4000,
                   seed: int = DRIFT_SEED) -> dict:
    """sigma2_mr, the model x round (whole-round drift) variance component of the composite, from the
    rounds of an anchored subset. Point = max(0, (MS_mr - MS_mcr) / C) on the jackknife pseudo-values;
    interval = case bootstrap percentiles (2.5, 97.5) of the same estimator, floored at 0; df = (M-1)(R-1).
    Pass the ranked models only: unanswered cells are zero rows, so a model that misses cases in one
    round reads as whole-round drift (A-block 2: glm-5.3-flash alone lifts the workup point 0 -> 0.105)."""
    import numpy as np
    score_fn = sr.composite_scores if score_fn is None else score_fn
    models, cases, rounds, Y = pseudo_values(anchor_rows, score_fn=score_fn, round_key=round_key)
    M, C, R = Y.shape
    if M < 2 or C < 2 or R < 2:
        raise ValueError(f"drift_variance needs >= 2 models, cases and rounds; got {M} x {C} x {R}")
    V, _ = _ems(Y)
    rng = np.random.default_rng(seed)
    bs = np.array([_ems(Y[:, rng.integers(0, C, C), :])[0]["mr"] for _ in range(boot)])
    lo, hi = (max(0.0, float(np.percentile(bs, q))) for q in (2.5, 97.5))
    return {"drift_var": max(0.0, float(V["mr"])), "drift_var_raw": float(V["mr"]), "drift_var_ci": [lo, hi],
            "drift_df": (M - 1) * (R - 1), "drift_sd": float(np.sqrt(max(0.0, V["mr"]))),
            "drift_sd_ci": [float(np.sqrt(lo)), float(np.sqrt(hi))], "n_models": M, "n_cases": C, "n_rounds": R,
            "residual_var": float(V["mcr_e"]), "boot": boot, "seed": seed}


def _anchored_drift_board(rows, *, score_fn, boot, seed, workers, rule, round_key, anchor_rows, drift):
    """M2 spec section 10: case bootstrap on the main batch plus an N(0, sigma2_mr) offset per model and draw."""
    first = min((_round_of(r, round_key) for r in rows), default=1)
    main = [r for r in rows if _round_of(r, round_key) == first]
    models = sorted({r["solver"] for r in main})
    keep, excluded = sr.common_complete_cases(main)
    if isinstance(drift, (int, float)) and not isinstance(drift, bool):
        est = {"drift_var": float(drift), "drift_var_ci": None, "drift_df": None}
        var, used = float(drift), "given"
    else:
        if anchor_rows is None:
            raise ValueError("anchored_drift needs anchor_rows (or a numeric drift variance)")
        est = drift_variance(anchor_rows, score_fn=score_fn, round_key=round_key)
        if drift not in ("point", "upper"):
            raise ValueError(f"drift must be 'point', 'upper' or a variance, got {drift!r}")
        var, used = (est["drift_var"], "point") if drift == "point" else (est["drift_var_ci"][1], "upper")
    out = {"n_common_complete": len(keep), "excluded_cases": excluded, "boot": boot, "seed": seed,
           "rule_parameters": rule, "n_rounds": 1, "bootstrap": "anchored_drift",
           "drift_var": est["drift_var"], "drift_var_ci": est["drift_var_ci"], "drift_df": est["drift_df"],
           "drift_used": used, "drift_var_used": var, "drift_estimate": est}
    if len(keep) < rule["min_common_cases"]:
        out.update(rule="R1", ranked=False, n_cases=len(keep), models=[{"model": m} for m in models],
                   why=f"{len(keep)} common complete cases < {rule['min_common_cases']}")
        return out
    scored = [{**r, round_key: first} for r in main if r["case"] in set(keep)]
    point = {m: s for m, s in score_fn(scored).items() if s is not None}
    samples = bootstrap_scores_twolevel(scored, keep, [first], score_fn=score_fn, boot=boot, seed=seed,
                                        mode="cases_only", workers=workers, round_key=round_key)
    sd = var ** 0.5
    if sd > 0:
        rng = random.Random(f"{seed}:anchored_drift")
        samples = [{m: (v + rng.gauss(0.0, sd) if v is not None else None)
                    for m, v in ((m, s.get(m)) for m in sorted(s))} for s in samples]
    sig = sr._significant(point, samples, rule["alpha"])
    out.update(rule="R0", ranked=True, n_cases=len(keep),
               why=f"{len(keep)} common complete cases, one round, drift offset sd {sd:.4f} ({used})")
    out["models"] = [{"model": m, "score": None if m not in point else round(point[m], 4),
                      "ci95": sr._interval([s[m] for s in samples if s.get(m) is not None]) if m in point else None,
                      "tier": 1 + sum(w == m for _, w in sig) if m in point else None} for m in models]
    out["significant_pairs"] = [{"better": a, "worse": b, "diff": round(point[a] - point[b], 4),
                                 "p_holm": round(p, 4)} for (a, b), p in sorted(sig.items())]
    return out


def track_board_twolevel(rows: list[dict], *, score_fn=None, boot: int | None = None, seed: int | None = None,
                         workers: int = 1, rule: dict | None = None, mode: str = "crossed",
                         round_key: str = "rep", anchor_rows: list[dict] | None = None,
                         drift="point") -> dict:
    """Apply the board rule to one track's real-model rows, with a case x round bootstrap.
    `mode="anchored_drift"`: `rows` is the main batch, `anchor_rows` the repeated rounds (module docstring)."""
    rule = {**sr.BOARD_RULE, **(rule or {})}
    score_fn = sr.composite_scores if score_fn is None else score_fn
    boot = rule["boot"] if boot is None else boot
    seed = rule["seed"] if seed is None else seed
    if mode == "anchored_drift":
        return _anchored_drift_board(rows, score_fn=score_fn, boot=boot, seed=seed, workers=workers, rule=rule,
                                     round_key=round_key, anchor_rows=anchor_rows, drift=drift)
    models = sorted({r["solver"] for r in rows})
    keep, excluded, rounds = complete_cases(rows, round_key)
    out = {"n_common_complete": len(keep), "excluded_cases": excluded, "boot": boot, "seed": seed,
           "rule_parameters": rule, "n_rounds": len(rounds), "bootstrap": mode}
    if len(keep) < rule["min_common_cases"]:
        out.update(rule="R1", ranked=False, n_cases=len(keep), models=[{"model": m} for m in models],
                   why=(f"{len(keep)} common complete cases < {rule['min_common_cases']}; the fill-interval branch "
                        "of track_board is not reproduced in the two-level board"))
        return out
    cases = set(keep)
    scored = [r for r in rows if r["case"] in cases]
    point = {m: s for m, s in score_fn(pooled_rows(scored, round_key)).items() if s is not None}
    samples = bootstrap_scores_twolevel(scored, keep, rounds, score_fn=score_fn, boot=boot, seed=seed, mode=mode,
                                        workers=workers, round_key=round_key)
    sig = sr._significant(point, samples, rule["alpha"])
    out.update(rule="R0", ranked=True, n_cases=len(keep),
               why=f"{len(keep)} common complete cases x {len(rounds)} rounds >= {rule['min_common_cases']} cases")
    out["models"] = [{"model": m, "score": None if m not in point else round(point[m], 4),
                      "ci95": sr._interval([s[m] for s in samples if s.get(m) is not None]) if m in point else None,
                      "tier": 1 + sum(w == m for _, w in sig) if m in point else None} for m in models]
    out["significant_pairs"] = [{"better": a, "worse": b, "diff": round(point[a] - point[b], 4),
                                 "p_holm": round(p, 4)} for (a, b), p in sorted(sig.items())]
    return out


def pair_se(samples: list[dict], models: list[str]) -> dict:
    """Standard deviation over the resamples of each paired model difference."""
    import itertools
    import statistics
    out = {}
    for a, b in itertools.combinations(sorted(models), 2):
        d = [s[a] - s[b] for s in samples if s.get(a) is not None and s.get(b) is not None]
        if len(d) > 1:
            out[(a, b)] = statistics.stdev(d)
    return out
