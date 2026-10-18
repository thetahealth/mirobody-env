"""ranking.py -- rank intervals, not ranks.

With a few dozen questions, repeated runs reorder much of a board, so a table of
ranks 1..N is false precision. This module reports, per model, a rank interval from
a paired bootstrap over questions: questions are drawn with replacement (every
solver's rows for a question move together), the drawn rows go through the
production aggregation (`rank_ddx` / `rank_models`), and the interval is the
[2.5%, 97.5%] quantile of the resulting ranks. Pairwise distinguishability is
reported too; it is not transitive, so no tiers are formed.

The seed is derived from the sorted question ids, so the same data gives the same
resamples. The interval does not include run-to-run variance of the models (it reuses
stored responses) and is tagged `bootstrap_only`; the full interval also needs
`pass^k` repeats.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import collections
import hashlib
import random

#: Bootstrap resamples; only affects Monte Carlo error of the interval endpoints.
N_BOOTSTRAP = 200

#: Interval coverage (95%).
INTERVAL_Q = (0.025, 0.975)

#: Below this success rate the quantiles degenerate, and results are tagged `unreliable`.
MIN_BOOT_SUCCESS = 0.5


def _seed_for(cases: list[str]) -> int:
    """The seed, derived from the set of question ids -- the same data yields the same
    set of resamples, and a different dataset automatically gets a different seed."""
    h = hashlib.sha256("|".join(sorted(cases)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def rank_intervals(rows: list[dict], ranker, *, pool: set[str] | None = None,
                   n_boot: int = N_BOOTSTRAP) -> dict:
    """Returns `{model: {...}}` -- each model's rank interval, score interval, and
    pairwise distinguishability.

    - `ranker(rows) -> list[rec]` -- the production aggregation function (`rank_ddx` or
      `functools.partial(rank_models, multiround=...)`).
    - `pool` -- the solvers ranked; `None` means everyone with a score.

    Per model: `rank_lo`/`rank_hi` (report these, not a point rank), `score_lo`/`score_hi`,
    `beats` (models it is distinguishably better than at 95%), `n_cases`/`n_boot`, and a
    `note` when the interval covers most of the board.
    """
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(str(r.get("case")), []).append(r)
    cases = sorted(by_case)
    if len(cases) < 2:
        return {}

    base = {r["model"]: r for r in ranker(rows)}
    live = [m for m, r in base.items()
            if r.get("score") is not None and (pool is None or m in pool)]
    if len(live) < 2:
        return {}

    rnd = random.Random(_seed_for(cases))
    n_attempted = 0
    failures: list[str] = []
    ranks: dict[str, list[int]] = {m: [] for m in live}
    scores: dict[str, list[float]] = {m: [] for m in live}
    #: pairwise win/loss counts within each resample
    wins: dict[tuple[str, str], int] = {}
    losses: dict[tuple[str, str], int] = {}
    seen_pair: dict[tuple[str, str], int] = {}

    for _ in range(n_boot):
        pick = [rnd.choice(cases) for _ in cases]        # resample questions
        boot: list[dict] = []
        for i, c in enumerate(pick):
            for r in by_case[c]:
                # a repeated question needs a new case name, or downstream pairing
                # would merge the copies and shrink the variance
                boot.append({**r, "case": f"{c}#b{i}"})
        n_attempted += 1
        try:
            recs = {x["model"]: x for x in ranker(boot)}
        except Exception as _e:                          # noqa: BLE001
            # failures are counted; a low success rate marks the result unreliable
            failures.append(f"{type(_e).__name__}: {str(_e)[:60]}")
            continue
        cur = [(m, recs.get(m, {}).get("score")) for m in live]
        cur = [(m, s) for m, s in cur if s is not None]
        if len(cur) < 2:
            continue
        order = sorted(cur, key=lambda t: (-t[1], t[0]))
        for i, (m, s) in enumerate(order):
            ranks[m].append(i + 1)
            scores[m].append(s)
        d = dict(cur)
        for a in d:
            for b in d:
                if a >= b:
                    continue
                seen_pair[(a, b)] = seen_pair.get((a, b), 0) + 1
                # a tie counts for neither side
                if d[a] > d[b]:
                    wins[(a, b)] = wins.get((a, b), 0) + 1
                elif d[b] > d[a]:
                    losses[(a, b)] = losses.get((a, b), 0) + 1

    def _q(v: list, lo_hi: tuple[float, float]):
        if not v:
            return (None, None)
        s = sorted(v)
        n = len(s)
        return (s[max(0, int(lo_hi[0] * n))], s[min(n - 1, int(lo_hi[1] * n))])

    out: dict[str, dict] = {}
    for m in live:
        rlo, rhi = _q(ranks[m], INTERVAL_Q)
        slo, shi = _q(scores[m], INTERVAL_Q)
        beats = []
        for other in live:
            if other == m:
                continue
            key = (m, other) if m < other else (other, m)
            tot = seen_pair.get(key, 0)
            if not tot:
                continue
            # times `m` won; ties count for neither side
            w = wins.get(key, 0) if m < other else losses.get(key, 0)
            if w / tot >= 0.975:             # 95% two-sided => 97.5% one-sided
                beats.append(other)
        out[m] = {"score": base[m].get("score"),
                  # only questions were resampled, not responses (see module docstring)
                  "uncertainty": "bootstrap_only",
                  "uncertainty_note": ("The bootstrap resamples questions only; variance from "
                                       "rerunning the model on the same questions is not included, "
                                       "so the true interval is wider (k=3 reruns widen the rank). "
                                       "A full interval needs two-level resampling "
                                       "(questions x responses)."),
                  "rank_lo": rlo, "rank_hi": rhi,
                  "score_lo": round(slo, 4) if slo is not None else None,
                  "score_hi": round(shi, 4) if shi is not None else None,
                  "beats": sorted(beats), "n_beats": len(beats),
                  "n_cases": len(cases), "n_boot": len(ranks[m]),
                  "n_attempted": n_attempted,
                  "n_succeeded": len(ranks[m]),
                  "boot_success_rate": (round(len(ranks[m]) / n_attempted, 3)
                                        if n_attempted else None),
                  "boot_failures": (collections.Counter(failures).most_common(3)
                                    if failures else None),
                  "n_ranked": len(live)}
        if n_attempted and len(ranks[m]) / n_attempted < MIN_BOOT_SUCCESS:
            out[m]["unreliable"] = (
                f"Interval not trustworthy: only {len(ranks[m])} of {n_attempted} resamples succeeded "
                f"({len(ranks[m]) / n_attempted:.0%} < {MIN_BOOT_SUCCESS:.0%}). "
                f"With this few samples the quantiles degenerate to [min,max], and the \"distinguishable\" "
                f"criterion requires winning every one of them, "
                f"while the display format looks identical to a normal bootstrap; do not read this table. "
                f"Failure modes: {collections.Counter(failures).most_common(3)}")
        if rlo is not None and (rhi - rlo + 1) > len(live) / 2:
            out[m]["note"] = (f"Rank interval #{rlo}-#{rhi} covers {rhi - rlo + 1}/{len(live)} positions"
                              f"; this batch cannot separate it from most other models")
    return out


def distinguishable_pairs(iv: dict) -> dict:
    """Compresses `rank_intervals`'s output into one publishable sentence: how many pairs
    of models are distinguishable.
    """
    ms = sorted(iv)
    tot = len(ms) * (len(ms) - 1) // 2
    ok = sum(1 for m in ms for o in iv[m]["beats"] if o in iv)
    return {"n_models": len(ms), "n_pairs": tot, "n_distinguishable": ok,
            "frac": round(ok / tot, 3) if tot else None,
            "note": (f"{ok}/{tot} model pairs are distinguishable under a 95% paired bootstrap"
                     + ("" if not tot else f"({ok / tot:.0%})")
                     + "; the remaining pairs do not support a stronger/weaker conclusion")}
