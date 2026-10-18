"""Rank intervals for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.dimensions import real_solver_pool
import logging

log = logging.getLogger("haenv.report")


def rank_interval_block(rows: list[dict], ranker, *, title: str) -> list[str]:
    """The rank-interval section (paired bootstrap), shared by both boards; the
    interval is printed before point ranks. See `haenv/scoring/ranking.py`.
    """
    from haenv.world.generation.baselines import BASELINE_NAMES as _BN_RI
    from haenv.scoring.ranking import MIN_BOOT_SUCCESS as _RI_MIN
    from haenv.scoring.ranking import distinguishable_pairs as _dp
    from haenv.scoring.ranking import rank_intervals as _ri
    pool, _ = real_solver_pool(rows)
    try:
        iv = _ri(rows, ranker, pool=pool)
    except Exception as e:                                      # noqa: BLE001
        log.warning("[report] rank interval could not be computed (%s), skipping this section", type(e).__name__)
        return []
    if len(iv) < 2:
        return []
    dp = _dp(iv)
    # Below the bootstrap success floor the table is not printed.
    _unrel = [m for m in iv if iv[m].get("unreliable")]
    if _unrel:
        _one = iv[_unrel[0]]
        return [f"### {title}\n",
                f"> **This section prints no table** -- paired bootstrap succeeded only "
                f"{_one.get('n_succeeded')}/{_one.get('n_attempted')} times ({_one.get('boot_success_rate')}), "
                f"below the floor of {_RI_MIN:.0%}.\n",
                f"> With this few successful samples, quantiles degenerate into a bare [min,max] and \"distinguishable\" "
                f"degenerates into \"beats everyone\", while the rendered table would look like a normal bootstrap. "
                f"Failure modes: `{_one.get('boot_failures')}`\n",
                "> ⇒ Fix the aggregator first, then look at the interval.\n"]
    L = [f"### {title}\n",
         f"> **Report the interval, not the rank.** {dp['note']}\n",
         "> This interval is a **lower bound** -- the bootstrap only resamples **items**; "
         "the variance from rerunning the model on the same batch of items is not included (measured `pass^k` rank swings are wider).\n",
         "| Model | Score | **Rank interval** | Score interval (95%) | Distinguishably beats |",
         "|---|---|---|---|---|"]
    for m in sorted(iv, key=lambda x: (iv[x]["rank_lo"], -(iv[x]["score"] or 0), x)):
        d = iv[m]
        _b = f"{d['n_beats']}" + (f" ({'·'.join(d['beats'][:3])}…)" if d["beats"] else "")
        L.append(f"| `{m}` | {d['score']:.3f} | **#{d['rank_lo']}–#{d['rank_hi']}** "
                 f"| {d['score_lo']:.3f}–{d['score_hi']:.3f} | {_b} |")
    L.append("")
    _wide = [m for m in iv if iv[m].get("note")]
    if _wide:
        L.append(f"> **{len(_wide)}/{len(iv)} models have a rank interval spanning more than half the board** -- "
                 f"this data **cannot separate them from most of the rest**.\n")
    _v0 = next(iter(iv.values()))
    L.append(f"> Denominator: {_v0['n_cases']} items × {_v0['n_boot']} paired bootstrap draws "
             f"({_v0.get('n_attempted')} attempted · success rate "
             f"**{_v0.get('boot_success_rate')}**).\n")
    return L
