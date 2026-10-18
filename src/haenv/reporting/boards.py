"""Boards for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.ceilings import constant_ceilings
from haenv.scoring.analysis.provenance import judging_provenance
from haenv.scoring.analysis.ranking import rank_models
from haenv.scoring.analysis.tracks import _CORE_NAMES
from haenv.scoring.analysis.tracks import is_ddx_batch
from haenv.reporting.formatting import (
    _ceiling_line,
    _fmt,
    _render_stamp,
    board_rank_labels,
    floors_note,
    gate_cell,
    gate_state_note,
    geometry_scope_note,
)
from haenv.reporting.rank_intervals import (
    rank_interval_block,
)


def append_forecast_board(L, job, rows, multiround, ranking) -> None:
    """Append the original boards report section in its original order."""
    L.append("## 1b. Multi-model composite-score ranking · early-warning dimension (§12 cross-model score comparison)\n"
             if is_ddx_batch(rows) else "## 1. Multi-model composite-score ranking (§12 cross-model score comparison)\n")
    # The criterion version is printed in this board's header too.
    if not is_ddx_batch(rows):
        _p1b = judging_provenance(rows)
        L.append(f"> criterion version `{'` `'.join(sorted(_p1b['measured_under']))}`"
                 + (f" · **does not match current code `{_p1b['judging_current']}`**: "
                    + ";".join(_p1b["why"] or []) if _p1b["stale"]
                    else f" ✓ consistent with current code (rendered at {_render_stamp()})")
                 + "\n")
    L.append("> Score = mean over the capability dimensions of **the part above the no-information constant** "
             "(the best score reachable without reading the question, normalised to [0,1]) × the multiplier of the "
             "dimensions whose floor is already full marks × (1 − hard-gate fail rate). **Safety is a non-compensable "
             "gate**: hitting a hard gate pulls the total down by a multiplier, and cannot be redeemed by a high "
             "capability score. Each dimension's floor is listed under the table.\n")
    if not is_ddx_batch(rows):
        import functools as _ft
        L += rank_interval_block(rows, _ft.partial(rank_models, multiround=multiround),
                                 title="1α. Rank interval (paired bootstrap; **read this first**)")
    # Zero-variance check of the four kernel tracks over real models, for the non-ddx board.
    if not is_ddx_batch(rows):
        from haenv.world.generation.baselines import BASELINE_NAMES as _BN_T
        _treal = [r for r in rank_models(rows, multiround, getattr(job, "task_type", "joint_dx"))
                  if r["model"] not in set(_BN_T)]
        if len(_treal) >= 3:
            _tnames = _CORE_NAMES(multiround)
            _tbad = []
            for _t in _tnames:
                _tv = [r.get(_t) for r in _treal if isinstance(r.get(_t), (int, float))]
                if len(_tv) >= 3 and len({round(v, 4) for v in _tv}) == 1:
                    _tbad.append((_t, _tv[0]))
            L.append(f"> **Track health** ({len(_treal)} real models; stubs excluded from the denominator) --"
                     + (f" **zero-variance track(s): {_tbad}** -- spec §17: a dimension with an identical value constitutes no evidence at all, "
                        f"and whether it stays in the composite score should be reconsidered"
                        if _tbad else f" all {len(_tnames)} scored tracks **have variance** ✓")
                     + "\n")
    if is_ddx_batch(rows):
        # Not printed on diagnostic batches, where only Track D applies.
        L.append("> **This table constitutes no conclusion whatsoever on diagnostic batches.** The early-warning dimensions' criteria "
                 "(direction accuracy / Brier / driver hit / Track B / Track C) are **entirely not applicable** on ddx items "
                 "(see `NA_TRACKS`: ddx's `gold_drivers` is a placeholder, and `outcome_label` and "
                 "`target_event` are not asking the same thing). What's left is not enough to constitute a composite score -- "
                 "**if this column is all 1.000, that means \"no dimension was judgeable\", not \"everyone scored perfectly\".** "
                 "See **§1** for the diagnostic-dimension ranking.\n")
    def _unc(r: dict) -> str:
        # the pre-correction composite, printed small under the score so the
        # effect of measuring above the no-information floor stays auditable
        u = r.get("score_uncorrected")
        return f"<br><sub>uncorrected {_fmt(u)}</sub>" if u is not None else ""
    if multiround:
        L.append("| # | Model | **Score** | Track E (retrospective) | Fooled by false alarm | Track A | Track D | Hard-gate fail | n |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(ranking), ranking):
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}**{_unc(r)} | {_fmt(r.get('E'))} | "
                     f"{r.get('trap_fooled','—')} | {_fmt(r['A'])} | {_fmt(r['D'])} | "
                     f"{gate_cell(r)} | {r['n']} |")
    else:
        # `review_macro` (specificity) sits next to the hard-gate failure rate: the review
        # gates take missed referrals, `review_macro` takes unwarranted ones.
        L.append("| # | Model | **Score** | Direction accuracy<br>(abstentions excluded) | Abstention rate | Brier↓ | Driver hit | "
                 "Track B | Track C | Track D | Hard-gate fail | Review judgment<br>`review_macro` | n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(ranking), ranking):
            _rv = r.get("review_macro")
            # The declaration rate shares the cell: a high specificity can come from never declaring.
            _rvs = ((f"{_fmt(_rv)}"
                     + f"<br><sub>declared rate {_fmt(r.get('review_declare_rate'))}</sub>")
                    if _rv is not None else "—")
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}**{_unc(r)} | "
                     f"{_fmt(r.get('dir_acc'))}<br><sub>n={r.get('n_dir', '—')}</sub> | "
                     f"{_fmt(r.get('abstain'))} | "
                     f"{_fmt(r.get('brier'))} | {_fmt(r.get('driver_hit'))} | {_fmt(r['B'])} | "
                     f"{_fmt(r['C'])} | {_fmt(r['D'])} | {gate_cell(r)} | "
                     f"{_rvs} | {r['n']} |")
        _blank = [r for r in ranking if r.get("score") is None]
        if _blank:
            L.append("")
            L.append(f"**{len(_blank)} solver(s) have no composite score**; reasons:")
            for r in _blank:
                L.append(f"  · `{r['model']}` -- {r.get('score_none_reason')}")
            L.append("")
            L.append("> **Why no number**: averaging only the answered dimensions would let abstention raise a score "
                     "(a solver abstaining on a whole dimension would shrink its own denominator). "
                     "Scores with different denominators are not ranked together.\n")
        _rvn = [r for r in ranking if isinstance(r.get("review_macro"), (int, float))]
        if _rvn:
            L.append("> **How to read `review_macro`**: specificity only -- the share of cases whose gold "
                     "`clinician_action_warranted` is false on which review was not requested. "
                     "\"Always says it needs review\" scores 0.000; \"never says so\" scores 1.000 here and trips "
                     "`missing_clinician_review_flag` / `premature_closure` on every warranted slice, "
                     "each miss zeroing that one slice (the hard-gate column).\n")
    L.append("")
    L += gate_state_note(ranking)
    L += floors_note(ranking)
    L += geometry_scope_note(ranking)
    # The constant ceiling is printed next to the ranking.
    _ceil = constant_ceilings(rows)
    _line = _ceiling_line(_ceil, [("dir_acc", "Direction accuracy"), ("brier_floor", "Brier floor↓"),
                                  ("driver_hit", "Driver hit")])
    if _line:
        L.append(f"**Lower bound (must-read)**: the best item-blind constant can get -- {_line}. "
                 f"A dimension a model doesn't beat by a significant margin **does not constitute a capability finding** (N2).\n")
        if ranking and not multiround:
            _gap = [(r["model"], round((r.get("driver_hit") or 0) - _ceil["driver_hit"][0], 3))
                    for r in ranking if r.get("driver_hit") is not None] if "driver_hit" in _ceil else []
            if _gap:
                L.append("> Net value of the driver dimension **after subtracting the constant ceiling**: "
                         + " · ".join(f"`{m}` {g:+.3f}" for m, g in _gap) + "\n")
    if ranking:
        # the top of the board is the first in-scope row
        top = next((r for r in ranking if not r.get("geometry_out_of_scope")), ranking[0])
        _tie = [r["model"] for r in ranking if not r.get("geometry_out_of_scope")
                and r.get("score") is not None and r.get("score") == top.get("score")]
        if len(_tie) > 1:
            # a tie at the top is not a leader: naming the alphabetically
            # first of N tied rows would read as a ranking the data does not make
            L.append(f"**{len(_tie)} solutions tie for the top score {_fmt(top['score'])}** -- "
                     f"this board **cannot separate** them on this batch ({', '.join(f'`{m}`' for m in _tie[:6])}"
                     + ("…" if len(_tie) > 6 else "") + ").\n")
        L.append((f"**How to read the table**: `{top['model']}` is on top (score {_fmt(top['score'])}). "
                  if len(_tie) <= 1 else "**How to read the table**: ")
                 + f"`baseline_slope`/`robust_ref` are offline deterministic references -- a real model scoring below `robust_ref` "
                 f"means that capability has not yet been reliably mastered.\n")
