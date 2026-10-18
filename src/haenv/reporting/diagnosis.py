"""Diagnosis for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.ceilings import degenerate_ceiling_gate
from haenv.scoring.analysis.ceilings import degenerate_ceilings
from haenv.scoring.analysis.dimensions import scored_dims_still_healthy
from haenv.scoring.analysis.discrimination import item_discrimination
from haenv.scoring.analysis.provenance import judging_provenance
from haenv.scoring.analysis.ranking import rank_ddx
from haenv.scoring.analysis.tracks import is_ddx_batch
from haenv.reporting.formatting import (
    _fmt,
    _render_stamp,
    board_rank_labels,
    gate_cell,
    gate_state_note,
    geometry_scope_note,
)
from haenv.reporting.rank_intervals import (
    rank_interval_block,
)
from haenv.reporting.readouts import (
    noise_floor_for,
)


def append_diagnosis_board(L, job, rows, built, _disc_gate) -> list[dict] | None:
    """Append the original diagnosis report section in its original order."""
    if is_ddx_batch(rows):                       # diagnostic items get their own table
        dd = rank_ddx(rows)
        L.append("## 1. Multi-model composite-score ranking · **diagnostic dimension** (§12 cross-model score comparison)\n")
        L += rank_interval_block(rows, rank_ddx, title="1α. Rank interval (paired bootstrap; **read this first**)")
        # Gold-standard source and criterion version go in the board header.
        from haenv.reporting.provenance_report import one_line as _prov_1
        L.append(_prov_1(dict(getattr(job, "provenance", None) or {}), len(built or rows)))
        # The consistency statement carries the render timestamp.
        _p1 = judging_provenance(rows)
        L.append(f" · criterion version `{'` `'.join(sorted(_p1['measured_under']))}`"
                 + (f" · **does not match current code `{_p1['judging_current']}`**: "
                    + ";".join(_p1["why"] or []) if _p1["stale"]
                    else f" ✓ consistent with current code (rendered at {_render_stamp()})"))
        L.append("")
        # The score caption is generated from the scoring profile (`load_profile()`,
        # the same entry point as `rank_ddx`).
        from haenv.scoring.legacy import load_profile as _lp_cap
        _cap_prof = _lp_cap()
        _cap_pairs = {a: b for a, b in _cap_prof.harmonic_pairs()}
        _cap_slots = []
        for _d in _cap_prof.scored_dims:
            if _d in {v for v in _cap_pairs.values()}:
                continue                                  # the paired side does not
                                                            # claim its own slot
            _nm = f"F1({_d} ⊕ {_cap_pairs[_d]})" if _d in _cap_pairs else _d
            _dr = _cap_prof.direction(_d)
            if _dr == "lower":
                _nm = f"1 − {_nm}"
            elif _dr == "non_monotone":
                _nm = f"~~{_nm}~~ (non-monotone, not scored)"
            _cap_slots.append(_nm)
        _cap_eff = sum(1 for s in _cap_slots if "not scored" not in s)
        L.append(f"> Score = mean({' · '.join(_cap_slots)}) × (1 − hard-gate fail rate).\n"
                 f"> **{len(_cap_slots)}** slots, of which "
                 f"**{len(_cap_slots) - _cap_eff}** are non-monotone and not scored ⇒ **{_cap_eff}** actually count.\n"
                 f"> The slot count above comes from the **profile's full slot roster**; while `n_core_total` in the result rows "
                 f"is the dimension count **actually applicable to this batch** (different geometries apply to different dimensions, "
                 f"only 3 on the single-shot geometry), **the two numbers are not supposed to be equal in the first place** -- they answer "
                 f"\"how many dimensions this scoring scheme defines\" versus \"how many dimensions this batch uses\".\n"
                 f"> profile `{_cap_prof.profile_id}` / fingerprint "
                 f"`{_cap_prof.content_sha256[:16]}` taken live, **never hand-written**.\n"
                 "> `join` is **reported per class** (each class equally weighted, insensitive to item-pack mix), but has been downgraded to "
                 "`diagnostic` and **does not count toward the composite score**.\n"
                 "> This sentence shares its source with the actual computed formula -- "
                 "changing the profile changes this sentence too.\n")
        L.append("> **This batch is diagnostic items, the early-warning table cannot be used here**: ddx items' `gold_drivers` is the placeholder "
                 "`unknown_or_multifactorial`, and a constant baseline always answers it -- on the early-warning table it would get a perfect Track C score and "
                 "rank first. That table is measuring the placeholder here, not diagnostic capability.\n")
        # Each score is printed with the ruler (`_apply_one_ruler`) that measured it.
        def _ruler_cell(r: dict) -> str:
            _u, _rl = r.get("n_core_used"), (r.get("ruler_dims") or [])
            if _u is None:
                return "not measured"                # absence must not be folded into 0,
                                                      # and must not print as `-` either
            _txt = f"{_u}/{len(_rl)}" if _rl else f"{_u}/{r.get('n_core_total')}"
            if r.get("score_offruler") is not None:
                return f"{_txt} extra {r.get('ruler_mismatch', {}).get('extra')}"
            if r.get("ruler_imputed_zero"):
                return f"{_txt} zero-filled×{len(r['ruler_imputed_zero'])}"
            return _txt
        L.append("| # | Model | **Score** | Ruler | dx_hit | of which top1 | held independent | join_hit | "
                 "urgency_ok | tests recall | tests precision | hard-gate fail | n |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in zip(board_rank_labels(dd), dd):
            L.append(f"| {i} | **{r['model']}** | **{_fmt(r['score'])}** | "
                     f"{_ruler_cell(r)} | "
                     f"{_fmt(r['dx_hit'])} ({r['n_dx']}) | {_fmt(r['dx_top1'])} | "
                     f"{_fmt(r['held_ind'])} ({r['n_ind']}) | {_fmt(r['join_hit'])} | "
                     f"{_fmt(r['urgency_ok'])} | {_fmt(r['tests_recall'])} | "
                     f"{_fmt(r['tests_prec'])} | {gate_cell(r)} | {r['n']} |")
        L.append("")
        L += gate_state_note(dd)
        L += geometry_scope_note(dd)
        L.append("> **How to read the \"Ruler\" column**: `dims used/dims in this board's ruler`. "
                 "`zero-filled×k` = this row is missing k dimensions and they were backfilled as 0 into the denominator before scoring (the shortfall is **its own behavior**, "
                 "not a measurement gap; dropping the denominator would mean \"not answering the hard question costs nothing\"). "
                 "`… extra …` = it claimed slots this board doesn't have ⇒ **given no score**, the original value is in `score_offruler`. "
                 "Two rows with a different \"ruler\" **cannot have their scores compared directly**.\n")
        # ---- join per class: the macro-average here is what enters the score ----
        _j0 = dd[0] if dd else {}
        if _j0.get("join_by_class"):
            _cls = list(_j0["join_by_class"])
            _scored = set(_j0.get("join_scored_classes") or [])
            L.append("**join per class** (each class equally weighted toward the score, insensitive to item-pack mix; "
                     "classes marked ✕ **do not count** but are still reported):\n")
            L.append("| Model | " + " | ".join(
                f"{k}{'' if k in _scored else ' ✕'} ({_j0['join_n_by_class'][k]})" for k in _cls)
                + " | **Macro-average (scored)** | Lower bound without looking at the item |")
            L.append("|---|" + "---|" * (len(_cls) + 2))
            for r in dd:
                L.append(f"| {r['model']} | " + " | ".join(
                    _fmt(r["join_by_class"].get(k)) for k in _cls)
                    + f" | **{_fmt(r.get('join_macro'))}** | {_fmt(r.get('join_macro_const_floor'))} |")
            L.append("")
            # ---- the behavioral-rule join dimension: printed side by side
            # with join_hit, never merged ----
            _sc_scored = set(_j0.get("scope_scored_classes") or [])
            _sc_floor = _j0.get("scope_macro_const_floor")
            L.append("**`join_scope_ok` per class** (behavioral criterion: whether top1's true-symptom coverage falls within that class's band. "
                     "**Read side by side with the table above, never merged** -- on this batch the two rank models in almost opposite orders):\n")
            L.append("| Model | " + " | ".join(
                f"{k}{'' if k in _sc_scored else ' ✕'}" for k in _cls)
                + " | **Macro-average (scored)** | Lower bound without looking at the item | Above the lower bound? |")
            L.append("|---|" + "---|" * (len(_cls) + 3))
            for r in dd:
                _m = r.get("scope_macro")
                _ok = ("—" if _m is None or _sc_floor is None
                       else ("✓" if _m > _sc_floor else "**No**"))
                L.append(f"| {r['model']} | " + " | ".join(
                    _fmt(r.get("scope_by_class", {}).get(k)) for k in _cls)
                    + f" | **{_fmt(_m)}** | {_fmt(_sc_floor)} | {_ok} |")
            L.append("")
            if _j0.get("scope_anchor_note"):
                L.append(f"> {_j0['scope_anchor_note']}\n")
            L.append("> **A model marked \"No\" -- this dimension does not constitute a capability finding for it** -- it did not beat "
                     "the item-blind strategy of \"always cite exactly 1 true symptom\".\n")
            for k, why in (_j0.get("reported_not_scored") or {}).items():
                if k.startswith("join:"):
                    L.append(f"> ✕ **{k[5:]}** not scored: {why}\n")
            # Classes pinned against the ceiling but still counted in the score.
            for k, why in (_j0.get("join_ceiling_warnings") or {}).items():
                L.append(f"> **{k}** is still counted toward the score, but {why}\n")
            L.append("> **A solution that always answers one class has a macro-average that is always 1/k** (k = number of scored classes) -- so "
                     "the \"lower bound without looking at the item\" column is an **analytic value**, and a real model must genuinely beat it for the reading to count.\n")
        # ---- join self-consistency, next to join_hit ----
        _sc = {}
        for r in rows:
            # Uses the coverage-based version; `join_self_contradiction` also fires on
            # correct comorbidity answers and is not reported.
            v = r.get("join_contradiction_cover")
            if v is None:
                continue
            a = _sc.setdefault(r.get("solver"), [0, 0])
            a[0] += 1 if v else 0
            a[1] += 1
        if _sc:
            L.append("**join self-consistency** (doesn't read the gold standard, only checks whether the model's own differential and `join_type` contradict each other):\n")
            L.append("| Model | Self-contradictory | How to read it |")
            L.append("|---|---|---|")
            for m, (bad, tot) in sorted(_sc.items(), key=lambda kv: (-(kv[1][0] / max(1, kv[1][1])), kv[0])):
                rate = bad / tot if tot else 0
                how = ("**`join_hit` does not constitute a capability finding**, check the convention first" if rate >= 0.30 else
                       "convention is basically self-consistent, `join_hit` can be read as capability" if rate <= 0.10 else "in between")
                L.append(f"| {m} | {bad}/{tot} = {rate:.0%} | {how} |")
            L.append("")
            L.append("> **Criterion**: the model's **top candidate alone covers every true symptom** -- meaning it internally believes `unified`, "
                     "yet `join_type` answered comorbidity/independent.\n"
                     "> Judged on \"coverage\", not \"how many were cited\": the latter cannot tell a **wrongly-answered** `unified` apart from a **correctly-answered** comorbidity.\n"
                     "> **Does not read `join_gold`**, so it does not beg the \"capability or convention\" question it is meant to answer; "
                     "but it does read the true-symptom roster (verifier side) -- the convention is \"doesn't read the **answer**\", not \"doesn't read any ground truth\".\n")

        cd = next((r for r in dd if r["model"] == "const_ddx"), None)
        if cd:
            L.append(f"**Lower bound (must-read)**: `const_ddx` **doesn't look at the item** and always answers `unified` + `A3`, getting "
                     f"join_hit {_fmt(cd['join_hit'])} · urgency_ok {_fmt(cd['urgency_ok'])}. "
                     f"**A real model that doesn't beat this row by a significant margin -- that dimension does not constitute a capability finding** (N2).\n")
        else:
            L.append("> This batch **has no constant baseline** `const_ddx`; the join / urgency dimensions lack a lower bound, "
                     "and their readings do not constitute a capability finding.\n")
        # ---- discrimination loop: whether this item set actually measures
        # capability ----
        for _dim in ("dx_hit", "join_hit", "urgency_ok"):
            _it = item_discrimination(rows, _dim)
            if not _it:
                continue
            _low = [i for i in _it if i["low_information"]]
            _ns = [i for i in _it if i["no_model_spread"] and not i["low_information"]]
            _single = [i for i in _it if i["no_model_spread"] is None]
            _dv = [i["discrimination"] for i in _it if i["discrimination"] is not None]
            L.append(f"- **{_dim}**: {len(_it)} items · **low-information {len(_low)}**"
                     f"(every model agrees and none beats the baseline, **excluded from the capability finding**)"
                     f" · no spread across models but beats the baseline {len(_ns)}"
                     f" · median discrimination {sorted(_dv)[len(_dv)//2] if _dv else '—'}"
                     + (f"\n  - **this batch has only 1 model under test**, so \"spread across models\" cannot be judged "
                        f"for {len(_single)} items (needs ≥2 models); only the \"relative to baseline\" half of the reading applies here"
                        if _single else ""))
            if _low:
                L.append(f"  - Items excluded: {', '.join(i['case'] for i in _low[:8])}"
                         f"{'…' if len(_low) > 8 else ''}"
                         f" (listed so that the item count stays accurate)")
        for _g in _disc_gate:
            L.append(f"\n> **Batch gate {_g['kind']}**: {_g['detail']}")
        # ---- 1a^8. degenerate ceiling: does any real solution beat the best degenerate strategy ----
        if is_ddx_batch(rows):
            from haenv.scoring.legacy import load_profile as _lp_c
            _cdims = tuple(_lp_c().scored_dims)
            _nz = noise_floor_for(job)
            _ceil = degenerate_ceilings(rows, _cdims, noise=_nz or None)
            if _ceil:
                _nok = [d for d, e in _ceil.items() if e["verdict"] == "ok"]
                _ncmp = [d for d, e in _ceil.items() if e["margin"] is not None]
                _nexp = [d for d, e in _ceil.items()
                         if e["verdict"] in ("declared_not_applicable", "declared_aggregate")]
                L.append(f"\n#### 1a⁸. Degenerate ceiling -- **{len(_nok)}/{len(_ncmp)}** "
                         f"scored dimensions **with a computable ceiling** have a real solution significantly beating the best degenerate strategy\n")
                L.append(f"> {len(_ceil)} scored dimensions total: **{len(_ncmp)} have a computable** ceiling · "
                         f"{len(_nexp)} are declared not applicable on this geometry · "
                         f"**{len(_ceil) - len(_ncmp) - len(_nexp)} are uncomputable and undeclared** -- "
                         f"the latter are listed one by one in the table.\n")
                L.append("| Dim | Degenerate ceiling (who) | Best real solution (who) | Margin | ×noise | Verdict |")
                L.append("|---|---|---|---|---|---|")
                _vzh = {"ok": "✅ measuring capability", "not_a_capability_dim": "cannot separate",
                        "unresolved": "unreadable", "noise_unknown": "noise not measured",
                        "no_degenerate_floor": "**no floor** (stub reads zero)",
                        "no_real_solver": "no real solution",
                        "undeclared_empty": "both sides empty and undeclared",
                        "declared_not_applicable": "— declared not applicable on this geometry",
                        "declared_aggregate": "— aggregate dimension, not a per-row field"}
                # Dimensions with a computable ceiling sort first (ascending
                # margin), uncomputable ones sort last -- but all get printed
                for _d, _e in sorted(_ceil.items(),
                                     key=lambda kv: (kv[1]["margin"] is None,
                                                     kv[1]["margin"] if kv[1]["margin"] is not None else 0)):
                    _cl = ("—" if _e["degenerate_ceiling"] is None
                           else f"{_e['degenerate_ceiling']}(`{_e['ceiling_by']}`)")
                    _bl = ("—" if _e["best_real"] is None
                           else f"{_e['best_real']}(`{_e['best_real_by']}`)")
                    L.append(f"| `{_d}` | {_cl} | {_bl} | {_fmt(_e['margin'])} "
                             f"| {_fmt(_e['margin_over_noise'])} "
                             f"| {_vzh.get(_e['verdict'], _e['verdict'])} |")
                L.append("")
                L.append("> **Degenerate ceiling** = the best score on that dimension by a **non-model, non-oracle** solution. "
                         "An oracle looks at the gold standard by design ⇒ excluded; a model not beating the oracle is expected.\n"
                         "> **The margin must be compared against noise, not against 0** -- noise is read **per dimension** from that job's "
                         "`reliability-*.json` (measured to differ by nearly 4x across dimensions in the same batch); "
                         "**a dimension whose noise was never measured is verdicted `noise not measured`, not `passing`**.\n"
                         "> **Reports, does not block**: a dimension not constituting a capability finding doesn't mean the item is wrong, "
                         "it means **that dimension's reading cannot be used as capability**.\n")
                for _g in degenerate_ceiling_gate(_ceil):
                    L.append(f"> {_g['detail']}\n")
            else:
                L.append("\n> **Degenerate ceiling could not be computed** -- this batch is missing a degenerate stub or a real solution "
                         "(both are needed to compare against the ceiling). Read it as **not measured**.\n")
            # ---- scored-dimension checkup: zero variance among real models ----
            _hw = scored_dims_still_healthy(rows)
            if _hw:
                L.append(f"> **Scored-dimension checkup: {len(_hw)} dimensions should have their scoring status reconsidered**\n")
                for _w in _hw:
                    L.append(f"> - {_w['detail']}\n")
            else:
                L.append("> ✅ Scored-dimension checkup: all still measure discriminating information.\n")
        L.append("")
        L.append("---\n")
    return dd if is_ddx_batch(rows) else None
