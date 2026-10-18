"""Validity for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.decisions import answer_space_diff
from haenv.scoring.analysis.dimensions import _REC_KEY
from haenv.scoring.analysis.dimensions import _rowdim
from haenv.scoring.analysis.dimensions import dimension_health
from haenv.scoring.analysis.ranking import rank_ddx
from haenv.scoring.analysis.tracks import is_ddx_batch
from haenv.reporting.formatting import (
    _fmt,
)


def append_validity_profile(L, rows, dd) -> None:
    """Append the original validity report section in its original order."""
    if is_ddx_batch(rows):
        from haenv.scoring.legacy import load_profile
        from haenv.scoring.legacy import profile_drift
        from haenv.scoring.semantic.report import profile_for_rows
        _pf = profile_for_rows(load_profile(), rows)
        L.append(f"### 1a′. Scoring profile `{_pf.profile_id}` -- which dimensions count toward the score, and **on what basis**\n")
        L.append("> Roles are pinned in `registry/scoring.yaml`; the drift check below reports on every batch "
                 "whether the pinned role still matches this batch.\n")
        L.append("| Dim | role | Validity (blind agreement rate) | Statistical health in this batch | Note |")
        L.append("|---|---|---|---|---|")
        _hmap = {"join_macro": "join_hit", "scope_macro": "join_scope_ok",
                 "tests_recall": "tests_recall", "tests_precision": "tests_precision",
                 "dx_listed": "dx_listed", "dx_hit": "dx_hit", "dx_top1": "dx_hit_top1",
                 "urgency_ok": "urgency_ok",
                 "join_hit_total": "join_hit", "held_ind": "held_independent"}
        _health = {}
        for _k, _src in _hmap.items():
            try:
                _health[_k] = dimension_health(rows, _src)
            except Exception:
                pass
        for _name, _m in _pf.metrics.items():
            _st, _ag = _pf.validity_state(_name)
            _vs = {"passing": f"✓ {_ag}", "failing": f"**{_ag}** < {_pf.threshold}",
                   "unmeasured": "**not measured**"}[_st]
            _h = _health.get(_name)
            _hs = ("—" if _h is None else
                   ("✓ usable as capability" if _h.get("usable_as_capability") else
                    f"{str(_h.get('verdict'))[:100]}"))
            _note = _m.get("pending_decision") or _m.get("superseded_reason") or ""
            L.append(f"| `{_name}` | **{_m['role']}** | {_vs if _m['role'] == 'dim' else '—'} "
                     f"| {_hs} | {str(_note)[:60]} |")
        L.append("")
        # ---- `revisit_when`: per-batch check whether a dimension should count again ----
        _rv = _pf.revisit_conditions()
        if _rv:
            L.append("#### 1a″. `revisit_when` -- under what condition this dimension's status should be reconsidered\n")
            L.append("> Conditions are written in `registry/scoring.yaml`. **Meeting a condition raises an alert; it does not change the role** -- "
                     "changing roles automatically would let the composite score drift with the models in the batch, so this section only "
                     "reports that a role should be reconsidered.\n")
            L.append("| Dim | Current role | Condition to re-enter scoring | Met in this batch? |")
            L.append("|---|---|---|---|")
            for _c in _rv:
                _nm = _c["metric"]
                _h2 = _health.get(_nm) or {}
                _st2, _ag2 = _pf.validity_state(_nm)
                _hit, _why, _w = None, "", _c["when"]
                # `revisit_when` values in `registry/scoring.yaml` are Chinese, so these
                # comparisons match Chinese substrings.
                if "极差" in _w and _h2:
                    _sp = _h2.get("spread")
                    if _sp is not None:
                        _hit = float(_sp) > 0.15
                        _why = f"spread {_sp}"
                if "一致率" in _w:
                    _hit = (_ag2 is not None and _ag2 >= _pf.threshold)
                    _why = f"agreement rate {'not measured' if _ag2 is None else _ag2}"
                if "行使面" in _w:
                    _n_nn = sum(1 for r in rows if r.get(_REC_KEY.get(_nm, _nm)) is not None)
                    _hit = _n_nn > 0
                    _why = f"{_n_nn} non-null cells"
                L.append(f"| `{_nm}` | {_c['role']} | {_w} | "
                         f"{'**met -- reconsider**' if _hit else 'No'}"
                         f"{(' · ' + _why) if _why else ''} |")
            L.append("")
        # ---- scored vs. publishable ----
        # `failing` kinds: `construct` (excluded from the score), `defect` and `disputed`
        # (kept in the score, block publication).
        _comp, _judg = _pf.computable_dims(), _pf.judgment_dims()
        L.append("### 1a⁗. Two kinds of dimensions in the composite score -- **computable dimensions (the backbone) + judgment dimensions**\n")
        L.append(f"> Computable **{len(_comp)}** dims · judgment **{len(_judg)}** dims. "
                 "The two kinds have criteria of a different nature, so they're **reported separately**:\n"
                 ">\n"
                 "> · **Computable dimensions**: the criterion is an identity relation (is the citation in the ledger / is the timestamp ≤ T / "
                 "did it exceed budget / did the query ground) -- **the construct and its implementation are the same thing**, "
                 "so **no blind anchor is needed**: there is no \"does this quantity measure that thing\" to ask.\n"
                 "> · **Judgment dimensions**: the criterion is a **proxy** (the gold standard is a human-written checklist, the answer is free text) ⇒ "
                 "must pass a blind anchor + clinical review.\n"
                 ">\n"
                 "> **Computable does not mean exempt from checks**: the spread self-check still runs on computable dimensions too, "
                 "since an item that hands over a legitimate roster for free can make a dimension structurally impossible to fail.\n"
                 ">\n"
                 "> Hard gates are **unchanged, not one word**: the non-compensable multiplier stays as is. What's being promoted into the composite score is the **continuous quantity**, not that multiplier.\n")
        L.append("| Class | Dimensions |")
        L.append("|---|---|")
        L.append(f"| **Computable (backbone)** | {', '.join('`' + x + '`' for x in _comp) or '(none)'} |")
        L.append(f"| **Judgment** | {', '.join('`' + x + '`' for x in _judg) or '(none)'} |")
        L.append("")
        if not _comp:
            L.append("> **Not one computable dimension** -- the composite score hangs 100% on the matching layer.\n")
        _pub = _pf.publishable_dims()
        _blk = _pf.blocked_from_publish()
        L.append("### 1a‴. Scored vs. **publishable externally**\n")
        L.append(f"> Scored **{len(_pf.scored_dims)}** dims · publishable **{len(_pub)}** dims. "
                 "The counts differ when a dimension has an open maintainer-side issue (criterion not final / gold standard awaiting clinical review): "
                 "it **stays in the composite score** but is **withheld from external publication**.\n")
        if _blk:
            L.append("| Dim | Validity | Unresolved issue | Who can resolve it |")
            L.append("|---|---|---|---|")
            for _b in _blk:
                L.append(f"| `{_b['metric']}` | {_b['state']} {_b['agreement']} | "
                         f"**{_b['failing_kind'] or '—'}** | {_b['who']} |")
            L.append("")
            L.append(f"> **This batch's composite score cannot be published externally**: {len(_blk)} dimensions carry unresolved validity issues. "
                     f"The publishable dimensions are {', '.join('`' + x + '`' for x in _pub) or '(none)'}.\n")
        else:
            L.append("> ✅ Every dimension in the composite score has no unresolved validity issue ⇒ this batch's composite score **can go external**.\n")
        # ---- held-back conditions (blocked or draft) make the batch unpublishable ----
        # Matched by diagnosis name, since eval rows carry no `spec_id`.
        try:
            from haenv.world.overlay import condition_registry as _cr_pub
            _all_r, _live_r = _cr_pub(include_draft=True), _cr_pub()
            _draft_dx = {str((_all_r[s] or {}).get("diagnosis") or ""): s
                         for s in (set(_all_r) - set(_live_r))}
            _draft_dx.pop("", None)
        except Exception as _e:                            # noqa: BLE001
            # Not silent: when the registry cannot be read, say so -- do not
            # report "publishable" just because no draft was found.
            _draft_dx = None
            L.append(f"> **The draft-condition gate could not be evaluated** (the condition registry couldn't be read: {_e}) -- "
                     "do not take this to mean this batch contains no draft conditions.\n")
        if _draft_dx:
            _hit = sorted({_draft_dx[d] for r in rows
                           if (d := str(r.get("ddx_diagnosis") or "")) in _draft_dx})
            if _hit:
                L.append(f"> **This batch's readings cannot be published externally**: hit {len(_hit)} "
                         f"condition(s) held back from publication (`clinical_review: blocked` or `draft: true`) -- "
                         f"{', '.join('`' + x + '`' for x in _hit)}. "
                         "A condition is released by its registry entry (`clinical_review: done`), not by a code default.\n")
        # ---- 1a^4a. per-dimension separation (see `haenv/scoring/separation.py`) ----
        try:
            from haenv.world.generation.baselines import BASELINE_NAMES as _sp_bn
            from haenv.world.generation.baselines import STUB_GEOMETRIES as _sp_geo
            from haenv.scoring.separation import PROCESS_DIMS as _sp_dims
            from haenv.scoring.separation import SEP_THRESHOLD as _sp_thr
            from haenv.scoring.separation import separation_report as _sp_rep
            from haenv.scoring.separation import unregistered_saturating as _sp_unreg
            _sp_rows = _sp_rep(rows, baseline_names=_sp_bn, stub_geometries=_sp_geo)
            # Stubs measured as saturating but missing from the exclusion list are printed here.
            _sp_un = {d: _sp_unreg(rows, d, baseline_names=_sp_bn) for d in _sp_dims}
            _sp_un = {d: v for d, v in _sp_un.items() if v}
        except Exception as _sp_e:            # a wrong scoring rule must be visible,
                                               # never silently swallowed
            _sp_rows = []
            L.append(f"### 1a⁴ᵃ. Per-dimension separation -- **could not be computed**: `{_sp_e}`\n")
        if _sp_rows:
            L.append("### 1a⁴ᵃ. Per-dimension separation -- **best real model − best degenerate baseline**\n")
            L.append(f"> Threshold **{_sp_thr}** (a single frozen constant, does not multiply per dimension -- a proliferating threshold is overfitting). "
                     "Paired dimensions are only computed on the **paired quantity**; stubs that saturate by construction and stubs that **don't match the geometry** are excluded and "
                     "reported separately in the \"best of the excluded\" column -- **excluding without reporting would turn separation into a number you can tune**.\n")
            L.append("| Dim | Direction | Best model | Best baseline | Separation | Verdict | Best of the excluded | Excluded |")
            L.append("|---|---|---|---|---|---|---|---|")
            for _r in _sp_rows:
                if _r["saturated"]:
                    _v = "zero-variance saturated"          # a constant value provides
                                                  # no evidence at all
                elif _r["vacuous"]:
                    _v = "◻️ no baseline (vacuous)"
                elif _r["passes"]:
                    _v = "✅ passes"
                elif (_r["separation"] or 0) < 0:
                    _v = "negative"
                else:
                    _v = "positive but doesn't pass"
                L.append(f"| `{_r['dim']}` | {_r['direction']} | "
                         f"{_fmt(_r['model_max'])} | {_fmt(_r['baseline_max'])} | "
                         f"**{_fmt(_r['separation'])}** | {_v} | "
                         f"{_fmt(_r['process_baseline_max'])} | "
                         f"{len(_r['excluded'])} |")
            L.append("")
            _sp_sat = [r["dim"] for r in _sp_rows if r["saturated"]]
            _sp_vac = [r["dim"] for r in _sp_rows if r["vacuous"] and not r["saturated"]]
            _sp_neg = [r["dim"] for r in _sp_rows
                       if r["separation"] is not None and r["separation"] < 0]
            if _sp_sat:
                L.append("> **Zero-variance saturated**: " + " · ".join(f"`{d}`" for d in _sp_sat)
                         + " -- the real-model side has an identical value throughout. Per spec §17, **an identical value constitutes no evidence at all**, "
                         "and this dimension's separation should not be read as either \"separated\" or \"not separated\".\n")
            if _sp_vac:
                L.append("> ◻️ **No baseline in this batch**: " + " · ".join(f"`{d}`" for d in _sp_vac)
                         + " -- `vacuous`. **\"No baseline\" does not mean \"wins\"**; "
                         "reading this dimension requires first getting its negative-control stub into this batch.\n")
            if _sp_neg:
                L.append("> **Negative separation**: " + " · ".join(f"`{d}`" for d in _sp_neg)
                         + " -- the best degenerate baseline outscores the best real model. "
                         "Check the \"best of the excluded\" column and the exclusion list first: "
                         "the most common cause of a negative sign is not a weak model, it's **something that shouldn't be in the pool got mixed in**.\n")
            if _sp_un:
                L.append("> **Stubs measured as saturated but not registered in `negative_controls`**: "
                         + " · ".join(f"`{d}` ← {', '.join(v)}" for d, v in _sp_un.items())
                         + " -- the exclusion list is hand-written, and it goes stale whenever the stub pool or the geometry changes. "
                         "**This dimension's separation is unreadable until the registration is filled in** "
                         "(the pool still has a stub that saturates by construction, and it is the baseline's best value).\n")
            for _r in _sp_rows:
                if _r["excluded"]:
                    L.append(f"<details><summary>`{_r['dim']}` -- which "
                             f"{len(_r['excluded'])} were excluded, and why</summary>\n")
                    for _s, _w in _r["excluded"].items():
                        L.append(f"- `{_s}` —— {_w}")
                    L.append("\n</details>\n")

        # ---- 1a^5. reason distribution of null values ----
        from haenv.scoring.absence import NULL_RATE_GATE
        from haenv.scoring.absence import SENTINEL_DIMS
        from haenv.scoring.absence import absence_gate
        from haenv.scoring.absence import absence_report
        from haenv.world.generation.baselines import BASELINE_NAMES as _BN
        # Null rate over real models only.
        _abs_rows = [r for r in rows if r.get("solver") not in set(_BN)]
        # Aggregate-layer dimensions are resolved through `rank_ddx` on diagnostic rows
        # (by `gold_kind`); other item types pass None.
        _is_ddx = any(str(r.get("gold_kind", "")).startswith("ddx:") for r in rows)
        _abs_recs = ([r for r in rank_ddx(rows) if r.get("model") not in set(_BN)]
                     if _is_ddx else None)
        _abs_rep = absence_report(_abs_rows, sorted(_pf.metrics),
                                  recs=_abs_recs) if _abs_rows else []
        if _abs_rep:
            L.append("### 1a⁵. Reason distribution of null values -- **not applicable / judged negative / never ran are three different things**\n")
            L.append(f"> Classification is cut by **the measured null rate** (§33); the threshold **{NULL_RATE_GATE}** is a frozen constant. "
                     f"\"Should this dimension have a value on this geometry\" reads the **profile's declaration** "
                     f"(`applies_to` / `aggregate`), it is not inferred by this section. This batch's geometry is "
                     f"`{_abs_rep[0]['geometry']}`.\n")
            L.append("> The five classes' readings are **not interchangeable**: `applicability` over threshold ⇒ judged negative (no judgeable object) · "
                     f" `sentinel` allows a high null rate by design (**{len(SENTINEL_DIMS)} currently registered**) · "
                     " `geometry_not_applicable` the profile explicitly states this geometry doesn't apply (**not a debt**) · "
                     " `declared_aggregate` declares `aggregate: per_solver` (no key on the row is normal) · "
                     " `undeclared_aggregate` the aggregator can compute it **but it isn't declared** ⇒ **a declaration gap** · "
                     " `not_wired` a genuine debt.\n")
            L.append("| Dim | Null rate | Class | Reason distribution |")
            L.append("|---|---|---|---|")
            for _a in _abs_rep:
                _rz = " · ".join(f"`{k}`×{v}" for k, v in _a["reasons"].items()) or "—"
                L.append(f"| `{_a['dim']}` | {_a['null_rate']:.3f} "
                         f"({_a['n_null']}/{_a['n']}) | {_a['class']}"
                         f"{' (over threshold)' if _a['fails'] else ''} | {_rz} |")
            L.append("")
            _ag = absence_gate(_abs_rep)
            _nw = [a["dim"] for a in _abs_rep if a["class"] == "not_wired"]
            _uc = sum(a["unclassified"] for a in _abs_rep)
            if _ag:
                L.append("> **A dimension's null rate is over threshold**: "
                         + " · ".join(f"`{a['dim']}` {a['null_rate']:.3f}" for a in _ag)
                         + " -- this dimension has no judgeable object in this batch of items, and should not be read as "
                         "\"it was tested\".\n")
            if _nw:
                L.append(f"> **{len(_nw)} dimension(s) have not one reading on this batch of items, and the profile does not declare them not applicable**"
                         " (a genuine debt): " + " · ".join(f"`{d}`" for d in _nw)
                         + ". **Must not be written up as \"this dimension is not applicable\"** -- the former needs wiring up, the latter needs a different criterion, "
                         "and conflating the two would read a wiring debt as a criterion debt.\n")
            _ua = [a["dim"] for a in _abs_rep if a["class"] == "undeclared_aggregate"]
            if _ua:
                L.append(f"> **{len(_ua)} dimension(s) the aggregator can compute, but the profile does not declare "
                         "`aggregate: per_solver`**: " + " · ".join(f"`{d}`" for d in _ua)
                         + ". They **have values** so they aren't judged negative, but the missing declaration means "
                         "the `reachability_check` gate **does not take effect** on them -- "
                         "add the declaration so that a future gap is reported as debt.\n")
            if _uc:
                L.append(f"> **{_uc} unclassified null value(s)** -- `absence.DENOM_FIELD` needs its table extended. "
                         "A count > 0 is an open item for this table.\n")
        # ---- 1a^6. answer space vs. the values the gold standard actually
        # uses ----
        _asd = answer_space_diff(rows)
        if _asd and _asd.get("n_all"):
            L.append("### 1a⁶. Answer space vs. the values the gold standard actually uses\n")
            L.append(f"> Menu **{_asd['menu']}** items · gold standard uses **{_asd['gold_used']}** items ⇒ "
                     f"**dead set of {_asd['dead']} items** (can never be correct on this pack).\n")
            L.append("| Convention | Falls in the dead set | Rate |")
            L.append("|---|---|---|")
            L.append(f"| Top driver | {_asd['d_top']}/{_asd['n_top']} | "
                     f"**{_asd['top_rate']:.3f}** |")
            L.append(f"| All nominations | {_asd['d_all']}/{_asd['n_all']} | "
                     f"**{_asd['all_rate']:.3f}** |")
            L.append("")
            L.append("> **The denominator only includes real models** -- degenerate stubs always answer with a driver that's in the gold standard, "
                     "and including them would pull this rate down. "
                     "The two rates **are not merged into one**: they differ by almost a factor of two, and merging them would conflate \"willing to list it\" with \"believes it\".\n")
            if _asd["hot"]:
                L.append("> Most-answered items in the dead set: "
                         + " · ".join(f"`{k}`×{v}" for k, v in _asd["hot"]) + ".\n")
            L.append("> **A large dead set does not mean the items are wrong**: it may mean the menu should be narrowed, or it may mean those drivers "
                     "**should genuinely be turned into items**. This section only lays out the dead set; which one it is depends on the item pack's intent.\n")
        # ---- 1a^7. `refuting_evidence` / `what_not_to_do` ----
        # Read through `_rowdim`, which also resolves the former `ref_*` names.
        _rw = [r for r in rows if _rowdim(r, "sce_n_excluded") or r.get("wnd_n_items")]
        if _rw:
            _real = {r["model"] for r in dd if not r.get("is_baseline")} or None
            L.append("### 1a⁷. Counter-evidence cited for exclusions / the prohibitions column\n")
            L.append("| Solution | Excluded items | With supporting evidence | **Self-contradictory** | Rate | Counter-evidence can't point to it | Prohibitions/cell | After dedup |")
            L.append("|---|---|---|---|---|---|---|---|")
            _agg: dict = {}
            for r in _rw:
                a = _agg.setdefault(str(r.get("solver")), [0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0])
                a[0] += int(_rowdim(r, "sce_n_excluded") or 0)
                a[1] += int(_rowdim(r, "sce_n_with_support") or 0)
                a[2] += int(r.get("sce_self_contradictory") or 0)
                a[3] += int(_rowdim(r, "sce_unrevealed") or 0)
                # `sce_unrevealed_rate` aggregates as the worst cell.
                a[7] = max(a[7], float(_rowdim(r, "sce_unrevealed_rate") or 0.0))
                a[4] += int(r.get("wnd_n_items") or 0)
                a[5] += 1 if r.get("wnd_n_items") else 0
                a[6] += int(r.get("sce_n_no_support") or 0)
                # The deduplicated count is printed too; its denominator is counted separately.
                if r.get("wnd_n_distinct") is not None:
                    a[8] += int(r.get("wnd_n_distinct") or 0)
                    a[9] += 1
            for s, a in sorted(_agg.items(),
                               key=lambda kv: (-(kv[1][2] / (kv[1][1] or 1)), kv[0])):
                _rate = f"**{a[2] / a[1]:.3f}**" if a[1] else "not applicable"
                _wnd = f"{a[4] / a[5]:.1f}" if a[5] else "not applicable"
                _wndd = f"{a[8] / a[9]:.1f}" if a[9] else "not applicable"
                _unrev = (f"{a[3]} (worst cell {a[7]:.2f})" if a[3] else "—")
                L.append(f"| `{s}` | {a[0]} | {a[1]} | {a[2]} | {_rate} | "
                         f"{_unrev} | {_wnd} | {_wndd} |")
            L.append("")
            L.append("> **Self-contradictory** = `ruled_out_by` lands inside this same exclusion's own "
                     "`supporting_evidence` -- **the same piece of evidence both supports it and excludes it**.\n"
                     "> **The denominator only counts items that were given supporting evidence**: an empty set cannot be self-contradictory, "
                     "and counting \"cited nothing at all\" into the denominator would give it a free pass, reading out as **fake model differentiation**. "
                     "\"Gave no supporting evidence\" is tracked separately in `sce_n_no_support`, not merged into this column.\n")
            _ns = sum(a[6] for a in _agg.values())
            if _ns:
                L.append(f"> This batch has **{_ns}** exclusion item(s) that **gave not one piece of supporting evidence** -- "
                         f"they do not enter the denominator of the rate above.\n")
            L.append("> **The prohibitions column is not scored**: case-specificity saturates on real models and no stub produces the field "
                     "(no lower bound), so per N2 it is not a capability finding. Its cross-case 3-gram overlap is low, so the column is "
                     "case-specific rather than templated.\n")
        # Records per segment whether it was judged substantive because a lab
        # crossed a boundary (magnitude > 30% of the reference range width) or
        # because of a new true symptom.
        _abn_seg = sum(int(_d.get("n_new_abnormal_lab") or 0)
                       for r in rows for _d in (r.get("rev_transitions") or []))
        if _abn_seg:
            L.append(f"> Tiering-basis audit: this batch has **{_abn_seg}** transition(s) judged substantive by an **abnormal lab value** "
                     f"(out of range by > 30% of the reference-range width) -- not by a new true symptom.\n")
        # ---- provisional / illegal admission / inconsistent denominator:
        # all three must be printed up front ----
        _prov = _pf.provisional_dims()
        _illegal = _pf.illegal_dims()
        if _illegal:
            L.append("> **A dimension violates the admission rule**: validity `failing` (measured and below threshold) yet still in `dim` -- "
                     + " · ".join(f"`{d['metric']}` {d['agreement']}<{d['threshold']}" for d in _illegal)
                     + ". This is a profile error, not an annotation on the reading.\n")
        if _prov:
            L.append(f"> **{len(_prov)}/{len(_pf.scored_dims)} scored dimension(s) are `provisional`** "
                     f"({' · '.join(f'`{x}`' for x in _prov)}) -- validity **not measured**, admitted under the rule of "
                     "\"known-bad is barred, not-yet-known may be provisionally admitted\".\n>\n"
                     "> **Provisional dimensions must not be used on an externally published board** -- they have no validity evidence. "
                     "They also enter the queue of \"who the next blind anchor should cover\".\n")
        # Inconsistent denominator: baseline and real models must be scored over the
        # same set of dimension names (as in `_apply_one_ruler`).
        _den = {r["model"]: (r.get("n_core_used"), r.get("n_core_total"),
                             tuple(sorted(r.get("core_used_dims") or ())))
                for r in dd if r.get("n_core_used") is not None}
        _dsets = {v[2] for v in _den.values()}
        if len(_dsets) > 1:
            _ruler = (dd[0].get("ruler_dims") if dd else None) or []
            _rset = tuple(sorted(_ruler))
            L.append("> **Inconsistent denominator**: rows use different **sets of scored dimensions** -- "
                     + " · ".join(f"`{m}` {a}/{b}" for m, (a, b, _s) in _den.items())
                     + ". **When a baseline and a real model are not scored on the same set of dimensions, that comparison does not hold**.\n>\n")
            _samecount = sorted(
                m for m, (a, _b, s) in _den.items()
                if s != _rset and _rset and a == len(_rset))
            if _samecount:
                L.append(f"> **{_samecount} use the same dimension count as this board's ruler, but a different set of dimensions** -- "
                         f"\"both use {len(_rset)} dimensions\" is not the same as \"use the same {len(_rset)} dimensions\"; "
                         f"two rulers of equal length are still not comparable.\n>\n")
            # Grouped by difference rather than listed per solver.
            _grp: dict[tuple, list[str]] = {}
            for _m, (_a, _b, _s) in sorted(_den.items()):
                if _rset and _s != _rset:
                    _grp.setdefault((tuple(sorted(set(_rset) - set(_s))),
                                     tuple(sorted(set(_s) - set(_rset)))), []).append(_m)
            for (_miss, _extra), _who in sorted(_grp.items(),
                                                key=lambda kv: (-len(kv[1]), kv[0])):
                L.append(f"> * missing {list(_miss) or '—'} · extra {list(_extra) or '—'} -- "
                         f"{len(_who)} solver(s): {'`' + '` `'.join(_who[:6]) + '`'}"
                         f"{' …' if len(_who) > 6 else ''}\n")
            L.append(">\n> This board's ruler (the mode) is " + (f"{list(_rset)}" if _rset else "**undetermined**")
                     + " -- see `_apply_one_ruler` for the verdict and disposition (an extra dimension ⇒ no score; "
                       "a missing dimension ⇒ backfilled as 0 into the denominator).\n")
        _unmet = _pf.unmet_validity()
        if _unmet:
            L.append(f"> **{len(_unmet)}/{len(_pf.scored_dims)} scored dimension(s) do not meet the validity precondition** "
                     f"(threshold {_pf.threshold}).\n>\n"
                     "> The precondition to enter `dim` is \"has a blind-anchor validity reading with agreement ≥ threshold\" -- "
                     "because `dimension_health` measures **statistical health**, it cannot see "
                     "\"whether this dimension measures that thing\". Evidence: `join_scope_ok` is statistically healthy, "
                     "yet the blind anchor measured its agreement with humans at only 0.667, and on the independent class it penalizes answering style.\n>\n"
                     "> **This batch's composite score is still computed per the profile** -- downgrading is a scoring-convention change that needs a decision; "
                     "this layer's job is to report it, not to decide on someone's behalf.\n")
        _drift = profile_drift(_pf, _health)
        if _drift:
            L.append("> **Profile drift** (pinned role vs. this batch's measurements):\n>\n")
            for _d in _drift:
                _kn = {"stale_scored": "pinned as scored, yet fails this batch's measurement",
                       "stale_excluded": "pinned as not scored, yet usable again per this batch's measurement"}[_d["kind"]]
                L.append(f"> * `{_d['metric']}`: {_kn} -- {_d['detail']}\n")
            L.append("")
        else:
            L.append("> Profile and this batch's measurements have **no drift**.\n")
        L.append("---\n")
