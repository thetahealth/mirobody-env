"""Forecast and diagnostic score aggregation over the domain measurements."""
from __future__ import annotations

from haenv.scoring.quantities import QUANTITIES as _QUANTITIES
from haenv.scoring.analysis.decisions import JOIN_CLASSES
from haenv.scoring.analysis.decisions import _scope_const_floor
from haenv.scoring.analysis.decisions import abstention_utility_of
from haenv.scoring.analysis.decisions import join_class_of
from haenv.scoring.analysis.decisions import join_scored_classes
from haenv.scoring.analysis.decisions import review_macro_of
from haenv.scoring.analysis.dimensions import MIN_GRIDS_FOR_SCORE
from haenv.scoring.analysis.dimensions import _QNAMES
from haenv.scoring.analysis.dimensions import _REC_KEY
from haenv.scoring.analysis.dimensions import _rowdim
from haenv.scoring.analysis.dimensions import real_solver_pool
from haenv.scoring.analysis.dimensions import unexercised_dims
from haenv.scoring.analysis.gates import gate_multiplier
from haenv.scoring.analysis.precision import _fsum_mean
from haenv.scoring.analysis.precision import _mean
from haenv.scoring.analysis.precision import _rd
from haenv.scoring.analysis.precision import exact_precision
from haenv.scoring.analysis.provenance import judging_provenance
from haenv.scoring.analysis.tracks import _CORE_NAMES
from haenv.scoring.analysis.tracks import _core_from_profile
from haenv.scoring.analysis.tracks import applicable_tracks
from haenv.scoring.analysis.tracks import noinfo_floors
from haenv.scoring.analysis.tracks import skill_over_floor
from haenv.scoring.analysis.tracks import stub_geometry_scope
import logging

log = logging.getLogger("haenv.analytics")


def rank_ddx_exact(rows: list[dict]) -> list[dict]:
    """`rank_ddx` with unrounded scores and dimensions: the input of the board bootstrap."""
    with exact_precision():
        return rank_ddx(rows)


def _board_order(r: dict) -> tuple:
    """Sort key: in-scope rows first, then unscored, then by score."""
    return (bool(r.get("geometry_out_of_scope")), r["score"] is None,
            -(r["score"] or 0), r["model"])


def rank_models(rows: list[dict], multiround: bool, task_type: str = "joint_dx") -> list[dict]:
    """Composite = mean skill above this batch's no-information floor over capability dimensions x product of zero-headroom dimensions (Track D / A) x (1 - hard-gate failure rate).

    The raw-mean composite is kept as `score_uncorrected`. `task_type` selects the
    scoring profile (see `_core_from_profile`).
    """
    # Differences from `rank_ddx`: a missing dimension already withholds the score
    # here (`len(_used) == len(core)`), and track health is reported by the report
    # layer instead of `unexercised_dims`.
    _prov_m = judging_provenance(rows)
    # Floors are batch-level (gold is solver-independent).
    _floors = noinfo_floors(rows, multiround)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("solver", "?"), []).append(r)
    out = []
    for name, rs in by.items():
        n = len(rs)
        # `gate_fail` reads `gate_multiplier`'s output so the column and the multiplier
        # agree; `n_unknown` travels with it.
        gate_fail = len(gate_multiplier(rs).get("gated_units") or [])
        abort = sum(1 for r in rs if str(r.get("overall", "")).startswith("ABORT"))
        tA = _mean([applicable_tracks(r).get("A") for r in rs])
        tB = _mean([applicable_tracks(r).get("B") for r in rs])
        tC = _mean([applicable_tracks(r).get("C") for r in rs])
        tD = _mean([applicable_tracks(r).get("D") for r in rs])
        rec = {"model": name, "n": n, "gate_fail": gate_fail, "abort": abort,
               "A": tA, "B": tB, "C": tC, "D": tD}
        if multiround:
            rec["E"] = _mean([r.get("trackE") for r in rs])
            rec["trap_fooled"] = f"{sum(r.get('trap_fooled', 0) or 0 for r in rs)}/" \
                                 f"{sum(r.get('n_trap', 0) or 0 for r in rs)}"
            core = [rec["E"], tD]
        else:
            # abstentions are excluded from the denominator (see judges.judge_forecast); the abstention rate is its own column
            _d = [r.get("direction_ok") for r in rs if r.get("direction_ok") is not None]
            rec["dir_acc"] = round(sum(1 for x in _d if x) / len(_d), 3) if _d else None
            # `dir_acc` is class-macro-averaged, so both degenerate strategies (always /
            # never) score 0.500.
            _byc: dict[float, list[bool]] = {}
            for r in rs:
                v, y = r.get("direction_ok"), r.get("y")
                if v is None or y is None:
                    continue
                _byc.setdefault(1.0 if float(y) >= 0.5 else 0.0, []).append(bool(v))
            rec["dir_by_class"] = {("event" if k else "no_event"):
                                   round(sum(1 for x in vs if x) / len(vs), 3)
                                   for k, vs in sorted(_byc.items())}
            rec["dir_n_by_class"] = {("event" if k else "no_event"): len(vs)
                                     for k, vs in sorted(_byc.items())}
            # Only one class present => no macro average (a constant answer would score 1.000).
            if len(_byc) >= 2:
                _rates = [sum(1 for x in vs if x) / len(vs) for vs in _byc.values()]
                rec["dir_acc_macro"] = _fsum_mean(_rates)
                rec["dir_macro_const_floor"] = 0.500
                rec["dir_macro_note"] = None
            else:
                rec["dir_acc_macro"] = None
                rec["dir_macro_const_floor"] = None
                rec["dir_macro_note"] = (
                    "Only one class of outcome in this batch, so no class macro average: with only one class, \"always answer that class\" "
                    "would score 1.000, handing a degenerate strategy full marks.")
            # No `abstained` field on any row => unmeasured, not 0.000.
            _ab = [r for r in rs if "abstained" in r]
            rec["abstain"] = (_mean([1.0 if r.get("abstained") else 0.0 for r in _ab])
                              if _ab else None)
            rec["n_abstain_scored"] = len(_ab)  # denominator of the abstention rate
            rec["n_dir"] = len(_d)
            rec["brier"] = _mean([r.get("brier") for r in rs])
            rec["driver_hit"] = _mean([1.0 if r.get("driver_hit") else 0.0 for r in rs])
            # The score uses the macro average; naive `dir_acc` is kept for comparison.
            core = [rec["dir_acc_macro"], tB, tC, tD]
        # `core` comes from the registry when the question type has a profile whose
        # scored dimensions all map through `CORE_DIM_TO_REC`; the hard-coded sets above
        # are the fallback.
        _pcore = _core_from_profile(task_type, rec)
        if _pcore is not None:
            core = _pcore
        _used = [c for c in core if isinstance(c, (int, float))]
        rec["n_core_used"], rec["n_core_total"] = len(_used), len(core)
        # Missing even one dimension => no composite score (`core_missing` names it);
        # dropping it would let abstention shrink the denominator. The review-flag
        # readings are reported on the same row (only early-warning questions have them).
        rec.update(review_macro_of(rs))
        rec["core_missing"] = [k for k, c in zip(_CORE_NAMES(multiround), core)
                               if not isinstance(c, (int, float))]
        # Hard-gate multiplier per scoring unit.
        _gm = gate_multiplier(rs)
        rec.update({"gate_unit": _gm["unit"], "n_units": _gm["n_units"],
                    "n_gated_units": _gm["n_gated_units"], "gated_units": _gm["gated_units"],
                    "n_gate_unknown": _gm["n_gate_unknown"],
                    "gate_unknown_cases": _gm["gate_unknown_cases"],
                    "gate_verified": _gm["verified"]})
        # Capability dimensions enter as skill above floor; Track D / A multiply;
        # `score_uncorrected` keeps the raw mean.
        _names = _CORE_NAMES(multiround)
        _comp = {k: skill_over_floor(c, (_floors.get(k) or {}).get("floor"))
                 for k, c in zip(_names, core)}
        rec["core_skill"] = {k: (None if v is None else round(v, 4))
                             for k, (_kind, v) in _comp.items()}
        rec["core_kind"] = {k: _kind for k, (_kind, _v) in _comp.items()}
        _cap = [v for (_kind, v) in _comp.values() if _kind in ("skill", "raw")]
        _ded = [v for (_kind, v) in _comp.values() if _kind == "deduct"]
        _dmul = 1.0
        for _v in _ded:
            _dmul *= _v
        rec["deduct_mult"] = round(_dmul, 4)
        rec["noinfo_floors"] = {k: _floors[k] for k in _names if k in _floors}
        _complete = bool(n and len(_used) == len(core))
        rec["score_uncorrected"] = (round(_mean(_used) * _gm["mult"], 3) if _complete else None)
        rec["score"] = (round(_mean(_cap) * _dmul * _gm["mult"], 3)
                        if (_complete and _cap) else None)
        rec["score_none_reason"] = (None if rec["score"] is not None else
                                    (f"denominator incomplete: missing {rec['core_missing']}"
                                     f"({len(_used)}/{len(core)} dimensions have a value)"
                                     if rec["core_missing"] else
                                     ("no dimension in this batch has headroom (each "
                                      "dimension's no-information constant already scores full marks)"
                                      if _complete else "no cell to judge")))
        rec.update(stub_geometry_scope(name, rs))
        # Same field names as `rank_ddx`.
        rec["measured_under"] = _prov_m["measured_under"]
        rec["judging_current"] = _prov_m["judging_current"]
        rec["judging_stale"] = _prov_m["stale"]
        rec["judging_stale_why"] = _prov_m["why"]
        # Names of the dimensions actually used (same field as `rank_ddx`'s `core_used_dims`).
        rec["core_used_dims"] = [k for k, c in zip(_CORE_NAMES(multiround), core)
                                 if isinstance(c, (int, float))]
        from haenv.scoring.semantic.coverage import protect_composite
        protect_composite(rec, rs)
        out.append(rec)
    # `score is None` sorts last, apart from genuine zeros.
    return sorted(out, key=_board_order)


def _apply_one_ruler(recs: list[dict]) -> None:
    """One board, one ruler: withhold scores computed against a different dimension set. Modifies `recs` in place.

    The ruler is the mode of real models' dimension sets (ties -> more dimensions,
    then name); fewer than 3 real models => unchanged. An extra dimension => score
    `None` (original kept in `score_offruler`); a missing dimension is imputed as 0
    (`score x n_used / (n_used + k)`).
    """
    from haenv.world.generation.baselines import BASELINE_NAMES as _BN_R
    _stub = set(_BN_R)
    scored = [r for r in recs if r.get("score") is not None]
    # Real-model pool via `real_solver_pool`, as in `unexercised_dims`.
    _pool, _ = real_solver_pool([{"solver": r["model"]} for r in scored])
    real = [r for r in scored if r["model"] in _pool]
    if len({r["model"] for r in real}) < 3:
        return
    tally: dict[tuple, int] = {}
    for r in real:
        k = tuple(sorted(r.get("core_used_dims") or ()))
        tally[k] = tally.get(k, 0) + 1
    ruler = max(tally, key=lambda k: (tally[k], len(k), k))
    for r in recs:
        if r.get("score") is None:
            continue
        mine = tuple(sorted(r.get("core_used_dims") or ()))
        r["ruler_dims"] = list(ruler)
        if mine == ruler:
            continue
        _extra = sorted(set(mine) - set(ruler))
        _miss = sorted(set(ruler) - set(mine))
        r["ruler_mismatch"] = {"extra": _extra or None, "missing": _miss or None}
        # Extra dimension => `None`. Missing dimension => imputed as 0, since not
        # answering must not remove a dimension from the denominator.
        if _extra:
            r["score_offruler"] = r["score"]
            r["score"] = None
            r["score_none_reason"] = (
                f"ruler mismatch (extra dimensions): this row has {_extra} extra, so it is a score on a different ruler; "
                f"this board's ruler has {len(ruler)} dimensions. There is no algorithm to fold the extra slots back, so no score is given "
                f"(the original value {r['score_offruler']} is kept in `score_offruler`)")
            continue
        _nu = int(r.get("n_core_used") or 0)
        if _nu and _miss:
            r["score_before_imputation"] = r["score"]
            r["score"] = round(r["score"] * _nu / (_nu + len(_miss)), 3)
            r["ruler_imputed_zero"] = _miss
            r["ruler_note"] = (
                f"missing {_miss} (other solvers on this board can compute it) => imputed as 0 into the denominator, "
                f"not removed from the denominator: removing it would mean \"not answering a hard dimension costs no points\". "
                f"before imputation {r['score_before_imputation']}, after imputation {r['score']}")


REPORTED_DIAGNOSTICS = {
    "dx_listed_n0": "A6 profile: dx_listed under the strict cap n + 0 (scored under n + 1)",
    "action_consistency": "A4 profile: semantic auxiliary atom, the action fits the model's own differential "
                          "(99% of cells tie across the top eight)",
    "tool_grounded_joint": "measures the propensity to order checks (corr. 0.91 with mean checks ordered; "
                           "a question-blind cheapest-first stub scores 0.546) and double-counts tests F1",
}


def rank_ddx(rows: list[dict]) -> list[dict]:
    """Ranking for diagnosis questions; the early-warning table does not apply (a constant answering the placeholder driver maxes out its Track C).

    Judges by gold kind: dx_hit (unified/comorbidity), held_independent
    (independent), join_hit, and the disposition dimension.
    """
    # One board, one judging version (`anchor.judging_vintages()`). Reports only:
    # unstamped historical batches must stay readable.
    from haenv.provenance.anchor import judging_vintages as _jvint
    from haenv.provenance.anchor import world_vintages as _wvint
    _vint = _jvint(rows)
    if len(_vint) > 1:
        log.error("[report] this batch of eval has %d judging vintages: %s -- "
                  "the numbers on the board come from different judging code and are not comparable. "
                  "Did a resumed run cross a judging change? Clear eval.jsonl and rerun that batch.",
                  len(_vint), _vint)
    # The world stamp is reported the same way (`world_vintages`).
    _wv = _wvint(rows)
    if len(_wv) > 1:
        # Grouping key: `world_sha . world_knobs . job=<job_sha256[:8]>`; some patient
        # inputs travel through the job yaml, not the GENERATION segment.
        log.error("[report] this batch of eval has %d world vintages: %s -- "
                  "the items come from different item-generation code/knobs/patients; the rows on the board are not comparable. "
                  "Read them grouped by the `world_sha . world_knobs . job` triple, or regenerate items and rerun the whole batch.",
                  len(_wv), _wv)

    from haenv.scoring.legacy import load_profile
    _prof = load_profile()
    _join_sel = join_scored_classes(rows)
    _scope_sel = join_scored_classes(rows, metric="join_scope_ok")
    # The anti-gaming anchor is batch-level, computed once outside the per-model loop.
    _scope_scored = list(_scope_sel["scored"])
    if "unified" not in _scope_scored and "unified" in JOIN_CLASSES:
        _scope_scored = ["unified"] + _scope_scored
    _scope_floor = _scope_const_floor(_scope_scored)
    # Unexercised dimensions (strict zero variance) are removed from the
    # denominator batch-wide (`unexercised_dims`). `per_solver` dimensions are
    # computed first with `review_macro_of` and checked at the aggregate layer.
    _by_pre: dict[str, list[dict]] = {}
    for _r in rows:
        if str(_r.get("gold_kind", "")).startswith("ddx:"):
            _by_pre.setdefault(_r.get("solver", "?"), []).append(_r)
    _real_pre, _ = real_solver_pool(rows)
    _ps_vals: dict[str, dict] = {"review_macro": {}, "review_utility": {}, "abst_utility": {},
                                 "review_utility_cc": {}, "abst_utility_cc": {}}
    for _s, _rs in _by_pre.items():
        if _s not in _real_pre:
            continue  # stubs excluded (designed-in variance)
        _rv_pre = review_macro_of(_rs) or {}
        _ps_vals["review_macro"][_s] = _rv_pre.get("review_macro")
        _ps_vals["review_utility"][_s] = _rv_pre.get("review_utility")
        _ps_vals["review_utility_cc"][_s] = _rv_pre.get("review_utility_cc")
        _ab_pre = abstention_utility_of(_rs)
        _ps_vals["abst_utility"][_s] = _ab_pre.get("abst_utility")
        _ps_vals["abst_utility_cc"][_s] = _ab_pre.get("abst_utility_cc")
    # M1 D3: a scored dimension enters the composite only on the geometries its profile
    # entry lists (`scored_on`); elsewhere it is reported, not scored.
    try:
        from haenv.scoring.absence import geometry_of as _geom_of
        _batch_geom = _geom_of(rows)
    except Exception:                                  # noqa: BLE001
        _batch_geom = None
    _report_only = {d for d in _prof.scored_dims if not _prof.scored_on(d, _batch_geom)}
    _unex = unexercised_dims(rows, per_solver=_ps_vals)
    # Judging provenance, batch-level (see `judging_provenance`).
    _prov = judging_provenance(rows)
    by: dict[str, list[dict]] = {}
    for r in rows:
        if str(r.get("gold_kind", "")).startswith("ddx:"):
            by.setdefault(r.get("solver", "?"), []).append(r)
    out = []
    for name, rs in by.items():
        n = len(rs)
        # Hard-gate column counts `gated_units`, matching the multiplier; on sliced packs
        # `overall` never shows per-slice hits. `gate_fail_rows` keeps the per-row count.
        gate_rows = sum(1 for r in rs if str(r.get("overall", "")).startswith("FAIL"))
        # Spec §34: zeroing scope is the scored unit (see `gate_multiplier`).
        _gm = gate_multiplier(rs)
        dxr = [r for r in rs if isinstance(r.get("dx_hit"), (int, float, bool))]
        indr = [r for r in rs if "held_independent" in r]
        jr = [r for r in rs if r.get("join_said") is not None or "join_hit" in r]
        ur = [r for r in rs if "urgency_ok" in r]
        gate = len(_gm.get("gated_units") or [])
        rec = {"model": name, "n": n, "gate_fail": gate, "gate_fail_rows": gate_rows,
               "gate_unit": _gm["unit"], "n_units": _gm["n_units"],
               "n_gated_units": _gm["n_gated_units"], "gated_units": _gm["gated_units"],
               "n_gate_unknown": _gm["n_gate_unknown"],
               "gate_unknown_cases": _gm["gate_unknown_cases"],
               "gate_verified": _gm["verified"],
               "dx_hit": _mean([1.0 if r.get("dx_hit") else 0.0 for r in dxr]),
               "dx_top1": _mean([1.0 if r.get("dx_hit_top1") else 0.0 for r in dxr]),
               "n_dx": len(dxr),
               "held_ind": _mean([1.0 if r.get("held_independent") else 0.0 for r in indr]),
               "n_ind": len(indr),
               "join_hit": _mean([1.0 if r.get("join_hit") else 0.0 for r in jr]),
               "urgency_ok": _mean([1.0 if r.get("urgency_ok") else 0.0 for r in ur]),
               # read through `_rowdim` (sliced geometry writes `wk_*` names)
               "tests_recall": _mean([_rowdim(r, "tests_recall") for r in rs]),
               "tests_prec": _mean([_rowdim(r, "tests_precision") for r in rs]),
               # Per-dimension aggregation; a dimension absent on this batch stays `None`, never 0.
               "disc_recall": _mean([_rowdim(r, "disc_recall") for r in rs]),
               "rev_stability": _mean([r.get("rev_stability") for r in rs]),
               "rev_responsiveness": _mean([r.get("rev_responsiveness") for r in rs]),
               # Every profile `role: dim` must be read here (else it lands in
               # `core_absent_dims`); `n_*` fields record each dimension's judgeable cells.
               "noop_ok": _mean([1.0 if r.get("noop_ok") else 0.0
                                 for r in rs if isinstance(r.get("noop_ok"), (int, float, bool))]),
               "quant_ok": _mean([1.0 if r.get("quant_ok") else 0.0
                                  for r in rs if "quant_ok" in r]),
               "excl_grounded_rate": _mean([r.get("excl_grounded_rate") for r in rs]),
               # ---- tool track T1-T3 ----
               # Direction is folded by `_signed()` (`tool_dup_rate` lower-is-better,
               # `tool_budget_used` non-monotone). Absent on non-gated geometries (`applies_to:
               # [gated]`) => not applicable. Read via `_rowdim`, which resolves the
               # `tool_grounded_rate` alias.
               "tool_target_grounded_rate": _mean(
                   [_rowdim(r, "tool_target_grounded_rate") for r in rs]),
               "tool_dup_rate": _mean([_rowdim(r, "tool_dup_rate") for r in rs]),
               "tool_budget_used": _mean([_rowdim(r, "tool_budget_used") for r in rs]),
               "n_tool_grounded": sum(1 for r in rs if isinstance(
                   _rowdim(r, "tool_target_grounded_rate"), (int, float))),
               "n_tool_dup": sum(1 for r in rs if isinstance(
                   _rowdim(r, "tool_dup_rate"), (int, float))),
               "n_tool_budget": sum(1 for r in rs if isinstance(
                   _rowdim(r, "tool_budget_used"), (int, float))),
               "n_noop_ok": sum(1 for r in rs if isinstance(r.get("noop_ok"), (int, float, bool))),
               "n_quant_ok": sum(1 for r in rs if "quant_ok" in r),
               "n_excl_grounded": sum(1 for r in rs
                                      if isinstance(r.get("excl_grounded_rate"), (int, float)))}
        # Dimensions recovered from slice names are recorded in the row. `_dim`'s
        # record-level `wk_*` fallback below does not fire (`rec` has no `wk_*` keys).
        rec["dims_from_slices"] = [
            q.canonical for q in _QUANTITIES if (q.row_aliases or q.renamed_from)
            if all(r.get(q.canonical) is None for r in rs)
            and any(r.get(a) is not None for a in (q.row_aliases + q.renamed_from) for r in rs)]
        # ---- composite dimensions ----
        # Excluded from the composite (still reported): `dx_hit` (saturates on a small
        # panel), `urgency_ok` (near its constant ceiling), and `tests_recall` alone
        # (rewards verbosity; paired with `tests_precision`).
        rec["join_by_class"] = {k: _mean([r.get("join_hit") for r in rs
                                          if join_class_of(r) == k]) for k in JOIN_CLASSES}
        rec["join_n_by_class"] = {k: sum(1 for r in rs if join_class_of(r) == k)
                                  for k in JOIN_CLASSES}
        _sc = [rec["join_by_class"][k] for k in _join_sel["scored"]
               if rec["join_by_class"].get(k) is not None]
        rec["join_macro"] = _fsum_mean(_sc)
        rec["join_scored_classes"] = list(_join_sel["scored"])
        rec["join_macro_const_floor"] = (round(1.0 / len(_join_sel["scored"]), 3)
                                         if _join_sel["scored"] else None)

        # ---- `join_scope_ok` also enters the composite ----
        # `join_hit` scores the `join_type` pick; `join_scope_ok` scores whether top-1
        # coverage of true symptoms falls in the class band. They can rank models
        # differently and enter side by side.
        rec["scope_by_class"] = {k: _mean([r.get("join_scope_ok") for r in rs
                                           if join_class_of(r) == k]) for k in JOIN_CLASSES}
        # `unified` stays in this macro average despite zero variance: without it,
        # "always cite one real symptom" would reach 1.000 on the remaining classes. It
        # serves as a normalization anchor and is labelled "not evidence" in the report.
        _sp = [rec["scope_by_class"][k] for k in _scope_scored
               if rec["scope_by_class"].get(k) is not None]
        rec["scope_macro"] = _fsum_mean(_sp)
        rec["scope_scored_classes"] = _scope_scored
        rec["scope_anchor_note"] = ("unified is zero-variance and stays in the macro average as an anti-gaming anchor: "
                                    "it adds the same constant to every model, changes no ranking, and keeps the question-blind floor "
                                    "at 0.667 instead of 1.000. Not evidence.")
        # The constant floor is not 1/k; see `_scope_const_floor`.
        rec["scope_macro_const_floor"] = _scope_floor

        rec.update(review_macro_of(rs))
        rec.update(abstention_utility_of(rs))
        # Roles come from `registry/scoring.yaml`; `scoring.profile_drift` reports when a batch disagrees.
        rec["scoring_profile"] = _prof.profile_id
        # `scoring_profile_sha` changes with profile content even when `profile_id` does not.
        rec["scoring_profile_sha"] = _prof.content_sha256[:16]
        rec["scored_dims"] = list(_prof.scored_dims)
        rec["validity_unmet"] = _prof.unmet_validity()
        rec["reported_not_scored"] = {
            "join_hit(total)": "sensitive to mix ratio: a model that always answers one class scores off the item pack's class ratio; read per class only",
            "dx_top1": "subset of dx_hit; kept diagnostic to avoid double weight",
            "urgency_ok": "fails the full dimension checkup (ceiling/floor/mid-band)",
            "held_ind": "n_ind is too small, cannot be read as its own dimension"}
        rec["reported_not_scored"].update(
            {f"join:{k}": v for k, v in _join_sel["excluded"].items()})
        # Ceiling-warned classes are scored but reported apart from excluded ones.
        if _join_sel.get("ceiling"):
            rec["join_ceiling_warnings"] = dict(_join_sel["ceiling"])
        # Core dimensions come from the profile in its order; on sliced batches the
        # tests/disc dimensions are read from the last slice (`wk_*`), and this is
        # recorded.
        _SLICE_ALIAS = {"tests_recall": "wk_tests_recall_last",
                        "tests_precision": "wk_tests_precision_last",
                        "disc_recall": "wk_disc_recall_last"}
        # Every `role: dim` in the profile is read from rows here (adds, never
        # overrides values computed above). `aggregate: per_solver` dimensions are
        # computed separately.
        for _d in _prof.scored_dims:
            _rk = _REC_KEY.get(_d, _d)
            if _rk in rec and rec[_rk] is not None:
                continue
            if str((_prof.metrics.get(_d) or {}).get("aggregate") or "") == "per_solver":
                continue
            _v = _mean([_rowdim(r, _d) for r in rs]) if _d in _QNAMES else None
            if _v is not None:
                rec[_rk] = _v
                rec.setdefault("dims_from_registry", []).append(_d)

        # Diagnostics that are read every time although they do not score (M1 amendment
        # 2026-10-01): the record carries the mean so the board can print it.
        for _d, _why in REPORTED_DIAGNOSTICS.items():
            _rk = _REC_KEY.get(_d, _d)
            if rec.get(_rk) is None and _d in _QNAMES:
                _v = _mean([_rowdim(r, _d) for r in rs])
                if _v is not None:
                    rec[_rk] = _v
            rec["reported_not_scored"].setdefault(_d, _why)

        def _dim(d):
            v = rec.get(_REC_KEY.get(d, d))
            if v is None and d in _SLICE_ALIAS:
                v = rec.get(_SLICE_ALIAS[d])
                if v is not None:
                    rec.setdefault("dims_from_slices", []).append(d)
            return v
        # ---- harmonic aggregation (`pair_aggregate: harmonic`) ----
        # The pair takes one slot as F1, so maxing recall alone cannot game it; both raw
        # readings stay reported.
        _pairs = {a: b for a, b in _prof.harmonic_pairs()}
        _paired = set(_pairs) | set(_pairs.values())
        _f1: dict[str, float | None] = {}
        for _a, _b in _pairs.items():
            # F1 per cell, then averaged (not F1 of the means), via
            # `separation.paired_grid_values`.
            from haenv.scoring.separation import paired_grid_values as _pgv
            # Insufficient-tier pre-registration 2026-10-01: the pair is not applicable on the
            # gold kinds its profile entries list (`not_scored_on_kinds`); raw recall/precision stay reported.
            _rs_f1 = [r for r in rs if _prof.scored_on_kind(_a, r.get("gold_kind"))
                      and _prof.scored_on_kind(_b, r.get("gold_kind"))]
            _vals = _pgv(_rs_f1, _a, _b, _rowdim)
            _ok = [v for v in _vals if v is not None]
            rec[f"n_{_a}_f1_kind_excluded"] = len(rs) - len(_rs_f1)
            _f1[_a] = _fsum_mean(_ok, 4)
            rec[f"{_a}_f1"] = _f1[_a]
            rec[f"n_{_a}_f1"] = len(_ok)
        # ---- direction ----
        # The composite is a higher-is-better mean: `lower` dimensions enter as `1 - v`,
        # `non_monotone` ones never enter.
        def _signed(_name: str, _v):
            """Fold one dimension to higher-is-better per the profile's direction; `None` for `non_monotone`."""
            if _v is None:
                return None
            # Also records out-of-range `score01` values (`note_out_of_range`); reports only.
            _prof.note_out_of_range(_name, _v)
            _d = _prof.direction(_name)
            if _d == "lower":
                return _rd(1.0 - float(_v), 4)
            if _d == "non_monotone":
                return None  # recorded in core_nonmonotone_dims
            return _v

        # ---- `paired_with` without `pair_aggregate`: both sides must be present ----
        # E.g. `tool_dup_rate = 0` is free without real queries, so it scores only when
        # its partner can be computed.
        def _pair_missing(_name: str) -> bool:
            _m = _prof.metrics.get(_name) or {}
            _other = str(_m.get("paired_with") or "")
            if not _other or _m.get("pair_aggregate"):
                return False
            return _dim(_other) is None

        # Unexercised dimensions take no slot. Kept apart from `core_absent_dims`
        # (not computable in this cell) and `core_nonmonotone_dims` (declared
        # non-monotone): each has a different fix.
        _core_pairs = [(d, (_f1[d] if d in _f1 else _dim(d)))
                       for d in _prof.scored_dims
                       if not (d in _paired and d not in _f1) and d not in _unex
                       and d not in _report_only]
        rec["core_report_only_dims"] = sorted(_report_only) or None
        rec["batch_geometry"] = _batch_geom
        rec["retired_dims"] = _prof.retired() or None
        rec["core_unexercised_dims"] = [d for d in _prof.scored_dims if d in _unex]
        rec["core_unexercised_why"] = {d: _unex[d] for d in rec["core_unexercised_dims"]} or None
        # Judging provenance travels with every reading.
        rec["measured_under"] = _prov["measured_under"]
        rec["judging_current"] = _prov["judging_current"]
        rec["judging_stale"] = _prov["stale"]
        rec["judging_stale_why"] = _prov["why"]
        core = [(None if _pair_missing(d) else _signed(d, v)) for d, v in _core_pairs]
        _core_names = [(f"{d}+{_pairs[d]}" if d in _f1 else d) for d, _ in _core_pairs]
        # Non-monotone dimensions are listed separately from missing ones.
        rec["core_nonmonotone_dims"] = [
            d for d, v in _core_pairs
            if v is not None and _prof.direction(d) == "non_monotone"]
        rec["core_inverted_dims"] = [
            d for d, v in _core_pairs
            if v is not None and _prof.direction(d) == "lower"
            and not _pair_missing(d)]
        # Scored alone but partner missing => listed separately.
        rec["core_unpaired_dims"] = [d for d, v in _core_pairs
                                     if v is not None and _pair_missing(d)]
        # The number of items used is recorded, and all-`None` gives `None`, not 0.0.
        used = [c for c in core if c is not None]
        rec["n_core_used"] = len(used)
        rec["n_core_total"] = len(core)
        # Names, not just a count: the ruler check compares sets.
        rec["core_used_dims"] = [d for d, c in zip(_core_names, core) if c is not None]
        # Scored dimensions this cell could not compute (non-monotone ones excluded).
        _nonmono = set(rec["core_nonmonotone_dims"]) | set(rec["core_unpaired_dims"])
        rec["core_absent_dims"] = [d for d, c in zip(_core_names, core)
                                   if c is None and d not in _nonmono]
        # Fewer than `MIN_GRIDS_FOR_SCORE` comparable cells => score `None` with the reason.
        rec["n_grids"] = n
        if n < MIN_GRIDS_FOR_SCORE:
            rec["score"] = None
            rec["score_none_reason"] = (
                f"only {n} comparable cells (< {MIN_GRIDS_FOR_SCORE}) -- a score computed from one cell "
                f"cannot sit in the same column as a score computed from a whole pack")
        elif not used:
            rec["score"] = None
            rec["score_none_reason"] = "not one scored dimension could be computed"
        else:
            rec["score"] = _rd(_mean(used) * _gm["mult"], 3) if n else None
        # ---- `score_full`: every score01/binary dimension, equal weight, any role ----
        # Goes through `_signed` like `score`, so its ranking is comparable.
        _full = [_signed(d, _dim(d)) for d in _prof.full_dims()]
        _fu = [c for c in _full if c is not None]
        rec["n_full_used"], rec["n_full_total"] = len(_fu), len(_full)
        rec["score_full"] = (_rd(_mean(_fu) * _gm["mult"], 3) if (_fu and n) else None)
        from haenv.scoring.semantic.coverage import protect_composite
        protect_composite(rec, rs)
        # Out-of-geometry stubs sort after in-scope rows (as in `rank_models`).
        rec.update(stub_geometry_scope(name, rs))
        out.append(rec)
    # The batch-level out-of-range ledger is attached to every row as
    # `batch_out_of_range` so the report can print it.
    _oor = _prof.out_of_range_report()
    for rec in out:
        rec["batch_out_of_range"] = _oor or None  # None when clean
    _apply_one_ruler(out)
    # Same sort key as `rank_models`: `None` after genuine zeros.
    return sorted(out, key=_board_order)
