"""Dimension values, statistical health and exercise eligibility."""
from __future__ import annotations

from haenv.scoring.quantities import BY_NAME as _QBY
from haenv.scoring.quantities import resolve as _qresolve
import logging

log = logging.getLogger("haenv.analytics")


METRIC_KINDS: dict[str, str] = {
    "dx_hit": "binary", "dx_hit_top1": "binary", "dx_mentioned": "binary",
    "dx_ruled_out_gold": "binary", "dx_all_threads": "binary",
    "join_hit": "binary", "join_top1_covers_all": "binary", "join_self_contradiction": "binary",
    # Behavioral-convention join dimension; sits alongside `join_hit`.
    "join_scope_ok": "binary",
    "join_contradiction_cover": "binary", "top1_claims_unified": "binary",
    "top1_compound": "binary", "specialty_ok": "binary",
    "urgency_ok": "binary", "urgency_within1": "binary", "a1": "binary",
    "tests_recall": "score01", "tests_precision": "score01",
    "join_top1_cover_frac": "score01", "dx_coverage": "score01",
    "n_drivers": "count", "n_drivers_with_evidence": "count", "tests_proposed": "count",
    "tests_named": "count", "tests_total": "count", "top1_n_evidence": "count",
    "dx_n_candidates": "count", "dx_threads_matched": "count",
    "dx_rank": "count", "dx_mentioned_rank": "count",
    "urgency_gap": "error",
    "rival_considered": "binary", "rival_ruled_out": "binary", "rival_top1_live": "binary",
    # `rival_recall` divides per case; `rival_recall_capped2` fixes the denominator at 2.
    "rival_recall": "score01", "rival_recall_capped2": "score01",
    # Discriminative tooling. `disc_n_*` is background, not correctness (read with
    # the order count). The two revision diagonals are registered separately: no
    # degenerate strategy scores well on both.
    "rev_stability": "score01", "rev_responsiveness": "score01",
    "rev_n_neutral": "count", "rev_n_substantive": "count",
    "disc_recall": "score01", "disc_covered": "count",
    "disc_n_rivals": "count", "disc_n_tests_proposed": "count",
    "rival_n_declared": "count",
}


CORRECTNESS_METRICS: frozenset = frozenset({
    "dx_hit", "dx_hit_top1", "dx_mentioned", "dx_ruled_out_gold", "dx_all_threads",
    "dx_coverage", "dx_listed", "join_hit", "join_top1_covers_all", "join_top1_cover_frac",
    "specialty_ok", "urgency_ok", "urgency_within1", "a1",
    "join_scope_ok",
    "tests_recall", "tests_precision",
    "rival_ruled_out",
    # `disc_recall` has no anchor reading, so it stays diagnostic.
    "disc_recall",
    "rev_stability", "rev_responsiveness",
    # Excluded: `rival_considered` (must be read with `rival_ruled_out`),
    # `rival_top1_live` (lower is better), `rival_n_declared` (a property of the
    # question).
})


def dimension_health(rows: list[dict], key: str, models: tuple | None = None,
                     baseline_solvers: tuple | None = None) -> dict:
    """Whether one scored dimension can be read as capability.

    The panel excludes every registered stub (`BASELINE_NAMES`), whose variance is
    designed in. Needs >= 3 real models; otherwise reports none, not 0. When all
    models' totals are (nearly) identical, rest-score r_pb is uninterpretable and is
    reported as such rather than as negative discrimination.
    """
    import statistics as _st
    # The profile's `metric_type` is used when `METRIC_KINDS` has no entry.
    if baseline_solvers is None:
        from haenv.world.generation.baselines import BASELINE_NAMES as _BN_H
        baseline_solvers = tuple(_BN_H)
    kind = METRIC_KINDS.get(key)
    if kind is None:
        try:
            from haenv.scoring.legacy import load_profile as _lp_k
            kind = ((_lp_k().metrics.get(key) or {}).get("metric_type")) or None
        except Exception:
            kind = None
    if kind is None:
        return {"key": key, "usable_as_capability": False,
                "verdict": f"metric type not registered ({key}); cannot be judged. "
                                      f"Declare one of binary / score01 / count / error "
                                      f"in report.METRIC_KINDS"}
    per: dict = {}
    n_frac = 0
    for r in rows:
        # Values go through `_rowdim` (the quantity registry) to pick up aliases and derived fields.
        v = _rowdim(r, key)
        if v is None or r.get("solver") in baseline_solvers:
            continue
        # Branch on the registered kind; continuous values are never coerced to bool.
        if kind == "binary":
            per.setdefault(r.get("case"), {})[r.get("solver")] = bool(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            n_frac += 1
            per.setdefault(r.get("case"), {})[r.get("solver")] = float(v)
        else:
            continue
    ms = tuple(models) if models else tuple(sorted({m for v in per.values() for m in v}))
    items = {c: v for c, v in per.items() if all(m in v for m in ms)}
    # Two denominators: `n_items` (items every model answered) for paired
    # statistics, and valued cells for the zero-variance judgment.
    _allcells = [per[c][m] for c in per for m in ms if m in per[c]]
    _allset = {v for v in _allcells}
    out: dict = {"key": key, "n_models": len(ms), "models": list(ms),
                 "n_items": len(items), "n_cells": len(_allcells)}
    if len(ms) < 3:
        out["usable_as_capability"] = False
        out["verdict"] = f"insufficient panel ({len(ms)} < 3 answering models) -- this dimension cannot be computed, it is not 0"
        return out
    # ---- a constant dimension is not evidence ----
    # Zero variance is its own tier, separate from saturation: across the whole batch
    # (needs a different question) or across models (models cannot be told apart).
    # Checked before the pairing threshold.
    out["distinct_values"] = len(_allset)
    if _allcells and len(_allset) == 1:
        out["zero_variance"] = "all"
        out["usable_as_capability"] = False
        out["verdict"] = (f"zero variance: {len(_allcells)} value-bearing cells are constantly "
                          f"{next(iter(_allset))}; per spec §17 this is no evidence "
                          f"(not 'saturated' or 'achieved')")
        return out
    if not items:
        out["usable_as_capability"] = False
        out["verdict"] = (f"no fully-paired item ({len(_allcells)} value-bearing cells, but not one case has "
                          f"a value for all {len(ms)} models)")
        return out
    n = len(items)
    _pm = {m: sum(items[c][m] for c in items) for m in ms}
    if len(set(_pm.values())) == 1:
        out["zero_variance"] = "across_models"
        out["per_model"] = {m: f"{_pm[m]}/{n}" for m in ms}
        out["usable_as_capability"] = False
        out["verdict"] = (f"zero variance across models: all {len(ms)} models are identical ({next(iter(_pm.values()))}/{n})"
                          f"; this dimension cannot tell models apart (questions still differ from each other); no capability conclusion under spec §17")
        return out

    # Continuous quantity: report distribution and range.
    if n_frac:
        out["value_type"] = kind
        pm = {m: _st.mean(items[c][m] for c in items) for m in ms}
        out["per_model"] = {m: round(pm[m], 3) for m in ms}
        out["spread"] = round(max(pm.values()) - min(pm.values()), 3)
        out["headroom"] = round(1 - max(pm.values()), 3)
        item_mean = {c: _st.mean(items[c][m] for m in ms) for c in items}
        out["items_at_1.0"] = round(sum(1 for v in item_mean.values() if v >= 0.999) / n, 3)
        out["items_at_0.0"] = round(sum(1 for v in item_mean.values() if v <= 0.001) / n, 3)
        # The detail names which check failed (spread or headroom).
        if kind == "score01":
            # Saturation = headroom < 0.15 AND most questions at the ceiling, matching the
            # binary branch. Low headroom alone is printed but does not disqualify.
            _saturated = out["headroom"] < 0.15 and out["items_at_1.0"] >= 0.5
            _bad_c = ([] if out["spread"] >= 0.05 else [f"spread={out['spread']}<0.05"]) + \
                     ([] if not _saturated else
                      [f"saturated: headroom={out['headroom']}<0.15 and "
                       f"items@1.0={out['items_at_1.0']}>=0.5"])
            out["saturated"] = bool(_saturated)
            out["usable_as_capability"] = bool(not _bad_c and key in CORRECTNESS_METRICS)
            _tight = ("" if out["headroom"] >= 0.15 else
                      f"; headroom at the top is only {out['headroom']} (not saturated: items@1.0="
                      f"{out['items_at_1.0']}<0.5)")
            out["verdict"] = ((("can be read as capability" if key in CORRECTNESS_METRICS else "a behavior-descriptive quantity, not right/wrong")
                               + " ([0,1]: has a range and is not saturated)" + _tight) if not _bad_c
                              else "score01, does not qualify: " + " / ".join(_bad_c))
        elif kind == "count":
            out.pop("headroom", None)  # a count has no upper bound
            out["usable_as_capability"] = bool(out["spread"] >= 0.05
                                               and key in CORRECTNESS_METRICS)
            out["verdict"] = ((("can be read as capability" if key in CORRECTNESS_METRICS
                                else "a behavior-descriptive quantity, not right/wrong") + " (count: has a range)")
                              if out["spread"] >= 0.05
                              else f"count, range too narrow (spread={out['spread']})")
        else:  # error: ideal value is 0
            out.pop("headroom", None)
            out["worst_abs_mean"] = round(max(abs(v) for v in out["per_model"].values()), 3)
            # a bias quantity measures calibration and never enters the composite
            out["usable_as_capability"] = False
            out["verdict"] = (f"a bias quantity (ideal 0): worst |mean| {out['worst_abs_mean']}, "
                              f"range {out['spread']}; measures calibration, does not enter the composite score")
        out["note"] = "the ceiling/floor share convention does not apply to non-binary quantities; binarizing it would misjudge it as saturated"
        return out
    out["value_type"] = "binary"
    ps = {c: sum(v[m] for m in ms) / len(ms) for c, v in items.items()}
    out["ceiling_frac"] = round(sum(1 for p in ps.values() if p == 1.0) / n, 3)
    out["floor_frac"] = round(sum(1 for p in ps.values() if p == 0.0) / n, 3)
    out["mid_frac"] = round(sum(1 for p in ps.values() if 0.2 < p < 0.8) / n, 3)
    tot = {m: sum(items[c][m] for c in items) for m in ms}
    out["per_model"] = {m: f"{tot[m]}/{n}" for m in ms}
    out["headroom"] = round(1 - max(tot.values()) / n, 3)
    # Rest-score r_pb needs a real capability ordering: when best and worst models
    # differ by fewer than 2 cells (or under 10% of the questions), it is noise.
    _span = max(tot.values()) - min(tot.values())
    if _span < max(2, 0.10 * n):
        out["rpb_median"] = None
        out["rpb_span"] = _span
        out["rpb_note"] = (f"the spread of per-model totals is only {_span}/{n} (best {max(tot.values())}, worst "
                           f"{min(tot.values())}) -> rest-score r_pb is dominated by individual cells, "
                           f"there is no comparable capability ordering, so this quantity is uninterpretable (neither positive nor negative discrimination)")
    else:
        rs = []
        for c in items:
            xs = [1.0 if items[c][m] else 0.0 for m in ms]
            ys = [sum(items[k][m] for k in items if k != c) for m in ms]
            if len(set(xs)) < 2 or len(set(ys)) < 2:
                continue
            mx, my = _st.mean(xs), _st.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
            if den:
                rs.append(num / den)
        out["rpb_median"] = round(_st.median(rs), 3) if rs else None
        out["rpb_n"] = len(rs)
    gates = {"ceiling≤0.15": out["ceiling_frac"] <= 0.15,
             "floor≤0.15": out["floor_frac"] <= 0.15,
             "mid≥0.50": out["mid_frac"] >= 0.50,
             "headroom≥0.25": out["headroom"] >= 0.25}
    if out.get("rpb_median") is not None:
        gates["r_pb≥0.15"] = out["rpb_median"] >= 0.15
    out["gates"] = gates
    bad = [k for k, v in gates.items() if not v]
    _cap = "can be read as capability" if key in CORRECTNESS_METRICS else "a behavior-descriptive quantity, not right/wrong"
    # Explicit boolean for `scoring.profile_drift`, so callers never match on `verdict` wording.
    out["usable_as_capability"] = bool(not bad and key in CORRECTNESS_METRICS)
    out["verdict"] = (_cap if not bad else
                      ("saturated, should not enter the composite score" if out["ceiling_frac"] >= 0.5 and out["headroom"] < 0.15
                       else "does not qualify: " + " / ".join(bad)))
    return out


_REC_KEY = {"tests_precision": "tests_prec"}


def _rowdim(r: dict, d: str):
    """Read one scored dimension from one cell, falling back to the slice (`wk_*`) name."""
    # Aliases come from the quantity registry, shared by every extraction path.
    if d in _QBY:
        return _qresolve(r, d)
    return r.get(d)


MIN_GRIDS_FOR_SCORE = 20


_QNAMES = frozenset(_QBY)


MIN_ITEMS_FOR_EXERCISE = 10


MIN_CELLS_FOR_EXERCISE = 10


def real_solver_pool(rows: list[dict]) -> tuple[set[str], list[str]]:
    """Who counts as a real model: returns `(pool, unregistered_names)`.

    The pool is every solver not in `BASELINE_NAMES`; names not in the configured
    models are returned (and logged) so any extra vote is visible. Shared by
    `unexercised_dims` and `_apply_one_ruler`.
    """
    from haenv.world.generation.baselines import BASELINE_NAMES as _BN_P
    _stub = set(_BN_P)
    pool = {str(r.get("solver")) for r in rows if str(r.get("solver")) not in _stub}
    known: set[str] = set()
    try:
        from haenv.app.config import load_cfg as _lc_p
        known = {str(k) for k in (_lc_p().get("models") or {})}
    except Exception:                                          # noqa: BLE001
        pass
    unknown = sorted(pool - known) if known else []
    if unknown:
        log.warning("[report] unregistered solver names entered the \"real model pool\": %s -- "
                    "they will take part in zero-variance judging and ruler majority voting. "
                    "Not in `config.yaml:models` (%d registered), and not in the stub list either.",
                    unknown, len(known))
    return pool, unknown


def unexercised_dims(rows: list[dict], per_solver: dict | None = None) -> dict[str, str]:
    """Scored dimensions not exercised in this batch, as `{dimension: reason}`; these are removed from the composite denominator.

    Only strict zero variance counts, judged on real models; saturation
    is a convention decision and is left to the profile. Batches with fewer than
    `MIN_ITEMS_FOR_EXERCISE` items or 3 real models are not judged.
    """
    try:
        from haenv.scoring.legacy import load_profile as _lp_u
        dims = tuple(_lp_u().scored_dims)
    except Exception:                                          # noqa: BLE001
        return {}
    _pool, _ = real_solver_pool(rows)
    real = [r for r in rows if str(r.get("solver")) in _pool]
    # Fewer than 3 real models => no judgment (an offline smoke batch is all stubs).
    _real_models = {str(r.get("solver")) for r in real}
    if len(_real_models) < 3:
        return {}
    _prof_u = _lp_u()
    _per_solver = per_solver or {}
    out: dict[str, str] = {}
    for d in dims:
        # `aggregate: per_solver` dimensions are not row fields; they are skipped
        # here and checked at the aggregate layer instead.
        if str((_prof_u.metrics.get(d) or {}).get("aggregate") or "") == "per_solver":
            _pv = [v for m, v in (_per_solver or {}).get(d, {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(_pv) >= 3 and len({round(x, 4) for x in _pv}) == 1:
                out[d] = (f"this batch cannot separate the models ({len(_pv)} real models are constantly {_pv[0]}); "
                          f"an `aggregate: per_solver` dimension, invisible to the per-row checkup, "
                          f"so it is checked at the aggregate layer. No capability conclusion under spec §17; excluded from the denominator")
            continue
        h = dimension_health(real, d)
        # A dimension fewer than 3 real models can answer does not enter the composite
        # (a symptom of mixed judging versions). Values missing only on stubs are fine.
        if h.get("n_models", 0) < 3:
            out[d] = (f"this batch only has values on stubs: {h.get('n_models', 0)} real models can answer "
                      f"(< 3), so it can only separate stubs, not models. "
                      f"In the composite score, solvers with a value would get one extra dimension over the ones without "
                      f"(different rulers), so it is excluded from the denominator")
            continue
        zv = h.get("zero_variance")
        # The `all` tier uses valued cells (`MIN_CELLS_FOR_EXERCISE`); the
        # `across_models` tier uses paired items.
        if zv == "all":
            if h.get("n_cells", 0) < MIN_CELLS_FOR_EXERCISE:
                continue
            out[d] = (f"this batch never exercised it: {h['n_cells']} value-bearing cells ({h['n_models']} real models) "
                      f"are constantly the same number; spec §17: \"a dimension whose value never varies constitutes no evidence at all\". "
                      f"In the composite score it would only dilute the dimensions with real discrimination, so it is excluded from the denominator")
            continue
        if h.get("n_items", 0) < MIN_ITEMS_FOR_EXERCISE:
            continue
        if zv == "across_models":
            out[d] = (f"this batch cannot separate the models: all {h['n_models']} real models have identical totals "
                      f"({h.get('per_model')}); the questions still differ from each other, but this dimension carries no information between models, "
                      f"so it is excluded from the denominator")
    return out


def scored_dims_still_healthy(rows: list[dict]) -> list[dict]:
    """Warn when a scored dimension is no longer healthy on this batch (saturated or zero variance); never changes the score."""
    # Watches every scored dimension the profile declares.
    warn = []
    try:
        from haenv.scoring.legacy import load_profile as _lp_h
        _keys = tuple(_lp_h().scored_dims)
    except Exception:
        _keys = ("join_hit", "tests_recall", "tests_precision")
    for k in _keys:
        h = dimension_health(rows, k)
        if "verdict" not in h or h.get("n_items", 0) < 10:
            continue
        v = h["verdict"]
        if v.startswith("can be read as capability"):
            continue
        warn.append({"kind": "scored_dim_degraded", "severity": "warn", "dim": k,
                     "detail": f"`{k}` in the composite score is now judged \"{v[:40]}\""
                               f" (r_pb={h.get('rpb_median')}, range={h.get('spread')},"
                               f" headroom={h.get('headroom')}) -- whether it should stay in the composite score should be reconsidered"})
    return warn
