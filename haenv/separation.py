"""Per-dimension separation: the best real model minus the best degenerate baseline.

Three rules keep the sign right: stubs that saturate a dimension by construction are excluded
as negative controls (and reported separately), a paired dimension is computed only on its
paired quantity, and a stub is used as a baseline only in the geometries it was written for.
A dimension passes at `SEP_THRESHOLD`; it is `vacuous` when no degenerate baseline is left
after exclusion. This module only reports; it never enters the headline score or a gate.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

SEP_THRESHOLD = 0.30

#: A stub whose mean on a dimension is >= this counts as saturated; unregistered ones are
#: reported by `unregistered_saturating`.
SATURATION_EPS = 0.999


class SeparationError(RuntimeError):
    """The convention was used incorrectly (unregistered dimension, paired dimension computed
    per-side, or a stub with no declared geometry).
    """


@dataclass(frozen=True)
class DimSpec:
    """The separation convention for one process dimension. `negative_controls` maps a stub to the
    reason it saturates by construction; the reason is mandatory.
    """
    direction: str                                  # "higher" | "lower"
    why: str                                        # what this dimension is asking
    paired_with: str | None = None                  # if paired, computable only on the paired quantity
    negative_controls: Mapping[str, str] = field(default_factory=dict)
    #: Stubs that saturate on a single side only and stop once paired. Documentation only; they are
    #: not excluded.
    single_side_saturating: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("higher", "lower"):
            raise SeparationError(
                f"direction must be higher/lower, got {self.direction!r} -- "
                "without a declared direction, a dimension where lower is better (e.g. trace_pseudo_excluded) gets its sign backwards")
        if not str(self.why).strip():
            raise SeparationError("DimSpec.why is required -- a dimension that cannot say what it measures should not be in this table")
        for _s, _w in (self.negative_controls or {}).items():
            if not str(_w).strip():
                raise SeparationError(
                    f"negative control {_s} has no reason -- an exclusion with no stated reason cannot be audited by the next person")


PROCESS_DIMS: dict[str, DimSpec] = {
    "rev_stability": DimSpec(
        direction="higher",
        why="in a neutral stretch (no new substantive information), whether the judgment stays stable",
        paired_with="rev_responsiveness",
        single_side_saturating={
            "no_revision": "never revises => stability is always 1.0 (StubbornSolver's definition)",
            "const_ddx": "always answers the same diagnosis => output is constant across slices",
            "baseline_slope": "the kernel only looks at the weight slope and has no belief-revision channel => constant across slices",
            "common_panel": "always orders the same routine panel => output is constant across slices",
            "shotgun_tests": "always orders the entire catalog => output is constant across slices",
            "humble_ref": "always declares insufficient information => output is constant across slices",
            "oracle_tests": "reads the gold standard, then outputs a constant",
            "oracle_probe_right": "reads the gold standard, then outputs a constant",
            "oracle_probe_wrong": "reads the gold standard, then outputs a constant",
            "oracle_review_right": "reads the gold standard, then outputs a constant",
            "oracle_review_wrong": "reads the gold standard, then outputs a constant",
            "gatetrip_treatment": "hits the same gate by construction every time => output is constant across slices",
            "gatetrip_invasive": "same as above",
        },
        note="14 of the 17 stubs are constantly 1.0 on this dimension -- on this dimension \"doing nothing wins\", "
             "so it can only be read paired with rev_responsiveness",
    ),
    "rev_responsiveness": DimSpec(
        direction="higher",
        why="when new substantive information appears, whether the judgment keeps up",
        paired_with="rev_stability",
        negative_controls={
            "flip_flop": "flips on every slice => responsiveness is always 1.0 (FlipFlopSolver's definition)",
        },
    ),
    "rev_list_stability": DimSpec(
        direction="higher",
        why="in a neutral stretch, whether the whole differential list (as a set of candidates) stays stable",
        paired_with="rev_list_responsiveness",
        # Excluding these would leave no baseline at all, so they are not excluded.
        single_side_saturating={
            "no_revision": "never revises => the list is constant across slices",
            "const_ddx": "always answers the same diagnosis => the list is constant",
            "flip_flop": "constant 1.0 on the list dimension (what flips is the stated position, not the set of candidate concepts)",
            "baseline_slope": "has no differential-list channel => constant",
            "common_panel": "output is constant across slices",
            "shotgun_tests": "output is constant across slices",
            "humble_ref": "output is constant across slices",
            "robust_ref": "the heuristic's output is constant across slices",
            "oracle_tests": "reads the gold standard, then outputs a constant",
            "oracle_probe_right": "reads the gold standard, then outputs a constant",
            "oracle_probe_wrong": "reads the gold standard, then outputs a constant",
            "oracle_review_right": "reads the gold standard, then outputs a constant",
            "oracle_review_wrong": "reads the gold standard, then outputs a constant",
            "gatetrip_treatment": "hits the same gate by construction every time => output is constant",
            "gatetrip_invasive": "same as above",
        },
    ),
    "rev_list_responsiveness": DimSpec(
        direction="higher",
        why="when new information appears, whether the whole differential list keeps up",
        paired_with="rev_list_stability",
        note="no saturated stubs on this dimension; the only one of the four rev dimensions "
             "that is positive computed per-side on a single pool (+0.6530), usable as a positive control",
    ),
    # ---- Tool track: grounded rate paired with budget thrift ----
    # The two point the same way only for a solver that queries selectively. `gated_probe` is a
    # scripted positive control and stays in the pool.
    "tool_target_grounded_rate": DimSpec(
        direction="higher",
        why="what fraction of the queried targets are signals this patient actually has (not decoys)",
        paired_with="tool_budget_thrift",
        negative_controls={},
        note="Paired with budget thrift; the best in-pool baseline is gated_probe "
             "(a scripted positive control, not excluded). A solver that makes zero "
             "queries has an entirely empty paired quantity and is dropped rather "
             "than scored: this pair cannot see the \"queries nothing\" end.",
    ),
    "tool_budget_thrift": DimSpec(
        direction="higher",
        why="the fraction of budget saved (= 1 - budget used) -- it only makes sense to pair with the grounded rate because they point in the same direction",
        paired_with="tool_target_grounded_rate",
        negative_controls={},
        note="`gated_shotgun` spends its full budget on this dimension -- "
             "it lands on the worst side, so it is a legitimate "
             "baseline, not excluded.",
    ),
    # ---- The process-record (trace_*) family ----
    "trace_evidence_revealed": DimSpec(
        direction="higher",
        why="whether evidence cited in the process record was actually revealed",
        note="Constant 1.0000 across 4 models on 152 cells (range 0.0000): zero-variance saturation. "
             "Per spec §17 a constant value is no evidence, so it is flagged `saturated`",
    ),
    "trace_commitments_falsifiable": DimSpec(
        direction="higher",
        why="whether each commitment states a falsification condition",
        note="Same as trace_evidence_revealed: constant 1.0000, range 0.0000 => zero-variance saturation",
    ),
    "trace_pseudo_excluded": DimSpec(
        direction="lower",           # lower is better -- fewer pseudo-exclusions
        why="the count of times a hypothesis was claimed excluded without citing a rebuttal",
        # `trace_junk` claims exclusions without a rebuttal, so it sits at the worst side: a legitimate
        # baseline, not a negative control.
        negative_controls={},
        note="A count where lower is better; without a declared direction the separation's sign is reversed. "
             "Cross-model range 0.3421 on 152 cells: the only process-record dimension with cross-model separation",
    ),
    "trace_exclusion_grounded": DimSpec(
        direction="higher",
        why="what fraction of the rebuttals cited when excluding a hypothesis resolve to visible evidence",
        note="6 distinct values on 152 cells (range 0.1088): the only continuous column with discrimination",
    ),
    "trace_n_hypotheses": DimSpec(
        direction="higher",
        why="the number of hypotheses put on the table (the scale term for trace_pseudo_excluded)",
        note="A count. Range 3.6579 over 152 cells, 10 distinct values",
    ),
}


# ---------------------------------------------------------------- pairing aggregation
def paired_grid_values(rows: Sequence[Mapping], a: str, b: str,
                       get: Callable[[Mapping, str], object] | None = None
                       ) -> list[float | None]:
    """Per-cell harmonic mean (F1) of the pair, the same convention as `report.rank_ddx`.

    Computed per cell, then averaged (not F1 of the averages). One side missing while the other is
    0 scores 0. `report` imports this implementation.
    """
    _g = get or (lambda r, k: r.get(k))
    out: list[float | None] = []
    for r in rows:
        x, y = _g(r, a), _g(r, b)
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            continue                                   # the whole pair is n/a for this cell
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            out.append(0.0 if float(x) == 0.0 else None)
            continue
        out.append(round(2 * x * y / (x + y), 4) if (x + y) > 0 else 0.0)
    return out


#: Fallback to an older key name, `{new: old}`. Shared by `report._rowdim` and `rowval` so both
#: lookups agree.
RENAMED_KEYS: dict[str, str] = {
    # Tool-call-level grounding; older batches used `tool_grounded_rate`.
    "tool_target_grounded_rate": "tool_grounded_rate",

    # `self_contradictory_exclusion` fields are stored as `ref_*` in older batches.
    "sce_n_excluded": "ref_n_excluded",
    "sce_note": "ref_note",
    "sce_contradiction_rate": "ref_contradiction_rate",
    "sce_n_with_support": "ref_n_with_support",
    "sce_unrevealed_rate": "ref_unrevealed_rate",
    "sce_unrevealed": "ref_unrevealed",
}

#: Keys derived on read from quantities already in the row (`tool_budget_thrift = 1 -
#: tool_budget_used`), so older batches remain readable.
DERIVED: dict[str, Callable[[Mapping], object]] = {
    "tool_budget_thrift": lambda r: (
        None if not isinstance(r.get("tool_budget_used"), (int, float))
        or isinstance(r.get("tool_budget_used"), bool)
        else round(1.0 - float(r["tool_budget_used"]), 3)),
}


def rowval(r: Mapping, k: str):
    """Read one quantity off a row: same name, then renamed-key fallback, then read-side derivation."""
    from .quantities import BY_NAME as _QBY
    from .quantities import resolve as _qresolve
    if k in _QBY:
        return _qresolve(r, k)
    v = r.get(k)
    if v is None and k in RENAMED_KEYS:
        v = r.get(RENAMED_KEYS[k])
    if v is None and k in DERIVED:
        v = DERIVED[k](r)
    return v


def _mean(vs: Sequence[float]) -> float | None:
    ok = [v for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(ok) / len(ok), 4) if ok else None


def _per_solver(rows: Sequence[Mapping], dim: str, spec: DimSpec,
                *, unpaired: bool = False) -> dict[str, float]:
    """This dimension's value per solver (paired dimensions through the paired quantity).
    `unpaired=True` switches to per-side aggregation.
    """
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(str(r.get("solver") or ""), []).append(r)
    out: dict[str, float] = {}
    for s, rs in by.items():
        if spec.paired_with and not unpaired:
            m = _mean([v for v in paired_grid_values(rs, dim, spec.paired_with, rowval)
                       if v is not None])
        else:
            m = _mean([rowval(r, dim) for r in rs
                       if isinstance(rowval(r, dim), (int, float))
                       and not isinstance(rowval(r, dim), bool)])
        if m is not None:
            out[s] = m
    return out


def separation(rows: Sequence[Mapping], dim: str, *,
               baseline_names: Sequence[str],
               stub_geometries: Mapping[str, frozenset] | None = None,
               geometry: str | None = None,
               allow_unpaired: bool = False) -> dict:
    """One dimension's separation. Positive means the real models beat the best degenerate
    baseline, whatever the dimension's direction.
    """
    spec = PROCESS_DIMS.get(dim)
    if spec is None:
        raise SeparationError(
            f"{dim!r} is not registered in PROCESS_DIMS -- fail-closed. "
            "Without registration there is no direction, and a dimension where lower is better gets its sign backwards")
    if spec.paired_with and not allow_unpaired:
        if dim > spec.paired_with:      # compute only on the lexicographically earlier side, to avoid computing both
            raise SeparationError(
                f"{dim!r} and {spec.paired_with!r} are a paired dimension; call it on the {spec.paired_with!r} side instead "
                "(separation is only computed on the paired quantity; a number computed per-side has no construct validity)")

    geo = geometry
    if geo is None:
        from .absence import geometry_of          # reuse the single implementation, don't write a second
        geo = geometry_of(list(rows))

    per = _per_solver(rows, dim, spec, unpaired=allow_unpaired)
    bset = set(baseline_names)
    models = {s: v for s, v in per.items() if s not in bset}

    excluded: dict[str, str] = {}
    pool: dict[str, float] = {}
    for s, v in per.items():
        if s not in bset:
            continue
        if s in (spec.negative_controls or {}):
            excluded[s] = f"negative control: {spec.negative_controls[s]}"
            continue
        if stub_geometries is not None:
            gs = stub_geometries.get(s)
            if gs is None:
                raise SeparationError(
                    f"stub {s!r} does not declare which geometries it has construct validity on -- fail-closed. "
                    "An instrument built for a different geometry mixed into the pool does not measure \"how much better the model is than a degenerate strategy\"")
            if geo not in gs:
                excluded[s] = f"geometry mismatch: it was written for {sorted(gs)}, this batch is {geo}"
                continue
        pool[s] = v

    better = max if spec.direction == "higher" else min
    m_best = better(models.values()) if models else None
    b_best = better(pool.values()) if pool else None
    p_best = better([per[s] for s in excluded if s in per]) if excluded else None

    sep = None
    if m_best is not None and b_best is not None:
        sep = round((m_best - b_best) if spec.direction == "higher" else (b_best - m_best), 4)

    mvals = sorted(models.values())
    saturated = bool(models) and (len(models) > 1) and abs(mvals[-1] - mvals[0]) < 1e-9

    return {
        "dim": dim if not spec.paired_with else f"{dim}+{spec.paired_with}",
        "direction": spec.direction,
        "geometry": geo,
        "model_max": m_best,                 # "the best real model" (per direction)
        "baseline_max": b_best,              # "the best degenerate baseline" (after exclusion)
        "separation": sep,
        "vacuous": b_best is None,
        "process_baseline_max": p_best,      # best among the excluded -- must be read together with vacuous
        "saturated": saturated,
        "n_models": len(models),
        "n_baselines": len(pool),
        "excluded": dict(sorted(excluded.items())),
        "threshold": SEP_THRESHOLD,
        "passes": (sep is not None and sep >= SEP_THRESHOLD),
        "models": dict(sorted(models.items(), key=lambda kv: -kv[1])),
    }


def separation_report(rows: Sequence[Mapping], *, baseline_names: Sequence[str],
                      stub_geometries: Mapping[str, frozenset] | None = None) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for dim, spec in PROCESS_DIMS.items():
        if dim in seen:
            continue
        if spec.paired_with:
            if spec.paired_with in seen or dim > spec.paired_with:
                continue
            seen.add(spec.paired_with)
        seen.add(dim)
        try:
            r = separation(rows, dim, baseline_names=baseline_names,
                           stub_geometries=stub_geometries)
        except SeparationError:
            raise
        if r["n_models"] or r["n_baselines"]:
            out.append(r)
    return out


def unregistered_saturating(rows: Sequence[Mapping], dim: str, *,
                            baseline_names: Sequence[str]) -> list[str]:
    """Stubs saturated on this dimension but not registered in `negative_controls`, so the
    exclusion list is checked against the data on every batch.
    """
    spec = PROCESS_DIMS.get(dim)
    if spec is None:
        raise SeparationError(f"{dim!r} is not registered")
    per = _per_solver(rows, dim, spec)
    bset, out = set(baseline_names), []
    for s, v in per.items():
        if s not in bset or s in (spec.negative_controls or {}):
            continue
        hit = (v >= SATURATION_EPS) if spec.direction == "higher" else (v <= 1 - SATURATION_EPS)
        if hit:
            out.append(s)
    return sorted(out)
