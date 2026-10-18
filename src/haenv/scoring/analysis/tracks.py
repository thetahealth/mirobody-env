"""Track applicability, baseline floors and forecast dimension contracts."""
from __future__ import annotations



NA_TRACKS: dict[str, tuple[str, ...]] = {"ddx": ("B", "C")}


def task_kind_of(row: dict) -> str:
    """Question type of this row, from `gold_kind`."""
    gk = str(row.get("gold_kind") or "")
    return "ddx" if gk.startswith("ddx") else ("forecast" if gk else "unknown")


def applicable_tracks(row: dict) -> dict:
    """Set tracks not applicable to this question type to `None`."""
    tk = dict(row.get("tracks") or {})
    for t in NA_TRACKS.get(task_kind_of(row), ()):
        if t in tk:
            tk[t] = None
    return tk


def _CORE_NAMES(multiround: bool) -> list[str]:
    """Names of the composite's core dimensions, in `rank_models`'s `core` order (so a missing one can be named)."""
    return (["trackE", "trackD"] if multiround
            else ["dir_acc_macro", "trackB", "trackC", "trackD"])


CORE_DIM_TO_REC = {
    "dir_acc_macro": "dir_acc_macro",
    "trackA": "A", "trackB": "B", "trackC": "C", "trackD": "D", "trackE": "E",
}


def _core_from_profile(task_type: str, rec: dict) -> list | None:
    """`core` from the profile's scored dimensions; `None` falls back to the hard-coded set.

    Both profiles match the hard-coded set item for item (`early_warning`:
    dir_acc_macro/trackB/trackC/trackD; `tracking_review`: trackE/trackD). Raises
    when a declared dimension has no bridge entry or no `rec` value.
    """
    from haenv.scoring.legacy import load_profile
    from haenv.scoring.legacy import ScoringError
    try:
        prof = load_profile(task_type=task_type)
    except ScoringError:
        return None  # no profile for this question type => hard-coded set
    if (prof.task_type or task_type) != task_type:
        return None
    dims = [d for d in prof.scored_dims]
    if not dims:
        return None
    if any(d not in CORE_DIM_TO_REC for d in dims):
        return None  # not the four-track family (e.g. ddx)
    missing = [d for d in dims if CORE_DIM_TO_REC[d] not in rec]
    if missing:
        raise ScoringError(
            f"[rank_models] profile {task_type} declares scored dimensions {missing} with no corresponding key in the result row -- "
            f"either the registry is wrong, or the code never filled that dimension. Silent skipping is not allowed: one fewer dimension = a smaller denominator = "
            f"deleting the dimension that would have lost points from the composite score")
    return [rec[CORE_DIM_TO_REC[d]] for d in dims]


_NO_HEADROOM_EPS = 1e-9


def _b_floor(ys: list[float]) -> tuple[float, str] | None:
    """Best constant Track B over this batch's outcomes.

    Mirrors `haenv_kernel/verifier.py::_track_B` (including 3-dp rounding):
    `max(0, 1 - (risk - y)^2) x (1 if the category side matches y else 0.6)`, on a
    0.001 grid of `risk`.
    """
    if not ys:
        return None
    n1 = sum(1 for y in ys if y >= 0.5)
    n0 = len(ys) - n1
    best, arg = -1.0, None
    for side in ("pos", "neg"):
        for k in range(1001):
            r = k / 1000
            b1 = round(max(0.0, 1 - (r - 1.0) ** 2) * (1.0 if side == "pos" else 0.6), 3)
            b0 = round(max(0.0, 1 - (r - 0.0) ** 2) * (1.0 if side == "neg" else 0.6), 3)
            m = (n1 * b1 + n0 * b0) / len(ys)
            if m > best + 1e-12:
                best, arg = m, (r, side)
    cat = "high" if arg[1] == "pos" else "low"
    return round(best, 4), f"constant risk={arg[0]:.3f} · {cat}"


def _c_floor(golds: list[list[str]]) -> tuple[float, str] | None:
    """Best constant Track C: one fixed top-2 driver list for every case (`|top2 & gold| / max(1, |gold|)`)."""
    if not golds:
        return None
    vocab = sorted({d for g in golds for d in g if d})
    if not vocab:
        return 0.0, "(no drivers in this batch's gold)"
    from itertools import combinations
    cands = [(d,) for d in vocab] + list(combinations(vocab, 2))
    best, arg = -1.0, None
    for s in cands:
        ss = set(s)
        m = sum(round(len(ss & set(g)) / max(1, len(g)), 3) for g in golds) / len(golds)
        if m > best + 1e-12:
            best, arg = m, s
    return round(best, 4), "constant " + " + ".join(arg)


def noinfo_floors(rows: list[dict], multiround: bool) -> dict[str, dict]:
    """The best score a solver that never reads the question can reach, per core dimension of `rank_models`, for this batch.

    Several kernel tracks give such a solver free points (e.g. Track D 1.000), so
    the composite is measured above these floors.

    | dim | floor | how |
    |---|---|---|
    | `dir_acc_macro` | 0.500 | class-macro average of any constant category |
    | `trackB` | analytic | `_b_floor` -- best constant `(risk, category)` |
    | `trackC` | analytic | `_c_floor` -- best constant top-2 driver list |
    | `trackD` | 1.000 | `_track_D` only deducts |
    | `trackA` | 1.000 | any accepted constant enum scores 1.0 |
    | `trackE` | empirical | best in-geometry answer-blind baseline that ran (`BLIND_BASELINES`); a lower bound |

    Returns `{dim: {"floor", "source", "policy"}}`; `floor is None` = not
    measurable, never 0.
    """
    by_case: dict = {}
    for r in rows:
        by_case.setdefault(r.get("case"), r)
    cases = list(by_case.values())
    out: dict[str, dict] = {}
    ys = [float(r["y"]) for r in cases if r.get("y") is not None]
    two_class = len({1 if y >= 0.5 else 0 for y in ys}) >= 2
    out["dir_acc_macro"] = ({"floor": 0.5, "source": "analytic",
                             "policy": "any constant category (two-class macro average is always 0.500)"}
                            if two_class else
                            {"floor": None, "source": "n/a", "policy": "only one outcome class in this batch"})
    _yb = [float(r["y"]) for r in cases
           if r.get("y") is not None and applicable_tracks(r).get("B") is not None]
    _b = _b_floor(_yb)
    out["trackB"] = ({"floor": _b[0], "source": "analytic", "policy": _b[1]} if _b else
                     {"floor": None, "source": "n/a", "policy": "no applicable cells in this batch"})
    _gc = [list(r.get("gold_drivers") or []) for r in cases
           if applicable_tracks(r).get("C") is not None]
    _c = _c_floor(_gc)
    out["trackC"] = ({"floor": _c[0], "source": "analytic", "policy": _c[1]} if _c else
                     {"floor": None, "source": "n/a", "policy": "no applicable cells in this batch"})
    for _k, _t, _pol in (("trackD", "D", "always fill what_not_to_do + always pick the referral class (A3)"),
                         ("trackA", "A", "always answer data_sufficiency=sufficient")):
        _app = any(applicable_tracks(r).get(_t) is not None for r in cases)
        out[_k] = ({"floor": 1.0, "source": "analytic", "policy": _pol} if _app else
                   {"floor": None, "source": "n/a", "policy": "no applicable cells in this batch"})
    if multiround:
        _blind = {}
        for r in rows:
            s = r.get("solver")
            if s in BLIND_BASELINES and not stub_geometry_scope(s, [r])["geometry_out_of_scope"]:
                if isinstance(r.get("trackE"), (int, float)):
                    _blind.setdefault(s, []).append(float(r["trackE"]))
        if _blind:
            _bm = {s: sum(v) / len(v) for s, v in _blind.items()}
            _s = max(sorted(_bm), key=lambda s: _bm[s])
            out["trackE"] = {"floor": round(_bm[_s], 4), "source": "empirical",
                             "policy": f"blind baseline `{_s}` (covers only those run in this batch, {sorted(_bm)}; true floor >= this value)"}
        else:
            out["trackE"] = {"floor": None, "source": "unmeasured",
                             "policy": "no blind baseline ran in this batch => floor not measured"}
    return out


def skill_over_floor(value, floor) -> tuple[str, float | None]:
    """Map one core-dimension value onto the no-information floor.

    * `floor < 1`: `("skill", clip((v - c) / (1 - c), 0, 1))`;
    * `floor >= 1` (Track D / Track A): `("deduct", v)` -- multiplies the composite;
    * `floor is None`: `("raw", v)`, and the caller says so.
    """
    if value is None:
        return ("missing", None)
    if floor is None:
        return ("raw", float(value))
    if floor >= 1 - _NO_HEADROOM_EPS:
        return ("deduct", max(0.0, min(1.0, float(value))))
    return ("skill", max(0.0, min(1.0, (float(value) - floor) / (1 - floor))))


def stub_geometry_scope(name: str, rs: list[dict]) -> dict:
    """Whether a solver ran outside the geometries declared in `baselines.STUB_GEOMETRIES`.

    E.g. `no_revision` and `flip_flop` are declared `{slices, multi}` and mean
    nothing under `single`. Solvers not in the table (real models) are always in
    scope.
    """
    from haenv.world.generation.baselines import STUB_GEOMETRIES as _SG
    geos = sorted({str(r.get("geometry")) for r in rs if r.get("geometry")})
    decl = _SG.get(name)
    if decl is None:
        return {"geometry": geos, "geometry_declared": None,
                "geometry_out_of_scope": False, "geometry_off": []}
    off = [g for g in geos if g not in decl]
    return {"geometry": geos, "geometry_declared": sorted(decl),
            "geometry_out_of_scope": bool(off), "geometry_off": off}


BLIND_BASELINES = ("const_ddx", "flip_flop")


HEURISTIC_BASELINES = ("baseline_slope", "robust_ref", "no_revision")


def is_ddx_batch(rows: list[dict]) -> bool:
    return any(str(r.get("gold_kind", "")).startswith("ddx:") for r in rows)
