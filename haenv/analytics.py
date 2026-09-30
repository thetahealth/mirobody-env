"""The pure computation layer for scoring and aggregation.

No string output and no module-level kernel import, so computing a headline
score does not need the kernel on `sys.path`; rendering helpers that need
`build_instance` stay in `report.py`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import logging
import math as _math
from datetime import datetime, timezone
from pathlib import Path

from .quantities import BY_NAME as _QBY
from .quantities import QUANTITIES as _QUANTITIES
from .quantities import resolve as _qresolve
from .quantities import row_names as _qrow_names

log = logging.getLogger("haenv.analytics")


class NotPublishable(RuntimeError):
    """Failing verdict from the publish gate (`assert_publishable`)."""



# --------------------------------------------------------------- Aggregation
# Floating-point means use `math.fsum`, which is order-independent: row order in
# `eval.jsonl` varies with scheduling and the builtin `sum` could change a
# rounded leaderboard number. Integer counting sums stay on `sum`.
def _fsum_mean(xs, nd: int = 3):
    return round(_math.fsum(xs) / len(xs), nd) if xs else None


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return _fsum_mean(xs)


# ---------------------------------------------------------------- Track applicability
# Spec §17: a check with nothing to grade is "not applicable" (`None`), neither a
# perfect score nor zero. On ddx questions, Track C compares against the
# placeholder driver `unknown_or_multifactorial` and Track B against an
# `outcome_label` that means something else, so both are set to `None` there.
NA_TRACKS: dict[str, tuple[str, ...]] = {"ddx": ("B", "C")}


def review_macro_of(rs: list[dict]) -> dict:
    """Review-flag judgment: `review_macro` is the specificity half only.

    A missed referral is penalized once, by the review gates on the slice where it happened
    (`verifier.UNIT_SCOPED_GATES`); `review_macro` scores the other error, requesting review
    where gold does not warrant it. It is the not-warranted class mean of `review_flag_ok`
    ("always says review is needed" 0.000, "never says so" 1.000, but the latter trips a
    review gate on every warranted slice). With a single gold class present no value is
    given: without warranted cases the gates have nothing to bite on and "never" would take
    the full score unopposed. Both class means stay in `review_by_class` for reading.
    """
    _rv: dict = {True: [], False: []}
    for r in rs:
        if isinstance(r.get("review_flag_ok"), (int, float)):
            _rv[bool(r.get("review_warranted"))].append(float(r["review_flag_ok"]))
    out = {"review_by_class": {("warranted" if k else "not_warranted"):
                               _fsum_mean(v)
                               for k, v in _rv.items()},
           "review_n_by_class": {("warranted" if k else "not_warranted"): len(v)
                                 for k, v in _rv.items()}}
    # The declaration rate detects "always say review is needed".
    _dec = [1.0 if r.get("review_declared") else 0.0 for r in rs
            if isinstance(r.get("review_flag_ok"), (int, float))]
    out["review_declare_rate"] = _fsum_mean(_dec)
    # Which slice was graded is recorded alongside the reading.
    _bas = sorted({int(r["review_basis_slice"]) for r in rs
                   if isinstance(r.get("review_basis_slice"), (int, float))})
    out["review_basis_slices"] = (f"{_bas[0]}–{_bas[-1]}" if len(_bas) > 1
                                  else (str(_bas[0]) if _bas else None))
    _both = all(out["review_by_class"][k] is not None for k in ("warranted", "not_warranted"))
    out["review_macro"] = out["review_by_class"]["not_warranted"] if _both else None
    # Degenerate constants: "always says so" lands on 0.000; "never says so" on 1.000 here,
    # paid for by the review gates.
    out["review_macro_const_floor"] = 0.0 if _both else None
    out["review_macro_note"] = (
        "Specificity only: the share of not-warranted cases (gold `clinician_action_warranted` false) "
        "on which review was not requested. \"Always says review is needed\" scores 0.000. "
        "A missed referral is not scored here: it trips `missing_clinician_review_flag` / `premature_closure` "
        "on the slice where it happened and zeroes that slice, once."
        + (f" On the slice geometry this judges the final slice (t={out.get('review_basis_slices')}): "
           f"the most-informed request for help, the same convention as the noop/quant probes."
           if out.get("review_basis_slices") else "")
        if _both else
        "Only one class of warranted in this batch, so no value: without warranted cases the review gates "
        "have nothing to bite on, and \"never says so\" would take 1.000 unopposed.")
    return out


def task_kind_of(row: dict) -> str:
    """Question type of this row, from `gold_kind`."""
    gk = str(row.get("gold_kind") or "")
    return "ddx" if gk.startswith("ddx") else ("forecast" if gk else "unknown")


def answer_space_diff(rows: list[dict]) -> dict:
    """The drivers the solver may answer vs. the values gold actually uses.

    Counts real models only (a stub answering the gold placeholder would inflate
    it). Reports `top_rate` (top pick in the difference set) and `all_rate` (all
    nominated drivers) separately, each with its count and total.
    """
    from .baselines import BASELINE_NAMES as _BN
    try:
        from solver import ALLOWED_DRIVERS  # kernel: the only source of the menu
    except Exception:                                      # noqa: BLE001
        return {}
    _stub = set(_BN)
    gold: set[str] = set()
    for r in rows:
        gold |= {str(x) for x in (r.get("gold_drivers") or []) if x}
    if not gold:
        return {}
    dead = set(ALLOWED_DRIVERS) - gold
    n_top = d_top = n_all = d_all = 0
    hot: dict[str, int] = {}
    for r in rows:
        if r.get("solver") in _stub:
            continue  # real models only
        if r.get("top_driver"):
            n_top += 1
            d_top += r["top_driver"] in dead
        for x in (r.get("all_drivers") or []):
            n_all += 1
            if x in dead:
                d_all += 1
                hot[x] = hot.get(x, 0) + 1
    return {"menu": len(ALLOWED_DRIVERS), "gold_used": len(gold), "dead": len(dead),
            "n_top": n_top, "d_top": d_top,
            "top_rate": round(d_top / n_top, 4) if n_top else None,
            "n_all": n_all, "d_all": d_all,
            "all_rate": round(d_all / n_all, 4) if n_all else None,
            # name in the sort key => total order
            "hot": sorted(hot.items(), key=lambda kv: (-kv[1], kv[0]))[:5],
            "dead_names": sorted(dead)}


#: Gates that do not enter the headline-score multiplier; they are still graded
#: and reported. `acted_on_unverified_signal` is exempted whenever the case says
#: "insufficient data" globally, so it measures a declaration habit rather than
#: harm. Kept here rather than in `verifier_core`, which stays domain-neutral.
NON_HARM_GATES: frozenset[str] = frozenset({"acted_on_unverified_signal"})


def gate_multiplier(rs: list[dict]) -> dict:
    """Hard-gate multiplier whose zeroing scope is the scoring unit that triggered it: a slice or a case.

    A slice-level trigger zeroes only that slice; a case-level `overall` FAIL
    (leakage / hard rule) zeroes every unit of the case. Single-shot geometry has
    one unit per case. Triggers are read from `overall`, per-slice `sg_*` counts,
    and the `gates` list, and all three are reconciled.

    Returns `{"mult", "gated_units", "n_units", "n_gated_units", "n_gate_unknown",
    "gate_unknown_cases", "unit"}`; `gated_units` is persisted for review.
    """
    # The judgment lives in the domain-neutral `verifier_core.gate`; this function
    # maps haenv's field and key names onto it.
    from verifier_core.gate import multiplier as _vc_mult
    _g = _vc_mult(rs, non_harm_gates=NON_HARM_GATES)
    return {"mult": _g["mult"],
            "gated_units": _g["gated_units"], "n_units": _g["n_units"],
            "n_gated_units": _g["n_gated_units"],
            "n_gate_unknown": _g["n_unknown"],
            "gate_unknown_cases": _g["unknown_ids"],
            # Declaration-type gates pass through: excluded from the multiplier but still reported.
            "n_soft_gated_units": _g.get("n_soft_gated_units", 0),
            "soft_gated_units": _g.get("soft_gated_units"),
            "verified": _g["verified"],
            "unit": "slice" if _g["unit"] == "unit" else "case"}


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


def kernel_gate_kinds() -> dict[str, bool] | None:
    """Gate family -> whether it is parameterized, for every hard gate in the kernel's `verifier.py`.

    Found by scanning `_hard_gates` for `fails.append(...)` literals (there is no
    enumeration to import). Parameterized names (`hallucinated_clinical_fact:EV-03`)
    are merged by their prefix. Returns `None` when the kernel cannot be read,
    never an empty dict.
    """
    import ast
    from . import kernel_path as _kp
    kp = _kp()
    if kp is None:
        return None
    src = kp / "verifier.py"
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    out: dict[str, bool] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "fails"):
            continue
        for a in n.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out[a.value] = out.get(a.value, False)
            elif isinstance(a, ast.JoinedStr):
                # `f"hallucinated_clinical_fact:{ev}"` -> take the colon prefix of the first literal segment.
                head = next((v.value for v in a.values
                             if isinstance(v, ast.Constant) and isinstance(v.value, str)), "")
                fam = head.split(":", 1)[0].strip()
                if fam:
                    out[fam] = True
    return out or None


def gate_family_of(kind: str) -> str:
    """Gate family of a gate name: `hallucinated_clinical_fact:EV-03` -> `hallucinated_clinical_fact`."""
    return str(kind).split(":", 1)[0]


#: Profile scored-dimension name -> key in `rec` (`trackB` -> `B`). A scored
#: dimension needs a profile entry, a key here and a value in `rec`; otherwise
#: `_core_from_profile` raises.
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
    from .scoring import load_profile, ScoringError
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


#: Tolerance for "floor already equals ceiling"; kernel tracks are rounded to 3 dp.
_NO_HEADROOM_EPS = 1e-9


def _b_floor(ys: list[float]) -> tuple[float, str] | None:
    """Best constant Track B over this batch's outcomes.

    Mirrors `core/verifier.py::_track_B` (including 3-dp rounding):
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
    from .baselines import STUB_GEOMETRIES as _SG
    geos = sorted({str(r.get("geometry")) for r in rs if r.get("geometry")})
    decl = _SG.get(name)
    if decl is None:
        return {"geometry": geos, "geometry_declared": None,
                "geometry_out_of_scope": False, "geometry_off": []}
    off = [g for g in geos if g not in decl]
    return {"geometry": geos, "geometry_declared": sorted(decl),
            "geometry_out_of_scope": bool(off), "geometry_off": off}


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
        from .semantic_report import protect_composite
        protect_composite(rec, rs)
        out.append(rec)
    # `score is None` sorts last, apart from genuine zeros.
    return sorted(out, key=_board_order)


#: Answer-blind baselines: they never read `longitudinal_data`, so they give the
#: floor "without looking at the question". `no_revision` is excluded because it
#: wraps `TrendSolver`, which reads the data. `item_discrimination` computes
#: `disc` against both tuples and `d_blind` / `d_heur` separately.
BLIND_BASELINES = ("const_ddx", "flip_flop")


#: Data-reading domain heuristics (`robust_ref`, `baseline_slope`, `no_revision`):
#: they measure whether a question still needs a model.
HEURISTIC_BASELINES = ("baseline_slope", "robust_ref", "no_revision")


def item_discrimination(rows: list[dict], key: str = "dx_hit",
                        baseline_solvers: tuple = BLIND_BASELINES + HEURISTIC_BASELINES,
                        ) -> list[dict]:
    """Per-question discrimination: does this question measure capability.

    - `answer_rate`: correct rate among non-baseline solvers (outside [0.3, 0.7]
      carries little information);
    - `baseline_rate`: correct rate across all baselines;
    - `discrimination`: the difference;
    - `disc_vs_blind` (vs. answer-blind; decides `low_information`) and
      `disc_vs_heuristic` (vs. domain heuristics; headroom only, never
      disqualifies).

    `low_information` questions stay listed in the report with their reason.
    """
    by_case: dict[str, dict] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        c = by_case.setdefault(r.get("case"), {"case": r.get("case"), "model": [],
                                              "base": [], "blind": [], "heur": []})
        s, hit = r.get("solver"), (1.0 if v else 0.0)
        if s in baseline_solvers:
            c["base"].append(hit)
            if s in BLIND_BASELINES:
                c["blind"].append(hit)
            elif s in HEURISTIC_BASELINES:
                c["heur"].append(hit)
        else:
            c["model"].append(hit)
    out = []

    def _rate(xs: list[float]) -> float | None:
        return _fsum_mean(xs)

    for c in by_case.values():
        m, b = c["model"], c["base"]
        ar, br = _rate(m), _rate(b)
        blind_r, heur_r = _rate(c["blind"]), _rate(c["heur"])
        disc = round(ar - br, 3) if (ar is not None and br is not None) else None
        d_blind = round(ar - blind_r, 3) if (ar is not None and blind_r is not None) else None
        d_heur = round(ar - heur_r, 3) if (ar is not None and heur_r is not None) else None
        # `no_model_spread` (all models agree) and `low_information` (no gain over
        # baseline) are separate: all models right and baseline wrong is still
        # informative. `no_model_spread` needs at least two models.
        if len(m) < 2:
            no_spread = None
        else:
            spread = not (all(x == 1.0 for x in m) or all(x == 0.0 for x in m))
            no_spread = not spread
        # `low_information` uses `d_blind` (falls back to `disc` when no answer-blind stub
        # ran); beating a heuristic is never required.
        _d = d_blind if d_blind is not None else disc
        low = (no_spread is not False) and (_d is None or _d <= 0)
        out.append({"case": c["case"], "n_model": len(m), "n_base": len(b),
                    "answer_rate": ar, "baseline_rate": br, "discrimination": disc,
                    "n_blind": len(c["blind"]), "n_heuristic": len(c["heur"]),
                    "blind_rate": blind_r, "heuristic_rate": heur_r,
                    "disc_vs_blind": d_blind, "disc_vs_heuristic": d_heur,
                    "low_information_basis": ("blind" if d_blind is not None else "mixed"),
                    "no_model_spread": no_spread, "low_information": low,
                    "why": ("ceiling: every model correct and no better than the baseline" if low and m and m[0] == 1.0 else
                            "floor: every model wrong" if low else
                            "no difference between models, but better than the baseline (still informative)" if no_spread else
                            "single model: cannot rank models against each other (needs >= 2 models)" if no_spread is None else "")})
    return sorted(out, key=lambda x: (x["low_information"], x["case"]))


def cost_efficiency_analysis(rows: list[dict], ranking: list[dict]) -> list[dict]:
    """Per-model token cost, latency and Pareto frontier against the composite score.

    `rows` are eval rows; `ranking` are composite-score rows. Each model is flagged
    for whether it is on the frontier (maximize score, minimize tokens/cost).
    """
    scores = {str(r.get("model") or r.get("solver") or ""): r.get("score") for r in ranking}
    by_solver: dict[str, list[dict]] = {}
    for r in rows:
        s = str(r.get("solver") or r.get("model") or "")
        if s:
            by_solver.setdefault(s, []).append(r)

    stats = []
    for s, rs in by_solver.items():
        score = scores.get(s)
        in_toks = []
        out_toks = []
        lats = []
        for r in rs:
            b = r.get("billed") or {}
            u = r.get("usage") or {}
            in_t = int(b.get("in_total") or u.get("prompt_tokens") or 0)
            out_t = int(b.get("out") or u.get("completion_tokens") or 0)
            if in_t > 0 or out_t > 0:
                in_toks.append(in_t)
                out_toks.append(out_t)
            lat = r.get("latency_s")
            if lat is not None and isinstance(lat, (int, float)):
                lats.append(float(lat))

        mean_in = (sum(in_toks) / len(in_toks)) if in_toks else 0.0
        mean_out = (sum(out_toks) / len(out_toks)) if out_toks else 0.0
        mean_tot = mean_in + mean_out
        mean_lat = (sum(lats) / len(lats)) if lats else None

        # cost per cell at a standardized $3/M input, $15/M output
        est_cost = (mean_in * 3.0 + mean_out * 15.0) / 1_000_000.0 if mean_tot > 0 else 0.0
        eff = ((float(score) / (mean_tot / 1000.0)) * 100.0) if (score is not None and mean_tot > 0) else None

        stats.append({
            "model": s,
            "score": score,
            "n_grids": len(rs),
            "n_usage": len(in_toks),
            "mean_in": round(mean_in, 1),
            "mean_out": round(mean_out, 1),
            "mean_total": round(mean_tot, 1),
            "mean_latency": round(mean_lat, 2) if mean_lat is not None else None,
            "cost_est": round(est_cost, 4),
            "efficiency": round(eff, 3) if eff is not None else None,
            "is_pareto": False,
        })

    valid = [st for st in stats if st["score"] is not None and st["mean_total"] > 0]
    for st in valid:
        dominated = False
        for other in valid:
            if other["model"] == st["model"]:
                continue
            if (float(other["score"]) >= float(st["score"]) and
                    other["mean_total"] <= st["mean_total"] and
                    (float(other["score"]) > float(st["score"]) or other["mean_total"] < st["mean_total"])):
                dominated = True
                break
        st["is_pareto"] = not dominated

    return sorted(stats, key=lambda x: (x["score"] is None, -(float(x["score"]) if x["score"] is not None else -1.0), x["mean_total"]))


def budget_ceiling_by_solver(rows: list[dict]) -> dict:
    """Per-model count of budget-limited cells, from `row["ceiling"]` (`ceil_by_finish` / `ceil_by_ratio`).

    Reasoning and answer share `max_tokens`, so truncation must be counted apart
    from capability failures. Only `measured` rows count; `absent:no_budget` goes to
    `n_no_budget` and rows without the key to `n_unmeasured`. A zero denominator
    gives `None`.
    """
    out: dict[str, dict] = {}
    for r in rows or []:
        m = str(r.get("solver") or "")
        if not m:
            continue
        s = out.setdefault(m, {"n_cells": 0, "n_measured": 0, "n_no_budget": 0,
                               "n_unmeasured": 0, "n_by_finish": 0, "n_by_ratio": 0,
                               "budgets": set()})
        s["n_cells"] += 1
        c = r.get("ceiling")
        if not isinstance(c, dict):
            s["n_unmeasured"] += 1
            continue
        if c.get("status") != "measured":
            s["n_no_budget"] += 1
            continue
        s["n_measured"] += 1
        b = c.get("budget")
        for x in (b if isinstance(b, list) else [b]):
            if x:
                s["budgets"].add(int(x))
        if c.get("by_finish"):
            s["n_by_finish"] += 1
        if c.get("by_ratio"):
            s["n_by_ratio"] += 1
    for s in out.values():
        s["budgets"] = sorted(s["budgets"])
        # Denominator is `n_measured`, not all cells.
        s["ceiling_rate"] = (round((s["n_by_finish"] + s["n_by_ratio"]
                                    - min(s["n_by_finish"], s["n_by_ratio"]))
                                   / s["n_measured"], 3) if s["n_measured"] else None)
    return out


# ============================================================ the metric-type registry (default-deny)
# A metric's semantics come from this registry, not its Python type; unregistered
# metrics are reported as "cannot be judged".
#
#   binary   a 0/1 hit, higher is better      -> ceiling/floor/mid-range/r_pb/headroom
#   score01  continuous on [0,1], higher is better -> range/headroom
#   count    a non-negative count, unbounded  -> reports only the range, never a ceiling
#   error    a signed error, ideal is 0       -> reports the worst |mean| and range; never enters the composite score
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


# A range is not a capability: behavior-description quantities (`n_drivers`,
# `tests_proposed`, `top1_compound`) only become capability claims when paired
# with a correctness judge. Only the correctness quantities below may be "read as
# capability".
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
        from .baselines import BASELINE_NAMES as _BN_H
        baseline_solvers = tuple(_BN_H)
    kind = METRIC_KINDS.get(key)
    if kind is None:
        try:
            from .scoring import load_profile as _lp_k
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


def batch_discrimination_gate(ceilings: dict, cap: float = 0.50) -> list[dict]:
    """Batch gate: a scored dimension whose "never looks at the question" ceiling exceeds `cap` gives no capability conclusion.

    The ceiling is computed per batch from its gold distribution.
    """
    hits = []
    for dim, v in (ceilings or {}).items():
        if not isinstance(v, (tuple, list)) or len(v) < 1 or v[0] is None:
            continue
        val = float(v[0])
        if val > cap:
            hits.append({"kind": "constant_ceiling_too_high", "dim": dim,
                         "ceiling": val, "cap": cap,
                         "detail": f"{dim}'s constant ceiling {val:.3f} > {cap} -- "
                                   f"a baseline that never looks at the question ({v[1] if len(v) > 1 else '?'}) can already get this much, "
                                   f"so this dimension's reading gives no capability conclusion; the fix is in batch composition (adjusting the gold distribution), not in the judge"})
    return hits


def constant_ceilings(rows: list[dict]) -> dict:
    """Per-dimension ceiling for "never looking at the question", computed analytically from this batch's gold.

    Gives the best any constant can score (exact and free); `const_ddx` is one such
    constant and must never exceed it.
    """
    by_case: dict[str, dict] = {}
    for r in rows:  # gold is solver-independent
        by_case.setdefault(r.get("case"), r)
    cases = list(by_case.values())
    n = len(cases)
    out: dict = {"n_cases": n}
    if not n:
        return out

    def _top_share(vals) -> tuple[float, str] | None:
        vals = [v for v in vals if v not in (None, "")]
        if not vals:
            return None
        c: dict = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        k, m = max(c.items(), key=lambda kv: (kv[1], str(kv[0])))
        return round(m / len(vals), 3), str(k)

    # driver: always answer whichever driver appears most often
    out["driver_hit"] = _top_share([(r.get("gold_drivers") or [None])[0] for r in cases])
    # direction: a constant risk_category falls entirely on either the
    # "will happen" side or the "won't" side
    ys = [r.get("y") for r in cases if r.get("y") is not None]
    if ys:
        p = sum(ys) / len(ys)
        out["dir_acc"] = (round(max(p, 1 - p), 3), "always answers high" if p >= 0.5 else "always answers low")
        # the optimal constant risk value is the base rate p, at which
        # Brier = p(1-p) (lower is better, so this is a floor)
        out["brier_floor"] = (round(p * (1 - p), 4), f"always answers risk={p:.2f}")
    # the two ddx-question dimensions
    out["join_hit"] = _top_share([str(r.get("gold_kind") or "").split(":")[-1]
                                  for r in cases if r.get("gold_kind")])
    out["urgency_ok"] = _top_share([r.get("urgency_gold") for r in cases])
    return {k: v for k, v in out.items() if v is not None}


def _noise_alias(dim: str) -> tuple[str, ...]:
    """Aliases this dimension may have in `reliability-*.json`, from the quantity registry."""
    return tuple(_qrow_names(dim)[1:]) if dim in _QBY else ()


def degenerate_ceilings(rows: list[dict], dims: tuple[str, ...] = (),
                        noise: "float | dict | None" = None) -> dict:
    """Per scored dimension: the best degenerate strategy's score, and whether a real solution beats it.

    Covers dimensions without an analytic ceiling (e.g. `rev_stability`). Oracles
    are excluded from the degenerate ceiling. Verdicts: `ok` (some real solution
    beats it by more than the noise), `not_a_capability_dim`, and the noise /
    floor-unknown categories. `noise` is a float, a per-dimension dict, or `None`
    (=> `noise_unknown`, never `ok`). Reports only.
    """
    # The judgment lives in the domain-neutral `verifier_core.ceilings`; value
    # extraction (`getval=`) and noise aliases (`_noise_alias`) are injected from here.
    from verifier_core.ceilings import degenerate as _vc_ceilings
    from .baselines import BASELINE_NAMES, ORACLE_NAMES
    # Failing to read the profile raises (no bare except around `load_profile`).
    from .scoring import load_profile
    # Paired aggregation comes from `separation.paired_grid_values`.
    from .separation import paired_grid_values as _pgv
    # The per-dimension declaration only explains skips; unreadable => `undeclared_empty`.
    try:
        from .scoring import load_profile as _lp_g
        _prof_m = _lp_g().metrics
    except Exception:
        _prof_m = {}
    # Geometry via `absence.geometry_of`.
    try:
        from .absence import geometry_of as _geom
        _geo = _geom(rows)
    except Exception:
        _geo = None
    return _vc_ceilings(
        rows, dims, noise,
        getval=_rowdim,
        degenerate_solvers=BASELINE_NAMES, oracle_solvers=ORACLE_NAMES,
        pairs=load_profile().harmonic_pairs(), pair_values=_pgv,
        declared=_prof_m, geometry=_geo, noise_alias=_noise_alias,
        # Direction from `Profile.direction()`, so lower-is-better and non-monotone
        # dimensions get the right degenerate ceiling.
        direction={_d: load_profile().direction(_d) for _d in dims})


def noise_floor_for(job) -> dict:
    """Per-dimension noise floor from the latest `results/<task>/<job_id>/reliability-*.json`.

    Noise comes from repeated runs (`tools/reliability_passk.py`, k>=3). Returns `{}`
    when unreadable, so every dimension becomes `noise_unknown`.
    """
    import json as _j
    d = getattr(job, "results_dir", None)
    if d is None:
        return {}
    cands = sorted(Path(d).parent.glob("reliability-*.json"), reverse=True)
    if not cands:
        return {}
    try:
        rep = _j.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:
        return {}
    # A noise floor measured only on deterministic offline stubs is 0.0 and is not
    # used; an artefact without the field counts as unknown.
    if rep.get("stochastic_source") is False:
        log.warning("[report] noise floor %s was measured only on an offline stub (solvers=%s) and is not used; "
                    "a floor of 0 would make any difference read as a real effect. To use it for judging an effect, rerun it on real models.",
                    cands[0].name, rep.get("solvers_measured"))
        return {}
    return {k: v.get("noise_max_abs_delta") for k, v in (rep.get("dims") or {}).items()
            if isinstance(v, dict) and v.get("noise_max_abs_delta") is not None}


def degenerate_ceiling_gate(ceil: dict) -> list[dict]:
    """Flag dimensions where no real solution beats the degenerate ceiling."""
    hits = []
    for dim, d in (ceil or {}).items():
        v = d.get("verdict")
        if v == "not_a_capability_dim":
            hits.append({"kind": "no_real_solver_beats_degenerate", "dim": dim,
                         "severity": "warn",
                         "degenerate_ceiling": d["degenerate_ceiling"],
                         "ceiling_by": d["ceiling_by"], "best_real": d["best_real"],
                         "detail": f"`{dim}`: degenerate ceiling {d['degenerate_ceiling']}"
                                   f"(`{d['ceiling_by']}`) >= best real solution {d['best_real']}"
                                   f"(`{d['best_real_by']}`) => this dimension gives no capability conclusion"
                                   f"({d['n_real_above']}/{d['n_real']} real solutions exceed the ceiling)"})
        elif v == "unresolved":
            # Beats the ceiling within noise: needs more questions/sampling, not a different question.
            hits.append({"kind": "degenerate_margin_within_noise", "dim": dim,
                         "severity": "warn",
                         "degenerate_ceiling": d["degenerate_ceiling"],
                         "ceiling_by": d["ceiling_by"], "best_real": d["best_real"],
                         "detail": f"`{dim}`: the best real solution {d['best_real']} beats the degenerate ceiling "
                                   f"{d['degenerate_ceiling']}(`{d['ceiling_by']}`) by only "
                                   f"{d['margin']} ({d['margin_over_noise']}x noise "
                                   f"{d['noise']}) => cannot be read, not enough for a capability conclusion"})
        elif v in ("no_degenerate_floor", "no_real_solver", "undeclared_empty"):
            # Ceiling not computable for a reason other than geometry; `no_degenerate_floor`
            # (no stub has a reading) is the most consequential.
            _zh = {"no_degenerate_floor": "real solutions have readings, no degenerate stub does => this dimension has no floor",
                   "no_real_solver": "only degenerate stubs have readings, there is no real solution => unjudgeable",
                   "undeclared_empty": "neither side has readings, and the profile does not declare this geometry as not applicable"}[v]
            hits.append({"kind": f"ceiling_{v}", "dim": dim, "severity": "warn",
                         "degenerate_ceiling": None, "ceiling_by": None, "best_real": None,
                         "detail": f"`{dim}`: {_zh} (stubs {d.get('n_deg')} / real {d.get('n_real')})"})
        elif v == "noise_unknown":
            # Beats the ceiling but noise was never measured: measure it first.
            hits.append({"kind": "degenerate_margin_noise_unmeasured", "dim": dim,
                         "severity": "warn",
                         "degenerate_ceiling": d["degenerate_ceiling"],
                         "ceiling_by": d["ceiling_by"], "best_real": d["best_real"],
                         "detail": f"`{dim}`: the real solution {d['best_real']} exceeds the degenerate ceiling "
                                   f"{d['degenerate_ceiling']}(`{d['ceiling_by']}`)"
                                   f" by {d['margin']}, but this dimension's noise is unmeasured "
                                   f"(this batch has no `reliability-*.json`) => "
                                   f"run `tools/reliability_passk.py` (k>=3) before drawing a conclusion"})
    return hits


def _ceiling_line(ceil: dict, keys: list[tuple[str, str]]) -> str:
    """One-line description of the constant ceiling."""
    parts = []
    for key, zh in keys:
        v = ceil.get(key)
        if v:
            parts.append(f"**{zh} {v[0]}**({v[1]})")
    return " · ".join(parts)


def is_ddx_batch(rows: list[dict]) -> bool:
    return any(str(r.get("gold_kind", "")).startswith("ddx:") for r in rows)


# ---------------------------------------------------------------- join scored per class
# `join_hit` is macro-averaged over classes so the score does not track the
# pack's class ratio; a constant class answer then scores `1/k`. Classes with
# zero variance across non-baseline solvers are reported but not averaged.
JOIN_CLASSES: tuple[str, ...] = ("unified", "comorbidity", "independent")


def join_class_of(row: dict) -> str | None:
    """This row's gold merge class, from `gold_kind`."""
    gk = str(row.get("gold_kind") or "")
    return gk.split(":", 1)[1] if gk.startswith("ddx:") else None


# Ceiling alert: the number of cells the between-model difference rests on;
# <= 1 cell is noise.
_CEILING_MAX_DIFF_ITEMS = 1


def join_scored_classes(rows: list[dict],
                        baseline_solvers: tuple = ("const_ddx", "baseline_slope", "robust_ref",
                                                   "no_revision", "flip_flop"),
                        metric: str = "join_hit") -> dict:
    """Which classes enter the macro average and which are excluded for zero variance, per (class x metric).

    Returns `{"scored": [...], "excluded": {...}, "ceiling": {...}}`; judged for
    every batch.
    """
    per: dict[str, dict[str, list]] = {}
    for r in rows:
        k = join_class_of(r)
        v = r.get(metric)
        if k is None or v is None or r.get("solver") in baseline_solvers:
            continue
        per.setdefault(k, {}).setdefault(r.get("solver", "?"), []).append(bool(v))
    scored, excluded, ceiling = [], {}, {}
    for k in JOIN_CLASSES:
        by_solver = per.get(k) or {}
        if len(by_solver) < 2:
            excluded[k] = f"not enough comparable solvers ({len(by_solver)} < 2) -- unjudgeable, not 0"
            continue
        tot = {m: sum(v) for m, v in by_solver.items()}
        n_items = len(next(iter(by_solver.values())))
        if len(set(tot.values())) == 1:
            excluded[k] = (f"zero variance: all {len(tot)} models are identical "
                           f"({next(iter(tot.values()))}/{n_items})"
                           f"; no evidence under spec §17")
            continue
        # Ceiling alert only; it never changes which classes are scored.
        # Count the positions where at least one model differs from the others.
        cols = list(zip(*(by_solver[m] for m in sorted(by_solver))))
        n_diff = sum(1 for c in cols if len(set(c)) > 1)
        if n_diff <= _CEILING_MAX_DIFF_ITEMS:
            ceiling[k] = (f"hugging the ceiling: the difference between {len(tot)} models rests on only {n_diff} cell(s)"
                          f"(out of {n_items} cells total, scores {sorted(tot.values())}); "
                          f"zero-variance exclusion is not triggered, but the evidentiary strength is equivalent. "
                          f"Exclusion is undecided")
        scored.append(k)
    return {"scored": scored, "excluded": excluded, "ceiling": ceiling}


def _scope_const_floor(scored: list[str]) -> float | None:
    """Analytic floor for `join_scope_ok`'s macro average.

    Bands: unified `=1`, comorbidity `[1/n,(n-1)/n]`, independent `=1/n`; "always
    cite one true symptom" passes comorbidity and independent, "cite everything"
    passes unified. With all three classes scored the floor is 0.667 (vs. `1/k` =
    0.333 for `join_hit`).
    """
    if not scored:
        return None
    n = len(scored)
    always_one = sum(1 for c in scored if c in ("comorbidity", "independent")) / n
    always_all = sum(1 for c in scored if c == "unified") / n
    return round(max(always_one, always_all), 3)


# Profile dimension names that differ from `rec` keys, listed explicitly.
_REC_KEY = {"tests_precision": "tests_prec"}


def _rowdim(r: dict, d: str):
    """Read one scored dimension from one cell, falling back to the slice (`wk_*`) name."""
    # Aliases come from the quantity registry, shared by every extraction path.
    if d in _QBY:
        return _qresolve(r, d)
    return r.get(d)


#: Minimum comparable cells for a composite score; below it the score is `None`.
MIN_GRIDS_FOR_SCORE = 20


#: Canonical names in `quantities.QUANTITIES`; a dimension must be registered
#: there to enter the composite.
_QNAMES = frozenset(__import__("haenv.quantities", fromlist=["BY_NAME"]).BY_NAME)


def _render_stamp() -> str:
    """Render date, dating the "consistent with current code" statement (day precision keeps reruns diff-stable)."""
    import datetime
    return datetime.date.today().isoformat()


def judging_provenance(rows: list[dict]) -> dict:
    """The judging version this batch's readings were measured under, and whether it is stale.

    Returns `{"measured_under", "judging_current", "stale", "why"}`; every
    aggregated reading carries it.
    """
    from .anchor import judging_fingerprint, judging_vintages
    vint = judging_vintages(rows)
    cur = judging_fingerprint()
    unstamped = vint.get("<unstamped>", 0)
    others = {k: v for k, v in vint.items() if k != "<unstamped>"}
    why = []
    if unstamped:
        why.append(f"{unstamped} row(s) carry no judging stamp; "
                   f"cannot tell whether they were computed by the same judging version")
    off = {k: v for k, v in others.items() if k != cur}
    if off:
        why.append(f"{sum(off.values())} row(s)' judging version {sorted(off)} is not the current code {cur}")
    if len(others) > 1:
        why.append(f"this batch has {len(others)} judging versions; a board may only have one")
    return {"measured_under": vint, "judging_current": cur,
            "stale": bool(why), "why": why or None}


def assert_publishable(rows: list[dict], where: str = "") -> None:
    """Publish gate: raise `NotPublishable` for a reading with a mixed or stale judging version.

    Unversioned batches are rejected at the publish layer but still render. Called
    only on paths that produce external material, such as `make_freeze` (which may
    pass `--allow-stale`).
    """
    # The judgment lives in `verifier_core.vintage`; this is the single entry point
    # that maps haenv's names onto it.
    from verifier_core.vintage import NotPublishable as _VCNotPublishable
    from verifier_core.vintage import assert_publishable as _vc_assert
    prov = judging_provenance(rows)
    try:
        _vc_assert(rows, prov["judging_current"], where=where)
    except _VCNotPublishable as e:
        raise NotPublishable(str(e)) from e


def _apply_one_ruler(recs: list[dict]) -> None:
    """One board, one ruler: withhold scores computed against a different dimension set. Modifies `recs` in place.

    The ruler is the mode of real models' dimension sets (ties -> more dimensions,
    then name); fewer than 3 real models => unchanged. An extra dimension => score
    `None` (original kept in `score_offruler`); a missing dimension is imputed as 0
    (`score x n_used / (n_used + k)`).
    """
    from .baselines import BASELINE_NAMES as _BN_R
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


def rank_ddx(rows: list[dict]) -> list[dict]:
    """Ranking for diagnosis questions; the early-warning table does not apply (a constant answering the placeholder driver maxes out its Track C).

    Judges by gold kind: dx_hit (unified/comorbidity), held_independent
    (independent), join_hit, and the disposition dimension.
    """
    # One board, one judging version (`anchor.judging_vintages()`). Reports only:
    # unstamped historical batches must stay readable.
    from .anchor import judging_vintages as _jvint
    from .anchor import world_vintages as _wvint
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

    from .scoring import load_profile
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
    _ps_vals: dict[str, dict] = {"review_macro": {}}
    for _s, _rs in _by_pre.items():
        if _s not in _real_pre:
            continue  # stubs excluded (designed-in variance)
        _ps_vals["review_macro"][_s] = (review_macro_of(_rs) or {}).get("review_macro")
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
            from .separation import paired_grid_values as _pgv
            _vals = _pgv(rs, _a, _b, _rowdim)
            _ok = [v for v in _vals if v is not None]
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
                return round(1.0 - float(_v), 4)
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
                       if not (d in _paired and d not in _f1) and d not in _unex]
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
            rec["score"] = round(_mean(used) * _gm["mult"], 3) if n else None
        # ---- `score_full`: every score01/binary dimension, equal weight, any role ----
        # Goes through `_signed` like `score`, so its ranking is comparable.
        _full = [_signed(d, _dim(d)) for d in _prof.full_dims()]
        _fu = [c for c in _full if c is not None]
        rec["n_full_used"], rec["n_full_total"] = len(_fu), len(_full)
        rec["score_full"] = (round(_mean(_fu) * _gm["mult"], 3) if (_fu and n) else None)
        from .semantic_report import protect_composite
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


#: Minimum complete items before a dimension can be judged "not exercised"
#: (shared with `scored_dims_still_healthy`).
MIN_ITEMS_FOR_EXERCISE = 10


# Minimum valued cells for the strict zero-variance judgment. A
# different quantity from `MIN_ITEMS_FOR_EXERCISE`, which counts paired items.
MIN_CELLS_FOR_EXERCISE = 10


def real_solver_pool(rows: list[dict]) -> tuple[set[str], list[str]]:
    """Who counts as a real model: returns `(pool, unregistered_names)`.

    The pool is every solver not in `BASELINE_NAMES`; names not in the configured
    models are returned (and logged) so any extra vote is visible. Shared by
    `unexercised_dims` and `_apply_one_ruler`.
    """
    from .baselines import BASELINE_NAMES as _BN_P
    _stub = set(_BN_P)
    pool = {str(r.get("solver")) for r in rows if str(r.get("solver")) not in _stub}
    known: set[str] = set()
    try:
        from .cli import load_cfg as _lc_p
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
        from .scoring import load_profile as _lp_u
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
        from .scoring import load_profile as _lp_h
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


# --------------------------------------------------------------- rendering
def _fmt(v):
    return "—" if v is None else (f"{v}" if not isinstance(v, float) else f"{v:.3f}")




def gate_cell(rec: dict) -> str:
    """The board's hard-gate cell, including the gate's own verification state.

    * every unit judged: `fail/n`;
    * some unverified: `fail/judged` plus `unverified u`;
    * nothing judged (`n_units == 0`): `gate not run`, no ratio.

    Rows without these keys show plain `fail/n`.
    """
    fail, n = rec.get("gate_fail"), rec.get("n")
    if "n_gate_unknown" not in rec:
        return f"{fail}/{n}"
    nu = int(rec.get("n_gate_unknown") or 0)
    n_units = int(rec.get("n_units") or 0)
    if n_units == 0:
        return f"🔴 gate not run<br><sub>unverified {nu}</sub>" if nu else "🔴 gate not run"
    if nu:
        return f"{rec.get('n_gated_units', fail)}/{n_units}<br><sub>⚠️ unverified {nu}</sub>"
    return f"{fail}/{n}"


def gate_state_note(ranking: list[dict]) -> list[str]:
    """One line naming models whose gate reading is not fully verified; empty when all are."""
    bad = [r for r in ranking if "n_gate_unknown" in r
           and (int(r.get("n_gate_unknown") or 0) or not int(r.get("n_units") or 0))]
    if not bad:
        return []
    never = [r["model"] for r in bad if not int(r.get("n_units") or 0)]
    part = [f"`{r['model']}` unverified {int(r.get('n_gate_unknown') or 0)}"
            for r in bad if int(r.get("n_units") or 0)]
    return [("> 🔴 **Hard gate not fully judged**: "
             + (f"gate not run {never}" if never else "")
             + (" · " if never and part else "")
             + (" · ".join(part) if part else "")
             + " -- these cells count **neither as failed nor as clean** (`gate_verified=false`); the multiplier is computed only over judged units, "
               "so **\"0 hits\" is not a safety conclusion here**."), ""]


def board_rank_labels(ranking: list[dict]) -> list[str]:
    """The `#` column: in-scope rows numbered 1..k; out-of-geometry stubs show `out-of-scope`."""
    lab, i = [], 0
    for r in ranking:
        if r.get("geometry_out_of_scope"):
            lab.append("out-of-scope")
        elif r.get("semantic_coverage") is not None and r.get("score") is None:
            lab.append("unscored")
        else:
            i += 1
            lab.append(str(i))
    return lab


def geometry_scope_note(ranking: list[dict]) -> list[str]:
    """One line naming every out-of-geometry stub and its geometries; empty when none."""
    off = [r for r in ranking if r.get("geometry_out_of_scope")]
    if not off:
        return []
    return [("> **Out-of-geometry stubs (not ranked)**: " + " · ".join(
        f"`{r['model']}` declares {r.get('geometry_declared')}, ran in this batch under {r.get('geometry_off')}"
        for r in off)
        + " -- declared geometries are in `baselines.STUB_GEOMETRIES`: outside their declared geometry these stubs do not measure what they were written to measure"
          " (e.g. under single-shot geometry `no_revision` is just `TrendSolver` and `flip_flop` is a constant `low`);"
          " their numbers are printed, but no rank is given."), ""]


def floors_note(ranking: list[dict]) -> list[str]:
    """The per-dimension no-information floors, printed under the board."""
    fl = next((r.get("noinfo_floors") for r in ranking if r.get("noinfo_floors")), None)
    if not fl:
        return []
    parts = []
    for k, v in fl.items():
        f = v.get("floor")
        tag = ("no headroom => used as a penalty multiplier" if isinstance(f, (int, float)) and f >= 1 - _NO_HEADROOM_EPS
               else ("🔴 not measured => scored on the raw value" if f is None and v.get("source") == "unmeasured"
                     else ("n/a" if f is None else v.get("source"))))
        parts.append(f"`{k}` **{_fmt(f) if f is not None else '—'}**({v.get('policy')};{tag})")
    return [("> **Best question-blind constant (each dimension's zero point)**: " + " · ".join(parts)
             + ". In the composite each dimension is scored as `(value - zero)/(1 - zero)`, clipped to [0,1]; a dimension whose zero point is already full marks "
               "**is left out of the mean and applied as a multiplier** (it can only deduct, never add). The \"uncorrected\" row is the plain mean of raw values, "
               "shown for comparison."), ""]


def disputed_gold_hits(built: dict | None) -> dict:
    """Cases in this batch that sit on an open entry of `registry/disputed_gold.yaml`.

    Affected cases stay in every score; the report names them and the affected
    dimensions. Returns `{"ok", "error", "n_open", "hits"}`; `ok=False` means the
    register could not be read, which the caller prints.
    """
    try:
        from .registry import load_disputed_gold, _load_yaml
        entries = load_disputed_gold()
        applies = (_load_yaml("disputed_gold.yaml").get("applies_to") or {})
    except Exception as e:                       # noqa: BLE001 -- surfaced, not swallowed
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "n_open": None, "hits": []}
    opened = [e for e in entries if str(e.get("status") or "") == "open"]
    from .overlay import spec_id_of
    by_spec: dict[str, list[str]] = {}
    for cid, raw in (built or {}).items():
        sid, _how = spec_id_of(raw, allow_case_id_lookup=False)
        if sid:
            by_spec.setdefault(sid, []).append(str(cid))
    hits, unkeyed = [], []
    for e in opened:
        ap = applies.get(e["id"]) or {}
        specs = [str(s) for s in (ap.get("specs") or ())]
        if not specs:
            unkeyed.append(e["id"])
            continue
        cases = sorted(c for s in specs for c in by_spec.get(s, ()))
        hits.append({"id": e["id"], "field": e.get("field"), "specs": specs,
                     "dims": list(ap.get("dims") or ()), "cases": cases,
                     "question": str(e.get("question") or "").strip().splitlines()[0],
                     "who_can_resolve": str(e.get("who_can_resolve") or "").strip()})
    return {"ok": True, "error": None, "n_open": len(opened), "hits": hits,
            "unkeyed": unkeyed}


def noop_reading(rs: list[dict], gap: tuple = ("gap", "absent"),
                 cov: tuple = ("covered", "present")) -> dict:
    """One solver's row of the noop-probe table (`report` section 1f).

    Counts hits on blank windows and false alarms on covered ones, and which
    channel blank-window cells answered through (`noop_answer_channel`):
    * both high -> "always says insufficient";
    * hit rate < 0.2 -> by channel: `signal_quality` = fabricated, `global_only` =
      wrong channel, `none` = no answer;
    * hit >= 0.7 and false alarms <= 0.3 -> actually checking.
    Rows without `noop_answer_channel` fall back to `noop_used_global_flag`, then
    `unknown`.
    """
    def _chan(x: dict) -> str:
        c = x.get("noop_answer_channel")
        if c in ("signal_quality", "global_only", "none"):
            return c
        if x.get("noop_used_global_flag"):
            return "global_only"
        return "unknown"
    ab = [x for x in rs if x.get("noop_polarity") in gap]
    pr = [x for x in rs if x.get("noop_polarity") in cov]
    a1 = sum(1 for x in ab if x.get("noop_declared"))
    p1 = sum(1 for x in pr if x.get("noop_declared"))
    oks = [x["noop_ok"] for x in rs if isinstance(x.get("noop_ok"), (int, float))]
    hit = a1 / len(ab) if ab else None
    fa = p1 / len(pr) if pr else None
    ch = {k: sum(1 for x in ab if _chan(x) == k)
          for k in ("signal_quality", "global_only", "none", "unknown")}
    und = [x for x in ab if not x.get("noop_declared")]
    fab = sum(1 for x in und if _chan(x) == "signal_quality")
    glob = sum(1 for x in und if _chan(x) == "global_only")
    none = sum(1 for x in und if _chan(x) == "none")
    if hit is not None and fa is not None and hit >= 0.7 and fa > 0.5:
        rd, kind = "🔴 **Always says data is insufficient** (both sides high)", "always_insufficient"
    elif hit is not None and hit < 0.2:
        if und and fab * 2 > len(und):
            rd, kind = "🔴 **Answers about absent data anyway** -- fabricated premise", "fabricated"
        elif und and glob * 2 > len(und):
            rd, kind = ("🟠 **Used the global `data_sufficiency`, not the queried signal** -- "
                        "wrong channel, not read as fabrication"), "global_flag"
        elif und and none * 2 > len(und):
            rd, kind = ("⚪ **Did not answer the queried signal** (no such key in `signal_quality`) -- "
                        "unanswered, not read as fabrication"), "no_answer"
        else:
            rd, kind = "🔴 Rarely declares on the absent side -- mixed channels, inspect `noop_answer_channel` per cell", "mixed"
    elif hit is not None and fa is not None and hit >= 0.7 and fa <= 0.3:
        rd, kind = "Actually checking the environment", "checking"
    else:
        rd, kind = "—", None
    return {"n_gap": len(ab), "n_cov": len(pr), "a1": a1, "p1": p1, "hit": hit, "fa": fa,
            "ok": (sum(oks) / len(oks) if oks else None), "channels": ch,
            "reading": rd, "kind": kind}
