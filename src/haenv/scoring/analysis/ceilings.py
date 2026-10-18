"""Constant/degenerate score ceilings and discrimination gates."""
from __future__ import annotations

from haenv.scoring.quantities import BY_NAME as _QBY
from haenv.scoring.quantities import row_names as _qrow_names
from haenv.scoring.analysis.dimensions import _rowdim


def batch_discrimination_gate(ceilings: dict, cap: float = 0.50) -> list[dict]:
    """Batch gate: a scored dimension whose "never looks at the question" ceiling exceeds `cap` gives no capability conclusion.

    The ceiling is computed per batch from its gold distribution.
    """
    hits = []
    for dim, v in (ceilings or {}).items():
        if not isinstance(v, (tuple, list)) or len(v) < 1 or v[0] is None:
            continue
        if dim == "tool_grounded_joint" or str(dim).startswith("tool_stub_"):
            continue      # reported, not scored (M1 amendment 2026-10-01): no acceptance line
        val = float(v[0])
        if val > cap:
            hits.append({"kind": "constant_ceiling_too_high", "dim": dim,
                         "ceiling": val, "cap": cap,
                         "detail": f"{dim}'s constant ceiling {val:.3f} > {cap} -- "
                                   f"a baseline that never looks at the question ({v[1] if len(v) > 1 else '?'}) can already get this much, "
                                   f"so this dimension's reading gives no capability conclusion; the fix is in batch composition (adjusting the gold distribution), not in the judge"})
    return hits


def constant_ceilings(rows: list[dict], gold: dict | None = None) -> dict:
    """Per-dimension ceiling for "never looking at the question", computed analytically from this batch's gold.

    Gives the best any constant can score (exact and free); `const_ddx` is one such
    constant and must never exceed it. M1 (2026-10-01): every core dimension of the
    joint-diagnosis board is covered (`blind_ceilings.core_constant_ceilings`); `gold`
    (`blind_ceilings.gold_from_cases_jsonl`) adds the dx_listed and tests-F1 ceilings,
    which need the gold lists rather than row fields. The retired `join_hit` /
    `urgency_ok` are no longer printed.
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
    # ddx-question core dimensions (M1): question-blind optimal constants
    if any(str(r.get("gold_kind") or "").startswith("ddx") for r in cases):
        from haenv.scoring.blind_ceilings import core_constant_ceilings as _ccc
        for k, v in _ccc(rows, gold).items():
            if isinstance(v, tuple):
                out[k] = v
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
    from haenv.world.generation.baselines import BASELINE_NAMES
    from haenv.world.generation.baselines import ORACLE_NAMES
    # Failing to read the profile raises (no bare except around `load_profile`).
    from haenv.scoring.legacy import load_profile
    # Paired aggregation comes from `separation.paired_grid_values`.
    from haenv.scoring.separation import paired_grid_values as _pgv
    # The per-dimension declaration only explains skips; unreadable => `undeclared_empty`.
    try:
        from haenv.scoring.legacy import load_profile as _lp_g
        _prof_m = _lp_g().metrics
    except Exception:
        _prof_m = {}
    # Geometry via `absence.geometry_of`.
    try:
        from haenv.runtime.row_store import geometry_of as _geom
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
