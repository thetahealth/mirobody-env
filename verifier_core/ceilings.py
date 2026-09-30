"""Degenerate-strategy ceilings: how much a strategy that understands nothing
scores on each dimension, and whether any real solver beats it by more than noise.

A dimension no real solver beats cannot be read as capability. Value
extraction, rosters, pairings and noise are injected by the caller; this
module holds only the judgement.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

#: Suffix of a combined pair's pseudo-dimension, `f"{a}+{b}{HARMONIC_SUFFIX}"`;
#: `resolve_noise` parses the same shape.
HARMONIC_SUFFIX = "(调和)"


def resolve_noise(dim: str, noise, *,
                  alias: Callable[[str], Iterable[str]] | None = None):
    """Noise floor for one dimension: exact name, then caller-supplied aliases,
    then, for a combined pair, the larger of the two sides' noise.
    """
    if not isinstance(noise, dict):
        return noise
    if dim in noise:
        return noise[dim]
    if alias is not None:
        for _alias in alias(dim):
            if _alias in noise:
                return noise[_alias]
    if dim.endswith(HARMONIC_SUFFIX) and "+" in dim:
        a, b = dim[:-len(HARMONIC_SUFFIX)].split("+", 1)
        xs = [x for x in (resolve_noise(a.strip(), noise, alias=alias),
                          resolve_noise(b.strip(), noise, alias=alias)) if x is not None]
        return max(xs) if xs else None
    return None


def degenerate(rows: Sequence[Mapping], dims: tuple[str, ...] = (),
               noise: "float | dict | None" = None, *,
               getval: Callable[[Mapping, str], object] | None = None,
               degenerate_solvers: Iterable[str] = (),
               oracle_solvers: Iterable[str] = (),
               pairs: Iterable[tuple[str, str]] = (),
               pair_values: Callable[..., Sequence] | None = None,
               declared: Mapping[str, Mapping] | None = None,
               direction: Mapping[str, str] | None = None,
               geometry: str | None = None,
               noise_alias: Callable[[str], Iterable[str]] | None = None,
               solver_field: str = "solver") -> dict:
    """Best degenerate score per dimension, and whether a real solver beat it.

    Oracles are excluded from the degenerate pool. Verdicts: `ok` (beaten by at
    least 2x noise), `unresolved` (beaten within noise), `not_a_capability_dim`
    (not beaten), `noise_unknown` (no noise floor), `direction_non_monotone`, or a
    reason for skipping (`declared_aggregate`, `declared_not_applicable`,
    `no_degenerate_floor`, `no_real_solver`, `undeclared_empty`). Reports only;
    blocking is the caller's decision. `noise` may be a float, a per-dimension
    dict, or None.
    """
    _get = getval or (lambda r, k: r.get(k))
    _stub, _orc = set(degenerate_solvers), set(oracle_solvers)
    _prof_m = declared or {}
    _geo = geometry
    per: dict[str, dict[str, list[float]]] = {}
    # Register every requested dimension so an empty one is reported, not dropped.
    if dims:
        per = {k: {} for k in dims}
    for r in rows:
        sname = r.get(solver_field)
        if dims:
            # Declared dimensions go through the extractor: under slice geometry the field
            # is `wk_<dim>_last`.
            for k in dims:
                v = _get(r, k)
                if isinstance(v, bool) or isinstance(v, (int, float)):
                    per.setdefault(k, {}).setdefault(sname, []).append(float(v))
        else:
            for k, v in r.items():
                if isinstance(v, bool) or isinstance(v, (int, float)):
                    per.setdefault(k, {}).setdefault(sname, []).append(float(v))

    # Each side of a pair is saturated by the other side's degenerate strategy, so
    # a declared pair is compared only as its combined pseudo-dimension.
    _means_of = {d: {s: sum(v) / len(v) for s, v in bys.items() if v}
                 for d, bys in per.items()}
    # Paired aggregation combines per cell, then averages (F1-type combinators are
    # non-linear).
    _by_solver: dict[str, list[Mapping]] = {}
    for _r in rows:
        _by_solver.setdefault(_r.get(solver_field), []).append(_r)
    for a, b in pairs:
        if a not in _means_of or b not in _means_of:
            continue
        ma, mb = _means_of[a], _means_of[b]
        combo: dict[str, list[float]] = {}
        for sname in set(ma) & set(mb):
            _vals = [v for v in pair_values(_by_solver.get(sname, []), a, b, _get)
                     if v is not None]
            if _vals:
                combo[sname] = [sum(_vals) / len(_vals)]
        if combo:
            per[f"{a}+{b}{HARMONIC_SUFFIX}"] = combo
            per.pop(a, None)
            per.pop(b, None)

    out: dict = {}
    for dim, bys in per.items():
        means = {s: sum(v) / len(v) for s, v in bys.items() if v}
        deg = {s: m for s, m in means.items() if s in _stub and s not in _orc}
        real = {s: m for s, m in means.items() if s not in _stub}
        if not deg or not real:
            _decl = (_prof_m.get(dim) or {})
            _ap = _decl.get("applies_to")
            if _decl.get("aggregate"):
                _why = "declared_aggregate"      # not a per-row field; expected
            elif _ap and _geo and _geo not in _ap:
                _why = "declared_not_applicable"  # profile says this geometry yields none
            elif real and not deg:
                _why = "no_degenerate_floor"     # real data, no stub ⇒ no floor at all
            elif deg and not real:
                _why = "no_real_solver"          # stubs but no real solver ⇒ equally undecidable
            else:
                _why = "undeclared_empty"        # both sides empty and nobody declared it
            out[dim] = {"verdict": _why, "n_deg": len(deg), "n_real": len(real),
                        "degenerate_ceiling": None, "ceiling_by": None,
                        "best_real": None, "best_real_by": None,
                        "margin": None, "margin_over_noise": None, "noise": None}
            continue
        _dir = str((direction or {}).get(dim) or "higher")
        if _dir == "non_monotone":
            out[dim] = {"verdict": "direction_non_monotone",
                        "n_deg": len(deg), "n_real": len(real),
                        "degenerate_ceiling": None, "ceiling_by": None,
                        "best_real": None, "best_real_by": None,
                        "margin": None, "margin_over_noise": None, "noise": None}
            continue
        _pick = min if _dir == "lower" else max
        bd, bdn = _pick(deg.values()), _pick(deg, key=lambda k: deg[k])
        br, brn = _pick(real.values()), _pick(real, key=lambda k: real[k])
        _nz = resolve_noise(dim, noise, alias=noise_alias)
        _m = (bd - br) if _dir == "lower" else (br - bd)
        if _m <= 0:
            _v = "not_a_capability_dim"
        elif _nz is None:
            _v = "noise_unknown"
        elif _m < 2 * _nz:
            _v = "unresolved"          # beat it, but not by more than noise ⇒ no capability conclusion
        else:
            _v = "ok"
        out[dim] = {
            "degenerate_ceiling": round(bd, 4), "ceiling_by": bdn,
            "best_real": round(br, 4), "best_real_by": brn,
            "margin": round(_m, 4),
            "margin_over_noise": (round(_m / _nz, 2) if _nz else None),
            "noise": _nz,
            "direction": _dir,
            "n_real_above": sum(1 for m in real.values()
                                if (m < bd if _dir == "lower" else m > bd)),
            "n_real": len(real),
            "verdict": _v,
        }
    return out
