"""Non-compensatory hard gates.

Hard gates enter the score as a multiplier, not a weighted term, so a failed
unit cannot be bought back::

    overall = weighted_mean * (1 - failed_units / total_units)

A unit is the zeroing scope (one answer or one time-slice); a case-level FAIL
fails every unit of that case.
"""
from __future__ import annotations

from typing import Iterable, Mapping

#: Default prefix for per-unit hit counters: `<prefix><gate>` is the hit
#: count, `<prefix><gate>_at` records where it hit.
UNIT_HIT_PREFIX = "sg_"

#: Gates recorded but not multiplied. Empty here: which gates those are is
#: task-layer knowledge, passed via `non_harm_gates=`.
DEFAULT_NON_HARM_GATES: frozenset[str] = frozenset()


def _gate_base(g) -> str:
    """Gate name with an instance suffix → base name.
    `some_gate:INSTANCE-03` → `some_gate`."""
    return str(g).split(":", 1)[0].strip()


def multiplier(rows: Iterable[Mapping], *,
               prefix: str = UNIT_HIT_PREFIX,
               units_field: str = "slice_gate_n_judged",
               gates_field: str = "gates",
               overall_field: str = "overall",
               unknown_field: str = "gates_recompute_skipped",
               id_field: str = "case",
               non_harm_gates: "frozenset[str] | set[str] | None" = None) -> dict:
    """The non-compensatory hard-gate multiplier.

    Returns `{"mult", "gated_units", "n_units", "n_gated_units", "n_soft_gated_units",
    "soft_gated_units", "n_unknown", "unknown_ids", "verified", "unit"}`.

    A row with no gate reading (no `gates` key, no `units_field`, no hit counters)
    or with untrustworthy readings counts as unknown, never as clean; `mult == 1.0`
    is only meaningful together with `verified` and `n_unknown`.
    """
    _non_harm = frozenset(non_harm_gates or DEFAULT_NON_HARM_GATES)
    rows = list(rows)
    # Per-unit counters of a non-harm gate are read like its list entries: recorded, not
    # multiplied (`soft_hit_fields`).
    _all_hit = sorted({k for r in rows for k in r
                       if k.startswith(prefix) and not k.endswith("_at")})
    hit_fields = [k for k in _all_hit if k[len(prefix):] not in _non_harm]
    soft_hit_fields = [k for k in _all_hit if k[len(prefix):] in _non_harm]
    n_units = n_failed = n_unknown = n_soft = 0
    gated: list[str] = []
    unknown: list[str] = []
    soft: list[str] = []   # declaration-type gates: listed, not multiplied
    for r in rows:
        ungraded = (gates_field not in r
                    and r.get(units_field) is None
                    and not any(k.startswith(prefix) for k in r)
                    and not str(r.get(overall_field, "")).startswith("FAIL"))
        # `gates_recompute_skipped` only marks rows unknown when no `gates` reading exists.
        if ungraded or (r.get(unknown_field) and gates_field not in r):
            n_unknown += int(r.get(units_field) or 0) or 1
            unknown.append(str(r.get(id_field)))
            continue
        u = int(r.get(units_field) or 0) or 1
        n_units += u
        # A missing `gates` list with `overall == FAIL` counts as failed: the gate is
        # unknown, and unknown must not fold into clean.
        _raw_gates = r.get(gates_field)
        if _raw_gates is not None:
            _harm = [g for g in _raw_gates if _gate_base(g) not in _non_harm]
            _soft = [g for g in _raw_gates if _gate_base(g) in _non_harm]
            if _soft:
                n_soft += u
                soft.append(f"{r.get(id_field)}:{';'.join(str(g) for g in _soft)}")
            if _harm:
                n_failed += u                  # instance scope ⇒ every unit fails
                gated.append(f"{r.get(id_field)}:{';'.join(str(g) for g in _harm)}")
                continue
        elif str(r.get(overall_field, "")).startswith("FAIL"):
            n_failed += u
            gated.append(f"{r.get(id_field)}:*")
            continue
        _soft_hit = min(sum(int(r.get(g) or 0) for g in soft_hit_fields), u)
        if _soft_hit:
            n_soft += _soft_hit
            soft.append(f"{r.get(id_field)}:" + ";".join(
                f"{g[len(prefix):]}@{r.get(g + '_at')}" for g in soft_hit_fields if r.get(g)))
        hit = sum(int(r.get(g) or 0) for g in hit_fields)
        # One unit can trip several gates => clamp, so the multiplier can
        # never go negative.
        hit = min(hit, u)
        if hit:
            n_failed += hit
            at = [f"{g[len(prefix):]}@{r.get(g + '_at')}" for g in hit_fields if r.get(g)]
            gated.append(f"{r.get(id_field)}:{';'.join(at)}")
    # Sorted so output does not depend on row order.
    return {"mult": (1.0 - n_failed / n_units) if n_units else 1.0,
            "gated_units": sorted(gated) or None,
            "n_units": n_units,
            "n_gated_units": n_failed,
            "n_soft_gated_units": n_soft,
            "soft_gated_units": sorted(soft) or None,
            "n_unknown": n_unknown,
            "unknown_ids": sorted(set(unknown))[:20] or None,
            "verified": bool(n_units) and not n_unknown,
            "unit": "unit" if any(r.get(units_field) for r in rows) else "whole"}
