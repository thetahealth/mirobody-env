"""Raw per-item diagnostic correctness from persisted production atoms.

This is an analysis convenience, not a new scoring dimension. Missing atoms
are unknown (None), never silently counted wrong or correct. An independent
case has no diagnosis to list: its production join judgement is the atom.
"""
from __future__ import annotations


def raw_correct(row: dict) -> bool | None:
    if str(row.get("overall") or "").startswith("ABORT"):
        return False
    kind = row.get("dx_kind") or str(row.get("gold_kind", "")).removeprefix("ddx:")
    if kind == "independent":
        if row.get("join_hit") is not None:
            return bool(row["join_hit"])
        if row.get("held_independent") is not None:
            return bool(row["held_independent"])
        return None
    if kind in ("unified", "comorbidity"):
        value = row.get("dx_listed")
        return float(value) == 1.0 if value is not None else None
    return None
