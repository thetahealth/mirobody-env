"""Reading a batch's result rows back: the last row per cell, and the geometry the rows were run on.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import json
from pathlib import Path


def geometry_of(rows: list[dict]) -> str:
    """This batch's execution geometry, used to tell a not-applicable dimension from a defect."""
    if any(r.get("slice_rows") for r in rows):
        return "slices"
    if any(r.get("rounds") for r in rows):
        return "multi"
    if any(r.get("tool_budget") is not None for r in rows):
        return "gated"
    return "single"


def load_rows(path: Path) -> list[dict]:
    """Read back a batch's result rows, keeping the last row per cell (a retried cell has two).
    Earlier rows stay in the file; statistics use this function.
    """
    out: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            for old_key, new_key in RENAMED_ROW_KEYS.items():
                if old_key in r and new_key not in r:
                    r[new_key] = r.pop(old_key)
            out[(r.get("case"), r.get("solver"))] = r      # later rows overwrite earlier ones
    return list(out.values())


#: Row keys renamed after batches were written with the old name. Stored batches are
#: read-only, so the old key is mapped here, at the one entry point that reads them.
RENAMED_ROW_KEYS: dict[str, str] = {"rival_recall_ang": "rival_recall_capped2"}
