"""Prospective candidate supply for the existing follow-up gold branches."""
from __future__ import annotations


def cover_branches(generate, probe, size, required, *, max_multiplier=8, minimum_emitted=0,
                   required_cells=None):
    """Expand all condition supplies together; never condition a draw on audit effect.

    The pool size is shared by nested publication sizes. Each attempted pool is
    retained by the probe. Patient effects, dates and errors are not redrawn.
    """
    attempts = []
    multiplier = 1
    while multiplier <= max_multiplier:
        text, stats = generate({k: v * multiplier for k, v in size.items()})
        result = probe(text, multiplier)
        counts = result["branches"] if required_cells is not None else result
        cells = result["cells"] if required_cells is not None else {}
        missing_cells = {k: n - cells.get(k, 0) for k, n in (required_cells or {}).items() if cells.get(k, 0) < n}
        missing = sorted(set(required) - {k for k, v in counts.items() if v})
        emitted = sum(counts.values())
        attempts.append({"multiplier": multiplier, "branches": dict(counts), "missing": missing,
                         "emitted": emitted, "minimum_emitted": minimum_emitted,
                         "cells": cells, "missing_cells": missing_cells})
        if not missing and not missing_cells and emitted >= minimum_emitted:
            return text, {**stats, "branch_supply": attempts}
        multiplier *= 2
    raise ValueError(f"candidate supply exhausted: {attempts}")
