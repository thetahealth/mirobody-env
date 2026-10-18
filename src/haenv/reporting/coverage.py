"""Coverage for the evaluation report."""
from __future__ import annotations

from haenv.reporting.gates import (
    _action_gate_surface,
    _case_gate_surface,
)


def append_mount_coverage(L, rows, built) -> None:
    """Append the original coverage report section in its original order."""
    from haenv.scoring.mounts.coverage import coverage as _mt_cov
    from haenv.scoring.mounts.table import known_gaps as _mt_gaps
    _cov = _mt_cov()
    L.append("### 1x. Cross-product coverage of judge \u00d7 geometry\n")
    L.append(f"> {_cov['n_judges']} judges \u00d7 {_cov['n_geometries']} geometries = "
             f"**{_cov['n_cells']} cells**; mounted **{_cov['mounted']}** \u00b7 "
             f"declared absent {_cov['declared_absent']} \u00b7 **unexplained holes {_cov['holes']}**\n")
    _g = _mt_gaps()
    if _g:
        # Separator hoisted out of the f-string: a backslash inside an f-string
        # expression is a SyntaxError before 3.12 (PEP 701), and requires-python
        # promises 3.10.
        _MT_GAP_SEP = "` \u00b7 `"
        L.append(f"> **{len(_g)} known gap(s)** "
                 f"(judge applies but is not wired on that geometry): "
                 f"`{_MT_GAP_SEP.join(_g)}`\n")
    L.append("> Every cell marked \"declared absent\" has a reason in `mount_table.WHY_NOT` -- "
             "\"too narrow\" and \"genuinely not applicable\" render identically on the table, and the reason is the only thing that tells them apart.\n")
    L += _case_gate_surface(rows, built)
    L += _action_gate_surface(rows)
