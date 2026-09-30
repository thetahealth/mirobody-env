"""Compare the judgeable cases and provenance of real-model and offline-stub rows.

Shared by batch-restamping tools and the maintainers' regression suite. The
comparison reads saved rows only and does not call models or change scores.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations


BOARD_HALVES_FIELDS = ("haenv_git_sha", "job_sha256", "generator_sha", "kernel_sha256")


def split_board_halves(rows_j: list[dict]) -> dict:
    """Split a batch into each half's provenance values and judgeable case set.

    Model membership uses the production ``real_solver_pool`` lookup. A case
    enters a half's denominator when at least one of its rows is not aborted;
    another model's failed response does not remove a usable case.
    """
    from haenv.report import real_solver_pool
    pool = real_solver_pool(rows_j)[0]
    out: dict = {}
    for half in ("real", "stub"):
        sel = [r for r in rows_j
               if (str(r.get("solver")) in pool) == (half == "real")]
        out[half] = {
            "n_rows": len(sel),
            "solvers": len({str(r.get("solver")) for r in sel}),
            "prov": {f: {str((r.get("world_ref") or {}).get(f)) for r in sel}
                     for f in BOARD_HALVES_FIELDS},
            "cases": {r.get("case") for r in sel
                      if not str(r.get("overall") or "").startswith("ABORT")},
        }
    return out


def board_halves_verdict(sp: dict) -> tuple[list[str], list[str]]:
    """Return denominator and provenance discrepancies separately.

    Different case sets are rejected, including sets with equal counts. Different
    provenance is reported separately because backfilling offline stubs is allowed.
    When one half is absent, the caller must report the comparison as unmeasured.
    """
    den, prov = [], []
    if not sp["real"]["n_rows"] or not sp["stub"]["n_rows"]:
        return den, prov
    r, s = sp["real"]["cases"], sp["stub"]["cases"]
    if r != s:
        den.append(f"可判例集不同 —— 真实模型 {len(r)} 例 vs 桩 {len(s)} 例 · "
                   f"只在真实模型那半可判的例:{sorted(r - s)} · "
                   f"只在桩那半:{sorted(s - r)}")
    for f in BOARD_HALVES_FIELDS:
        if sp["real"]["prov"][f] != sp["stub"]["prov"][f]:
            prov.append(f"{f}: 真实模型 {sorted(sp['real']['prov'][f])}"
                        f" vs 桩 {sorted(sp['stub']['prov'][f])}")
    return den, prov
