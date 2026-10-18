"""Formatting for the evaluation report."""
from __future__ import annotations

from haenv.scoring.analysis.tracks import _NO_HEADROOM_EPS


def _ceiling_line(ceil: dict, keys: list[tuple[str, str]]) -> str:
    """One-line description of the constant ceiling."""
    parts = []
    for key, zh in keys:
        v = ceil.get(key)
        if v:
            parts.append(f"**{zh} {v[0]}**({v[1]})")
    return " · ".join(parts)


def _render_stamp() -> str:
    """Render date, dating the "consistent with current code" statement (day precision keeps reruns diff-stable)."""
    import datetime
    return datetime.date.today().isoformat()


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
