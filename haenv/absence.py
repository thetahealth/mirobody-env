"""absence.py -- why a cell has no value, as a controlled vocabulary.

An empty set is not compliance: a cell with nothing to judge is recorded as
not-applicable and excluded from aggregation, never scored as full marks. This module
records why a value is missing (`REASONS`), so "the judge is not mounted", "nothing to
judge", "the cell crashed" and "never scored" stay distinguishable in the artifact.
Key absent from a row means the judge did not mount; key present with value None means
it mounted but had nothing to judge (its denominator field in `DENOM_FIELD` is empty).

`absence_report` classifies each dimension by its null rate over a batch:
  * `applicability` -- null rate above `NULL_RATE_GATE` is a hard fail;
  * `sentinel` -- registered in `SENTINEL_DIMS`, a high null rate is by design, report only;
  * `not_wired` -- no reading anywhere in the batch; a debt, never folded into the other two.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations
from .row_store import (  # noqa: F401
    geometry_of,
)

# ---------------------------------------------------------------- controlled vocabulary (changing it = changing the contract)
REASONS: dict[str, str] = {
    "not_scored":
        "This cell was never scored (response exhausted / leak / hard-rule violation). "
        "The judge never ran on it -- this must be kept separate from \"ran but could not "
        "judge\": the former is an environment problem, the latter is a question-or-answer problem",
    "errored":
        "This cell crashed (`overall == 'ERROR'`) -- a code exception, not an environment "
        "problem, and not a question-or-answer problem. A crash is a defect and is recorded "
        "separately from `not_scored` (response-exhausted/leak).",
    "judge_not_mounted":
        "The judge is not mounted for this gold kind or `when` precondition (`Judge.mounts` "
        "is False). Artefact shape: this dimension's key (including slice aliases) is absent "
        "from the row entirely. Example: an `independent`-class item has no unified diagnosis "
        "=> `dx_hit` does not mount",
    "no_target":
        "The judge is mounted, but this item has nothing to judge (no near-miss / no "
        "exclusion / not a single test ordered). Artefact shape: the key is present, value is "
        "None, and this dimension's denominator field is None or 0",
    "dim_not_wired":
        "Not a single reading across the whole batch -- the geometry is absent (e.g. the "
        "tools track needs `gated`) or the aggregator never reads it. This is debt, not "
        "not-applicable, and is kept distinct from `no_target`",
    "absent_unclassified":
        "Key is present, value is None, while the denominator field has a value -- "
        "unclassified. A count > 0 means this table needs extending",
}

#: Each dimension's denominator field, registered explicitly (the names follow no pattern).
#: None/0 => nothing to judge.
DENOM_FIELD: dict[str, str] = {
    "disc_recall": "disc_n_rivals",
    "excl_grounded_rate": "excl_n",
    "rev_responsiveness": "rev_n_substantive",
    "rev_stability": "rev_n_neutral",
    # List-layer dimensions share the denominators of their counterparts.
    "rev_list_responsiveness": "rev_n_substantive",
    "rev_list_stability": "rev_n_neutral",
    "tests_recall": "tests_total",
    "tests_precision": "tests_proposed",
    "noop_ok": "noop_n_pts_in_window",
    "quant_ok": "quant_n_visible",
}

#: Null-rate threshold for the hard fail; a frozen constant. Measured null rates are bimodal
#: (0.000-0.303 / 1.000), so 0.50 separates the two clusters.
NULL_RATE_GATE = 0.50

#: Sentinel dimensions: triggered by a specific condition, high null rate by design, report
#: only. An entry must state its triggering condition, its measured null rate and sample size,
#: and a degenerate stub that triggers it on a real batch.
SENTINEL_DIMS: dict[str, str] = {}


def absence_of(row: dict, dim: str) -> str | None:
    """Why does this cell of this dimension have no value? Returns None when it
    has one.

    Reads only production fields: `overall` (scoring status), this
    dimension's key and slice aliases (mounted or not), and the denominator
    field registered in `DENOM_FIELD` (whether there was anything to judge). No
    guessing, no separate lookup path.
    """
    from .quantities import BY_NAME as _QBY  # deferred import: avoids a cycle with report
    from .quantities import row_names as _qrow_names
    from .analytics import _rowdim
    if isinstance(_rowdim(row, dim), (int, float)):
        return None
    _ov = str(row.get("overall", ""))
    if _ov.startswith("ABORT"):
        return "not_scored"
    # Checked before `judge_not_mounted`: an ERROR row has no dimension keys, but the judge never
    # ran rather than not mounting.
    if _ov == "ERROR":
        return "errored"
    keys = list(_qrow_names(dim)) if dim in _QBY else [dim]
    if not any(k in row for k in keys):
        return "judge_not_mounted"
    den = DENOM_FIELD.get(dim)
    if den is not None:
        v = row.get(den)
        if v is None or (isinstance(v, (int, float)) and not v):
            return "no_target"
        return "absent_unclassified"
    # No denominator registered: cannot tell "nothing to judge" from anything else.
    return "absent_unclassified"


def absence_report(rows: list[dict], dims: list[str],
                   recs: list[dict] | None = None) -> list[dict]:
    """Per dimension: null rate, reason counts and class. `rows` should hold only real-model cells.

    Aggregate-layer dimensions (`dx_top1`, `join_macro`, ...) have no key in the rows; they
    are computed by `rank_ddx`. Pass its output as `recs` to tell them apart from
    `not_wired`; without `recs` such a dimension is classed `unknown_layer` and not failed.
    """
    from .analytics import _REC_KEY
    from .scoring import load_profile
    # Geometry applicability and aggregate status come from the profile's declarations
    # (`applies_on` / `is_aggregate`), not from inference.
    _prof = load_profile()
    _geo = geometry_of(rows)
    _agg: dict[str, bool] = {}
    if recs is not None:
        for d in dims:
            k = _REC_KEY.get(d, d)
            _agg[d] = any(isinstance(rc.get(k), (int, float)) for rc in recs)
    out = []
    for d in dims:
        reasons: dict[str, int] = {}
        n_have = 0
        for r in rows:
            why = absence_of(r, d)
            if why is None:
                n_have += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
        n = len(rows)
        n_null = n - n_have
        rate = (n_null / n) if n else 0.0
        if n_have == 0 and n and not _prof.applies_on(d, _geo):
            cls = "geometry_not_applicable"
        elif n_have == 0 and n and _prof.is_aggregate(d):
            cls = "declared_aggregate"
        elif n_have == 0 and n and recs is None:
            cls = "unknown_layer"
        elif n_have == 0 and n and _agg.get(d):
            # Computable by the aggregator but not declared `aggregate: per_solver` in the profile.
            cls = "undeclared_aggregate"
        elif n_have == 0 and n:
            cls = "not_wired"
        elif d in SENTINEL_DIMS:
            cls = "sentinel"
        else:
            cls = "applicability"
        out.append({
            "dim": d, "n": n, "n_have": n_have, "n_null": n_null,
            "null_rate": round(rate, 4), "reasons": dict(sorted(reasons.items())),
            "class": cls,
            # Only the applicability class hard-fails.
            "fails": cls == "applicability" and rate > NULL_RATE_GATE,
            "agg_has_value": _agg.get(d) if recs is not None else None,
            "geometry": _geo,
            "unclassified": reasons.get("absent_unclassified", 0),
        })
    return sorted(out, key=lambda x: (-x["null_rate"], x["dim"]))


def absence_gate(report: list[dict]) -> list[dict]:
    """The hard-fail entries. `applicability`-class null rate over threshold =>
    this dimension has nothing to judge in this batch."""
    return [r for r in report if r["fails"]]
