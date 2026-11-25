"""Read control measurements by collection protocol, independently of gold labels.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from statistics import mean


def usable(row, axis, *, last_change=None, steady_days=None):
    """Sample validity; the sourced constants are declared in p3_measurements.yaml."""
    if axis == "potassium" and row.get("haemolysed"):
        return False
    if axis == "LDL" and row.get("method") == "friedewald" and float(row.get("triglycerides", 0)) >= 4.5:
        return False
    if axis == "TSH" and last_change is not None:
        from haenv.world.labworld.meds import drug
        steady_days = int(drug("levothyroxine")["tau_rule_days"]) if steady_days is None else steady_days
        if int(row["day"]) - last_change < steady_days:
            return False
    return True


def reading(rows, axis, *, last_change=None, truth=False):
    """Return the valid mean/latest repeat, or None when reassessment is due.

    A superseding result only replaces an earlier sample if it is itself valid.
    First-day home BP readings are excluded by their collection date, not placement.
    """
    if not rows:
        return None
    field = "truth" if truth else "value"
    if axis == "home_sbp":
        first = min(int(r["day"]) for r in rows)
        valid = [r for r in rows if int(r["day"]) > first]
        # NICE NG136: at least four days, duplicate morning/evening readings.
        days = {int(r["day"]) for r in rows}
        if len(days) < 4 or any(sum(int(r["day"]) == d for r in rows) < 4 for d in days):
            return None
        return mean(float(r[field]) for r in valid)
    valid = [r for r in rows if usable(r, axis, last_change=last_change)]
    if not valid:
        return None
    replaced = {r.get("supersedes") for r in valid}
    valid = [r for r in valid if r["sample_id"] not in replaced]
    return float(max(valid, key=lambda r: (int(r["day"]), int(r["sequence"])))[field]) if valid else None


def effective_record(blk):
    """Hidden-truth gold reads the same collection protocol as the visible reader."""
    rec = next(r for r in blk["visits"] if int(r["day"]) == int(blk["T"]))
    rows = blk.get("control_measurements")
    if not rows:
        return rec
    from haenv.p3.gold import last_change
    kwargs = dict(last_change=last_change(blk["dose_line"], int(blk["T"])))
    tr = reading(rows, blk["axis"], truth=True, **kwargs)
    val = reading(rows, blk["axis"], **kwargs)
    # An early TSH still carries a safety observation; the existing early rule
    # holds the dose. It must not be treated as a steady-state dose target.
    return {**rec, "truth": tr if tr is not None else rec["truth"],
            "printed": val if val is not None else rec["printed"],
            "measurement_pending": val is None}
