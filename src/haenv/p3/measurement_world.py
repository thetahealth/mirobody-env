"""M3 truth-first construction of raw measurement records.

Coverage contrasts are authored world targets, never reader-based selection.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

from haenv.resources.regpath import load_registry, registry_cached


@registry_cached("p3_measurements.yaml")
def table():
    return load_registry("p3_measurements.yaml")


def attach(blk, key):
    from haenv.world.truth import coupling_enabled
    from haenv.p3.plan import h01
    if not coupling_enabled() or not blk.get("documented_course_only"):
        return
    axis = blk["axis"]
    if axis not in ("home_sbp", "LDL", "TSH"):
        return
    cfg = table()["generation"]
    rec = next(r for r in blk["visits"] if int(r["day"]) == int(blk["T"]))
    day, value, truth = int(blk["T"]), float(rec["printed"]), float(rec["truth"])
    rows = []
    if axis == "home_sbp":
        from haenv.world.labworld.meds import axis as axis_table
        bv = load_registry("biological_variation.yaml")["home_bp"]
        spread = float(bv["repeat_sdd_systolic"]["value"]) / math.sqrt(2)
        upper = float(blk.get("upper", axis_table(axis)["upper"]))
        # Balance the dated day means, preserving the established B3 mean.
        # A day-level contrast is allocated before rendering or reading.
        contrast = h01("m3-contrast", key) < float(cfg["contrast_share"])
        delta = math.ceil(abs(upper - value) + spread) if contrast else max(1, round(spread))
        direction = -1 if value > upper else 1
        days, per_day = int(cfg["bp_days"]), int(cfg["bp_readings_per_day"])
        earlier = days - 2
        quotient, remainder = divmod(delta, earlier)
        daily = [-direction * (quotient + int(i < remainder)) for i in range(earlier)]
        for i in range(earlier // 2):
            amplitude = max(1, round(spread * (0.5 + h01("m3-day-spread", key, i))))
            daily[2 * i] -= amplitude
            daily[2 * i + 1] += amplitude
        daily.append(direction * delta)
        # Duplicate readings stay close at the axis's existing instrument
        # resolution. The contrast is across days, never a within-minute jump.
        resolution = 10 ** -int(axis_table(axis)["ndigits"])
        offsets = [round(spread) + (-1 if i % 2 == 0 else 1) * resolution for i in range(per_day)]
        offsets.extend(offset + (-1 if i % 2 == 0 else 1) * resolution
                       for offset in daily for i in range(per_day))
        for i, offset in enumerate(offsets):
            rows.append(dict(day=day - days + 1 + i // per_day, sequence=i % per_day + 1,
                             sample_id=f"BP-{i + 1}", value=value + offset, truth=truth + offset))
    elif axis == "LDL":
        # Nonfasting alone is valid. Only the high-TG Friedewald estimate is
        # superseded, with a direct assay on the repeat, no invented wait time.
        from haenv.world.labworld.meds import axis as axis_table, band
        upper = float(blk.get("upper", axis_table(axis)["upper"]))
        contrast = h01("m3-contrast", key) < float(cfg["contrast_share"])
        initial = (round(upper * math.exp((-1.5 if value > upper else 1.5) * band(axis)), 1)
                   if contrast else value)
        rows = [dict(day=day, sequence=1, sample_id="L1", value=initial, truth=truth,
                     fasting=False, method="friedewald", triglycerides=cfg["lipid_initial_triglycerides"]),
                dict(day=day, sequence=2, sample_id="L2", value=value, truth=truth,
                     fasting=True, method="direct", supersedes="L1")]
    else:
        rows = [dict(day=day, sequence=1, sample_id="T1", value=value, truth=truth)]
    blk["control_measurements"] = rows


def replace_control(blk, value, printed):
    """Keep existing surgical control twins coherent with their raw samples."""
    rows = blk.get("control_measurements")
    if not rows:
        return
    from haenv.p3.measurements import reading
    old = reading(rows, blk["axis"])
    old_truth = reading(rows, blk["axis"], truth=True)
    # For early TSH the sample exists as an observation despite pending target assessment.
    if old is None:
        old, old_truth = rows[-1]["value"], rows[-1]["truth"]
    for r in rows:
        r["value"] += printed - old
        r["truth"] += value - old_truth
