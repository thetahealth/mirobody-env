"""Knife-edge check for P3 blood-pressure-target items (world 1.5, B3: fidelity F4).

An item's gold reads today's home systolic 7-day mean against the target (and, for an ACEI titrated on
target in CKD, against the clinic's floor). The gold must not rest on one reading's noise: re-measuring the
same truth should not put the printed mean on the other side of a threshold from the truth. With the
sourced within-person SD of one home mean (Stergiou 2002: SD of differences 6.9 mmHg between repeated
home means, so 6.9 / sqrt 2 = 4.88 mmHg per mean; `registry/biological_variation.yaml`), an item is
refused when a single redraw crosses a threshold with probability >= 2.5 % (margin < 1.96 SD). The
check also reports one seeded redraw per item. Report-only in 1.5 Phase B: it refuses nothing at
generation time (items are never dropped by a gate; Phase D switches gates on).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

#: Gate level: a redraw crosses a threshold with probability >= Phi(-Z_GATE) (2.5 %).
Z_GATE = 1.96
BP_AXIS = "home_sbp"
BLOCK = "meds_p3"


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def thresholds(blk: dict, truth: float | None = None) -> list[float]:
    """The lines today's reading is read against: the patient's target and, when the truth is on target
    below the top rung of a drug titrated on target with a registered floor (`gold.g3`), the floor."""
    from haenv.p3.gold import known_of
    from haenv.world.labworld.meds import drug
    d = drug(blk["drug"], known_of(blk))
    out = [float(blk["upper"])]
    from haenv.p3.gold import dose_at
    on_target = truth is None or float(truth) < float(blk["upper"])
    at_max = float(dose_at(blk["dose_line"], int(blk["T"]))) == float(d["ladder"][-1])
    if on_target and not at_max and d.get("on_target") == "titrate" and d.get("titrate_on_target_min"):
        out.append(float(d["titrate_on_target_min"]))
    return out


def check(blk: dict, sd: float | None = None) -> dict:
    """One P3 home-BP item: the truth's margin to its nearest threshold in SDs of one home mean, the
    probability that one redraw crosses it, whether the gate refuses it, and one seeded redraw."""
    from haenv.world.labworld import z
    if sd is None:
        from haenv.world.truth import variation as V
        sd = V.home_mean_repeat_sd()
    T = int(blk["T"])
    rec = next(r for r in blk["visits"] if int(r["day"]) == T)
    truth = float(rec["truth"])
    lines = thresholds(blk, truth)
    margin = min(abs(truth - t) for t in lines)
    p_flip = sum(_phi(-abs(truth - t) / sd) for t in lines)
    key = str(blk.get("record_key") or blk.get("course_documentation_key") or "")
    redraw = round(truth + sd * z("b3-knife-edge", key, T))
    crossed = any((redraw >= t) != (truth >= t) for t in lines)
    return {"truth": truth, "printed": rec["printed"], "thresholds": lines, "margin_mmHg": round(margin, 3),
            "margin_sd": round(margin / sd, 3), "p_flip": round(p_flip, 4),
            "refuse": margin < Z_GATE * sd, "redraw": redraw, "redraw_flips": crossed}


def bp_blocks(rows):
    """(case_id, block) of every P3 item judged on the home systolic axis."""
    for row in rows:
        case = row.get("case", row)
        blk = ((case.get("adjudication") or {}).get(BLOCK)) or {}
        if blk.get("axis") == BP_AXIS and blk.get("visits"):
            yield str(case.get("case_id")), blk


def report(rows, sd: float | None = None) -> dict:
    """Counts over a batch: items checked, refused by the gate, flipped by the seeded redraw."""
    res = {cid: check(blk, sd) for cid, blk in bp_blocks(rows)}
    return {"n_bp_items": len(res), "refused": sorted(c for c, r in res.items() if r["refuse"]),
            "redraw_flips": sorted(c for c, r in res.items() if r["redraw_flips"]),
            "z_gate": Z_GATE, "items": res}
