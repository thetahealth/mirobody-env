"""G4 -- the follow-up item's gold, recomputed from the per-point records (design 4.2, PREREG 2).

Inputs are only the two compared records (t0 = previous draw, t1 = today) and the analyte's
RCV: printed values give dlog_obs, the 4-decimal truths give dlog_true, and the perturbations
marked `acts` give the net bias B of each draw; an acting perturbation that carries less than half
of the observed change does not name the source (R15, `carries`). The four zones partition the admissible items;
anything outside them (gray band RCV/4 <= |dtrue| < RCV/2 included) returns `None` and is not
emitted.

This module is in both segments of pack 4: the generator calls it to accept an item, the
emission gate and the audit call it to re-derive the gold, and the judge reads its output.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

CLASSES = ("true_change", "analytic_biological_noise", "preanalytical", "method_difference")
#: |dobs| / RCV of a noise item: inside the reference change value, so the two readings do not differ
#: beyond what measurement error and within-person variation produce; 0.4 keeps the change visible, the
#: 0.9 ceiling leaves a margin for the printed rounding. The other classes sit above the analyte's band
#: (`band`, from 1.2), where the RCV reading calls the change significant.
NOISE_BAND = (0.4, 0.9)
#: The next step per class. A change inside the RCV is no significant change: noise takes `no_action`
#: (routine monitoring continues); the three classes with a change beyond RCV are acted on or repeated
#: under standard conditions (provisional, K5 lab sign-off pending).
NEXT_STEP = {"true_change": "act_on_change", "preanalytical": "repeat_standardized",
             "method_difference": "repeat_standardized", "analytic_biological_noise": "no_action"}
_KIND_TO_CLASS = {"preanalytical": "preanalytical", "method": "method_difference"}


def acting(rec: dict) -> list[dict]:
    return [p for p in rec.get("perturbations") or () if p.get("acts")]


def carries(p: dict, rec: dict, side: int, o: float) -> bool:
    """R15: an acting perturbation names the source only if it carries at least half of the observed
    change (`side` -1 on the previous draw, +1 on today's). A Jaffe switch on a CKD patient's raised
    creatinine acts, but by too little to be the source."""
    from ..labworld.observe import net_bias_log
    b = side * net_bias_log(rec["truth"], rec["e"], [p])
    return b * o > 0 and abs(b) >= abs(o) / 2


def point_bias(rec: dict) -> float:
    from ..labworld.observe import net_bias_log
    return net_bias_log(rec["truth"], rec["e"], rec.get("perturbations"))


def deltas(r0: dict, r1: dict) -> dict:
    return {"obs": math.log(float(r1["printed"]) / float(r0["printed"])),
            "true": math.log(float(r1["truth"]) / float(r0["truth"])),
            "bias": point_bias(r1) - point_bias(r0)}


def g4(r0: dict, r1: dict, rcv: float, band: tuple[float, float]) -> dict:
    """`{class | None, reason, dlog_obs, dlog_true, bias, ratio, factor}`."""
    d = deltas(r0, r1)
    o, t, b = d["obs"], d["true"], d["bias"]
    ratio = abs(o) / rcv
    out = {"class": None, "dlog_obs": round(o, 6), "dlog_true": round(t, 6), "bias": round(b, 6),
           "ratio": round(ratio, 6), "factor": None, "reason": ""}
    within = NOISE_BAND[0] <= ratio <= NOISE_BAND[1]
    if not (within or band[0] <= ratio <= band[1]):
        out["reason"] = f"|dobs|/RCV {ratio:.3f} outside band {band} and the within-RCV band {NOISE_BAND}"
        return out
    acts = [p for p in acting(r0) if carries(p, r0, -1, o)] + [p for p in acting(r1) if carries(p, r1, 1, o)]
    if within:
        # inside the RCV only a change nothing names is noise: no true change, no acting perturbation
        if acts or abs(t) >= rcv / 4:
            out["reason"] = "a change inside the RCV with a named source or a true change"
            return out
        out["class"] = "analytic_biological_noise"
        return out
    kinds = {p.get("kind") for p in acts}
    if len(acts) > 1 or (acts and len(kinds) != 1):
        out["reason"] = f"{len(acts)} acting perturbations"
        return out
    if abs(t) >= rcv / 2:
        if acts:
            out["reason"] = "true change under an acting perturbation"
            return out
        if t * o <= 0 or abs(t) < abs(o) / 2:
            out["reason"] = "true change does not carry the observed change"
            return out
        out["class"] = "true_change"
        return out
    if abs(t) >= rcv / 4:
        out["reason"] = f"gray band |dtrue|/RCV {abs(t) / rcv:.3f}"
        return out
    if not acts:
        out["reason"] = "a change beyond the RCV with no source"
        return out
    p = acts[0]
    cls = _KIND_TO_CLASS.get(str(p.get("kind")))
    if cls is None:
        out["reason"] = f"acting perturbation of kind {p.get('kind')}"
        return out
    if b * o <= 0 or abs(b) < abs(o) / 2:
        out["reason"] = "perturbation does not carry the observed change"
        return out
    out["class"], out["factor"] = cls, p.get("factor")
    return out


def true_change_pct(r0: dict, r1: dict) -> float:
    return round(100.0 * (float(r1["truth"]) / float(r0["truth"]) - 1.0), 3)


def gold_block_consistent(blk: dict) -> list[str]:
    """Re-derive G4 from the records stored in an `adjudication.followup_p4` block; the list of
    disagreements (empty = consistent). Used by the emission gate and the audit."""
    from ..labworld import analyte
    from ..labworld.observe import record_consistent
    bad: list[str] = []
    recs = {int(r["day"]): r for r in blk.get("records") or ()}
    r0, r1 = recs.get(int(blk["t0"])), recs.get(int(blk["t1"]))
    if r0 is None or r1 is None:
        return ["records of t0/t1 missing"]
    a = analyte(blk["analyte"])
    for r in recs.values():
        why = record_consistent(r, int(a["ndigits"]), tuple(a["range"]))
        if why:
            bad.append(why)
    g = g4(r0, r1, float(blk["rcv"]), tuple(blk["band"]))
    if g["class"] != blk.get("class"):
        bad.append(f"G4 recomputed {g['class']} ({g['reason']}) != gold {blk.get('class')}")
    if abs(true_change_pct(r0, r1) - float(blk.get("true_change_pct"))) > 1e-6:
        bad.append("true_change_pct does not follow from the truths")
    for p in r0.get("perturbations", []) + r1.get("perturbations", []):
        if p.get("role") == "decoy" and p.get("acts"):
            bad.append(f"decoy {p.get('factor')} marked acting")
    return bad
