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
#: 0.7 ceiling keeps it clearly inside the RCV for either method's analytical CV and for the printed
#: rounding. The other classes sit in the analyte's change band (`realize.change_band`, from 1.3), where
#: the RCV reading calls the change significant.
NOISE_BAND = (0.4, 0.7)
#: The default next step per class. A change inside the RCV is no significant change: noise takes
#: `no_action` (routine monitoring continues); the three classes with a change beyond RCV are acted on
#: or repeated under standard conditions. `next_step` refines it by the side of the source and by the
#: ACEI direction (provisional, K5 lab sign-off pending).
NEXT_STEP = {"true_change": "act_on_change", "preanalytical": "repeat_standardized",
             "method_difference": "repeat_standardized", "analytic_biological_noise": "no_action"}
#: KDIGO 2024 CKD PP 3.6.4: "Continue ACEi or ARB therapy unless serum creatinine rises by more than 30%
#: within 4 weeks following initiation of treatment" -- a measured rise, read on the printed values.
ACEI_RISE_MAX_PCT = 30.0
_KIND_TO_CLASS = {"preanalytical": "preanalytical", "method": "method_difference"}


def acting(rec: dict) -> list[dict]:
    return [p for p in rec.get("perturbations") or () if p.get("acts")]


def carries(p: dict, rec: dict, side: int, o: float) -> bool:
    """R15: an acting perturbation names the source only if it carries at least half of the observed
    change (`side` -1 on the previous draw, +1 on today's). A Jaffe switch on a CKD patient's raised
    creatinine acts, but by too little to be the source."""
    from haenv.world.labworld.observe import net_bias_log
    b = side * net_bias_log(rec["truth"], rec["e"], [p])
    return b * o > 0 and abs(b) >= abs(o) / 2


def point_bias(rec: dict) -> float:
    from haenv.world.labworld.observe import net_bias_log
    return net_bias_log(rec["truth"], rec["e"], rec.get("perturbations"))


def method_key(rec: dict) -> str | None:
    """`"jaffe"` when the draw is measured by the Jaffe method (as the cause or throughout), else None
    (the analyte's default method): the key of `labworld.cv_total`."""
    return "jaffe" if any(p.get("factor") == "jaffe_enzymatic_switch"
                          and (p.get("role") == "cause" or (p.get("params") or {}).get("method_throughout"))
                          for p in rec.get("perturbations") or ()) else None


def pair_rcv(analyte: str, r0: dict, r1: dict) -> float:
    """The RCV of the compared pair from the two draws' methods (`labworld.rcv95`)."""
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    return rcv95(cv_total(analyte, method_key(r0)), cv_total(analyte, method_key(r1)))


def deltas(r0: dict, r1: dict) -> dict:
    return {"obs": math.log(float(r1["printed"]) / float(r0["printed"])),
            "true": math.log(float(r1["truth"]) / float(r0["truth"])),
            "bias": point_bias(r1) - point_bias(r0)}


def g4(r0: dict, r1: dict, rcv: float, band: tuple[float, float], *, source_policy=None) -> dict:
    """`{class | None, reason, dlog_obs, dlog_true, bias, ratio, factor}`."""
    d = deltas(r0, r1)
    o, t, b = d["obs"], d["true"], d["bias"]
    ratio = abs(o) / rcv
    out = {"class": None, "dlog_obs": round(o, 6), "dlog_true": round(t, 6), "bias": round(b, 6),
           "ratio": round(ratio, 6), "factor": None, "reason": ""}
    within = NOISE_BAND[0] <= ratio <= NOISE_BAND[1]
    clinical = source_policy == "clinical"
    if source_policy not in (None, "clinical"):
        raise ValueError("unknown source interpretation policy")
    if not clinical and not (within or band[0] <= ratio <= band[1]):
        out["reason"] = f"|dobs|/RCV {ratio:.3f} outside band {band} and the within-RCV band {NOISE_BAND}"
        return out
    acts = [p for p in acting(r0) if carries(p, r0, -1, o)] + [p for p in acting(r1) if carries(p, r1, 1, o)]
    if within and not clinical:
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
    if clinical and acts and abs(t) >= rcv / 4:
        out["reason"] = "true change under an acting perturbation"
        return out
    # Clinical source interpretation asks which component carries the
    # observed change. The old RCV truth bands are an item construction
    # restriction, not a minimum ACEI response in a sourced clinical case.
    # Retaining them here preferentially emitted larger treatment responses.
    true_carries = abs(t) > 0 and abs(t) >= abs(o) / 2
    true_component = true_carries if clinical else abs(t) >= rcv / 2
    if true_component:
        if acts:
            out["reason"] = "true change under an acting perturbation"
            return out
        if t * o <= 0 or abs(t) < abs(o) / 2:
            out["reason"] = "true change does not carry the observed change"
            return out
        out["class"] = "true_change"
        return out
    if not clinical and abs(t) >= rcv / 4:
        out["reason"] = f"gray band |dtrue|/RCV {abs(t) / rcv:.3f}"
        return out
    if not acts:
        if clinical:
            out["class"] = "analytic_biological_noise"
            return out
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


def next_step(blk: dict) -> tuple[str, tuple[str, ...]]:
    """`(primary, accepted)` next steps of a gold block (or a realized item): its class, `cause`,
    `t0`/`t1` and the two compared records.

    * noise: no action, a standardized repeat accepted;
    * pre-analytical / method source on the previous draw: today's reading is the standard one, so no
      action (a repeat accepted); on today's draw: a standardized repeat;
    * ACEI stop: the fall is the expected reversal of the drug's haemodynamic effect, no action
      (acting on it accepted);
    * ACEI start: a printed rise of at most 30% takes a repeat as primary; no action remains
      accepted. Repeat renal function and potassium for clinician assessment, without autonomous
      stopping or dose changes. Timing follows indication and risk (NHS SPS ACEI/ARB monitoring;
      for CKD, KDIGO 2024 PP 3.6.2/3.6.4 and NICE NG203 1.6.15/1.6.21/1.6.22), not a universal
      CKD interval. Beyond 30% acted on, a repeat also accepted when the rise over the mean of the
      pre-treatment readings is within 30%;
    * any other true change: acted on."""
    cls = blk["class"]
    if cls == "analytic_biological_noise":
        return "no_action", ("no_action", "repeat_standardized")
    recs = {int(r["day"]): r for r in blk.get("records") or ()}
    r0, r1 = recs[int(blk["t0"])], recs[int(blk["t1"])]
    if cls in ("preanalytical", "method_difference"):
        if acting(r0) and not acting(r1):
            return "no_action", ("no_action", "repeat_standardized")
        return "repeat_standardized", ("repeat_standardized",)
    if blk.get("cause") == "acei_start_or_stop":
        obs = 100.0 * (float(r1["printed"]) / float(r0["printed"]) - 1.0)
        if obs < 0:
            return "no_action", ("no_action", "act_on_change")
        if obs <= ACEI_RISE_MAX_PCT:
            return "repeat_standardized", ("repeat_standardized", "no_action")
        if pretreatment_rise_pct(blk, recs) <= ACEI_RISE_MAX_PCT:
            return "act_on_change", ("act_on_change", "repeat_standardized")
    return "act_on_change", ("act_on_change",)


def pretreatment_rise_pct(blk: dict, recs: dict) -> float:
    """Printed rise of today's reading over the mean of the readings since the last ACEI event up to
    the previous draw (the off-drug run before the start)."""
    t0, t1 = int(blk["t0"]), int(blk["t1"])
    last = max((int(e["day"]) for e in (blk.get("acei") or {}).get("events") or () if int(e["day"]) <= t0),
               default=None)
    pre = [float(r["printed"]) for d, r in recs.items() if d <= t0 and (last is None or d > last)] \
        or [float(recs[t0]["printed"])]
    return 100.0 * (float(recs[t1]["printed"]) / (sum(pre) / len(pre)) - 1.0)


def true_change_pct(r0: dict, r1: dict) -> float:
    return round(100.0 * (float(r1["truth"]) / float(r0["truth"]) - 1.0), 3)


def gold_block_consistent(blk: dict) -> list[str]:
    """Re-derive G4 from the records stored in an `adjudication.followup_p4` block; the list of
    disagreements (empty = consistent). Used by the emission gate and the audit."""
    from haenv.world.labworld import analyte
    from haenv.world.labworld.observe import record_consistent
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
    if abs(pair_rcv(blk["analyte"], r0, r1) - float(blk["rcv"])) > 1e-5:
        bad.append("rcv does not follow from the two draws' methods")
    g = g4(r0, r1, float(blk["rcv"]), tuple(blk["band"]), source_policy=blk.get("source_policy"))
    if g["class"] != blk.get("class"):
        bad.append(f"G4 recomputed {g['class']} ({g['reason']}) != gold {blk.get('class')}")
    if abs(true_change_pct(r0, r1) - float(blk.get("true_change_pct"))) > 1e-6:
        bad.append("true_change_pct does not follow from the truths")
    from haenv.world.labworld.perturb import no_effect_violation
    for r in recs.values():
        for p in r.get("perturbations", []):
            if p.get("role") == "decoy" and p.get("acts"):
                bad.append(f"decoy {p.get('factor')} marked acting")
            elif p.get("role") in ("decoy", "background") and not p.get("acts"):
                # a factor recorded as not acting must not act under the world's own model (a wait
                # past the stability bound, a meat meal inside its window): the record would then
                # print a value that the perturbation model no longer reproduces
                why = no_effect_violation(p, blk["analyte"])
                if why:
                    bad.append(f"day {r['day']}: {p.get('role')} {p.get('factor')} does not rest: {why}")
    return bad
