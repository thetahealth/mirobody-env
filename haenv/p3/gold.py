"""G3 -- the medication-adjustment item's gold, recomputed from the world records (PREREG 2).

Inputs are only the records stored on the gold block: the dose line, the refill pickups and the
two adherence levels around them, the control-axis visit records (4-decimal truths), the diary
entries with their recorded cause, the patient's known comorbidity (A24: with CKD and
albuminuria the ACEI is raised to its top dose even on target), and the drug's tables. `g3(blk)`
returns the decision, the
set of active states and, when the item sits in a gray band, the reason (decision `None`: not
emitted). Rules are checked as a partition, so the decision does not depend on rule order.

This module is in both pack-3 segments: the generator accepts an item through it, the emission
gate and the audit re-derive the gold with it, the judge reads its output.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import math

CLASSES = ("uptitrate", "downtitrate", "maintain", "switch", "check_adherence_or_adverse_effect")
#: active states of each subtype (single activation, PREREG 2)
STATES = {"off": {"off"}, "top_off": {"off"}, "ae_dose": {"ae_dose"},
          "over": {"over"}, "early": {"off", "early"}, "on": set(), "adh_low": {"off", "adh_low"},
          "unattributed": {"ae_unattributed"}}


def expected_states(sub: str):
    """The subtype's active states (PREREG 2)."""
    s = STATES.get(sub)
    return None if s is None else set(s)


def dose_at(line, day: int) -> float:
    return [v for d, v in sorted(line, key=lambda x: int(x[0])) if int(d) <= int(day)][-1]


def last_increase(line, day: int) -> int:
    """Day of the last increase on or before `day`; the start day when there was none."""
    pts = sorted(((int(d), float(v)) for d, v in line), key=lambda x: x[0])
    s = pts[0][0]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d1 <= day and v1 > v0:
            s = d1
    return s


def last_change(line, day: int) -> int:
    pts = sorted(((int(d), float(v)) for d, v in line), key=lambda x: x[0])
    s = pts[0][0]
    for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
        if d1 <= day and v1 != v0:
            s = d1
    return s


def adherence_of(blk: dict) -> float:
    from ..labworld.meds import adherence_T, coverage_segments, rule
    seg = coverage_segments(list(blk["refills"]), float(blk["a_before"]), float(blk["a_last"]), int(blk.get("supply", 30)))
    return adherence_T(seg, int(blk["T"]), int(rule()["adherence_window_days"]))


def control_status(axis_name: str, y: float, upper: float | None = None) -> tuple[str | None, str]:
    """off / on / over, or None with the gray reason (every threshold at >= 1 RCV, log scale).
    `upper`: this patient's target (A12); the axis default when None."""
    from ..labworld.meds import axis, rcv, rule
    a = axis(axis_name)
    R = rcv(axis_name) * float(rule()["control_gray_rcv"])
    ly = math.log(float(y))
    up = math.log(float(a["upper"] if upper is None else upper))
    if a.get("over_below") is not None and ly <= math.log(float(a["over_below"])) - R:
        return "over", ""
    if ly >= up + R:
        return "off", ""
    if ly <= up - R and (a.get("lower") is None or ly >= math.log(float(a["lower"])) + R):
        return "on", ""
    return None, f"{axis_name} truth {y} within 1 RCV of a threshold"


def ae_cause(entry: dict, line, start: int, attribution: str) -> str | None:
    """What the visible record says about a diary entry of the drug's registered adverse effect:
    predating (onset before the drug), dose_related, unattributed, or None (gray). `exposure`
    drugs: any onset on the drug is dose-related; `timing` drugs: by the onset windows (A5)."""
    from ..labworld.meds import rule
    rl = rule()
    on = int(entry["onset"])
    if abs(on - start) < int(rl["start_gray_days"]):
        return None                     # onset within the gray band around the start (A15)
    if on < start:
        return "predating"
    if attribution == "exposure":
        return "dose_related"
    anchor = last_increase(line, on)
    lo, hi = rl["ae_dose_window"]
    if lo <= on - anchor <= hi:
        return "dose_related"
    if on - last_change(line, on) >= int(rl["ae_unattributed_min"]) and last_change(line, 10 ** 6) <= on:
        return "unattributed"
    return None


def known_of(blk: dict) -> tuple[str, ...]:
    """The patient's known comorbidity on the record (A24): it can change the drug's next step."""
    c = blk.get("comorbidity")
    return (str(c["code"]),) if isinstance(c, dict) and c.get("code") else ()


def g3(blk: dict) -> dict:
    from ..labworld.meds import drug, rule
    rl = rule()
    d = drug(blk["drug"], known_of(blk))
    L = [float(x) for x in d["ladder"]]
    T = int(blk["T"])
    line = blk["dose_line"]
    out = {"class": None, "states": [], "reason": ""}
    x = float(dose_at(line, T))
    s = last_increase(line, T)
    tau = float(d["tau_rule_days"])
    since = T - s
    a = adherence_of(blk)
    g_lo, g_hi = rl["adherence_gray"]
    if g_lo <= a < g_hi:
        out["reason"] = f"a(T) = {a:.3f} in the gray band"
        return out
    rec_T = next(r for r in blk["visits"] if int(r["day"]) == T)
    upper = blk.get("upper")
    status, why = control_status(d["axis"], rec_T["truth"], upper)
    if status is None:
        out["reason"] = why
        return out
    pst, _ = control_status(d["axis"], rec_T["printed"], upper)
    if pst != status:
        out["reason"] = f"printed {rec_T['printed']} on the other side of the target from the truth"
        return out
    early = False
    if status == "off":                 # "just stepped up" is read only off target (G3, A5)
        if since <= float(rl["early_max_frac"]) * tau:
            early = True
        elif since < tau:
            out["reason"] = f"T - s = {since} in the gray band ({rl['early_max_frac']} tau, tau)"
            return out
    ae = None
    start = int(min(int(dd) for dd, _ in line))
    for e in blk["diary"]:
        if e.get("kind") != "drug_symptom":
            continue
        seen = ae_cause(e, line, start, d["attribution"])
        if seen is None:
            out["reason"] = f"symptom onset {e['onset']} in the gray band of its timing"
            return out
        if seen != e["cause"]:
            out["reason"] = f"recorded cause {e['cause']} != timing {seen}"
            return out
        if seen in ("dose_related", "unattributed"):
            ae = seen
    over = status == "over"
    off = status == "off"
    adh_low = a < float(rl["adherence_line"])
    states = {k for k, v in (("off", off), ("over", over), ("early", early), ("adh_low", adh_low),
                             ("ae_dose", ae == "dose_related"), ("ae_unattributed", ae == "unattributed")) if v}
    out["states"] = sorted(states)
    at_min, at_max = x == L[0], x == L[-1]
    if (off and adh_low) or ae == "unattributed":
        c = "check_adherence_or_adverse_effect"
    elif ae == "dose_related" or over:
        if at_min:                      # no rung below (A12: not emitted)
            out["reason"] = "dose-related adverse effect or overtreatment on the lowest rung"
            return out
        c = "downtitrate"
    elif off and early:
        c = "maintain"
    elif off and at_max:
        c = "switch"
    elif off:
        # the next step the drug's guideline gives when the target is missed below the top rung (A15)
        c = "uptitrate" if d["when_off"] == "titrate" else "switch"
    elif d.get("on_target") == "titrate" and not at_max:
        c = "uptitrate"                 # A24: CKD with albuminuria -- the ACEI goes to its top dose on target too
    else:
        c = "maintain"
    if c == "switch" and not d.get("switch", True):
        out["reason"] = f"{blk['drug']} has no drug to switch to (not emitted)"
        return out
    out["class"] = c
    out.update(dose=x, at_min=at_min, at_max=at_max, since_increase=since, a_T=round(a, 4),
               control=status, ae=ae, tau=tau)
    return out


def gold_block_consistent(blk: dict) -> list[str]:
    """Re-derive G3 from the records of an `adjudication.meds_p3` block; the disagreements."""
    from ..labworld.meds import axis
    from ..labworld.observe import record_consistent
    bad: list[str] = []
    a = axis(blk["axis"])
    for r in blk.get("visits") or ():
        why = record_consistent(r, int(a["ndigits"]), tuple(a["range"]))
        if why:
            bad.append(why)
    g = g3(blk)
    if g["class"] != blk.get("class"):
        bad.append(f"G3 recomputed {g['class']} ({g['reason']}) != gold {blk.get('class')}")
    want = expected_states(str(blk.get("subtype")))
    if want is not None and set(g["states"]) != want:
        bad.append(f"active states {g['states']} != subtype {blk.get('subtype')} {sorted(want)}")
    return bad
