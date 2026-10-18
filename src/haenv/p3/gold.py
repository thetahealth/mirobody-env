"""G3 -- the medication-adjustment item's gold, recomputed from the world records (PREREG 2).

Inputs are only the records stored on the gold block: the dose line, the printed refill pickups and
today's pill count (adherence is read from them by the clinic's printed rule), the control-axis visit records (4-decimal truths), the diary
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


def switch_days(dose_line) -> set[int]:
    """The days of a dose change: the new strength is dispensed at the visit (the old stock set aside)."""
    pts = sorted(int(d) for d, _ in dose_line)
    return set(pts[1:])


def adherence_of(blk: dict) -> float:
    """a(T): the coverage the clinic's printed rule gives on the printed refill record and today's pill
    count (by day, `printed_coverage`). The world's adherence levels make the record; the gold reads the
    record."""
    return printed_readings(blk)[1]


def printed_readings(blk: dict) -> tuple[float, float]:
    start = min(int(d) for d, _ in blk["dose_line"])
    return printed_coverage(blk["refills"], int(blk.get("supply", 30)), int(blk["pill_count"]), int(blk["T"]), start,
                            switch_days(blk["dose_line"]))


def printed_coverage(pickups: list[int], supply: int, left: int, T: int, start: int,
                     reset=()) -> tuple[float, float]:
    """The adherence a reader computes from the printed record alone over the last
    `adherence_window_days` (from the start when the drug started inside them): (per interval, the
    share of days covered with an early refill's surplus carried forward; by day, a covered day while
    stock remains). Both take the days since the last pickup from today's pill count and count a day
    on the drug before the first printed pickup as uncovered. `reset`: the new-strength pickups, where
    the old strength's stock is set aside."""
    from haenv.world.labworld.meds import carry_in
    from haenv.world.labworld.meds import rule
    pick = [int(x) for x in pickups]
    lo = max(int(T) - int(rule()["adherence_window_days"]) + 1, int(start))
    n = int(T) + 1 - lo
    reset = {int(x) for x in reset}
    cin = carry_in(pick, supply, reset)
    taken = float(supply) + cin[-1] - float(left)
    a_last = min(1.0, max(0.0, taken / max(1, int(T) - pick[-1])))
    seg = 0.0
    for d in range(lo, int(T) + 1):
        if d < pick[0]:
            continue
        if d >= pick[-1]:
            seg += a_last
            continue
        k = max(i for i, x in enumerate(pick) if x <= d)
        seg += min(1.0, (float(supply) + cin[k]) / float(pick[k + 1] - pick[k]))
    stock, day = 0, 0
    for d in range(min(pick[0], lo), pick[-1]):
        if d in pick:
            stock = int(supply) + (0 if d in reset else stock)
        if stock > 0:
            stock -= 1
            day += d >= lo
    day += max(0.0, min(float(int(T) - pick[-1]), taken))
    return seg / n, day / n


def printed_adherence_why(blk: dict, a: float) -> str:
    """Why the printed record does not settle the side of the adherence line (empty when it does): a
    reading of it (per interval or by day) on the other side of the line from a(T), or inside the gray band."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    line, (lo, hi) = float(rl["adherence_line"]), rl["adherence_gray"]
    for v in printed_readings(blk):
        if (v < line) != (a < line) or lo <= v < hi:
            return f"printed coverage {v:.3f} does not give a(T) = {a:.3f}'s side of {line}"
    return ""


def control_status(axis_name: str, y: float, upper: float | None = None) -> tuple[str | None, str]:
    """off / on / over, or None with the gray reason (every threshold at one reading's reference
    change, 1.96 x CV on the log scale: `meds.band`).
    `upper`: this patient's target (A12); the axis default when None."""
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.meds import band
    a = axis(axis_name)
    R = band(axis_name)
    ly = math.log(float(y))
    up = math.log(float(a["upper"] if upper is None else upper))
    if a.get("over_below") is not None and ly <= math.log(float(a["over_below"])) - R:
        return "over", ""
    if ly >= up + R:
        return "off", ""
    if ly <= up - R and (a.get("lower") is None or ly >= math.log(float(a["lower"])) + R):
        return "on", ""
    return None, f"{axis_name} truth {y} within one reading band of a threshold"


def floor_side(axis_name: str, truth: float, printed: float, floor: float) -> tuple[str | None, str]:
    """above / below a clinic floor (the CKD SBP floor, A24), or None with the gray reason: the truth
    within one reading's reference change (`meds.band`, 1.96 x CV on the log scale) of the floor, or
    the printed reading on the other side of it from the truth (p3-fixes c: the floor is read as the
    target is, `control_status`)."""
    from haenv.world.labworld.meds import band
    R = band(axis_name)
    lt = math.log(float(truth)) - math.log(float(floor))
    if abs(lt) < R:
        return None, f"{axis_name} truth {truth} within one reading band of the floor {floor}"
    side = "above" if lt > 0 else "below"
    if ("above" if float(printed) >= float(floor) else "below") != side:
        return None, f"printed {printed} on the other side of the floor {floor} from the truth"
    return side, ""


def ae_cause(entry: dict, line, start: int, attribution: str) -> str | None:
    """What the visible record says about a diary entry of the drug's registered adverse effect:
    predating (onset before the drug), dose_related, unattributed, or None (gray). `exposure`
    drugs: any onset on the drug is dose-related; `timing` drugs: by the onset windows (A5)."""
    from haenv.world.labworld.meds import rule
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


def natural_status(onset: int, days: int, T: int) -> str:
    """A self-limited episode is ongoing while it lasts (`days` from its onset) and resolved afterwards."""
    return "持续至今" if int(onset) + int(days) >= int(T) else "已缓解"


def documented_contexts(d: dict) -> set[str]:
    """The non-drug settings a diary entry may name (the drug's own `predating_contexts` and the acute-illness
    settings); a drug-symptom entry that names one is not attributed to the drug (`render.attribution_rule`)."""
    from haenv.world.labworld.meds import meds
    names = {c if isinstance(c, str) else c["context"] for c in d.get("predating_contexts") or ()}
    return names | set(meds()["confounders"]["intercurrent"]["settings"])


def confounder_why(blk: dict, e: dict, d: dict) -> str:
    """Why a conspicuous background entry of the diary (`kind: confounder`) could act on the decision, or ""
    when it cannot. The entry is read from what the item shows: another medicine's common adverse effect
    (the medicine is on the visible list and no word of the managed drug's `decision_facts` is in it), an acute
    illness, or the drug's own adverse-effect wording with a documented non-drug setting (statin background
    symptom); the last two carry a status that follows the episode's registered duration. Any other entry is
    refused; a refused entry makes the item unrealizable (the realizer draws another attempt)."""
    from haenv.world.labworld.meds import meds
    from haenv.world.labworld.meds import rule
    reg = meds()["confounders"]
    sym, cause, T = str(e.get("symptom")), e.get("cause"), int(blk["T"])
    start = min(int(x) for x, _ in blk["dose_line"])
    if cause == "other_drug":
        src = e.get("source_drug")
        if sym in d["ae_symptoms"] or src not in (blk.get("other_drugs") or ()) \
                or sym not in ((reg["other_drug"].get(src) or {}).get("symptoms") or ()):
            return f"other-drug entry {sym!r}: {src} is not a visible medicine with that registered effect"
        hit = [w for w in d.get("decision_facts") or () if w in sym]
        return f"other-drug entry {sym!r} carries the managed drug's adverse-effect word {hit}" if hit else ""
    if cause == "intercurrent":
        ill = reg["intercurrent"]
        days = next((int(x["days"]) for x in ill["illnesses"] if x["symptom"] == sym), None)
        if days is None or e.get("context") not in ill["settings"]:
            return f"illness entry {sym!r} / {e.get('context')!r} is not a registered illness with a documented setting"
        want = natural_status(e["onset"], days, T)
    elif cause == "background_symptom":
        spec = reg["background_symptom"].get(blk["drug"])
        if not spec or sym not in d["ae_symptoms"] or e.get("context") not in documented_contexts(d):
            return f"background symptom {sym!r} / {e.get('context')!r} has no registered non-drug setting for {blk['drug']}"
        if int(e["onset"]) - start < int(rule()["start_gray_days"]):
            return "background symptom starts inside the start gray band"
        want = natural_status(e["onset"], spec["days"], T)
    else:
        return f"unknown background cause {cause!r}"
    return "" if e.get("status") == want else f"{cause} entry {sym!r} is {e.get('status')}, its course says {want}"


def edge_margin_why(blk: dict) -> str:
    """Why an item sits on a rule line the clinic itself chose (empty when it does not): a patient-reported
    onset within `rule.edge_margin_onset_days` of 3 / 28 / 56 days after the last increase or of the start
    gray band. Such lines carry no clinical boundary, so an item is not placed on them. (The interval lines
    0.5 tau and tau are recorded dates a visit schedule can sit on; those items carry an accepted set, `accepted`.)"""
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
    rl = rule()
    mo, md = int(rl["edge_margin_onset_days"]), int(rl["edge_margin_date_days"])
    d = drug(blk["drug"], known_of(blk))
    line = blk["dose_line"]
    start = int(min(int(x) for x, _ in line))
    for e in blk["diary"]:
        if e.get("kind") != "drug_symptom":
            continue
        on = int(e["onset"])
        if abs(on - start) < int(rl["start_gray_days"]) + mo:
            return f"symptom onset {on} within {mo} days of the start gray band"
        if d["attribution"] == "timing" and on > start:
            lo, hi = rl["ae_dose_window"]
            for edge, delta in ((lo, on - last_increase(line, on)), (hi, on - last_increase(line, on)),
                                (int(rl["ae_unattributed_min"]), on - last_change(line, on))):
                if abs(delta - int(edge)) < mo:
                    return f"symptom onset {delta} days after the dose change within {mo} days of the {edge}-day line"
    return ""


def accepted(blk: dict, g: dict | None = None) -> list[str]:
    """The decisions the item accepts: its gold, and, when an off-target `T - s` lies within
    `rule.edge_margin_date_days` of 0.5 tau or tau (the clinic's own interval lines, no clinical boundary),
    the decision of the neighbouring reading of that line ("effect not yet arrived" against "arrived"). Both
    are scored correct; an item away from the lines accepts its gold alone."""
    from haenv.world.labworld.meds import rule
    rl = rule()
    g = g or g3(blk)
    out = {g["class"]}
    if g.get("class") and g.get("control") == "off":
        since, tau = int(g["since_increase"]), float(g["tau"])
        half = float(rl["early_max_frac"]) * tau
        if min(abs(since - half), abs(since - tau)) < int(rl["edge_margin_date_days"]):
            alt = g3(blk, early_as="early" not in g["states"])["class"]
            if alt:
                out.add(alt)
    return sorted(out)


def known_of(blk: dict) -> tuple[str, ...]:
    """The patient's known comorbidity on the record (A24): it can change the drug's next step."""
    c = blk.get("comorbidity")
    return (str(c["code"]),) if isinstance(c, dict) and c.get("code") else ()


def g3(blk: dict, early_as: bool | None = None) -> dict:
    from haenv.world.labworld.meds import drug
    from haenv.world.labworld.meds import rule
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
    why = printed_adherence_why(blk, a)
    if why:
        out["reason"] = why
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
    if status == "off" and early_as is not None:
        early = early_as                # `accepted`: the neighbouring reading of the clinic's own interval line
    elif status == "off":               # "just stepped up" is read only off target (G3, A5)
        if since <= float(rl["early_max_frac"]) * tau:
            early = True
        elif since < tau:
            out["reason"] = f"T - s = {since} in the gray band ({rl['early_max_frac']} tau, tau)"
            return out
    ae = None
    start = int(min(int(dd) for dd, _ in line))
    for e in blk["diary"]:
        if e.get("kind") == "confounder":
            why = confounder_why(blk, e, d)
            if why:
                out["reason"] = why
                return out
            continue
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
    elif ae == "dose_related" and not over and d.get("ae_dose_decision") is False:
        out["reason"] = "a dose-related symptom with the control axis not suppressed: dosed by the axis, not emitted"
        return out
    elif ae == "dose_related" or over:
        if at_min:                      # no rung below (A12: not emitted)
            out["reason"] = "dose-related adverse effect or overtreatment on the lowest rung"
            return out
        c = "downtitrate"
    elif off and early:
        c = "maintain"
    elif off and historical_limit(blk, x):
        c = "switch"
    elif off and at_max:
        c = "switch"
    elif off:
        # the next step the drug's guideline gives when the target is missed below the top rung (A15)
        c = "uptitrate" if d["when_off"] == "titrate" else "switch"
    elif d.get("on_target") == "titrate" and not at_max:
        # CKD with albuminuria: the ACEI goes to its top dose on target too, unless today's reading is
        # under the registered floor; a reading within one band of the floor is not emitted (p3-fixes c)
        fl = d.get("titrate_on_target_min")
        side = "above"
        if fl:
            side, why = floor_side(d["axis"], rec_T["truth"], rec_T["printed"], float(fl))
            if side is None:
                out["reason"] = why
                return out
        c = "uptitrate" if side == "above" else "maintain"
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
    from haenv.world.labworld.meds import axis
    from haenv.world.labworld.observe import record_consistent
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
    if int((blk.get("plan") or {}).get("version") or 0) >= 16:
        why = edge_margin_why(blk)
        if why:
            bad.append(why)
        if blk.get("accept") != accepted(blk, g):
            bad.append(f"accepted set {blk.get('accept')} != recomputed {accepted(blk, g)}")
    return bad


def historical_limit(blk, dose):
    from haenv.p3.history import limiting_record
    return limiting_record(blk.get("adverse_event_record"), blk["drug"], dose)
