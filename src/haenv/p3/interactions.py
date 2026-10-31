"""Preregistered P3 intersections and surgical, two-factor gold controls.

These interventions change decision records, never rendered text or a stored
answer. They hold other premises fixed; they are local rule counterfactuals,
not a claim that a patient's entire physical trajectory stays unchanged.
"""
from __future__ import annotations

from copy import deepcopy


def family(drug, cls, subtype):
    from haenv.world.labworld.meds import drug as registered_drug
    if cls in ("switch", "uptitrate") and registered_drug(drug).get("off_also_accepts"):
        return None
    if cls == "check_adherence_or_adverse_effect":
        return {"adh_low": "control_adherence", "absorption": "control_administration",
                "unattributed": "symptom_timing"}.get(subtype)
    if cls == "switch":
        return "control_limit"
    if cls == "uptitrate" and drug != "levothyroxine":
        return "control_headroom"
    if cls == "downtitrate" and drug not in ("levothyroxine", "gliclazide"):
        return "symptom_timing"
    if cls == "maintain" and subtype == "on" and drug not in ("levothyroxine", "gliclazide"):
        return "course_control"
    return None


def quotas(n_items):
    """Nested quotas fixed before generation; exactly 40/100, 20/50.

    Split only the increment in each sourced drug/action/subtype cell, so
    neither the original cell mix nor the 50-item prefix can change.
    """
    from haenv.p3.plan import planned_scenarios
    from haenv.p3.policy import specification
    from haenv.world.validation.pack_size import apportion, nest_chain
    share = specification()["interactions"]["published_share"]
    previous, certified = {}, {}
    for n in nest_chain(n_items):
        cells = planned_scenarios(n)
        capacity = {cell: count - previous.get(cell, 0) for cell, count in cells.items()
                    if family(cell[0], cell[1], cell[3])}
        need = round(n * share) - sum(certified.values())
        if any(v < 0 for v in capacity.values()) or need > sum(capacity.values()):
            raise ValueError("non-monotone or insufficient P3 interaction quota")
        extra = apportion(need, capacity)
        for cell, count in extra.items():
            certified[cell] = certified.get(cell, 0) + count
        previous = cells
    return {(*cell, flag): count for cell, total in previous.items()
            for flag, count in ((True, certified.get(cell, 0)), (False, total - certified.get(cell, 0))) if count}


def _control(g, status):
    from haenv.world.labworld.meds import axis, band
    import math
    from haenv.world.labworld.meds import drug
    from haenv.p3.gold import known_of
    a = axis(g["axis"])
    upper = float(g.get("upper", a["upper"]))
    if status == "on":
        upper = min(upper, float(drug(g["drug"], known_of(g)).get("titrate_on_target_min") or upper))
    value = upper * math.exp((1.5 if status == "off" else -1.5) * band(g["axis"]))
    row = next(r for r in g["visits"] if int(r["day"]) == int(g["T"]))
    row.update(truth=value, printed=round(value, int(a["ndigits"])))


def _coverage(b):
    """Complete daily coverage, retaining the dose-change pickup dates."""
    start = min(int(d) for d, _ in b["dose_line"])
    end, supply = int(b["T"]), int(b.get("supply", 30))
    resets = sorted({start, *(int(d) for d, _ in b["dose_line"])})
    picks = []
    for i, day in enumerate(resets):
        stop = resets[i + 1] if i + 1 < len(resets) else end + 1
        picks += list(range(day, stop, supply))
    # Today's pill-count observation is not a new dispensing.
    if picks[-1] == end:
        picks.pop()
    b["refills"] = picks
    b["pill_count"] = supply - (end - picks[-1])


def _administration(b):
    b["administration_record"]["supplement_hour"] = b["administration_record"]["drug_hour"] + 5


def _dose_limit(b):
    from haenv.p3.gold import dose_at
    from haenv.p3.history import limiting_record
    x = dose_at(b["dose_line"], b["T"])
    b["adverse_event_record"] = [r for r in b.get("adverse_event_record", ())
                                 if not limiting_record([r], b["drug"], x)]


def _headroom(b, *, to_top=None):
    from haenv.p3.gold import dose_at, known_of
    from haenv.world.labworld.meds import drug
    ladder = drug(b["drug"], known_of(b))["ladder"]
    x = float(dose_at(b["dose_line"], b["T"]))
    if to_top is None:
        to_top = x != float(ladder[-1])
    b["dose_line"][-1][1] = ladder[-1] if to_top else ladder[-2]


def _symptoms(b):
    return [e for e in b["diary"] if e.get("kind") == "drug_symptom" and e.get("cause") != "predating"]


def _symptom_course(b, status=None):
    from haenv.i18n.text import L, is_text
    for e in _symptoms(b):
        key = status or ("p3.status.ongoing" if is_text(e.get("status"), "p3.status.resolved")
                         else "p3.status.resolved")
        e["status"] = L(key)


def _symptom_onset(b):
    start = min(int(d) for d, _ in b["dose_line"])
    for e in _symptoms(b):
        e.update(onset=start - 30, cause="predating")


def ablations(g):
    """Two named factors, with one inactive/alternative record per factor."""
    from haenv.p3.gold import dose_at
    name = g.get("interaction_family")
    if not name:
        return {}
    a, b = deepcopy(g), deepcopy(g)
    if name in ("control_adherence", "control_administration", "control_limit", "control_headroom"):
        _control(a, "on")
        if name == "control_adherence":
            _coverage(b)
            second = "coverage"
        elif name == "control_administration":
            _administration(b)
            second = "supplement_separation"
        elif name == "control_limit":
            from haenv.p3.gold import historical_limit
            x = dose_at(b["dose_line"], b["T"])
            if historical_limit(b, x):
                _dose_limit(b)
                second = "prior_tolerated_dose"
            else:
                _headroom(b, to_top=False)
                second = "current_dose"
        else:
            _headroom(b, to_top=True)
            second = "current_dose"
        return {"control": a, second: b}
    if name == "course_control":
        _control(a, "off")
        _symptom_course(b, "p3.status.ongoing")
        return {"control": a, "symptom_course": b}
    if name == "symptom_timing":
        # Retain the symptom and its onset. Removing the whole record would
        # remove both factors at once and would not be an independent control.
        _symptom_course(a, "p3.status.resolved")
        _symptom_onset(b)
        return {"symptom_course": a, "onset_relative_to_exposure": b}
    raise ValueError(f"unknown interaction family {name}")


def receipt(g):
    from haenv.p3.gold import g3, accepted
    base = g3(g)
    variants = ablations(g)
    controls = {}
    for factor, variant in variants.items():
        changed = g3(variant)
        original_set = set(accepted(g, base))
        changed_set = set(accepted(variant, changed)) if changed["class"] is not None else set()
        controls[factor] = {"class": changed["class"], "reason": changed["reason"],
                            "accepted": sorted(changed_set),
                            "pass": changed["class"] is not None and not original_set.intersection(changed_set)}
    return {"family": g.get("interaction_family"), "class": base["class"], "factors": controls,
            "pass": len(controls) == 2 and all(v["pass"] for v in controls.values())}
