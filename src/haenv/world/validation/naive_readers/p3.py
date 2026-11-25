"""Medication readers using the production reader with changed inputs.

SYNTHETIC data, evaluation only, not medical advice.
"""
from unittest.mock import patch

from haenv.p3 import features as F
from haenv.p3.history import acei_contraindicated, limiting_record
from haenv.world.labworld.meds import rule
from .core import Reader
from .decisive import FactOutcome

NAME = "p3"
FACTS = {
    "adherence": ("adherence", None), "early": ("visits", True),
    "ae": ("diary", True), "history": ("archive", True),
    "position": ("prescription", False), "comorbidity": ("profile", False),
    "absorption": ("administration", False),
}
FACT_NOTE = "Control status itself is not counted. M3 counts the collection protocol only when replacing its extraction by the latest raw sample changes the decision."


def with_inputs(sp, f2, blk2):
    with patch.object(F, "inputs", lambda _sp: f2), patch.object(F, "_blk", lambda _sp: blk2):
        return F.rule_reader(sp)


def latest_dispense_pillbox(sp):
    f = F.inputs(sp)
    return with_inputs(sp, {**f, "coverage90": f["pill_adherence"]}, F._blk(sp))


def latest_prescription_row(sp):
    f, b = F.inputs(sp), F._blk(sp)
    dose = b.get("latest_prescription_dose", f["dose"])
    position = "min" if dose == f["ladder"][0] else "top" if dose == f["ladder"][-1] else "mid"
    return with_inputs(sp, {**f, "dose": dose, "position": position}, b)


def today_only(sp):
    f = F.inputs(sp)
    f = {**f, "coverage90": f["pill_adherence"]}
    b = F._blk(sp)
    from haenv.world.labworld.meds import drug
    if b.get("control_measurements") and drug(f["drug"])["axis"] == "home_sbp":
        from statistics import mean
        y = mean(float(r["value"]) for r in b["control_measurements"] if int(r["day"]) == f["T"])
        f = {**f, "reading": y, "status": "off" if y > f["upper"] else "on"}
    if f["symptom_onset"] is not None and f["T"] - f["symptom_onset"] > rule()["adherence_window_days"]:
        f["drug_symptom"] = 0
    return with_inputs(sp, f, {**F._blk(sp), "adverse_event_record": []})


def no_history(sp):
    return with_inputs(sp, F.inputs(sp), {**F._blk(sp), "adverse_event_record": []})


def latest_control_measurement(sp):
    f = F.inputs(sp)
    return with_inputs(sp, {**f, "status": f["printed_status"], "reading": f["printed_reading"],
                            "measurement_pending": False}, F._blk(sp))


def printed_value_only(sp):
    f = F.inputs(sp)
    return {"off": "switch" if f["position"] == "top" else "uptitrate",
            "on": "maintain", "over": "downtitrate"}[f["printed_status"]]


def keyword_only(sp):
    return "downtitrate" if F.inputs(sp)["drug_symptom_any"] else printed_value_only(sp)


def _archive_event(rows, drug, dose):
    """`history_any_event`: any archived episode of the managed drug that shows an abnormal result or
    a symptom (raised potassium or creatinine, glucose under 4, CK over 10 x ULN, an oedema course)
    counts as a dose limit at the current dose; criteria, causes and doses are not read."""
    from haenv.p3.policy import identify_drug
    for r in rows or ():
        if r.get("drug_id") != drug and identify_drug(r.get("drug")) != drug:
            continue
        labs = list(r.get("labs") or ())
        base = labs[0] if labs else {}
        for lab in labs[1:2]:
            k, c = lab.get("potassium_mmol_L"), lab.get("creatinine_umol_L")
            if k is not None and float(k) >= 5.5:
                return True
            if c is not None and base.get("creatinine_umol_L") and float(c) >= 1.3 * float(base["creatinine_umol_L"]):
                return True
            if lab.get("glucose_mmol_L") is not None and float(lab["glucose_mmol_L"]) < 4.0:
                return True
            if lab.get("ck_uln") is not None and float(lab["ck_uln"]) > 10:
                return True
        if r.get("symptom") and drug == "amlodipine":
            return True
        if r.get("ck_uln") is not None and float(r["ck_uln"]) > 10:
            return True
    return False


def history_any_event(sp):
    import haenv.p3.history as H
    any_event = lambda rows, drug, dose: (H.acei_contraindicated(rows, drug) or _archive_event(rows, drug, dose))
    with patch.object(H, "limiting_record", any_event):
        return F.rule_reader(sp)


def causality_withdrawal_only(sp):
    f = F.inputs(sp)
    ev = f.get("causal_evidence")
    if ev is not None and "withdrawal" in ev:
        # Improvement during the hold read alone: improved -> attributable; not improved -> not.
        f = {**f, "causal_evidence": {**ev, "alternative": not ev["withdrawal"]}}
    return with_inputs(sp, f, F._blk(sp))


def coverage_min_reading(sp):
    f = F.inputs(sp)
    return with_inputs(sp, {**f, "coverage90": min(f["coverage90"], f["pill_adherence"])}, F._blk(sp))


SIGNAL_CLASS = {"hist": "switch", "ae": "downtitrate", "over": "downtitrate",
                "unatt": "check_adherence_or_adverse_effect", "low": "check_adherence_or_adverse_effect",
                "absorp": "check_adherence_or_adverse_effect", "early": "maintain"}
SALIENT_ORDER = ("hist", "ae", "over", "unatt", "low", "absorp", "early")


def salient_signals(sp) -> dict:
    """The class-pointing signals a reader notices on the visible record, each read by its surface
    criterion (an archived episode that looks like a limit, an improvement note or an ongoing drug
    symptom, coverage under the line on either reading, a review inside half the interval)."""
    f, b = F.inputs(sp), F._blk(sp)
    from haenv.p3.causality import assessment
    rl = rule()
    off = f["status"] == "off"
    line = float(rl["adherence_line"])
    ev = f.get("causal_evidence")
    ae = unatt = False
    if f["drug_symptom"] and not (f["start"] is not None and f["symptom_onset"] < f["start"]):
        if ev is not None:
            ae = ev.get("withdrawal") is True
            unatt = not ae and assessment(f["symptom_onset"], f["start"], ev.get("withdrawal"),
                                          ev.get("alternative")) == "unattributed"
        else:
            ae = True
    return {"hist": off and _archive_event(b.get("adverse_event_record"), f["drug"], f["dose"]),
            "ae": ae, "over": f["status"] == "over", "unatt": unatt,
            "low": off and min(f["coverage90"], f["pill_adherence"]) < line,
            "absorp": off and bool(f["absorption_interaction"]),
            "early": off and f["since_increase"] <= float(rl["early_max_frac"]) * f["tau"]}


def salient_order(sp, order=SALIENT_ORDER):
    sig = salient_signals(sp)
    for k in order:
        if sig[k]:
            return SIGNAL_CLASS[k]
    return F.rule_reader(sp)


def salient_first(sp):
    return salient_order(sp)


READERS = (
    Reader("full", "full", F.rule_reader, "Production rule reader on all visible evidence."),
    Reader("printed_rule", "full", F.printed_rule_reader, "Production reader of printed clinic rules."),
    Reader("latest_dispense_pillbox", "naive", latest_dispense_pillbox, "Latest pickup and pill count replace longitudinal coverage."),
    Reader("latest_prescription_row", "naive", latest_prescription_row, "Use the latest prescription dose without cross-source reconciliation; retain every other input."),
    Reader("today_only", "naive", today_only, "Latest BP day, no archived history, symptoms inside the protocol's adherence window, latest pickup only."),
    Reader("no_history", "naive", no_history, "All visible inputs except archived adverse-event records."),
    Reader("latest_control_measurement", "naive", latest_control_measurement, "Latest individual control sample replaces protocol extraction; all other protocol rules retained."),
    Reader("history_any_event", "naive", history_any_event, "Every rule, but any archived abnormal result or symptom episode of the drug is read as a dose limit."),
    Reader("causality_withdrawal_only", "naive", causality_withdrawal_only, "Every rule, but causality read from the hold response alone."),
    Reader("coverage_min_reading", "naive", coverage_min_reading, "Every rule, but coverage is the lower of the recent pill count and the 90-day record."),
    Reader("salient_first", "naive", salient_first, "First surface signal in the order archive event, adverse effect, overtreatment, unexplained symptom, low coverage, early review; else the full rules."),
    Reader("printed_value_only", "stub", printed_value_only, "Today's status and dose position; no protocol."),
    Reader("anomaly", "stub", F.anomaly_stub, "Production adherence and symptom anomaly stub; no protocol."),
    Reader("keyword_only", "stub", keyword_only, "Diary keyword presence and today's status; no protocol."),
)
#: The single premise each naive reader reads wrong, and the gold classes that premise can move.
#: `floor` = 1 - (share of published core items whose gold the premise can flip): a reader that reads
#: its premise wrong is correct on every item the premise cannot move, so that share is its floor.
#: The derivation and the measured values are in the world-1.5 difficulty design note.
def adherence_band_changes_decision(sp, inputs) -> bool:
    """Hold every other clinical input fixed and compare the two adherence bands."""
    block = F._blk(sp)
    low = with_inputs(sp, {**inputs, "coverage90": 0.0}, block)
    high = with_inputs(sp, {**inputs, "coverage90": 1.0}, block)
    return low != high


def flip_capable(sp, g, name) -> bool:
    """Whether this reader's one wrong premise can change this item's gold class."""
    from haenv.p3 import features as F
    from haenv.p3.gold import historical_limit
    from haenv.world.labworld.meds import rule
    line = float(rule()["adherence_line"])
    f = F.inputs(sp)
    cls = g.get("class")
    if name == "coverage_min_reading":
        return f["pill_adherence"] < line <= f["coverage90"] and adherence_band_changes_decision(sp, f)
    if name == "causality_withdrawal_only":
        # `alternative = not withdrawal` only changes the assessed cause on a check-class item whose
        # hold improved (or whose assessment of other causes is missing).
        a = g.get("ae_causality") or {}
        return (cls == "check_adherence_or_adverse_effect" and a.get("withdrawal") is True
                and a.get("alternative") in (True, None))
    if name == "history_any_event":
        # any archived episode read as a limit: errs only where the limit branch is reached and the
        # archive does not in fact limit today's dose.
        if f["status"] != "off" or f["coverage90"] < line or f["absorption_interaction"] or f["drug_symptom"]:
            return False
        if f.get("since_increase") is not None and f["since_increase"] <= float(rule()["early_max_frac"]) * f["tau"]:
            return False
        return bool(g.get("adverse_event_record")) and not historical_limit(g, g["g3"]["dose"])
    if name == "salient_first":
        # the first surface signal wins: errs only where a salient signal disagrees with the gold,
        # i.e. an item that carries a conflict/near-miss premise the printed priority order overrides.
        return bool(g.get("signal_conflict") or g.get("history_near_miss"))
    if name == "latest_dispense_pillbox":
        crosses_line = (f["pill_adherence"] < line) != (f["coverage90"] < line)
        return crosses_line and adherence_band_changes_decision(sp, f)
    if name == "latest_prescription_row":
        # the latest prescription row replaces cross-source reconciliation: errs only where the superseded
        # row states a different dose AND that dose moves the decision (the reconciled dose sits at a
        # different rung of the ladder, so the position the item is read from changes).
        from haenv.world.labworld.meds import drug as registered
        dose = g.get("dose_reconciliation") or {}
        if not dose.get("discrepancy"):
            return False
        ladder = [float(x) for x in registered(g["drug"], ())["ladder"]]
        truth, wrong = float(dose.get("truth_dose") or 0), float(dose.get("prescription_dose") or 0)
        pos = lambda v: 0 if v == ladder[0] else (len(ladder) - 1 if v == ladder[-1] else 1)
        return truth in ladder and wrong in ladder and pos(truth) != pos(wrong)
    if name == "no_history":
        # every archived record ignored: errs only where an archived record decides the item.
        return bool(g.get("adverse_event_record")) and historical_limit(g, g["g3"]["dose"])
    return False


#: Readers judged against a structural floor (see `flip_capable`).
FLOOR_READERS = ("coverage_min_reading", "causality_withdrawal_only", "history_any_event", "salient_first",
                 "latest_dispense_pillbox", "latest_prescription_row", "no_history")


def floors(rows) -> dict:
    """{reader: {floor, capable}} over one stratum: 1 - the share of items its wrong premise can move."""
    n = len(rows)
    out = {}
    for name in FLOOR_READERS:
        if not all("flip" in r for r in rows):
            return {}
        capable = sum(bool(r["flip"].get(name)) for r in rows)
        out[name] = {"capable": capable, "floor": round(1 - capable / n, 4) if n else None}
    out["_n"] = n
    return out


NOT_APPLICABLE = {
    "summary_sentence_only": "same as latest_dispense_pillbox on medication items",
    "trust_system_score": "No machine score on medication items.",
    "checklist_only": "No checklist on medication items.",
    "latest_row_only": "Dose shortcut: latest_prescription_row; dispensing shortcut: latest_dispense_pillbox.",
}


def gold(case):
    return case["adjudication"]["meds_p3"]


def constant_answers(g):
    return set(g.get("accept") or [g["class"]])


def ok(answer, g):
    return answer in constant_answers(g)


def gold_label(g):
    return g["class"]


def answer_from_label(label):
    return label


def group_key(sp):
    return F.drug_key(F._blk(sp))


def features(sp):
    f, b = F.inputs(sp), F._blk(sp)
    naive = {f"status:{f['printed_status']}": 1, f"position:{f['position']}": 1,
             f"drug:{f['drug']}": 1, "drug_symptom_any": f["drug_symptom_any"],
             "pill_adherence": f["pill_adherence"]}
    noise = {k: v for k, v in F.surface(sp, 0).items() if k not in {"H", "drug"}}
    return naive, noise


def extra_features(sp):
    f, b = F.inputs(sp), F._blk(sp)
    clean = {**features(sp)[0], **{k: f[k] for k in ("since_increase", "coverage90", "drug_symptom")},
             "absorption": int(bool(f["absorption_interaction"])),
             "limiting": int(bool(limiting_record(b.get("adverse_event_record"), f["drug"], f["dose"]))),
             "acei_ci": int(bool(acei_contraindicated(b.get("adverse_event_record"), f["drug"]))),
             "CKD": int("CKD" in f["known"])}
    clean.pop(f"status:{f['printed_status']}", None)
    clean[f"status:{f['status']}"] = 1
    return {"clean": clean}


def knockout(sp, full_answer):
    f, b = F.inputs(sp), F._blk(sp)
    edits = {"adherence": {"coverage90": 1.0}, "early": {"since_increase": 10**6},
             "ae": {"drug_symptom": 0}, "position": {"position": "mid"},
             "comorbidity": {"known": ()}, "absorption": {"absorption_interaction": None}}
    line = float(rule()["adherence_line"])
    # Coverage lives outside the latest row only when the latest pickup and pill count give another band.
    adherence_outside = (f["pill_adherence"] < line) != (f["coverage90"] < line)
    result = []
    for fact, (group, outside) in FACTS.items():
        if fact == "position" and "latest_prescription_dose" in b:
            continue  # Dose reconciliation below is one fact, not two edits of dose/position.
        if fact == "ae" and f.get("causal_evidence") is not None:
            continue  # Do not double-count the whole symptom and its causal elements.
        outside = adherence_outside if fact == "adherence" else outside
        try:
            b2 = {**b, "adverse_event_record": []} if fact == "history" else b
            answer = with_inputs(sp, {**f, **edits.get(fact, {})}, b2)
            result.append(FactOutcome(fact, group, outside, "flip" if answer != full_answer else "same"))
        except Exception as exc:
            result.append(FactOutcome(fact, group, outside, "error", error=f"{type(exc).__name__}: {exc}"))
    if "latest_prescription_dose" in b:
        answer = latest_prescription_row(sp)
        result.append(FactOutcome("dose_records", "prescription", b["latest_prescription_dose"] != f["dose"],
                                  "flip" if answer != full_answer else "same"))
    if b.get("control_measurements"):
        try:
            answer = latest_control_measurement(sp)
            outside = len({r["day"] for r in b["control_measurements"]}) > 1
            result.append(FactOutcome("measurement_protocol", "measurements", outside,
                                      "flip" if answer != full_answer else "same"))
        except Exception as exc:
            result.append(FactOutcome("measurement_protocol", "measurements", None, "error", error=str(exc)))
    if f.get("causal_evidence") is not None:
        for fact in ("onset", "withdrawal", "alternative"):
            changed = dict(f["causal_evidence"])
            f2 = {**f, "causal_evidence": changed}
            if fact == "onset":
                f2["symptom_onset"] = f["start"] - 1
            else:
                changed.pop(fact, None)
            answer = with_inputs(sp, f2, b)
            result.append(FactOutcome("causal_" + fact, "diary" if fact == "onset" else "causal_note",
                                      fact != "alternative", "flip" if answer != full_answer else "same"))
    return result
