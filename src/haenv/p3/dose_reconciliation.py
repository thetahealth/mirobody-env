"""Render reconciled dose sources from an already established medication course.

SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.i18n.text import J, L, T, is_text


def project(item, sources):
    """Keep treatment truth fixed; reconcile an imported documentation discrepancy."""
    from haenv.world.truth import coupling_enabled
    from haenv.world.validation.difficulty_r2 import enabled
    if not item.get("dose_reconciliation") and (not coupling_enabled() or not enabled(item.get("plan") or item)):
        return
    from haenv.resources.regpath import load_registry
    from haenv.p3.plan import h01
    from haenv.p3.policy import identify_drug
    from haenv.world.labworld.meds import drug
    registry = load_registry("p3_dose_reconciliation.yaml")
    cfg = registry["generation"]
    contract = registry["record_contract"]
    key, today = item["drug"], int(item["T"])
    d = drug(key)
    dose = float(sorted(item["dose_line"])[-1][1])
    ladder = [float(v) for v in d["ladder"]]
    # Opposite headroom, using only this medicine's existing registered doses.
    other = ladder[-2] if dose == ladder[-1] else ladder[-1]
    # A copied lower-dose prescription must be reconciled before applying a
    # dose-specific historical limit. At that lower dose the old episode does
    # not establish today's tolerated ceiling. This is an authored source
    # discrepancy, drawn before any reader runs, not a treatment change.
    from haenv.p3.history import limiting_record
    if (cfg.get("historical_limit_lower_prescription")
            and limiting_record(item.get("adverse_event_record"), key, dose)
            and dose > ladder[0]):
        other = ladder[ladder.index(dose) - 1]
    cid = item.get("record_key") or item.get("course_documentation_key") or str(item["dose_line"])
    previous = item.get("dose_reconciliation")
    discrepancy = previous["discrepancy"] if previous else h01("m2-dose-discrepancy", cid) < cfg["discrepancy_share"]
    list_current = previous["current_record"] == "L" if previous else h01("m2-dose-authority", cid) < cfg["patient_list_current_share"]
    rx_dose = other if discrepancy and list_current else dose
    list_dose = other if discrepancy and not list_current else dose
    strength = float(registry["sources"]["bottle_strengths"][key]["strength"])
    current, superseded = ("L", "R") if list_current else ("R", "L")
    rx = max((r for r in sources["prescription_record"]
              if identify_drug(r.get("drug")) == key and int(r["day"]) < today), key=lambda r: int(r["day"]))
    dispensing = max((r for r in sources["dispensing_record"]
                      if identify_drug(r.get("drug")) == key), key=lambda r: int(r["day"]))
    review = next(r for r in sources["medication_review"] if identify_drug(r.get("drug")) == key)
    name, unit = L(f"drug.{key}"), d["unit"].split("/")[0]
    # An imported prescription field must not repeat the actual dose in an
    # event column; retain that history in its original dated encounter note.
    from haenv.p3.textread import field, lines
    encounter = next(r for r in sources["encounter_record"] if int(r["day"]) == int(rx["day"])
                     and is_text(field(r.get("text"), "p3.emr.line.complaint"), "p3.emr.complaint.visit"))
    encounter["text"] = J("lines", [*lines(encounter["text"]), T("p3.emr.line.record", text=rx["event"])])
    rx["event"] = T("p3.dose.event_reference")
    rx.update(record_id="R", prescription=T("p3.rx.daily_unit", name=name, dose=f"{rx_dose:g}", unit=unit))
    dispensing.update(record_id="D", bottle_label=T(contract["strength_template"], value=f"{strength:g}", unit=unit))
    review.update(record_id="L", patient_list=T(contract["count_template"], count=f"{list_dose / strength:g}"))
    for row in (rx, review):
        row["record_status"] = T("p3.dose.record_status", record=row["record_id"],
                                 current=current, superseded=superseded, note="N")
    sources["encounter_record"].append({"day": today, "record_id": "N", "source": T("p3.dose.service"),
        "text": T(contract["note_template"], name=name, current=current, superseded=superseded, dispensing="D")})
    sources["encounter_record"].sort(key=lambda r: int(r["day"]))
    # The header lists medicines, with the regimen in its attributed sources.
    from haenv.p3.textread import slots
    def target_line(value):
        line = slots(value, "med.line") or slots(value, "med.line_per_day")
        return line is not None and identify_drug(line["name"]) == key
    sources["current_medications"] = [T("p3.dose.list_reference", name=name) if target_line(v) else v
                                     for v in sources["current_medications"]]
    item["dose_reconciliation"] = {"discrepancy": discrepancy, "current_record": current,
                                   "prescription_dose": rx_dose, "patient_list_dose": list_dose,
                                   "truth_dose": dose, "source": "p3_dose_reconciliation.yaml"}
