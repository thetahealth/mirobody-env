"""Resolve dose from visible source references, never from adjudication.

SYNTHETIC data, evaluation only, not medical advice.
"""
from haenv.p3.textread import slots, prescription_dose
from haenv.resources.regpath import load_registry, registry_cached


@registry_cached("p3_dose_reconciliation.yaml")
def record_contract():
    return load_registry("p3_dose_reconciliation.yaml")["record_contract"]


def reconcile(sources, key):
    from haenv.p3.policy import identify_drug
    contract = record_contract()
    notes = [(r, slots(r.get("text"), contract["note_template"])) for r in sources.get("encounter_record") or ()]
    notes = [(r, n) for r, n in notes if n and identify_drug(n["name"]) == key]
    if not notes:
        return {}
    if any(n != notes[0][1] for _, n in notes[1:]):
        raise ValueError("dose reconciliation note is ambiguous")
    _, note = notes[0]
    records = [r for field in ("prescription_record", "medication_review", "dispensing_record")
               for r in sources.get(field) or () if identify_drug(r.get("drug")) == key]
    def one(record):
        matches = [r for r in records if r.get("record_id") == record]
        if not matches or any(r != matches[0] for r in matches[1:]):
            raise ValueError("dose reconciliation source missing or ambiguous: " + record)
        return matches[0]
    chosen = one(note["current"])
    if "prescription" in chosen:
        dose = prescription_dose(chosen["prescription"])
    else:
        strength = slots(one(note["dispensing"]).get("bottle_label"), contract["strength_template"])
        count = slots(chosen.get("patient_list"), contract["count_template"])
        if not strength or not count:
            raise ValueError("dose reconciliation needs bottle strength and daily tablet count")
        dose = float(strength["value"]) * float(count["count"])
    return {"reconciled_dose": dose,
            "dose_source_record": note["current"],
            "latest_prescription_dose": prescription_dose(one("R")["prescription"])}
