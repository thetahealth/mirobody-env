"""Target-drug records, prior intolerance and the saved medication context."""

from haenv.i18n.text import text_of as _rt
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.judge import _clinical_sorted


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    drug = policy["drug_names"][f["drug"]]
    for section in ("current_medications", "prescription_record", "dispensing_record", "medication_review",
                    "encounter_record", "home_measurements", "lab_results", "symptom_diary"):
        if section in v:
            c.add_key(f"/visible_facts/{section}")
    for i in range(len(snapshot["protocols"])):
        c.add_key(f"/protocols/{i}")
    if f.get("plan", {}).get("history_active"):
        events = f.get("adverse_event_record") or []
        if not events:
            raise ValueError("Active old intolerance has no saved event")
        for event in events:
            days = {r["day"] for r in event.get("prescriptions", []) + event.get("labs", [])} | {event["day"]}
            for section in ("prescription_record", "lab_results", "encounter_record"):
                for ref, row in c.records(section, lambda r: r.get("day") in days):
                    c.add_history(ref, row)
            doses = [r.get("dose_mg", r.get("dose_ug")) for r in event.get("prescriptions", [])]
            if not doses:
                if "exposure_class" not in event or "reaction" not in event:
                    raise ValueError("Old event has neither a recorded dose nor a registered allergy")
                c.prohibit(_rt('judge.scoring.management.adapters.p3.facts.1', s0=event['exposure_class'], s1=event['reaction']),
                           [h["fact_id"] for h in c.history],
                           violation=_rt('judge.scoring.management.adapters.p3.facts.2', s0=event['exposure_class'], s1=event['reaction']))
                continue
            peak = max(doses)
            c.prohibit(_rt('judge.scoring.management.adapters.p3.facts.3', s0=drug, s1=peak),
                       [h["fact_id"] for h in c.history],
                       violation=_rt('judge.scoring.management.adapters.p3.facts.4', s0=drug, s1=peak))
    c.prohibit(_rt('judge.scoring.management.adapters.p3.facts.5'), ["/visible_facts/current_medications"],
               violation=_rt('judge.scoring.management.adapters.p3.facts.6'), actor="patient_self_adjustment")
    actions = {a: _rt('judge.scoring.management.adapters.p3.facts.7', s0=drug, s1=label)
               for a, label in policy["med_decisions"].items()}
    selected = snapshot["acceptable_management"]["med_decision"][0]
    explanation = (_rt('judge.scoring.management.adapters.p3.facts.8', s0=drug, s1=_clinical_sorted(v.get('home_measurements', []))))
    followup = _rt('judge.scoring.management.adapters.p3.facts.9', s0=drug)
    brief = _rt('judge.scoring.management.adapters.p3.facts.10', s0=drug)
    return c.finish(actions=actions, selected=selected, accepted=snapshot["acceptable_management"]["med_decision"],
                    explanation=explanation, brief_explanation=brief, followup=followup,
                    initial=_rt('judge.scoring.management.adapters.p3.facts.11'))
