"""Original comparison reports and applicable laboratory/clinical records."""

from haenv.i18n.runtime import registry_phrase
from haenv.i18n.text import text_of as _rt
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.judge import _clinical_sorted


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    pair = []
    for day in (f["t0"], f["t1"]):
        rows = c.records("report", lambda r: r["day"] == day)
        if len(rows) != 1:
            raise ValueError("Saved comparison report is not unique")
        c.add_key(rows[0][0])
        pair.append(rows[0][1])
    for section in ("medication_record", "outpatient_record", "diet_record", "laboratory_method_record",
                    "specimen_procedure", "patient_remarks"):
        if section in v:
            c.add_key(f"/visible_facts/{section}")
    for i in range(len(snapshot["protocols"])):
        c.add_key(f"/protocols/{i}")
    if "interpretation_rule" in v:
        c.add_key("/visible_facts/interpretation_rule")
    c.prohibit(_rt('judge.scoring.management.adapters.p4.facts.1'), ["/visible_facts/report"],
               violation=_rt('judge.scoring.management.adapters.p4.facts.2'), actor="patient_self_adjustment")
    accepted = snapshot["acceptable_management"]["next_step"]
    if accepted == ["repeat_standardized"] and f["analyte"] == "creatinine":
        c.prohibit(_rt('judge.scoring.management.adapters.p4.facts.3'),
                   [f"/protocols/{i}" for i in range(len(snapshot["protocols"]))],
                   violation=_rt('judge.scoring.management.adapters.p4.facts.4'), required=[_rt('judge.scoring.management.adapters.p4.facts.5'), _rt('judge.scoring.management.adapters.p4.facts.6')])
    name = v["name"]
    actions = {a: _rt('judge.scoring.management.adapters.p4.facts.7', s0=name, s1=label)
               for a, label in policy["next_steps"].items()}
    if accepted == ["repeat_standardized"] and f["analyte"] == "creatinine":
        actions["repeat_standardized"] += _rt('judge.scoring.management.adapters.p4.facts.8')
    selected = accepted[0]
    from haenv.resources.regpath import load_registry
    labels = load_registry("management_adapter_rules.yaml")["p4"]
    cause = registry_phrase(labels["source_labels"][f["cause"]] if f.get("cause") else labels["class_labels"][f["class"]])
    explanation = (_rt('judge.scoring.management.adapters.p4.facts.9', s0=name, s1=_clinical_sorted(pair), s2=cause))
    followup = _rt('judge.scoring.management.adapters.p4.facts.10', s0=name)
    brief = _rt('judge.scoring.management.adapters.p4.facts.11', s0=name, s1=pair[0]['value'], s2=pair[1]['value'], s3=pair[1]['unit'], s4=cause)
    return c.finish(actions=actions, selected=selected, accepted=accepted, explanation=explanation,
                    brief_explanation=brief, followup=followup,
                    initial=_rt('judge.scoring.management.adapters.p4.facts.12'))
