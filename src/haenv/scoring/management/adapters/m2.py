"""Saved workup, symptoms, prior reports and medication restrictions."""

from haenv.i18n.runtime import registry_phrase
from haenv.i18n.text import text_of as _rt
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.judge import _clinical_sorted
from haenv.scoring.management.rubric import canonical


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    for section in v:
        c.add_key(f"/visible_facts/{section}")
    for stream, rows in (f.get("world_observations", {}) if policy["version"] != "management-rubric-v7" else {}).items():
        for i, row in enumerate(rows):
            if isinstance(row, dict) and row.get("finding_id"):
                c.add_key(f"/decision_facts/world_observations/{stream}/{i}")
    context_ref = "/visible_facts/m2_frame" if "m2_frame" in v else "/visible_facts/archived_laboratory_record"
    for i in range(len(snapshot["protocols"])):
        c.add_key(f"/protocols/{i}")
    frame = f.get("m2_frame", {})
    state = (frame.get("f2_reading") or {}).get("dose_state", {})
    reduction = state.get("adverse_reduction")
    if reduction and reduction.get("blocks_next_step"):
        for ref, row in c.records("medication_record", lambda r: r.get("day") == reduction["day"]):
            c.add_history(ref, row)
        c.prohibit(_rt('judge.scoring.management.adapters.m2.facts.1', s0=reduction['day'], s1=reduction['reason'], s2=reduction['dose']),
                   ["/visible_facts/medication_record"],
                   violation=_rt('judge.scoring.management.adapters.m2.facts.2', s0=reduction['dose']))
    for test in snapshot["required_checks"]:
        allowed = [test] + snapshot["check_alternatives"].get(test, [])
        c.prohibit(_rt('judge.scoring.management.adapters.m2.facts.3', s0=test, s1=canonical(allowed[1:])),
                   [context_ref], violation=_rt('judge.scoring.management.adapters.m2.facts.4', s0=test),
                   required=["|".join(allowed)])
    urgency = snapshot["acceptable_management"].get("urgency")
    from haenv.resources.regpath import load_registry
    implementation = {k: registry_phrase(v) for k, v in load_registry("management_adapter_rules.yaml")["m2"]["urgency_implementation"].items()}
    actions = {u: _rt('judge.scoring.management.adapters.m2.facts.5', s0=text) for u, text in implementation.items()}
    if policy["version"] == "management-rubric-v7":
        # Existing task boundary, shared with the prescription profile adapter.
        c.prohibit(_rt("judge.scoring.management.adapters.p3.facts.5"), [context_ref],
                   violation=_rt("judge.scoring.management.adapters.p3.facts.6"),
                   actor="patient_self_adjustment")
        if urgency == "🔴":
            c.prohibit(_rt("judge.management.v7.emergency"), [context_ref],
                       violation=_rt("judge.management.v7.delayed_control"),
                       actor="patient", time_window="immediate_emergency_handover")
    if not urgency:
        raise ValueError("Saved M2 management has no registered urgency")
    medication = snapshot["acceptable_management"].get("medication")
    if medication:
        target = frame.get("target_drug") or state.get("drug")
        if not target:
            # Pre-i18n archives can lack f2_reading. Their stable English drug
            # key is the only target identifier retained by that old schema.
            scene = v.get('m2_frame', {}).get('scene', '')
            archived = [key for key in policy['drug_names'] if f' {key} ' in scene]
            target = archived[0] if len(archived) == 1 else None
        if not target:
            raise ValueError("Saved M2 target drug is not explicit in decision facts")
        med_text = _rt('judge.punctuation.m2.1', s0=policy['drug_names'].get(target, target), s1=policy['m2_med_decisions'][medication[0]])
        actions = {u: a + med_text for u, a in actions.items()}
    def with_checks(action, checks):
        tests = _rt('judge.punctuation.m2.2').join(_rt('judge.scoring.management.adapters.m2.facts.6', s0=test) for test in checks)
        return action + (_rt('judge.punctuation.m2.3') + tests if tests else "")

    alternatives = [with_checks(actions[urgency], [alternative if t == test else t for t in snapshot["required_checks"]])
                    for test, choices in snapshot["check_alternatives"].items()
                    for alternative in choices if test in snapshot["required_checks"]]
    actions = {u: with_checks(a, snapshot["required_checks"]) for u, a in actions.items()}
    from haenv.scoring.management.rubric import pointer
    explanation = _rt('judge.scoring.management.adapters.m2.facts.7', s0=_clinical_sorted(pointer(snapshot, context_ref)))
    followup = _rt('judge.scoring.management.adapters.m2.facts.8', s0=implementation[urgency])
    reading = frame.get("f2_reading") or {}
    brief = (_rt('judge.scoring.management.adapters.m2.facts.9', s0=reading['marker'], s1=reading['last'], s2=reading['target'])
             if reading.get("marker") else _rt('judge.scoring.management.adapters.m2.facts.10'))
    return c.finish(actions=actions, selected=urgency, explanation=explanation, brief_explanation=brief, followup=followup,
                    initial=_rt('judge.scoring.management.adapters.m2.facts.11'), management_alternatives=alternatives)
