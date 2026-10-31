"""Triage facts, registered urgency constraints and an example implementation."""

from haenv.i18n.text import text_of as _rt
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.adapters.common import applicable_guidelines
from haenv.scoring.management.judge import _clinical_sorted
import re


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    rid = f["rule"].split("_", 1)[0]
    from haenv.p2.clinical_rules import table
    matched = []
    for i, protocol in enumerate(snapshot["protocols"]):
        # Stable printed rule IDs locate the actually saved clause. Current
        # catalog positions may differ from an archived edition (e.g. R3d).
        for j, step in enumerate(protocol.get("steps") or []):
            if isinstance(step, str) and re.match(re.escape(rid) + r"(?=\s|[：:])", step):
                matched.append((f"/protocols/{i}/steps/{j}", step))
    if len(matched) != 1:
        raise ValueError("Saved triage rule has no unique visible protocol")
    for key in ("home_check_today", "intake_checklist", "home_bp_log", "morning_weight",
                "cgm_nightly", "wearable_daily", "clinical_history_record", "device_record"):
        if key in v:
            c.add_key(f"/visible_facts/{key}")
    c.add_key(matched[0][0])
    accepted = snapshot["acceptable_management"]["disposition"]
    order, names = list(policy["dispositions"]), policy["dispositions"]
    lower = order[max(order.index(a) for a in accepted) + 1:]
    if lower:
        c.prohibit(_rt('judge.scoring.management.adapters.p2.facts.1'),
                   [matched[0][0], "/visible_facts/intake_checklist"],
                   violation=_rt('judge.scoring.management.adapters.p2.facts.2', s0=names[lower[-1]]))
    if accepted == ["ed_now"]:
        c.prohibit(_rt('judge.scoring.management.adapters.p2.facts.3'), [matched[0][0]],
                   violation=_rt('judge.scoring.management.adapters.p2.facts.4'), required=[_rt('judge.scoring.management.adapters.p2.facts.5')])
    c.guidelines = applicable_guidelines(snapshot)
    implementation = {"ed_now": _rt('judge.scoring.management.adapters.p2.facts.6'),
                      "within_24h": _rt('judge.scoring.management.adapters.p2.facts.7'),
                      "routine_followup": _rt('judge.scoring.management.adapters.p2.facts.8'),
                      "watchful_waiting": _rt('judge.scoring.management.adapters.p2.facts.9')}
    low = table()["thresholds"]["hypothermia_temp_lt"]
    temperature = v["home_check_today"].get("temperature_c")
    if rid == "R3a" and temperature is not None and temperature < low:
        c.prohibit(_rt('judge.scoring.management.adapters.p2.facts.10'), [matched[0][0]],
                   violation=_rt('judge.scoring.management.adapters.p2.facts.11'), required=[_rt('judge.scoring.management.adapters.p2.facts.12')])
        implementation = {a: text + _rt('judge.scoring.management.adapters.p2.facts.13') for a, text in implementation.items()}
    actions, selected = implementation, accepted[0]
    followup = (_rt('judge.scoring.management.adapters.p2.facts.14', s0=implementation[selected]))
    explanation = _rt('judge.scoring.management.adapters.p2.facts.15', s0=_clinical_sorted(v['home_check_today']), s1=_clinical_sorted(v['intake_checklist']))
    initial = next((r["control"] for r in c.guidelines["initial"] if r["control"]), None)
    check = v["home_check_today"]
    brief = (_rt('judge.scoring.management.adapters.p2.facts.16', s0=check['heart_rate_bpm'], s1=check['spo2_pct']))
    return c.finish(actions=actions, selected=selected, accepted=accepted, explanation=explanation,
                    brief_explanation=brief, followup=followup, initial=initial)
