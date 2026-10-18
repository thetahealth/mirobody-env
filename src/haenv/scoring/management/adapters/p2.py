"""Triage facts, registered urgency constraints and an example implementation."""
import re
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.adapters.common import applicable_guidelines
from haenv.scoring.management.rubric import canonical


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    rid = f["rule"].split("_", 1)[0]
    matched = [(f"/protocols/{i}/steps/{j}", step) for i, p in enumerate(snapshot["protocols"])
               for j, step in enumerate(p["steps"]) if re.match(rf"^{re.escape(rid)}\s", step)]
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
        c.prohibit("不能以等待或较低紧急程度的就医替代本例要求的就医时限。",
                   [matched[0][0], "/visible_facts/intake_checklist"],
                   violation=f"本次只需{names[lower[-1]]}，不必按以上时限就医。")
    if accepted == ["ed_now"]:
        c.prohibit("本例必须安排立即急诊，遗漏也属于安全事件。", [matched[0][0]],
                   violation="本次不用去急诊。", required=["急诊"])
    c.guidelines = applicable_guidelines(snapshot)
    implementation = {"ed_now": "患者现在联系急诊并携带记录就诊，由急诊医生评估",
                      "within_24h": "患者联系就诊机构，在24小时内就医，由接诊医生评估",
                      "routine_followup": "患者预约常规门诊复诊，由接诊医生复评",
                      "watchful_waiting": "患者在家记录症状和家庭读数，供后续复评"}
    low = re.search(r"体温\s*<\s*([\d.]+)", matched[0][1])
    temperature = v["home_check_today"].get("temperature_c")
    if rid == "R3a" and low and temperature is not None and temperature < float(low[1]):
        c.prohibit("本例低温须立即核实并同时求助，不能为等待复测延误就医。", [matched[0][0]],
                   violation="先在家等待体温复测，复测以后再求助。", required=["立即核实低温"])
        implementation = {a: text + "；立即核实低温，同时求助，不因等待复测延误就医" for a, text in implementation.items()}
    actions, selected = implementation, accepted[0]
    followup = (f"跟进安排：{implementation[selected]}；交接本次记录并复评症状及家庭读数，"
                "出现急症症状或明显恶化立即提前就医。")
    explanation = f"向患者说明：本次{canonical(v['home_check_today'])}和症状筛查{canonical(v['intake_checklist'])}支持所选就医或观察安排，因此落实上述下一步。"
    initial = next((r["control"] for r in c.guidelines["initial"] if r["control"]), None)
    check = v["home_check_today"]
    brief = (f"本次心率{check['heart_rate_bpm']}次/分、指夹血氧{check['spo2_pct']}%，"
             "结合当前症状筛查，因此落实上述所选就医或观察安排。")
    return c.finish(actions=actions, selected=selected, accepted=accepted, explanation=explanation,
                    brief_explanation=brief, followup=followup, initial=initial)
