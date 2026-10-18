"""Target-drug records, prior intolerance and the saved medication context."""
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.rubric import canonical


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
                c.prohibit(f"已核实对{event['exposure_class']}发生{event['reaction']}，不能安排该过敏药物。",
                           [h["fact_id"] for h in c.history],
                           violation=f"本次直接使用{event['exposure_class']}，不必核对既往{event['reaction']}。")
                continue
            peak = max(doses)
            c.prohibit(f"本次不能恢复或超过已证实不耐受的{drug} {peak:g}剂量。",
                       [h["fact_id"] for h in c.history],
                       violation=f"本次恢复{drug} {peak:g}剂量，不必复评既往不耐受。")
    c.prohibit("患者不能自行开药、停药或改剂量。", ["/visible_facts/current_medications"],
               violation="患者可以自行调整所有药物剂量，无需医生审核。")
    actions = {a: f"{drug}：{label}，由患者落实处方并由医生复核执行情况"
               for a, label in policy["med_decisions"].items()}
    selected = snapshot["acceptable_management"]["med_decision"][0]
    explanation = (f"向患者说明：{drug}当前用药、控制指标和症状记录{canonical(v.get('home_measurements', []))}；"
                   f"结合实际服药覆盖、用药经过及既往耐受情况，因此落实上述方案。")
    followup = f"跟进安排：落实{drug}方案后复评疗效、控制指标及不适症状；出现明显不适或病情恶化提前就医。"
    brief = f"已结合{drug}本次控制指标、服药经过和既往耐受情况，因此落实上述所选用药方案。"
    return c.finish(actions=actions, selected=selected, accepted=snapshot["acceptable_management"]["med_decision"],
                    explanation=explanation, brief_explanation=brief, followup=followup,
                    initial="面诊时如有新的相关症状，由接诊医生评估是否需要定向初始检查。")
