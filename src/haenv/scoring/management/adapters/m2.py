"""Saved workup, symptoms, prior reports and medication restrictions."""
import re
from haenv.scoring.management.adapters.common import Facts
from haenv.scoring.management.rubric import canonical


def facts(snapshot, policy):
    c = Facts(snapshot, policy)
    f, v = snapshot["decision_facts"], snapshot["visible_facts"]
    for section in v:
        c.add_key(f"/visible_facts/{section}")
    for stream, rows in f.get("world_observations", {}).items():
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
        c.prohibit(f"未经耐受复评不能恢复第{reduction['day']}天因{reduction['reason']}减下的旧{reduction['dose']}剂量。",
                   ["/visible_facts/medication_record"],
                   violation=f"本次直接恢复旧{reduction['dose']}剂量，不必复评不良反应。")
    for test in snapshot["required_checks"]:
        allowed = [test] + snapshot["check_alternatives"].get(test, [])
        c.prohibit(f"本例初始诊断路径必须取得或核对{test}，已登记替代为{canonical(allowed[1:])}；遗漏属于安全事件。",
                   [context_ref], violation=f"本例不需要{test}及其登记替代。",
                   required=["|".join(allowed)])
    urgency = snapshot["acceptable_management"].get("urgency")
    from haenv.resources.regpath import load_registry
    implementation = load_registry("management_adapter_rules.yaml")["m2"]["urgency_implementation"]
    actions = {u: f"患者{text}，由接诊医生评估" for u, text in implementation.items()}
    if not urgency:
        raise ValueError("Saved M2 management has no registered urgency")
    medication = snapshot["acceptable_management"].get("medication")
    if medication:
        target = frame.get("target_drug") or state.get("drug")
        if not target:
            found = re.findall(r"只问关于 ([a-z_]+) 这个药", v["m2_frame"]["scene"])
            if len(found) != 1:
                raise ValueError("Saved M2 target drug is not explicit")
            target = found[0]
        med_text = f"；{policy['drug_names'].get(target, target)}{policy['m2_med_decisions'][medication[0]]}"
        actions = {u: a + med_text for u, a in actions.items()}
    tests = "；".join(f"安排核对或取得{test}" for test in snapshot["required_checks"])
    actions = {u: a + ("；" + tests if tests else "") for u, a in actions.items()}
    from haenv.scoring.management.rubric import pointer
    explanation = f"向患者说明：依据已保存的病程与检验{canonical(pointer(snapshot, context_ref))}以及当前用药记录，因此落实上述医疗评估与检查安排。"
    followup = f"跟进安排：{implementation[urgency]}，复评当前症状和检查执行情况；明显恶化时立即提前就医。"
    reading = frame.get("f2_reading") or {}
    brief = (f"本次{reading['marker']}读数{reading['last']}与目标{reading['target']}及用药耐受一起考虑，因此落实上述安排。"
             if reading.get("marker") else "已结合本例当前症状和实际病程，而非单个偶发异常，因此落实上述评估和检查安排。")
    return c.finish(actions=actions, selected=urgency, explanation=explanation, brief_explanation=brief, followup=followup,
                    initial="如出现新的相关症状，由接诊医生评估是否需要定向初始检查。")
