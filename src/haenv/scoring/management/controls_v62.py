"""Case-grounded positive controls; never a judge or a scoring rule.

Only saved visible records, published-in-case intervals and the saved decision
choose these renderings. Missing optional observations are omitted, never '无'.
"""
from __future__ import annotations
import re

NEGATIVE_CONTROLS = frozenset({
    'empty', 'generic', 'generic_correct', 'readings_only', 'verbose_unrelated',
    'gold_plus_contraindication', 'gold_plus_not_recommended_check', 'gold_wrong_decision',
})


def _last(rows):
    if not rows:
        raise ValueError('Positive control requires a saved observation')
    return max(rows, key=lambda row: row['day'])


def _drug_prescription(v, name):
    matches = [m for m in v['current_medications'] if name in m]
    if len(matches) != 1:
        raise ValueError('Target drug has no unique saved current prescription')
    return matches[0]


def _med_context(snapshot, control, policy):
    v, f = snapshot['visible_facts'], snapshot['decision_facts']
    name = policy['drug_names'][f['drug']]
    current = _drug_prescription(v, name)
    axis = {'LDL': 'LDL-C', 'TSH': 'TSH', 'HbA1c': 'HbA1c', 'home_sbp': '家庭收缩压'}[f['axis']]
    pool = v['home_measurements'] if f['axis'] == 'home_sbp' else v['lab_results']
    matches = [r for r in pool if axis in r['test']]
    observation = _last(matches)
    fact = f"当前处方为{current}；最近一次{observation['test']}为{observation['result']}"
    relevant_symptoms = {r['symptom'] for r in f.get('diary', []) if r.get('kind') == 'drug_symptom'}
    symptoms = [r for r in v.get('symptom_diary', []) if r['symptom'] in relevant_symptoms]
    if symptoms:
        symptom = max(symptoms, key=lambda r: r['onset_day'])
        fact += f"；症状记录为{symptom['symptom']}（第{symptom['onset_day']}天起）"
    steps = [step for p in snapshot['protocols'] for step in p.get('steps', [])
             if isinstance(step, str) and step.startswith(name) and '剂量阶梯' in step and '充分显现' in step]
    if len(steps) != 1:
        raise ValueError('No unique saved target-drug efficacy interval')
    interval = re.search(r'后约\s*(\d+)\s*天疗效才充分显现', steps[0])
    tiers = re.search(r'剂量阶梯\s*(.*?)（每日）', steps[0])
    dose = re.search(r'(\d+(?:\.\d+)?)\s*(mg|μg)', current)
    if not interval or not tiers or not dose:
        raise ValueError('Saved prescription, dose ladder or interval cannot be rendered')
    amounts = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*(?:mg|μg)', tiers[1])]
    value, unit = float(dose[1]), dose[2]
    actions = {}
    for decision, label in policy['med_decisions'].items():
        if decision in {'uptitrate', 'downtitrate'}:
            candidates = [x for x in amounts if x > value] if decision == 'uptitrate' else [x for x in amounts if x < value]
            if candidates:
                target = min(candidates) if decision == 'uptitrate' else max(candidates)
                action = f"{current}，经医生审核后将{name}调至{target:g} {unit}每日"
            else:
                if control['selected'] == decision:
                    raise ValueError('Selected dose change has no next tier in the saved protocol')
                # This action is not selected at a ladder boundary; do not invent a dose.
                action = f"{current}，由医生复核该药剂量，确认阶梯边界后再决定是否调整"
        elif decision == 'maintain':
            action = f"维持{current}，核对实际服药和本次控制指标"
        elif decision == 'switch':
            action = f"本次核对{current}的耐受及既往限制，由医生审核加用其他药物或更换方案，患者不自行加量"
        else:
            action = f"{current}暂不调整，先核对实际服药、取药与症状起病经过，再由医生复评"
        actions[decision] = action
    control['actions'] = actions
    control['management_alternatives'] = []
    control['followup_project'] = observation['test']
    control['followup'] = (f"本次先完成所选方案及症状核查；落实方案后约{interval[1]}天复查{observation['test']}并复评疗效和不适；"
                           "若出现明显不适、原症状加重或新的急性症状，提前就诊，不等到预约复查。")
    control['brief_explanation'] = f"向患者解释：{fact}，应结合当前控制情况和实际用药经过落实上述所选方案。"
    control['explanation'] = control['brief_explanation'] + '不能只看一次读数或只凭症状名称自行改药。'


def _lab_context(snapshot, control, policy):
    v, f = snapshot['visible_facts'], snapshot['decision_facts']
    pair = [_last([r for r in v['report'] if r['day'] == day]) for day in (f['t0'], f['t1'])]
    name = v['name']
    fact = f"第{pair[0]['day']}天{name}{pair[0]['value']}、第{pair[1]['day']}天{pair[1]['value']}{pair[1]['unit']}"
    selected = control['selected']
    if f['class'] == 'true_change':
        # Interpret the clinical record, never claim that a small numeric gap proves a drug cause.
        events = v.get('outpatient_record', [])
        relevant = [r for r in events if r.get('medication') and r.get('text') and r.get('course_action') != '开始服用']
        event = _last(relevant)['text'] if relevant else ''
        from haenv.resources.regpath import load_registry
        cause = load_registry('management_adapter_rules.yaml')['p4']['source_labels'][f['cause']]
        reason = (f"需核对的病程线索：{cause}；" + (f"病程记载“{event}”；" if event else '') +
                  '数值差本身不能单独证实病因，仍须结合本例已登记的用药及病程变化评估处置')
    elif f['class'] == 'method_difference':
        reason = '两份报告的方法、平台或批号及相应比对记录影响可比性，应在可比条件下复核，不能直接当成病情加重'
    elif f['class'] == 'preanalytical':
        reason = ('应核对这两次各自的进食、采样及标本条件，不能把受影响的报告直接当成病情变化；'
                  + ('本次按当前报告继续常规监测' if selected == 'no_action' else '先按标准条件重新采样确认'))
    else:
        reason = '结合本例已登记的分析及个体内波动范围，不能仅凭这次差值自行改药，按所选安排观察或复核'
    control['brief_explanation'] = f"向患者解释：{fact}；{reason}。"
    control['explanation'] = control['brief_explanation'] + '有新症状时结合病程重新评估。'
    project = name
    if f['analyte'] == 'creatinine' and snapshot['acceptable_management']['next_step'] == ['repeat_standardized']:
        project += '、肾功能和血钾'
    when = {'no_action': '下次常规复诊时', 'repeat_standardized': '下次按标准条件采样时',
            'act_on_change': '本次就诊先评估处置，并在下次复诊时'}[selected]
    control['followup_project'] = project
    control['followup'] = f"{when}复查或复评{project}及相关症状；出现明显不适、新症状或病情恶化，提前就诊。"
    control['actions'] = {
        'no_action': f"本次不因{name}前后差值自行改药，按原计划常规监测{name}",
        'repeat_standardized': f"预约下次标准条件采样复查{project}，携带两份原报告核对检测条件",
        'act_on_change': f"本次就诊复评{name}及相关症状，核对用药和病程变化，由医生据此处理并安排复查{name}",
    }


def improve_positive_context(pack, snapshot, control, policy, history):
    if history:
        pieces = []
        for h in history:
            r = h['record']
            if 'prescription' in r:
                text = r['prescription']
            elif 'drug' in r and 'event' in r:
                text = r['drug'] + r['event']
                if r.get('reason'):
                    text += '，原因是' + r['reason']
            elif 'value' in r and 'name' in r:
                text = f"{r['name']}{r['value']}{r['unit']}"
                if pack == 'p4' and snapshot['decision_facts']['class'] == 'method_difference':
                    text += f"，方法{r['method']}，平台{r['platform']}，批号{r['reagent_lot']}"
            elif 'text' in r:
                text = r['text'].replace('\n', '；')
            elif 'result' in r:
                text = (r.get('test') or r.get('name') or '') + str(r['result'])
                if r.get('conclusion'):
                    text += '；' + r['conclusion']
            else:
                raise ValueError('Unsupported decisive record for positive history rendering')
            pieces.append(f"第{r['day']}天{text}")
        control['history'] = '既往关键记录：' + '；'.join(dict.fromkeys(pieces)) + '；本次按上述经过核对处置。'
    if pack == 'p3':
        _med_context(snapshot, control, policy)
    elif pack == 'p4':
        _lab_context(snapshot, control, policy)
    elif pack == 'p2':
        if control['selected'] == 'watchful_waiting':
            control['actions']['watchful_waiting'] = '今天先在家观察，' + control['actions']['watchful_waiting']
        elif control['selected'] == 'routine_followup':
            control['actions']['routine_followup'] += '，按预约在下次常规复诊携带记录就诊'
        control['followup_project'] = '本次症状、体温、血压、心率和指夹血氧'
        when = ('今天继续记录，并在下次复诊时' if control['selected'] == 'watchful_waiting'
                else '按上述就医时限由接诊医生')
        control['followup'] = f"{when}复评{control['followup_project']}；出现急症症状或明显恶化，立即提前就医。"
        control['explanation'] = control['brief_explanation'] + '就医时请携带本次记录，供接诊医生复核。'
    else:
        v = snapshot['visible_facts']
        frame = v['m2_frame']
        records = frame.get('followup_record', [])
        reading = snapshot['decision_facts'].get('m2_frame', {}).get('f2_reading') or {}
        medications = v['current_medications']
        facts = []
        if records:
            row = _last(records)
            facts.append(f"本次{row['indicator']}为{row['value']}{row['unit']}")
        if medications:
            facts.append('本例当前用药为' + '、'.join(medications))
        if not facts:
            facts.append('本次患者描述“' + frame['patient_quote'] + '”')
        control['brief_explanation'] = '向患者解释：' + '；'.join(facts) + '；应结合本次症状及原报告先完成上述医疗评估与检查，不能凭单次异常自行调药。'
        control['explanation'] = control['brief_explanation'] + '已有报告应携带供接诊医生核对，之后结合结果复评。'
        project = '、'.join(snapshot['required_checks'])
        if records:
            project = '、'.join(filter(None, (project, records[-1]['indicator'])))
        elif reading.get('marker'):
            project = '、'.join(filter(None, (project, reading['marker'])))
        if not project:
            project = '本次症状和当前用药疗效'
        control['followup_project'] = project
        control['followup'] = f"先按上述就医时限落实评估和检查，下次复诊核对{project}及症状变化；明显恶化或新发急症症状立即提前就医。"
        if snapshot['acceptable_management'].get('medication') and medications:
            control['actions'] = {k: a + '；核对本例现用处方' + '、'.join(medications) for k, a in control['actions'].items()}
    control['management_alternatives'] = []
