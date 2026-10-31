"""Case-grounded positive controls; never a judge or a scoring rule.

Only saved visible records, published-in-case intervals and the saved decision
choose these renderings. Missing optional observations are omitted, never '无'.
"""
from __future__ import annotations

from haenv.i18n.runtime import registry_phrase
from haenv.i18n.text import text_of as _rt

NEGATIVE_CONTROLS = frozenset({
    'empty', 'generic', 'generic_correct', 'readings_only', 'verbose_unrelated',
    'gold_plus_contraindication', 'gold_plus_not_recommended_check', 'gold_wrong_decision',
})


def _last(rows):
    if not rows:
        raise ValueError('Positive control requires a saved observation')
    return max(rows, key=lambda row: row['day'])


def _current_dose(v, f, spec):
    from haenv.p3.gold import dose_at
    if 'dose_line' in f:
        return dose_at(f['dose_line'], f['T'])
    if 'current_dose' in f:
        return f['current_dose']
    for row in sorted(v.get('prescription_record', []), key=lambda r: r['day'], reverse=True):
        if 'dose_mg' in row or 'dose_ug' in row:
            return row.get('dose_mg', row.get('dose_ug'))
    # Compatibility with archived snapshots that retained only a current-medication
    # string. New snapshots always carry the numeric dose line above.
    from haenv.world.clinical.med_course import display
    from haenv.i18n.runtime import runtime_text
    matches = [dose for dose in spec['ladder']
               if runtime_text(display(f['drug'], dose)) in v.get('current_medications', [])]
    if len(matches) != 1:
        raise ValueError('Saved medication has no unique structured current dose')
    return matches[0]


def _drug_prescription(v, f, name, unit):
    from haenv.world.labworld.meds import drug
    from haenv.world.clinical.med_course import display
    from haenv.i18n.runtime import runtime_text
    return runtime_text(display(f['drug'], _current_dose(v, f, drug(f['drug']))))


def _med_context(snapshot, control, policy):
    v, f = snapshot['visible_facts'], snapshot['decision_facts']
    name = policy['drug_names'][f['drug']]
    from haenv.world.labworld.meds import drug
    spec = drug(f['drug'])
    unit = spec['unit'].split('/')[0]
    current = _drug_prescription(v, f, name, unit)
    axis = {'LDL': 'LDL-C', 'TSH': 'TSH', 'HbA1c': 'HbA1c', 'home_sbp': _rt('judge.scoring.management.controls_v62._med_context.1')}[f['axis']]
    pool = v['home_measurements'] if f['axis'] == 'home_sbp' else v['lab_results']
    matches = list(pool) if f['axis'] == 'home_sbp' else [r for r in pool if r.get('axis_key') == f['axis'] or axis in r['test']]
    observation = _last(matches)
    fact = _rt('judge.scoring.management.controls_v62._med_context.2', s0=current, s1=observation['test'], s2=observation['result'])
    # The visible diary orders saved events by onset; onset and prose may
    # be localized without changing which structured events are drug symptoms.
    symptoms = [row for event, row in zip(sorted(f.get('diary', []), key=lambda e: e['onset']), v.get('symptom_diary', []))
                if event.get('kind') == 'drug_symptom']
    if symptoms:
        symptom = max(symptoms, key=lambda r: r['onset_day'])
        fact += _rt('judge.scoring.management.controls_v62._med_context.3', s0=symptom['symptom'], s1=symptom['onset_day'])
    from haenv.p3.gold import dose_at
    amounts = [float(x) for x in spec['ladder']]
    value = _current_dose(v, f, spec)
    interval_days = int(spec['tau_rule_days'])
    actions = {}
    for decision, label in policy['med_decisions'].items():
        if decision in {'uptitrate', 'downtitrate'}:
            candidates = [x for x in amounts if x > value] if decision == 'uptitrate' else [x for x in amounts if x < value]
            if candidates:
                target = min(candidates) if decision == 'uptitrate' else max(candidates)
                action = _rt('judge.scoring.management.controls_v62._med_context.4', s0=current, s1=name, s2=target, s3=unit)
            else:
                if control['selected'] == decision:
                    raise ValueError('Selected dose change has no next tier in the saved protocol')
                # This action is not selected at a ladder boundary; do not invent a dose.
                action = _rt('judge.scoring.management.controls_v62._med_context.5', s0=current)
        elif decision == 'maintain':
            action = _rt('judge.scoring.management.controls_v62._med_context.6', s0=current)
        elif decision == 'switch':
            action = _rt('judge.scoring.management.controls_v62._med_context.7', s0=current)
        else:
            action = _rt('judge.scoring.management.controls_v62._med_context.8', s0=current)
        actions[decision] = action
    control['actions'] = actions
    control['management_alternatives'] = []
    control['followup_project'] = observation['test']
    control['followup'] = (_rt('judge.scoring.management.controls_v62._med_context.9', s0=str(interval_days), s1=observation['test']))
    control['followup_variants'] = {
        'relative': _rt('judge.control.med_followup_relative', project=observation['test']),
        'range': _rt('judge.control.med_followup_range', days=interval_days, project=observation['test']),
        'fixed': _rt('judge.control.med_followup_fixed', weeks=interval_days / 7, project=observation['test']),
    }
    control['brief_explanation'] = _rt('judge.scoring.management.controls_v62._med_context.10' , s0=fact)
    control['explanation'] = control['brief_explanation'] + _rt('judge.scoring.management.controls_v62._med_context.11')


def _medication_action(row, facts, visible):
    if 'course_action_key' in row:
        return row['course_action_key']
    for event in facts.get('acei', {}).get('events', []):
        if event['day'] == row.get('onset_day'):
            return event['action']
    from haenv.resources.regpath import load_registry
    cfg = load_registry('p4_diversity.yaml')
    records = [r for r in visible.get('medication_record', [])
               if r.get('day') == row.get('onset_day')]
    for record in records:
        if record.get('course_end_day') is not None:
            # An ended historical course is a start event. Adherence events use
            # their distinct structured dose and unit in the medication registry.
            for key in ('thyroxine_adherence', 'fibrate_adherence'):
                med = cfg['medication_world'][key]
                if record.get('dose') == med['dose'] and record.get('dose_unit') == med['unit']:
                    return 'adherence_start'
            return 'start'
    cause = facts.get('cause')
    if cause in cfg['rules']:
        rule = cfg['rules'][cause]
        if rule.get('section') == 'medication_record':
            if cause in {'thyroxine_adherence', 'fibrate_adherence'}:
                return 'adherence_start' if facts.get('sign', -rule['sign']) == rule['sign'] else 'adherence_stop'
            return 'start' if facts.get('sign', -rule['sign']) == rule['sign'] else 'stop'
    return None


def _lab_context(snapshot, control, policy):
    v, f = snapshot['visible_facts'], snapshot['decision_facts']
    pair = [_last([r for r in v['report'] if r['day'] == day]) for day in (f['t0'], f['t1'])]
    name = v['name']
    fact = _rt('judge.scoring.management.controls_v62._lab_context.1', s0=pair[0]['day'], s1=name, s2=pair[0]['value'], s3=pair[1]['day'], s4=pair[1]['value'], s5=pair[1]['unit'])
    selected = control['selected']
    if f['class'] == 'true_change':
        # Interpret the clinical record, never claim that a small numeric gap proves a drug cause.
        events = v.get('outpatient_record', [])
        relevant = [r for r in events if r.get('medication') and r.get('text') and _medication_action(r, f, v) != 'start']
        event = _last(relevant)['text'] if relevant else ''
        from haenv.resources.regpath import load_registry
        cause = registry_phrase(load_registry('management_adapter_rules.yaml')['p4']['source_labels'][f['cause']])
        reason = (_rt('judge.scoring.management.controls_v62._lab_context.2', s0=cause) + (_rt('judge.scoring.management.controls_v62._lab_context.3', s0=event) if event else '') +
                  _rt('judge.scoring.management.controls_v62._lab_context.4'))
    elif f['class'] == 'method_difference':
        reason = _rt('judge.scoring.management.controls_v62._lab_context.5')
    elif f['class'] == 'preanalytical':
        reason = (_rt('judge.scoring.management.controls_v62._lab_context.6')
                  + (_rt('judge.scoring.management.controls_v62._lab_context.7') if selected == 'no_action' else _rt('judge.scoring.management.controls_v62._lab_context.8')))
    else:
        reason = _rt('judge.scoring.management.controls_v62._lab_context.9')
    control['brief_explanation'] = _rt('judge.scoring.management.controls_v62._lab_context.10', s0=fact, s1=reason)
    control['explanation'] = control['brief_explanation'] + _rt('judge.scoring.management.controls_v62._lab_context.11')
    project = name
    if f['analyte'] == 'creatinine' and snapshot['acceptable_management']['next_step'] == ['repeat_standardized']:
        project += _rt('judge.scoring.management.controls_v62._lab_context.12')
    when = {'no_action': _rt('judge.scoring.management.controls_v62._lab_context.13'), 'repeat_standardized': _rt('judge.scoring.management.controls_v62._lab_context.14'),
            'act_on_change': _rt('judge.scoring.management.controls_v62._lab_context.15')}[selected]
    control['followup_project'] = project
    control['followup'] = _rt('judge.scoring.management.controls_v62._lab_context.16', s0=when, s1=project)
    control['actions'] = {
        'no_action': _rt('judge.scoring.management.controls_v62._lab_context.17', s0=name, s1=name),
        'repeat_standardized': _rt('judge.scoring.management.controls_v62._lab_context.18', s0=project),
        'act_on_change': _rt('judge.scoring.management.controls_v62._lab_context.19', s0=name, s1=name),
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
                    text += _rt('judge.scoring.management.controls_v62.improve_positive_context.1') + r['reason']
            elif 'value' in r and 'name' in r:
                text = f"{r['name']}{r['value']}{r['unit']}"
                if pack == 'p4' and snapshot['decision_facts']['class'] == 'method_difference':
                    text += _rt('judge.scoring.management.controls_v62.improve_positive_context.2', s0=r['method'], s1=r['platform'], s2=r['reagent_lot'])
            elif 'text' in r:
                text = r['text'].replace('\n', _rt('judge.punctuation.controls_v62.1'))
            elif 'result' in r:
                text = (r.get('test') or r.get('name') or '') + str(r['result'])
                if r.get('conclusion'):
                    text += _rt('judge.punctuation.controls_v62.2') + r['conclusion']
            else:
                raise ValueError('Unsupported decisive record for positive history rendering')
            pieces.append(_rt('judge.scoring.management.controls_v62.improve_positive_context.3', s0=r['day'], s1=text))
        control['history'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.4') + _rt('judge.punctuation.controls_v62.3').join(dict.fromkeys(pieces)) + _rt('judge.scoring.management.controls_v62.improve_positive_context.5')
    if pack == 'p3':
        _med_context(snapshot, control, policy)
    elif pack == 'p4':
        _lab_context(snapshot, control, policy)
    elif pack == 'p2':
        if control['selected'] == 'watchful_waiting':
            control['actions']['watchful_waiting'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.6') + control['actions']['watchful_waiting']
        elif control['selected'] == 'routine_followup':
            control['actions']['routine_followup'] += _rt('judge.scoring.management.controls_v62.improve_positive_context.7')
        control['followup_project'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.8')
        when = (_rt('judge.scoring.management.controls_v62.improve_positive_context.9') if control['selected'] == 'watchful_waiting'
                else _rt('judge.scoring.management.controls_v62.improve_positive_context.10'))
        control['followup'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.11', s0=when, s1=control['followup_project'])
        control['explanation'] = control['brief_explanation'] + _rt('judge.scoring.management.controls_v62.improve_positive_context.12')
    else:
        v = snapshot['visible_facts']
        frame = v['m2_frame']
        records = frame.get('followup_record', [])
        reading = snapshot['decision_facts'].get('m2_frame', {}).get('f2_reading') or {}
        medications = v['current_medications']
        facts = []
        if records:
            row = _last(records)
            facts.append(_rt('judge.scoring.management.controls_v62.improve_positive_context.13', s0=row['indicator'], s1=row['value'], s2=row['unit']))
        if medications:
            facts.append(_rt('judge.scoring.management.controls_v62.improve_positive_context.14') + _rt('judge.punctuation.controls_v62.4').join(medications))
        if not facts:
            facts.append(_rt('judge.scoring.management.controls_v62.improve_positive_context.15') + frame['patient_quote'] + '”')
        control['brief_explanation'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.16') + _rt('judge.punctuation.controls_v62.5').join(facts) + _rt('judge.scoring.management.controls_v62.improve_positive_context.17')
        control['explanation'] = control['brief_explanation'] + _rt('judge.scoring.management.controls_v62.improve_positive_context.18')
        project = _rt('judge.punctuation.controls_v62.6').join(snapshot['required_checks'])
        if records:
            project = _rt('judge.punctuation.controls_v62.7').join(filter(None, (project, records[-1]['indicator'])))
        elif reading.get('marker'):
            project = _rt('judge.punctuation.controls_v62.8').join(filter(None, (project, reading['marker'])))
        if not project:
            project = _rt('judge.scoring.management.controls_v62.improve_positive_context.19')
        control['followup_project'] = project
        control['followup'] = _rt('judge.scoring.management.controls_v62.improve_positive_context.20', s0=project)
        if snapshot['acceptable_management'].get('medication') and medications:
            control['actions'] = {k: a + _rt('judge.scoring.management.controls_v62.improve_positive_context.21') + _rt('judge.punctuation.controls_v62.9').join(medications) for k, a in control['actions'].items()}
    control['management_alternatives'] = []
    control.setdefault('followup_variants', {k: control['followup'] for k in ('relative', 'range', 'fixed')})
