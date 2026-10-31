"""Collect free-text plans and retain rubric inputs on the generation side."""
from __future__ import annotations

from haenv.i18n.runtime import runtime_text
from haenv.i18n.text import L as _L, text_of as _rt

from copy import deepcopy
import hashlib
import json

INSTRUCTION = (_L('runtime.scoring.management.plan.module.1'))
STUB_PLAN = (_L('runtime.scoring.management.plan.module.2'))
BLOCKS = {'p2': 'triage_p2', 'p3': 'meds_p3', 'p4': 'followup_p4', 'm2': 'm2_frame'}


def answer(data):
    return {**data, 'management_plan': runtime_text(STUB_PLAN)}


def _record_refs(block, refs, role):
    out = []
    for ref in refs:
        section, day = ref['section'], int(ref['day'])
        rows = block.get(section) or []
        if not isinstance(rows, list):
            rows = [rows]
        matches = [(i, r) for i, r in enumerate(rows) if isinstance(r, dict)
                   and r.get('day', r.get('draw_day', r.get('ts'))) == day]
        if not matches:
            raise ValueError(f'rubric source missing: {section}@{day}')
        for i, r in matches:
            out.append({'id': str(r.get('evidence_id') or f'{section}-{i}'), 'section': section, 'day': day,
                        'role': role, 'record': deepcopy(r)})
    return out


def finalize(raw, case_id, pack, *, derive_workup=True):
    if pack == 'm2' and not ((raw.adjudication or {}).get('m2_workup')
                             or (raw.prediction_context or {}).get('m2_frame')):
        return
    if pack == 'm2' and derive_workup:
        from haenv.m2.workup_generation import finalize as finalize_workup
        finalize_workup(raw, case_id)
    pc, adj = raw.prediction_context or {}, raw.adjudication or {}
    key = BLOCKS[pack]
    g = adj.get(key) or (adj if pack == 'm2' else {})
    block = pc.get(key) if pack != 'm2' else pc
    if not g or not block or (pack != 'm2' and 'class' not in g and 'floor' not in g):
        return
    from haenv.world.history.history_dependency import metadata
    meta = metadata(raw, pack)
    look = g.get('look_alike') or meta.get('look_alike') or {}
    decisive = _record_refs(block, meta.get('records') or [], 'decisive') if meta.get('active') else []
    inactive = _record_refs(block, [r for r in look.get('records') or [] if not r['acts']], 'inactive_similar')
    from haenv.world.protocols.library import bearing_ids
    from haenv.world.protocols.library import catalog
    from haenv.world.protocols.library import table
    ids = bearing_ids(pack, block)
    if pack == 'p2':
        rid = str(g['rule']).split('_', 1)[0]
        ids = {m['protocol_id'] for m in table()['triage'] if rid in m['rules']}
    protocols = [deepcopy(d) for d in catalog() if d['protocol_id'] in ids]
    checks, alternatives = [], {}
    if pack == 'm2':
        workup = adj.get('m2_workup') or {}
        checks, alternatives = workup.get('required') or [], workup.get('alternatives') or {}
        from haenv.m2.workup import pre_review_rules
        rules = pre_review_rules()
        protocols.append({'protocol_id': 'initial-workup', 'applies': _L('runtime.scoring.management.plan.finalize.1'),
                          'excludes': _L('runtime.scoring.management.plan.finalize.2'),
                          'steps': deepcopy(workup), 'sources': deepcopy(rules)})
        accepted = {'urgency': g.get('urgency', (adj.get('ddx') or {}).get('urgency')),
                    'medication': g.get('med_accept') or [], 'followup': g.get('f3_accept') or []}
        facts = {k: deepcopy(v) for k, v in adj.items() if k in ('ddx', 'm2_frame', 'm2_workup')}
        T = int(pc['prediction_time_T'])
        facts['world_observations'] = {s: [deepcopy(r) for r in rows if int(r.get('ts', T + 1)) <= T]
                                       for s, rows in (raw.longitudinal_data or {}).items()}
        if g.get('frame') == 'F3':
            from haenv.m2.core import f3_followup, f3_truth
            facts['followup_result'] = f3_followup(raw.longitudinal_data, g.get('f3_plan'), T)
            facts['followup_truth'] = f3_truth(raw.longitudinal_data, g, T)
            accepted['followup'] = deepcopy(facts['followup_truth'])
    else:
        facts = {k: deepcopy(v) for k, v in g.items() if k != 'open_management'}
        if pack == 'p2':
            accepted = {'disposition': deepcopy(g['set'])}
            checks = {'readings': deepcopy(g.get('readings') or {}), 'rule': g['rule']}
        elif pack == 'p3':
            accepted = {'med_decision': deepcopy(g.get('accept') or [g['class']])}
            checks = {'control_axis': g['axis'], 'renal_monitoring': deepcopy(g.get('renal') or []),
                      'adherence': deepcopy(g.get('g3') or {})}
        else:
            from haenv.p4.gold import next_step
            accepted = {'next_step': list(next_step(g)[1])}
            checks = {'repeat_analyte': g['analyte'], 'standardization': deepcopy(block.get('specimen_procedure') or {}),
                      'existing_records': deepcopy(g.get('records') or [])}
    data = {'version': 1, 'decisive_old_records': decisive, 'inactive_similar_records': inactive,
            'protocols': protocols, 'acceptable_management': accepted,
            'required_checks': deepcopy(checks), 'check_alternatives': deepcopy(alternatives),
            'decision_facts': facts, 'visible_facts': deepcopy(block)}
    data['facts_sha256'] = hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    adj['open_management'] = data


def render_for(next_call, solver, payload):
    text, pid = next_call(solver, payload)
    pc = getattr(payload, 'prediction_context', None) or {}
    library = any(isinstance(pc.get(k), dict) and pc[k].get('rules_delivery') == 'library' for k in BLOCKS.values())
    if not library and (any(k in pc for k in BLOCKS.values()) or 'archived_laboratory_record' in pc):
        text += runtime_text(INSTRUCTION)
    return text, pid


def protocol_question(next_call, pack, block):
    if block.get("rules_delivery") == "library" and pack in {"p2", "p3", "p4"}:
        from haenv.app.pack_hooks import HOOKS, Stage
        if pack == 'p2' and (Stage.PROTOCOL_QUESTION, 'p2:sources') in HOOKS:
            return next_call(pack, block)
        return _rt("runtime.protocol_question." + pack)
    return next_call(pack, block)


def install_prompt():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.PROMPT, "management_plan", render_for)
    register_hook(Stage.PROTOCOL_QUESTION, "management_plan:i18n", protocol_question)


def register(pack):
    if pack.name not in BLOCKS:
        return
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    def with_facts(next_call, cs, *args, **kwargs):
        raw, audit = next_call(cs, *args, **kwargs)
        if raw is not None:
            finalize(raw, str(cs.case_id), pack.name, derive_workup=False)
        return raw, audit
    register_hook(Stage.CASE, "management_plan:" + pack.name, with_facts)
    install_prompt()


def install():
    from haenv.app.pack_hooks import register_pack_extension
    register_pack_extension("management_plan", register)
