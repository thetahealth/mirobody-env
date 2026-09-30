"""Correct only the complete score-blind affected set, preserving prior evidence."""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from haenv.semantic_inputs import anonymous_prompt
from haenv.semantic_judge import JudgeTask
from haenv.semantic_rubric import load_policy, make_rubric
from haenv.semantic_pipeline import _reference, digest
from haenv.semantic_corrections import prepare_correction, validate_correction
from haenv.semantic_visibility import correct_reference


def reference(visible, truth=True):
    return {"kind":"ddx:unified","diagnosis":"A","aliases":[],"threads":[],"join_gold":"unified",
            "required_tests":[],"optional_tests":[],"rivals":[],
            "noop":{"target":"steps","window":[0,13],"truth_present":truth},
            "visible_case":{"longitudinal_data":visible}}


def base_run(tmp_path):
    base=tmp_path/'base';base.mkdir()
    policy=load_policy();rows=[]
    for i,ref in enumerate([reference({}), reference({'steps':[{'ts':0,'value':2}]})]):
        answer={'answer_text':'No data','differential':[{'diagnosis':'A'}]}
        criteria,rubric=make_rubric(ref,answer,policy)
        prompt,txt=anonymous_prompt(criteria,ref,answer,evidence_ids=True)
        from haenv.judge_evidence import evidence_catalogue
        task=JudgeTask('base',tuple(criteria),prompt,txt,policy['judge']['model_id'],policy['version'],
                       evidence_catalog=evidence_catalogue(txt)[0])
        rows.append({'key':str(i),'status':'ready','task':asdict(task),'rubric':rubric,
                     'task_sha256':task.fingerprint,'source':{'batch':str(tmp_path/'batch'),'case':str(i),
                        'solver':'s','geometry':'gated','endpoint':None,'raw_sha256':'answer'}})
    tasks=base/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (base/'manifest.json').write_text(json.dumps({'run_id':'base','sealed':True,'policy':policy,
        'tasks_sha256':digest(tasks),'sources':[],'code':{'judging_sha16':'base-sha'}}))
    return base,rows


def test_fix_uses_delivered_window_but_preserves_original_reference():
    r=reference({});original=deepcopy(r)
    fixed,change=correct_reference(r)
    assert fixed['noop']['truth_present'] is False and r==original
    assert change['source_truth_present'] is True and change['n_visible_in_window']==0
    r2=reference({'steps':[{'ts':0,'value':2}]})
    same,change=correct_reference(r2)
    assert same==r2 and change is None


def test_default_reference_uses_real_delivered_data(monkeypatch):
    from types import SimpleNamespace
    import haenv.semantic_pipeline as p
    import haenv.judges._helpers as h
    monkeypatch.setattr(p,'asdict',lambda sp:{'case_id':'c','longitudinal_data':{}})
    monkeypatch.setattr(h,'_gold',lambda vp,key,default=None: 'A' if key=='diagnosis' else default)
    monkeypatch.setattr(h,'gold_kind',lambda vp:'ddx:unified')
    monkeypatch.setattr(h,'_rivals_of',lambda vp:[])
    ref=_reference({'noop_truth_present':True,'noop_target':'steps','noop_window':[0,13]},object(),SimpleNamespace())
    assert ref['noop']['truth_present'] is False


def test_correction_selects_all_and_only_conflicts_without_reusing_votes(tmp_path):
    base,rows=base_run(tmp_path);before=digest(base/'tasks.jsonl')
    out=tmp_path/'correction'
    m=prepare_correction(base,out,run_id='correction')
    new=[json.loads(line) for line in (out/'tasks.jsonl').read_text().splitlines()]
    assert len(new)==1 and new[0]['source']==rows[0]['source']
    assert new[0]['task']['answer_text']==rows[0]['task']['answer_text']
    assert new[0]['task_sha256']!=rows[0]['task_sha256']
    assert digest(base/'tasks.jsonl')==before
    assert not (out/'samples').exists()
    assert m['correction']['keys']==['0']
    assert validate_correction(out)[0]['run_id']=='base'


def test_missing_or_extra_correction_is_refused_even_if_rehashed(tmp_path):
    base,_=base_run(tmp_path);out=tmp_path/'correction';prepare_correction(base,out,run_id='correction')
    (out/'tasks.jsonl').write_text('')
    m=json.loads((out/'manifest.json').read_text());m['tasks_sha256']=digest(out/'tasks.jsonl')
    (out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='complete affected'):
        validate_correction(out)


def test_changed_correction_answer_is_rejected_even_if_rehashed(tmp_path):
    base,_=base_run(tmp_path);out=tmp_path/'correction';prepare_correction(base,out,run_id='correction')
    rows=[json.loads(line) for line in (out/'tasks.jsonl').read_text().splitlines()]
    rows[0]['task']['answer_text']='{"answer_text":"modified"}'
    (out/'tasks.jsonl').write_text(json.dumps(rows[0])+'\n')
    m=json.loads((out/'manifest.json').read_text());m['tasks_sha256']=digest(out/'tasks.jsonl')
    (out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='expected correction'):
        validate_correction(out)


def _votes(out, row):
    from haenv.semantic_judge import JudgeReply, evaluate_consensus
    from haenv.semantic_rubric import summarize_verdicts
    task=JudgeTask(**{**row['task'],'item_ids':tuple(row['task']['item_ids'])})
    reply=json.dumps({'verdicts':{item:{'label':'yes','reason':'test','evidence_ids':[]}
                                  for item in task.item_ids}})
    consensus=evaluate_consensus(task,lambda req:JudgeReply(reply,False,None))
    (out/'results').mkdir(exist_ok=True);(out/'samples').mkdir(exist_ok=True)
    (out/'samples'/f"{row['key']}.jsonl").write_text(''.join(json.dumps(x)+'\n' for x in consensus['samples']))
    (out/'results'/f"{row['key']}.json").write_text(json.dumps({'source':row['source'],
        'consensus':consensus,'summary':summarize_verdicts(consensus['verdicts'],row['rubric'])}))


def _with_sources(base,tmp_path):
    m=json.loads((base/'manifest.json').read_text());m['sources']=[{'batch':str(tmp_path/'batch'),'files':{}}]
    (base/'manifest.json').write_text(json.dumps(m))


def test_old_report_masks_known_wrong_gold_without_altering_votes(tmp_path):
    from haenv.semantic_report import read_run
    base,rows=base_run(tmp_path);_with_sources(base,tmp_path)
    for row in rows:_votes(base,row)
    before=digest(base/'results/0.json')
    _,result=read_run(base,tmp_path/'batch')
    assert result[('0','s')]['status']=='needs_visibility_correction'
    assert result[('0','s')]['summary']['metrics']['noop_ok'] is None
    assert result[('1','s')]['summary']['metrics']['noop_ok']==1
    assert digest(base/'results/0.json')==before


def test_pending_correction_cannot_use_old_score_and_complete_overlay_preserves_unaffected(tmp_path):
    from haenv.semantic_report import view_for_batch
    base,records=base_run(tmp_path);_with_sources(base,tmp_path)
    for row in records:_votes(base,row)
    out=tmp_path/'correction';prepare_correction(base,out,run_id='correct')
    manifest=json.loads((out/'manifest.json').read_text());manifest['sealed']=True
    (out/'manifest.json').write_text(json.dumps(manifest))
    rows=[{'case':str(i),'solver':'s','geometry':'gated','judging_sha16':'old','noop_ok':1} for i in range(2)]
    viewed=view_for_batch(rows,tmp_path/'batch',out)
    assert viewed[0]['noop_ok'] is None and viewed[1]['noop_ok']==1
    correction=json.loads((out/'tasks.jsonl').read_text());_votes(out,correction)
    viewed=view_for_batch(rows,tmp_path/'batch',out)
    assert viewed[0]['noop_ok']==1 and viewed[1]['noop_ok']==1
    assert viewed[0]['semantic']['run_id']=='correct'
    assert viewed[1]['semantic']['run_id']=='base'
    assert viewed[0]['semantic']['correction']['base_run_id']=='base'
