"""Correct only the complete score-blind affected set, preserving prior evidence."""

from dataclasses import asdict

from copy import deepcopy

import json

from pathlib import Path

import pytest

from haenv.scoring.semantic.inputs import anonymous_prompt

from haenv.scoring.semantic.judge import JudgeTask

from haenv.scoring.semantic.rubric import load_policy

from haenv.scoring.semantic.rubric import make_rubric

from haenv.scoring.semantic.pipeline import _reference

from haenv.scoring.semantic.pipeline import digest

from haenv.scoring.semantic.corrections import prepare_correction

from haenv.scoring.semantic.corrections import validate_correction

from haenv.scoring.semantic.visibility import correct_reference

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
        from haenv.scoring.judge_evidence import evidence_catalogue
        task=JudgeTask('base',tuple(criteria),prompt,txt,policy['judge']['model_id'],policy['version'],
                       evidence_catalog=evidence_catalogue(txt)[0])
        rows.append({'key':str(i),'status':'ready','task':asdict(task),'rubric':rubric,
                     'task_sha256':task.fingerprint,'source':{'batch':str(tmp_path/'batch'),'case':str(i),
                        'solver':'s','geometry':'gated','endpoint':None,'raw_sha256':'answer'}})
    tasks=base/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (base/'manifest.json').write_text(json.dumps({'run_id':'base','sealed':True,'policy':policy,
        'tasks_sha256':digest(tasks),'sources':[],'code':{'judging_sha16':'base-sha'}}))
    return base,rows

def _votes(out, row):
    from haenv.scoring.semantic.judge import JudgeReply
    from haenv.scoring.semantic.judge import evaluate_consensus
    from haenv.scoring.semantic.rubric import summarize_verdicts
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


__all__ = [name for name in globals() if not name.startswith("__")]
