"""Render saved blind-review answers under the current Q policy, without calls.

The key/corpus are trusted local audit inputs, never included in judge prompts.
Original answer and obtained records are preserved in both language renderings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def render_one(job):
    review, key, snapshot, lang = job
    from haenv import i18n
    from haenv.i18n.text import judge_fingerprints
    from haenv.scoring.management.evidence import obtained_context
    from haenv.scoring.management.rubric import derive, load_policy
    from haenv.scoring.management.judge import make_task
    i18n.set_current(lang)
    messages = [{'role': 'user', 'content': v} for v in review['question']]
    messages += review['obtained_records']
    acquired = obtained_context(snapshot, trace={'messages': messages}, query_mode=True)
    policy = load_policy()
    rubric = derive(key['pack'], snapshot, policy, obtained_evidence=acquired)
    task = make_task(rubric, snapshot, review['answer'], policy, run_id='offline-input-audit')
    events = rubric['case_facts']['safety_events']
    assert all(all(k in event for k in ('id', 'applicability', 'actor', 'time_window', 'required_evidence')) for event in events)
    assert 'decision_facts' not in rubric['case_facts']
    return {'id': review['id'], 'language': lang, 'pack': key['pack'], 'case': key['case'],
            'item_ids': task.item_ids, 'rubric_version': rubric['version'],
            'management_rubric_sha16': judge_fingerprints(lang)['management_rubric_sha16'],
            'fingerprint': task.fingerprint, 'case_facts': rubric['case_facts'], 'prompt': task.prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('review', 'key', 'corpus', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    keys = {r['id']: r for r in read(args.key)}
    corpus = {(r['pack'], r['case']): r for r in read(args.corpus)}
    jobs = [(review, keys[review['id']],
             corpus[keys[review['id']]['pack'], keys[review['id']]['case']]['truth']['adjudication']['open_management'], lang)
            for review in read(args.review) for lang in ('zh', 'en')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(render_one, jobs, chunksize=1))
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in ('zh', 'en'):
        selected = [r for r in rows if r['language'] == lang]
        (args.out / (lang + '-inputs.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    facts = {'answers': len(jobs) // 2, 'rendered_inputs': len(rows), 'model_calls': 0,
             'languages': {lang: sum(r['language'] == lang for r in rows) for lang in ('zh', 'en')},
             'explicit_safety_lists': sum('safety_events' in r['case_facts'] for r in rows),
             'matching_item_ids': all(rows[i]['item_ids'] == rows[i+1]['item_ids'] for i in range(0, len(rows), 2))}
    (args.out/'summary.json').write_text(json.dumps(facts, indent=2)+'\n')
    print(json.dumps(facts))


if __name__ == '__main__':
    main()
