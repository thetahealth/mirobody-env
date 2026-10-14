"""Killed cells are retried through the existing scheduler and accounted in receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from haenv.cell_recovery import Recovery, recovery
from haenv import run_scheduler as scheduler


def append(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_missing_started_cells_reordered_and_completed_success_kept(tmp_path):
    append(tmp_path / 'cells.jsonl', [dict(event='started', case=c, solver='m') for c in ['done', 'lost']])
    append(tmp_path / 'eval.jsonl', [dict(case='done', solver='m', overall='SCORED')])
    tasks = [('new', 'm'), ('lost', 'm')]
    seen = []
    def run(t):
        seen.append(t[0]); return dict(case=t[0], solver=t[1], overall='SCORED')
    scheduler.run(run, tasks, lambda r:r, workers=1, key_of=lambda t:t[1], batch_dir=tmp_path)
    receipt = json.loads((tmp_path / 'resume.json').read_text())
    assert seen == ['lost', 'new']
    assert receipt['interrupted'] == [{'case':'lost', 'solver':'m'}]
    assert receipt['recovered'][0]['case'] == 'lost'
    assert receipt['recovered'][0]['attempts'] == 2
    assert receipt['pending'] == []


def test_old_partial_response_identifies_inflight_and_abort_report_keeps_policy(tmp_path):
    append(tmp_path / 'responses.jsonl', [dict(case='lost', solver='m', raw='saved slice')])
    state = Recovery(tmp_path, [('lost', 'm')], lambda t:t[1])
    assert state.interrupted == {('lost', 'm')}
    state.start(('lost', 'm'))
    state.emitted(dict(case='lost', solver='m', overall='ABORT(no_answer)'))
    state.finish()
    receipt = json.loads((tmp_path / 'resume.json').read_text())
    assert receipt['recovered'][0]['overall'] == 'ABORT(no_answer)'
    assert receipt['failed_cells'][0]['overall'] == 'ABORT(no_answer)'
    assert receipt['pending'] == []


def test_killed_process_leaves_start_and_next_run_fills_it(tmp_path):
    script = '''import os, sys
from pathlib import Path
from haenv.cell_recovery import recovery
with recovery(Path(sys.argv[1]), [('lost','m')], lambda t:t[1]) as r:
 r.start(('lost','m'))
 os._exit(137)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=False)
    assert result.returncode == 137
    with recovery(tmp_path, [('lost', 'm')], lambda t:t[1]) as r:
        assert ('lost', 'm') in r.interrupted
        r.start(('lost', 'm'))
        r.emitted(dict(case='lost', solver='m', overall='SCORED'))
    assert json.loads((tmp_path / 'resume.json').read_text())['recovered'][0]['attempts'] == 2


def test_live_scheduler_cannot_be_taken_over_and_stopped_run_keeps_pending(tmp_path):
    with recovery(tmp_path, [('c', 'm')], lambda t:t[1]) as r:
        r.start(('c','m'))
        with pytest.raises(ValueError, match='Another scheduler'):
            with recovery(tmp_path, [('c', 'm')], lambda t:t[1]):
                pass
    with recovery(tmp_path, [('c', 'm')], lambda t:t[1]) as r:
        assert r.interrupted == {('c', 'm')}
    receipt = json.loads((tmp_path / 'resume.json').read_text())
    assert receipt['pending'] == [{'case':'c', 'solver':'m'}]


def test_repeated_infra_errors_record_attempt_count_without_changing_verdict(tmp_path):
    append(tmp_path / 'cells.jsonl', [dict(event='started', case='c', solver='m')] * 3)
    append(tmp_path / 'eval.jsonl', [dict(case='c', solver='m', overall='ERROR')])
    with recovery(tmp_path, [('c', 'm')], lambda t:t[1]) as r:
        r.start(('c','m'))
        r.emitted(dict(case='c', solver='m', overall='ERROR'))
    receipt = json.loads((tmp_path / 'resume.json').read_text())
    assert receipt['failed_cells'][0]['attempts'] == 4
    assert receipt['interrupted'] == []


@pytest.mark.parametrize('suffix', ['', ':resend:1'])
def test_legacy_inflight_marker_maps_content_address_to_missing_cell(tmp_path, suffix):
    from haenv.solver_accounting import cell_v2
    root = tmp_path / 'solver-accounting'; (root / 'receipts').mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps({'batch_uid': 'batch', 'run_id': 'batch'}))
    identity = cell_v2('batch', 'lost', 'm')
    (root / 'receipts/marker.sent').write_text(json.dumps({'request_id': f'solver:v2:{identity}:prompt:1{suffix}'}))
    state = Recovery(tmp_path, [('lost', 'm')], lambda t:t[1])
    assert state.interrupted == {('lost', 'm')}


def test_kept_terminal_failures_are_still_listed_and_success_clears_old_error(tmp_path):
    append(tmp_path/'eval.jsonl',[dict(case='kept',solver='m',overall='ABORT(no_answer)',failed_attempts=[{},{}]),
                                 dict(case='retry',solver='m',overall='ERROR')])
    with recovery(tmp_path,[('retry','m')],lambda t:t[1]) as r:
        r.start(('retry','m'));r.emitted(dict(case='retry',solver='m',overall='SCORED'))
    report=json.loads((tmp_path/'resume.json').read_text())
    assert report['failed_cells']==[{'case':'kept','solver':'m','overall':'ABORT(no_answer)','attempts':0,'failed_requests':2}]


def test_parallel_stop_reports_draining_when_sibling_already_stopped():
    import threading
    import time
    from haenv.ops.runtime import Draining
    from haenv.semantic_budget import BudgetExceeded
    ready=threading.Barrier(2)
    def interrupted(task):
        ready.wait(timeout=2)
        if task==0:
            raise BudgetExceeded('sibling stop')
        time.sleep(.02)
        raise Draining('operator stop')
    with pytest.raises(Draining):
        scheduler._run(interrupted,[0,1],lambda r:r,workers=2,pooling='backend',
                       backend_limits={'b':2},backend_of=lambda m:'b')
