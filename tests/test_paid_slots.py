"""Independent evaluation processes share one paid-request concurrency ceiling."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

from haenv.paid_slots import request_slot


SCRIPT = '''
from pathlib import Path
import fcntl,json,sys,time
from haenv.paid_slots import request_slot
p=Path(sys.argv[1]); counter=p.with_suffix('.counter')
def change(delta):
    with counter.with_suffix('.lock').open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        d=json.loads(counter.read_text());d['live']+=delta;d['peak']=max(d['peak'],d['live'])
        counter.write_text(json.dumps(d))
with request_slot(p,limit=2):
    change(1);time.sleep(.05);change(-1)
'''


def test_independent_processes_never_exceed_global_slots(tmp_path):
    ledger=tmp_path/'budget.json';ledger.with_suffix('.counter').write_text('{"live":0,"peak":0}')
    jobs=[subprocess.Popen([sys.executable,'-c',SCRIPT,str(ledger)]) for _ in range(8)]
    assert [p.wait(timeout=10) for p in jobs]==[0]*8
    state=json.loads(ledger.with_suffix('.counter').read_text())
    assert state['live']==0 and state['peak']==2


def test_slot_capacity_is_live_not_refused(tmp_path):
    """The ceiling is read at each acquisition (runtime / capacity.json); a caller's own
    `limit` only narrows its lanes and never rewrites the shared ceiling."""
    from haenv.paid_slots import current_limits
    ledger=tmp_path/'budget.json'
    before=current_limits(ledger)
    with request_slot(ledger,limit=2):
        with request_slot(ledger,limit=3):pass
    assert current_limits(ledger)==before


def test_process_exit_releases_kernel_lock(tmp_path):
    ledger=tmp_path/'budget.json'
    code="from pathlib import Path; from haenv.paid_slots import request_slot; import os,sys; " \
         "c=request_slot(Path(sys.argv[1]),limit=1); c.__enter__(); os._exit(0)"
    assert subprocess.run([sys.executable,'-c',code,str(ledger)],timeout=5).returncode==0
    with request_slot(ledger,limit=1):pass


def test_solver_adapter_enters_the_shared_slots(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from decimal import Decimal
    from haenv.paid_completion import AccountedCompletion
    from haenv.semantic_transport import PriceSchedule
    from haenv.semantic_budget import BudgetLedger
    active=[]
    @contextmanager
    def slot(path,**kw):
        active.append(path)
        try:yield
        finally:active.pop()
    monkeypatch.setattr('haenv.paid_slots.request_slot',slot)
    ledger=BudgetLedger(tmp_path/'b.json',limit_usd='1')
    p=AccountedCompletion(PriceSchedule('m',Decimal('.000001'),Decimal('.000001'),Decimal(0),100000),ledger,tmp_path/'receipts')
    def dispatch(prompt):
        assert active==[ledger.path]
        return {'usage':{'cost':.001}}
    p.request('r','q',{'backend':'openrouter','model':'m','max_tokens':1000},dispatch)


def test_judge_enters_same_shared_slots(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from decimal import Decimal
    from haenv.semantic_transport import PricedJudge,PriceSchedule
    from haenv.semantic_judge import JudgeRequest
    from haenv.semantic_budget import BudgetLedger
    active=[]
    @contextmanager
    def slot(path,**kw):
        active.append(path)
        try:yield
        finally:active.pop()
    monkeypatch.setattr('haenv.paid_slots.request_slot',slot)
    ledger=BudgetLedger(tmp_path/'b.json',limit_usd='1')
    class Solver:
        model='m';reasoning_effort='high';pool=None;max_tokens=1000
        def _post(self,prompt):
            assert active==[ledger.path]
            return {'usage':{'cost':.001},'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}]}
    judge=PricedJudge(Solver(),PriceSchedule('m',Decimal('.000001'),Decimal('.000001'),Decimal(0),100000),ledger)
    judge(JudgeRequest('r',1,'q'))


def test_solver_lanes_leave_judge_lanes_free(tmp_path):
    import threading
    from haenv.paid_slots import GLOBAL_REQUEST_LIMIT, JUDGE_RESERVED_LANES, SOLVER_LANES
    assert SOLVER_LANES == GLOBAL_REQUEST_LIMIT - JUDGE_RESERVED_LANES >= 1
    ledger = tmp_path / "budget.json"
    held, release = [], threading.Event()
    def hold():
        with request_slot(ledger, limit=GLOBAL_REQUEST_LIMIT, lanes=SOLVER_LANES):
            held.append(1); release.wait(5)
    threads = [threading.Thread(target=hold) for _ in range(SOLVER_LANES + 3)]
    for t in threads: t.start()
    import time
    deadline = time.time() + 3
    while len(held) < SOLVER_LANES and time.time() < deadline: time.sleep(.01)
    time.sleep(.2)
    assert len(held) == SOLVER_LANES                   # extra solver callers wait
    done = []
    def judge():
        with request_slot(ledger, limit=GLOBAL_REQUEST_LIMIT): done.append(1)
    j = threading.Thread(target=judge); j.start(); j.join(2)
    assert done == [1]                                 # a judge still gets a lane
    release.set()
    for t in threads: t.join(5)
    assert len(held) == SOLVER_LANES + 3
