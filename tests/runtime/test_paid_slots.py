"""Independent evaluation processes share one paid-request concurrency ceiling."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

from haenv.runtime.slots import request_slot


SCRIPT = '''
from pathlib import Path
import fcntl,json,sys,time
from haenv.runtime.slots import request_slot
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
    from haenv.runtime.slots import current_limits
    ledger=tmp_path/'budget.json'
    before=current_limits(ledger)
    with request_slot(ledger,limit=2):
        with request_slot(ledger,limit=3):pass
    assert current_limits(ledger)==before


def test_process_exit_releases_kernel_lock(tmp_path):
    ledger=tmp_path/'budget.json'
    code="from pathlib import Path; from haenv.runtime.slots import request_slot; import os,sys; " \
         "c=request_slot(Path(sys.argv[1]),limit=1); c.__enter__(); os._exit(0)"
    assert subprocess.run([sys.executable,'-c',code,str(ledger)],timeout=5).returncode==0
    with request_slot(ledger,limit=1):pass


def test_solver_adapter_enters_the_shared_slots(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from decimal import Decimal
    from haenv.runtime.completion import AccountedCompletion
    from haenv.scoring.semantic.transport import PriceSchedule
    from haenv.scoring.semantic.budget import BudgetLedger
    active=[]
    @contextmanager
    def slot(path,**kw):
        active.append(path)
        try:yield
        finally:active.pop()
    monkeypatch.setattr('haenv.runtime.slots.request_slot',slot)
    ledger=BudgetLedger(tmp_path/'b.json',limit_usd='1')
    p=AccountedCompletion(PriceSchedule('m',Decimal('.000001'),Decimal('.000001'),Decimal(0),100000),ledger,tmp_path/'receipts')
    def dispatch(prompt):
        assert active==[ledger.path]
        return {'usage':{'cost':.001}}
    p.request('r','q',{'backend':'openrouter','model':'m','max_tokens':1000},dispatch)


def test_judge_enters_same_shared_slots(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from decimal import Decimal
    from haenv.scoring.semantic.transport import PricedJudge
    from haenv.scoring.semantic.transport import PriceSchedule
    from haenv.scoring.semantic.judge import JudgeRequest
    from haenv.scoring.semantic.budget import BudgetLedger
    active=[]
    @contextmanager
    def slot(path,**kw):
        active.append(path)
        try:yield
        finally:active.pop()
    monkeypatch.setattr('haenv.runtime.slots.request_slot',slot)
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
    from haenv.runtime.slots import GLOBAL_REQUEST_LIMIT, JUDGE_RESERVED_LANES, SOLVER_LANES
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


def _hold(ledger, n, **kw):
    """Start `n` threads that each hold a slot until released; return (held, release, threads)."""
    import threading
    held, release = [], threading.Event()
    def hold():
        with request_slot(ledger, **kw):
            held.append(1); release.wait(5)
    threads = [threading.Thread(target=hold) for _ in range(n)]
    for t in threads: t.start()
    return held, release, threads


def _settle(held, want):
    import time
    deadline = time.time() + 3
    while len(held) < want and time.time() < deadline: time.sleep(.01)
    time.sleep(.15)


def test_backend_pools_do_not_wait_on_each_other(tmp_path):
    """A backend with its own ceiling draws from its own pool: a full global pool does not
    hold it back, and its own ceiling still binds."""
    from haenv.runtime.slots import set_capacity, pool_of
    ledger = tmp_path / "budget.json"
    set_capacity(ledger, 2)                                 # global pool: 2 lanes
    set_capacity(ledger, 3, backend="bk-a")                 # bk-a: its own 3 lanes
    assert pool_of(ledger, "bk-a") == "bk-a" and pool_of(ledger, "bk-unnamed") is None
    g_held, g_rel, g_threads = _hold(ledger, 4, backend="bk-unnamed")
    _settle(g_held, 2)
    assert len(g_held) == 2                                 # the global pool is full
    a_held, a_rel, a_threads = _hold(ledger, 5, backend="bk-a")
    _settle(a_held, 3)
    assert len(a_held) == 3                                 # bk-a fills its own pool, no more
    g_rel.set(); a_rel.set()
    for t in g_threads + a_threads: t.join(5)
    assert len(g_held) == 4 and len(a_held) == 5


def test_backend_pool_keeps_judge_lanes(tmp_path):
    from haenv.runtime.slots import set_capacity, current_limits, JUDGE_RESERVED_LANES
    ledger = tmp_path / "budget.json"
    set_capacity(ledger, JUDGE_RESERVED_LANES + 2, backend="bk-j")
    assert current_limits(ledger, "bk-j") == (JUDGE_RESERVED_LANES + 2, JUDGE_RESERVED_LANES)
    held, release, threads = _hold(ledger, 4, backend="bk-j", role="solver")
    _settle(held, 2)
    assert len(held) == 2                                   # solvers leave the reserved lanes
    done = []
    import threading
    def judge():
        with request_slot(ledger, backend="bk-j"): done.append(1)
    j = threading.Thread(target=judge); j.start(); j.join(2)
    assert done == [1]
    release.set()
    for t in threads: t.join(5)


def test_old_capacity_file_keeps_working(tmp_path):
    """A ledger whose capacity.json predates backend pools ({"limit": N}) keeps its global
    ceiling; setting a backend pool later keeps that entry."""
    from haenv.runtime.slots import set_capacity, current_limits, _root
    ledger = tmp_path / "budget.json"
    (_root(ledger) / "capacity.json").write_text(json.dumps({"limit": 5}))
    assert current_limits(ledger)[0] == 5
    assert current_limits(ledger, "bk-unnamed")[0] == 5     # no pool of its own: global
    change = set_capacity(ledger, 7, backend="bk-b")
    assert change["new"] == {"limit": 5, "backends": {"bk-b": 7}}
    assert current_limits(ledger)[0] == 5 and current_limits(ledger, "bk-b")[0] == 7


def test_in_flight_reservations_count_across_pools(tmp_path):
    """Pools bound concurrency only: a reservation held in one backend's pool still counts
    against the ledger's cap for a request in another pool."""
    from haenv.runtime.slots import set_capacity
    from haenv.scoring.semantic.budget import BudgetLedger
    from haenv.scoring.semantic.budget import BudgetExceeded
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    set_capacity(ledger.path, 2, backend="bk-a"); set_capacity(ledger.path, 2, backend="bk-b")
    with request_slot(ledger.path, backend="bk-a"):
        ledger.reserve("r-a", "0.6")
        with request_slot(ledger.path, backend="bk-b"):
            with pytest.raises(BudgetExceeded):
                ledger.reserve("r-b", "0.6")


def test_in_flight_slots_sees_backend_pools(tmp_path):
    from haenv.runtime.slots import set_capacity, requests_in_flight
    from haenv.runtime.ops.runtime import in_flight_slots
    ledger = tmp_path / "budget.json"
    set_capacity(ledger, 2, backend="bk-c")
    with request_slot(ledger, backend="bk-c"):
        assert requests_in_flight(ledger) and in_flight_slots(ledger) == ["bk-c@0.lock"]
    assert not requests_in_flight(ledger) and in_flight_slots(ledger) == []


def test_runtime_capacity_names_backend_pools(tmp_path):
    from haenv.runtime.ops.runtime import validate
    from haenv.runtime.ops.runtime import RuntimeRejected
    got = validate({"capacity": {"global": 10, "judge_reserved": 2, "backends": {"bk-d": 4}}})
    assert got["capacity"]["backends"] == {"bk-d": 4}
    with pytest.raises(RuntimeRejected):
        validate({"capacity": {"global": 10, "backends": {"bk-d": 0}}})


def test_only_the_head_of_the_queue_looks_for_a_lane(tmp_path, monkeypatch):
    """With every lane taken, waiters sleep in the kernel: lane lookups grow with time,
    not with the number of waiters; a released lane goes to a waiter at once."""
    import threading
    import time
    import haenv.runtime.slots as ps
    looks = []
    real = ps.current_limits
    monkeypatch.setattr(ps, "current_limits", lambda *a: looks.append(1) or real(*a))
    monkeypatch.setattr(ps, "_POLL_S", .05, raising=False)
    ledger = tmp_path / "budget.json"
    held, got = threading.Event(), []
    def holder():
        with request_slot(ledger, limit=1):
            held.set()
            time.sleep(.5)
    def waiter():
        with request_slot(ledger, limit=1):
            got.append(time.monotonic())
    first = threading.Thread(target=holder); first.start(); held.wait()
    released = time.monotonic() + .5
    waiters = [threading.Thread(target=waiter) for _ in range(20)]
    for t in waiters: t.start()
    first.join(); [t.join(10) for t in waiters]
    assert len(got) == 20 and min(got) - released < .1
    assert len(looks) < 100          # one head polls for ~0.5 s, not each of 20 waiters
