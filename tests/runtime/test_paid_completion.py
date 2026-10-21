"""The common paid-call adapter preserves raw receipts and enforces one shared cap."""
import time
import threading
from decimal import Decimal

import pytest

from haenv.runtime.completion import AccountedCompletion
from haenv.scoring.semantic.budget import BudgetLedger
from haenv.scoring.semantic.budget import BudgetExceeded
from haenv.scoring.semantic.transport import PriceSchedule


def _sent(tmp_path, request_id):
    """The marker a request leaves on disk just before it is sent."""
    import hashlib
    from haenv.runtime.completion import mark_sent
    mark_sent(tmp_path / "receipts" / (hashlib.sha256(request_id.encode()).hexdigest() + ".json"),
              request_id=request_id)


def adapter(tmp_path, *, limit="1", model="model", effort=None):
    ledger = BudgetLedger(tmp_path / "ledger.json", limit_usd=limit)
    prices = PriceSchedule(model, Decimal(".000001"), Decimal(".000002"), Decimal(0), 100000)
    paid = AccountedCompletion(prices, ledger, tmp_path / "receipts")
    settings = {"model": model, "backend": "openrouter", "max_tokens": 1000,
                "reasoning_effort": effort, "response_format": None}
    return paid, ledger, settings


def test_solver_effort_is_preserved_not_forced_to_judge_effort(tmp_path):
    paid, ledger, settings = adapter(tmp_path, effort="medium")
    response = {"id": "id", "usage": {"cost": .001}, "choices": []}
    calls = []
    def post(prompt):calls.append(prompt);return response
    assert paid.request("solver:r1", "question", settings, post) == response
    assert settings["reasoning_effort"] == "medium" and calls == ["question"]
    assert ledger.snapshot()["settled_usd"] == "0.001"


def test_cap_refuses_before_network(tmp_path):
    paid, ledger, settings = adapter(tmp_path, limit=".000001")
    with pytest.raises(BudgetExceeded):
        paid.request("r1", "p", settings, lambda p:pytest.fail("Network started without budget"))


def test_receipt_replay_never_purchases_a_duplicate_request(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    response = {"usage": {"cost": .001}, "choices": [], "id": "provider-id"}
    assert paid.request("r1", "p", settings, lambda p:response) == response
    assert paid.request("r1", "p", settings, lambda p:pytest.fail("duplicate")) == response
    assert len(ledger.snapshot()["requests"]) == 1


def test_unknown_cost_retains_raw_response_and_new_requests_continue(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    raw = {"id": "provider-id", "choices": []}
    assert paid.request("r1", "p", settings, lambda p:raw) == raw
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"
    paid.request("r2", "p", settings, lambda p:{"usage": {"cost": .001}})


def test_same_id_with_changed_payload_refused(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    paid.request("r1", "p", settings, lambda p:{"usage": {"cost": .001}})
    with pytest.raises(BudgetExceeded):
        paid.request("r1", "different", settings, lambda p:pytest.fail("payload changed"))


def test_lost_receipt_never_reissued(tmp_path):
    from haenv.runtime.completion import ReconciledRequestFailure
    paid, ledger, settings = adapter(tmp_path)
    ledger.reserve("r1", ".1")
    _sent(tmp_path, "r1")                                # the request left, no answer came back
    with pytest.raises(ReconciledRequestFailure):
        paid.request("r1", "p", settings, lambda p:pytest.fail("lost request duplicated"))
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"


def test_transport_failure_stores_no_secret_exception_text(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    def fail(prompt):raise OSError("secret-token-do-not-log")
    from haenv.runtime.completion import UnknownCostAttemptFailure
    with pytest.raises(UnknownCostAttemptFailure) as error:
        paid.request("r1", "p", settings, fail)
    assert "secret-token" not in str(error.value)
    assert b"secret-token" not in (tmp_path / "ledger.json.sqlite").read_bytes()
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"


def test_schema_tokens_enter_the_reservation(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    settings["response_format"] = {"schema": "x" * 5000}
    paid.request("r1", "p", settings, lambda p:{"usage": {"cost": .001}})
    assert Decimal(ledger.snapshot()["requests"]["r1"]["reserved_usd"]) > Decimal(".009")


def test_parallel_same_id_dispatches_only_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    paid, ledger, settings = adapter(tmp_path)
    calls = []
    def post(prompt):calls.append(prompt);return {"usage": {"cost": .001}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(paid.request, "same", "p", settings, post) for _ in range(2)]
        assert futures[0].result() == futures[1].result()
    assert calls == ["p"] and ledger.snapshot()["settled_usd"] == "0.001"


def test_crash_after_receipt_before_settlement_recovers_without_network(tmp_path, monkeypatch):
    paid, ledger, settings = adapter(tmp_path)
    original = ledger.settle
    def interrupt(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(ledger, "settle", interrupt)
    with pytest.raises(KeyboardInterrupt):
        paid.request("r1", "p", settings, lambda p:{"usage": {"cost": .001}})
    assert ledger.snapshot()["requests"]["r1"]["status"] == "reserved"
    monkeypatch.setattr(ledger, "settle", original)
    paid.request("r1", "p", settings, lambda p:pytest.fail("resending already received answer"))
    assert ledger.snapshot()["settled_usd"] == "0.001"


def test_the_wait_for_a_paid_lane_is_recorded_per_thread(tmp_path, monkeypatch):
    """A request that queues for a lane leaves that wait for the solver to read once."""
    import contextlib
    import time
    import haenv.runtime.slots as paid_slots
    from haenv.runtime.completion import take_slot_wait_s

    @contextlib.contextmanager
    def slow_slot(*a, **k):
        time.sleep(0.2)
        yield
    monkeypatch.setattr(paid_slots, "request_slot", slow_slot)
    paid, ledger, settings = adapter(tmp_path)
    take_slot_wait_s()
    paid.request("r1", "p", settings, lambda p: {"usage": {"cost": .001}, "choices": []})
    assert take_slot_wait_s() >= 0.2
    assert take_slot_wait_s() is None


def test_a_reservation_whose_request_never_left_is_sent_on_resume(tmp_path):
    """Reserve, then die while waiting for a lane: nothing was sent, so the resume sends it
    under the same reservation instead of counting it lost."""
    paid, ledger, settings = adapter(tmp_path)
    ledger.reserve("r1", ".1")
    paid.request("r1", "p", settings, lambda p: {"usage": {"cost": .001}})
    record = ledger.snapshot()["requests"]["r1"]
    assert record["status"] == "settled" and len(ledger.snapshot()["requests"]) == 1


@pytest.mark.parametrize("role,global_pool", [("solver", False), ("judge", True)])
def test_route_role_survives_unsent_cancellation_resume_and_receipt_replay(
        tmp_path, monkeypatch, role, global_pool):
    """Route admission and graceful stop compose without charging an unsent attempt."""
    from contextlib import contextmanager
    import haenv.runtime.slots as paid_slots
    from haenv.runtime.ops.runtime import Draining

    paid, ledger, settings = adapter(tmp_path)
    paid.role, paid.global_pool, paid.request_limit = role, global_pool, 2
    admitted = []

    @contextmanager
    def stop_once(path, **pool):
        admitted.append(pool)
        assert ledger.record('queued')['role'] == role
        if len(admitted) == 1:
            raise Draining('stopped before dispatch')
        yield

    monkeypatch.setattr(paid_slots, 'request_slot', stop_once)
    with pytest.raises(Draining):
        paid.request('queued', 'question', settings, lambda p: pytest.fail('sent after stop'))
    assert ledger.record('queued')['status'] == 'cancelled'
    assert Decimal(ledger.summary()['counted_usd']) == 0
    calls = []

    def dispatch(prompt):
        calls.append(prompt)
        assert ledger.record('queued')['status'] == 'reserved'
        return {'usage': {'cost': .001}, 'choices': []}

    response = paid.request('queued', 'question', settings, dispatch)
    assert paid.request('queued', 'question', settings,
                        lambda p: pytest.fail('receipt replay dispatched again')) == response
    assert calls == ['question'] and len(admitted) == 2
    assert all(p['role'] == role and p['limit'] == 2
               and p['backend'] == (None if global_pool else 'openrouter') for p in admitted)
    assert ledger.record('queued')['role'] == role
    assert Decimal(ledger.summary()['counted_usd']) == Decimal('.001')


def test_the_ledger_is_never_called_while_a_lane_is_held(tmp_path, monkeypatch):
    """Reserve before the lane, settle after it: ledger latency must not hold a lane."""
    import contextlib
    import threading
    import haenv.runtime.slots as paid_slots
    held = threading.local()

    @contextlib.contextmanager
    def recording_slot(*args, **kwargs):
        held.now = True
        try:
            yield
        finally:
            held.now = False
    monkeypatch.setattr(paid_slots, "request_slot", recording_slot)
    paid, ledger, settings = adapter(tmp_path)
    for name in ("reserve", "reserve_or_pause", "settle", "record_unknown", "settle_rejected"):
        original = getattr(ledger, name)

        def guarded(*args, _original=original, _name=name, **kwargs):
            assert not getattr(held, "now", False), f"ledger.{_name} ran inside a lane"
            return _original(*args, **kwargs)
        monkeypatch.setattr(ledger, name, guarded)
    paid.request("r1", "p", settings, lambda p: {"usage": {"cost": .001}})
    with pytest.raises(Exception):
        paid.request("r2", "p", settings, lambda p: (_ for _ in ()).throw(OSError("down")))
    assert held.now is False


def test_requests_waiting_for_a_lane_still_respect_the_cap(tmp_path):
    """Reservations are taken before the lane, so the cap binds queued requests too."""
    from concurrent.futures import ThreadPoolExecutor
    from haenv.scoring.semantic.budget import BudgetCapReached
    paid, ledger, settings = adapter(tmp_path, limit=".006")      # room for about two bounds
    gate, entered = threading.Event(), threading.Semaphore(0)

    def dispatch(prompt):
        entered.release()
        gate.wait(5)
        return {"usage": {"cost": .001}}

    def one(i):
        try:
            return paid.request(f"r{i}", "p", settings, dispatch)
        except BudgetCapReached:
            return None
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(one, i) for i in range(8)]
        entered.acquire(timeout=5)
        gate.set()
        results = [f.result(10) for f in futures]
    snap = ledger.snapshot()
    assert 0 < sum(r is not None for r in results) < 8
    assert Decimal(snap["committed_usd"]) <= Decimal(".006")


def test_a_process_reserves_ahead_of_its_lanes_only_up_to_the_configured_bound(tmp_path, monkeypatch):
    """With no lane free, at most `paid.reserve_ahead` requests hold a reservation; the rest
    wait before the ledger. Once lanes open, every request completes."""
    import contextlib
    from concurrent.futures import ThreadPoolExecutor
    import haenv.runtime.slots as paid_slots
    open_lane = threading.Event()

    @contextlib.contextmanager
    def closed_until_opened(*args, **kwargs):
        open_lane.wait(5)
        yield
    monkeypatch.setattr(paid_slots, "request_slot", closed_until_opened)
    paid, ledger, settings = adapter(tmp_path)
    monkeypatch.setitem(paid_slots._AHEAD, str(ledger.path), threading.BoundedSemaphore(2))
    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(paid.request, f"r{i}", "p", settings, lambda p: {"usage": {"cost": .001}})
                   for i in range(6)]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(ledger.snapshot()["requests"]) < 2:
            time.sleep(.01)
        time.sleep(.2)
        assert len(ledger.snapshot()["requests"]) == 2
        open_lane.set()
        assert all(f.result(10) for f in futures)
    assert len(ledger.snapshot()["requests"]) == 6
