"""The SQLite ledger: JSON import, transactions across processes, quantile reservations."""
from decimal import Decimal
import json
import os
import subprocess
import sys

import pytest

import haenv.runtime.budget as sb
from haenv.runtime.budget import BudgetCapReached, BudgetExceeded, BudgetLedger, read_state


def _json_ledger(path, records, limit="10", tail=b""):
    """A JSON ledger in the snapshot-plus-change-lines form, optionally with a torn tail."""
    head = {"limit_usd": limit, "halt_reason": None, "requests": dict(records[:1])}
    lines = [json.dumps({"r": rid, "v": rec}) for rid, rec in records[1:]]
    path.write_bytes((json.dumps(head) + "\n" + "".join(l + "\n" for l in lines)).encode() + tail)


RECORDS = [("a", {"reserved_usd": "0.5", "actual_usd": "0.10", "status": "settled"}),
           ("b", {"reserved_usd": "0.3", "actual_usd": None, "status": "reserved"}),
           ("c", {"reserved_usd": "0.2", "actual_usd": None, "status": "unknown_capped"}),
           ("d", {"reserved_usd": "0.4", "actual_usd": None, "status": "tariff_capped"}),
           ("a", {"reserved_usd": "0.5", "actual_usd": "0.15", "status": "settled"})]


def test_json_ledger_is_imported_once_read_only_with_equal_totals(tmp_path):
    path = tmp_path / "budget.json"
    _json_ledger(path, RECORDS, tail=b'{"r":"z","v":{"reserved_usd":"9","act')   # torn append
    before = path.read_bytes()
    legacy = read_state(path)
    ledger = BudgetLedger(path)
    assert path.read_bytes() == before and ledger.store.is_file()
    snap = ledger.snapshot()
    assert snap["requests"] == legacy["requests"] and "z" not in snap["requests"]
    assert snap["committed_usd"] == "1.05" and snap["in_flight_usd"] == "0.3"
    assert {k: snap[k] for k in snap if k != "requests"} == ledger.summary()
    assert read_state(path)["requests"] == legacy["requests"]        # now read from the database


def test_archived_json_ledger_stays_readable_without_a_database(tmp_path):
    path = tmp_path / "budget.json"
    _json_ledger(path, RECORDS)
    assert read_state(path)["requests"]["a"]["actual_usd"] == "0.15"
    assert not sb.store_of(path).exists()


def test_json_ledger_written_after_import_is_refused(tmp_path):
    path = tmp_path / "budget.json"
    _json_ledger(path, RECORDS)
    BudgetLedger(path)
    with path.open("ab") as handle:                     # an old-format writer still running
        handle.write(b'{"r":"e","v":{"reserved_usd":"1","actual_usd":null,"status":"reserved"}}\n')
    with pytest.raises(BudgetExceeded, match="changed after its import"):
        BudgetLedger(path)


WORKER = r'''
import sys
from haenv.runtime.budget import BudgetCapReached, BudgetLedger
ledger = BudgetLedger(sys.argv[1])
for i in range(int(sys.argv[3])):
    rid = f"{sys.argv[2]}-{i}"
    try:
        ledger.reserve(rid, "0.05")
    except BudgetCapReached:
        continue
    ledger.settle(rid, "0.01")
'''


def test_processes_never_lose_an_update_or_pass_the_cap(tmp_path):
    path = tmp_path / "budget.json"
    BudgetLedger(path, limit_usd="1.5")
    jobs = [subprocess.Popen([sys.executable, "-c", WORKER, str(path), f"p{n}", "40"])
            for n in range(8)]
    assert [job.wait(timeout=120) for job in jobs] == [0] * 8
    snap = BudgetLedger(path).snapshot()
    settled = [r for r in snap["requests"].values() if r["status"] == "settled"]
    assert len(snap["requests"]) == len(settled) >= 146          # filled to within one bound
    assert Decimal(snap["committed_usd"]) == Decimal("0.01") * len(settled) <= Decimal("1.5")
    running = BudgetLedger(path).summary()                       # running totals = fresh sums
    assert all(Decimal(str(running[k])) == Decimal(str(snap[k])) for k in running
               if k.endswith(("_usd", "_n")))


CRASH = r'''
import os, sys
import haenv.runtime.budget as sb
ledger = sb.BudgetLedger(sys.argv[1])
sb._Txn.write_meta = lambda self, before: os._exit(3)   # die after the row, before COMMIT
ledger.reserve("half", "0.5")
'''


def test_a_process_killed_mid_transaction_leaves_the_previous_state(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path, limit_usd="1")
    ledger.reserve("whole", "0.2")
    assert subprocess.run([sys.executable, "-c", CRASH, str(path)], timeout=60).returncode == 3
    assert ledger.record("half") is None and ledger.summary()["committed_usd"] == "0.2"
    ledger.reserve("next", "0.8")                                 # the write lock was released


def _history(ledger, key, costs):
    for i, cost in enumerate(costs):
        ledger.reserve(f"h{i}", "1", cost_key=key)
        ledger.settle(f"h{i}", cost)


def test_reservation_is_the_quantile_of_settled_history_once_it_is_long_enough(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    key = sb.cost_key("solver", "openrouter", "m", "out6000:in4096")
    _history(ledger, key, ["0.01"] * 29)
    assert ledger.reservation(key, "1") == Decimal("1")              # 29 < 30 observations
    _history_more = [f"0.0{d}" for d in range(1, 10)] * 8           # 72 more, up to $0.09
    for i, cost in enumerate(_history_more):
        ledger.reserve(f"m{i}", "1", cost_key=key)
        ledger.settle(f"m{i}", cost)
    assert ledger.reservation(key, "1") == Decimal("0.09")           # p99 of 101
    assert ledger.reservation(key, "0.05") == Decimal("0.05")        # never above the bound
    assert ledger.reservation(sb.cost_key("judge", "openrouter", "m"), "1") == Decimal("1")
    ledger.reserve("live", "1", cost_key=key)
    assert ledger.record("live")["reserved_usd"] == "0.09" and ledger.record("live")["bound_usd"] == "1"


def test_overrun_settles_the_true_charge_and_unknown_counts_the_worst_case(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    key = sb.cost_key("solver", "b", "m")
    _history(ledger, key, ["0.01"] * 30)
    ledger.reserve("over", "1", cost_key=key)
    ledger.settle("over", "0.6")                                      # above 0.01, within 1: no stop
    assert ledger.record("over")["overrun_usd"] == "0.59" and ledger.summary()["halt_reason"] is None
    ledger.reserve("lost", "1", cost_key=key)
    ledger.record_unknown("lost", reason="provider_cost_missing")
    assert ledger.record("lost")["reserved_usd"] == "1"
    ledger.reserve("wrong", "1", cost_key=key)
    with pytest.raises(BudgetExceeded):
        ledger.settle("wrong", "1.5")                                # above the worst case


@pytest.mark.parametrize("slots", [1, 3])
def test_adversarial_overruns_never_exceed_the_cap(tmp_path, monkeypatch, slots):
    """Every request overruns to its worst case. With at most `overrun_slots` requests in
    flight at once, the committed total never passes the cap."""
    declared = sb.budget_settings()
    monkeypatch.setattr(sb, "budget_settings", lambda: {**declared, "overrun_slots": slots})
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="10")
    key = sb.cost_key("solver", "b", "m")
    _history(ledger, key, ["0.01"] * 30)
    admitted, n = 0, 0
    while True:
        batch = []
        for _ in range(slots):
            try:
                ledger.reserve(f"x{n}", "1", cost_key=key)
                batch.append(f"x{n}")
            except BudgetCapReached:
                pass
            n += 1
        if not batch:
            break
        for rid in batch:
            ledger.settle(rid, "1")
            assert Decimal(ledger.summary()["committed_usd"]) <= 10
        admitted += len(batch)
    assert admitted >= 8 and Decimal(ledger.summary()["committed_usd"]) <= 10


def test_judges_and_solvers_share_one_pause_and_retry_path(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="1")
    ledger.reserve("busy", "0.9")
    now = [0.0]

    def sleep(s):                       # another process settles while this one waits
        now[0] += s
        if now[0] >= 60:
            ledger.settle("busy", "0.1")

    pause = ledger.reserve_or_pause("judge:1", "0.5", sleep=sleep, clock=lambda: now[0],
                                    poll_s=30, timeout_s=600)
    assert pause["resumed_after_s"] == 60 and ledger.record("judge:1")["status"] == "reserved"
    with pytest.raises(BudgetCapReached):
        ledger.reserve_or_pause("judge:2", "0.9", sleep=sleep, clock=lambda: now[0],
                                poll_s=30, timeout_s=0)
    ops = (tmp_path / "budget.json.ops.jsonl").read_text()
    assert "budget_paused" in ops and "budget_resumed" in ops
