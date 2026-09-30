"""An unknown charge is counted at its reserved bound and the run continues.

The only stop is a new reservation that would exceed the cap. Answers and votes keep
their meaning: a complete answer with unknown cost is valid; a failed call is an
ordinary failure retried under a new id; nothing is bought twice.
"""
from decimal import Decimal
import json
import logging

import pytest

from haenv.paid_completion import AccountedCompletion, ReconciledRequestFailure
from haenv.semantic_budget import BudgetExceeded, BudgetLedger, cost_summary
from haenv.semantic_judge import JudgeRequest, JudgeTask, evaluate_consensus
from haenv.semantic_transport import PricedJudge, PriceSchedule, ReconciledCallFailure


# ---------------------------------------------------------------- ledger

def test_unknown_is_counted_at_its_bound_without_a_halt(tmp_path, caplog):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    ledger.reserve("a", ".3")
    with caplog.at_level(logging.WARNING):
        ledger.settle("a", None, reason="provider_cost_missing")
    snap = ledger.snapshot()
    record = snap["requests"]["a"]
    assert record["status"] == "unknown_capped" and record["actual_usd"] is None
    assert "provider_cost_missing" in record["reconciliation"]
    assert snap["halt_reason"] is None and snap["committed_usd"] == "0.3"
    assert snap["unknown_capped_n"] == 1 and snap["unknown_capped_usd"] == "0.3"
    assert any("cost unknown" in r.getMessage() and "a" in r.getMessage() for r in caplog.records)
    ledger.reserve("b", ".3")                     # the next request is not blocked


def test_cap_still_binds_with_unknowns_counted_at_their_bounds(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    for name in ("a", "b", "c"):
        ledger.reserve(name, ".3")
        ledger.settle(name, None, reason="dispatch_failed_no_receipt")
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve("d", ".11")                 # .9 + .11 > 1
    assert "budget" in str(error.value).lower() and "unknown" not in str(error.value).lower()
    ledger.reserve("e", ".1")                      # exactly at the cap is allowed
    assert ledger.snapshot()["committed_usd"] == "1.0"


def test_a_later_receipt_settles_the_bound_and_frees_the_difference(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    ledger.reserve("a", ".5")
    ledger.settle("a", None, reason="provider_cost_missing")
    ledger.settle_unknown_later("a", ".02", evidence="generation record gen-1 total_cost")
    snap = ledger.snapshot()
    record = snap["requests"]["a"]
    assert record["status"] == "settled" and record["actual_usd"] == "0.02"
    assert Decimal(record["settled_after_unknown"]["bound_usd"]) == Decimal(".5")
    assert snap["committed_usd"] == "0.02" and snap["unknown_capped_n"] == 0
    with pytest.raises(ValueError):
        ledger.settle_unknown_later("a", ".01", evidence="again")   # only unknown_capped
    ledger.reserve("b", ".9")                      # the freed difference is usable


def test_a_later_receipt_above_the_bound_still_stops(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    ledger.reserve("a", ".1")
    ledger.settle("a", None, reason="provider_cost_missing")
    with pytest.raises(BudgetExceeded):
        ledger.settle_unknown_later("a", ".2", evidence="generation record")
    assert ledger.snapshot()["halt_reason"] == "Provider charge exceeded the reserved upper bound"


def test_cost_summary_names_paid_and_unknown_separately(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    ledger.reserve("a", ".2"); ledger.settle("a", ".01")
    ledger.reserve("b", ".3"); ledger.settle("b", None, reason="provider_cost_missing")
    text = cost_summary(ledger.snapshot())
    assert "paid $0.01" in text and "1 unknown" in text and "$0.3" in text


def test_legacy_unknown_record_is_closed_on_encounter(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd="1")
    ledger.reserve("old", ".2")
    with ledger._locked() as state:                # a ledger carrying a halted "unknown" record
        state["requests"]["old"]["status"] = "unknown"
        state["halt_reason"] = "A billed request has unknown cost; reconciliation required"
        ledger._write(state)
    ledger.record_unknown("old", reason="legacy_unknown_carried")
    snap = ledger.snapshot()
    assert snap["requests"]["old"]["status"] == "unknown_capped" and snap["halt_reason"] is None


# ---------------------------------------------------------------- judge path

def _metadata():
    return {"id": "openai/gpt-6-luna", "context_length": 100000,
            "supported_parameters": ["reasoning"],
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}}


class _JudgeSolver:
    model, reasoning_effort, max_tokens, pool = "openai/gpt-6-luna", "high", 1000, None

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def _post(self, prompt):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _ok(content="{}", usage=None):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": usage}


def _judge(tmp_path, replies, limit="10"):
    ledger = BudgetLedger(tmp_path / "b.json", limit_usd=limit)
    solver = _JudgeSolver(replies)
    judge = PricedJudge(solver, PriceSchedule.from_metadata(_metadata(), solver.model), ledger,
                        receipts=tmp_path / "receipts")
    return judge, ledger, solver


def test_judge_vote_with_missing_cost_is_valid_and_the_next_call_goes_out(tmp_path):
    judge, ledger, solver = _judge(tmp_path, [_ok(usage={"prompt_tokens": 1}), _ok(usage={"cost": .0001})])
    assert judge(JudgeRequest("s:1", 1, "p")).raw == "{}"
    assert ledger.snapshot()["requests"]["s:1"]["status"] == "unknown_capped"
    assert judge(JudgeRequest("s:2", 2, "p")).raw == "{}" and solver.calls == 2
    assert ledger.snapshot()["halt_reason"] is None


def test_judge_failed_call_is_an_ordinary_failure_and_is_not_rebought(tmp_path):
    judge, ledger, solver = _judge(tmp_path, [OSError("secret-url"), _ok(usage={"cost": .0001})])
    with pytest.raises(RuntimeError) as error:
        judge(JudgeRequest("s:1", 1, "p"))
    assert not isinstance(error.value, BudgetExceeded) and "secret-url" not in str(error.value)
    assert ledger.snapshot()["requests"]["s:1"]["status"] == "unknown_capped"
    with pytest.raises(ReconciledCallFailure):          # resume: same id is never bought again
        judge(JudgeRequest("s:1", 1, "p"))
    judge(JudgeRequest("s:2", 2, "p"))
    assert solver.calls == 2


def test_judge_reservation_without_receipt_closes_at_bound_and_continues(tmp_path):
    judge, ledger, solver = _judge(tmp_path, [_ok(usage={"cost": .0001})])
    ledger.reserve("s:1", ".1")                         # issuing process died in flight
    with pytest.raises(ReconciledCallFailure):
        judge(JudgeRequest("s:1", 1, "p"))
    assert solver.calls == 0
    assert ledger.snapshot()["requests"]["s:1"]["status"] == "unknown_capped"
    judge(JudgeRequest("s:2", 2, "p"))
    assert solver.calls == 1


def test_judge_cost_unknown_vote_replays_without_a_second_purchase(tmp_path):
    judge, ledger, solver = _judge(tmp_path, [_ok(usage={"prompt_tokens": 1})])
    first = judge(JudgeRequest("s:1", 1, "p"))
    again = judge(JudgeRequest("s:1", 1, "p"))
    assert first.raw == again.raw and solver.calls == 1
    assert len(ledger.snapshot()["requests"]) == 1


def test_consensus_continues_past_a_failed_unknown_cost_call(tmp_path):
    verdict = json.dumps({"verdicts": {"a": {"label": "yes", "reason": "rubric applied",
                                              "evidence": ["Possible A."]}}})
    judge, ledger, solver = _judge(tmp_path, [OSError("x"), _ok(verdict, {"cost": .0001}),
                                              _ok(verdict, {"cost": .0001})])
    task = JudgeTask(run_id="run-a", item_ids=("a",), prompt="p",
                     answer_text="Possible A. Evidence E1 supports A.",
                     judge_model="test-judge", rubric_version="v1")
    try:
        result = evaluate_consensus(task, judge)
    except BudgetExceeded:                              # a stop here is a failure
        pytest.fail("an unknown charge stopped the consensus")
    statuses = [s["status"] for s in result["samples"]]
    assert statuses[0] == "call_error" and solver.calls == 3
    assert ledger.snapshot()["halt_reason"] is None


# ---------------------------------------------------------------- solver path

def _adapter(tmp_path, limit="1"):
    ledger = BudgetLedger(tmp_path / "ledger.json", limit_usd=limit)
    prices = PriceSchedule("model", Decimal(".000001"), Decimal(".000002"), Decimal(0), 100000)
    paid = AccountedCompletion(prices, ledger, tmp_path / "receipts")
    settings = {"model": "model", "backend": "openrouter", "max_tokens": 1000,
                "reasoning_effort": None, "response_format": None}
    return paid, ledger, settings


def test_solver_answer_with_unknown_cost_is_returned_and_the_next_request_goes_out(tmp_path):
    paid, ledger, settings = _adapter(tmp_path)
    answer = {"id": "gen", "choices": [{"message": {"content": "ok"}}]}   # stream lost usage
    assert paid.request("r1", "p", settings, lambda p: answer) == answer
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"
    assert paid.request("r2", "p", settings, lambda p: {"usage": {"cost": .0001}})
    assert paid.request("r1", "p", settings, lambda p: pytest.fail("re-bought")) == answer


def test_solver_failed_dispatch_is_an_ordinary_failed_attempt(tmp_path):
    paid, ledger, settings = _adapter(tmp_path)
    def fail(prompt): raise OSError("secret-token")
    with pytest.raises(Exception) as error:
        paid.request("r1", "p", settings, fail)
    assert not isinstance(error.value, BudgetExceeded) and "secret-token" not in str(error.value)
    assert "secret-token" not in (tmp_path / "ledger.json").read_text()
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"
    with pytest.raises(ReconciledRequestFailure):
        paid.request("r1", "p", settings, lambda p: pytest.fail("re-bought"))
    paid.request("r2", "p", settings, lambda p: {"usage": {"cost": .0001}})


def test_solver_reservation_without_receipt_closes_at_bound(tmp_path):
    paid, ledger, settings = _adapter(tmp_path)
    ledger.reserve("r1", ".1")
    with pytest.raises(ReconciledRequestFailure):
        paid.request("r1", "p", settings, lambda p: pytest.fail("lost request duplicated"))
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"


def test_solver_retry_loop_continues_under_a_new_request_id(tmp_path):
    import threading
    from haenv.solver_accounting import SolverRequestAccountant
    paid, ledger, settings = _adapter(tmp_path)
    stop = threading.Event()
    cursor = SolverRequestAccountant(paid, "cell", stop)
    class S: pool = None
    import haenv.solver_accounting as sa
    original = sa.wire_settings
    sa.wire_settings = lambda solver: settings
    try:
        def fail(prompt): raise TimeoutError("t")
        with pytest.raises(Exception):
            cursor.request(S(), "p", fail)
        assert not stop.is_set()
        assert cursor.request(S(), "p", lambda p: {"usage": {"cost": .0001}}) == {"usage": {"cost": .0001}}
    finally:
        sa.wire_settings = original
    assert set(ledger.snapshot()["requests"]) == {"solver:cell:1", "solver:cell:2"}


def test_solver_cap_is_the_only_stop(tmp_path):
    paid, ledger, settings = _adapter(tmp_path, limit=".005")
    paid.request("r1", "p", settings, lambda p: {"id": "g", "choices": []})    # unknown, bound ~.002
    with pytest.raises(BudgetExceeded) as error:
        paid.request("r2", "p" * 2000, settings, lambda p: pytest.fail("over the cap"))
    assert "budget" in str(error.value).lower()
