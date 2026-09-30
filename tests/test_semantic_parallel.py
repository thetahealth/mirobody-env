"""Parallel grading must preserve consensus, cost limits and deterministic task identity."""
from dataclasses import asdict
import json
import threading

from haenv.semantic_budget import BudgetLedger
from haenv.semantic_judge import JudgeTask, JudgeReply
from haenv.semantic_parallel import MAX_JUDGE_LANES, run_cells


def records(n):
    out = []
    for i in range(n):
        task = JudgeTask("run", ("dx",), f"prompt {i}", "A", "judge", "v1")
        out.append({"key": str(i), "status": "ready", "task": asdict(task), "source": {"case": str(i)},
                    "rubric": {"groups": {"dx_hit": ["dx"]}, "not_applicable": {}, "version": "v1"}})
    return out


def test_parallel_cells_keep_votes_isolated_and_resume_without_calls(tmp_path):
    barrier = threading.Barrier(4); calls = [];lock = threading.Lock()
    def factory():
        def call(request):
            if request.index == 1:barrier.wait(timeout=3)
            with lock:calls.append(request.sample_id)
            raw = json.dumps({"verdicts": {"dx": {"label":"yes", "reason":"A", "evidence":["A"]}}})
            return JudgeReply(raw, False, None)
        return call
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    result = run_cells(records(4), tmp_path, "run", ledger, factory, concurrency=4)
    assert result["states"] == {"resolved": 4} and len(calls) == len(set(calls)) == 8
    second = run_cells(records(4), tmp_path, "run", ledger, factory, concurrency=4)
    assert second["new_cells"] == 0 and len(calls) == 8


def test_parallel_reservations_cannot_overspend_and_leave_pending(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=".1")
    calls = []
    def factory():
        def call(request):
            ledger.reserve(request.sample_id, ".1")
            calls.append(request.sample_id)
            ledger.settle(request.sample_id, ".1")
            return JudgeReply('{"verdicts":{"dx":{"label":"yes","reason":"A","evidence":[]}}}',False,None)
        return call
    result = run_cells(records(4), tmp_path, "run", ledger, factory, concurrency=4)
    assert len(calls) == 1
    assert result["status"] == "incomplete" and result["states"] == {"pending": 4}
    assert result["budget"]["committed_usd"] == "0.1"


def test_max_lanes_dispatch_each_sample_once_and_persist_every_cell(tmp_path):
    n = MAX_JUDGE_LANES
    barrier = threading.Barrier(n); calls = []; lock = threading.Lock()
    def factory():
        def call(request):
            if request.index == 0:barrier.wait(timeout=20)
            with lock:calls.append(request.sample_id)
            raw = json.dumps({"verdicts": {"dx": {"label":"yes", "reason":"A", "evidence":["A"]}}})
            return JudgeReply(raw, False, None)
        return call
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    result = run_cells(records(n), tmp_path, "run", ledger, factory, concurrency=n)
    assert result["states"] == {"resolved": n} and result["new_cells"] == n
    assert len(calls) == len(set(calls)) == 2 * n
    assert sorted(p.stem for p in (tmp_path / "results").glob("*.json")) == sorted(str(i) for i in range(n))
    for i in range(n):
        journal = [json.loads(x) for x in (tmp_path / "samples" / f"{i}.jsonl").read_text().split("\n") if x]
        assert len(journal) == 2
    again = run_cells(records(n), tmp_path, "run", ledger, factory, concurrency=n)
    assert again["new_cells"] == 0 and len(calls) == 2 * n
