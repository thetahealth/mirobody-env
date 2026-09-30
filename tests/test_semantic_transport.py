"""The semantic transport binds fresh calls to verified pricing and the shared cap."""
from decimal import Decimal

import pytest

from haenv.semantic_budget import BudgetExceeded, BudgetLedger
from haenv.semantic_judge import JudgeRequest
from haenv.semantic_transport import PricedJudge, PriceSchedule


def metadata():
    return {"id": "openai/gpt-6-luna", "context_length": 100000,
            "supported_parameters": ["reasoning", "max_tokens", "response_format", "structured_outputs"],
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005",
                        "overrides": [{"min_prompt_tokens": 272000,
                                       "prompt": "0.0000002", "completion": "0.00000075"}]}}


class Solver:
    model = "openai/gpt-6-luna"
    reasoning_effort = "high"
    max_tokens = 12000
    pool = None

    def __init__(self, usage):
        self.usage = usage
        self.calls = 0

    def _post(self, prompt):
        self.calls += 1
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": self.usage}


def test_price_uses_all_tiers_conservatively():
    price = PriceSchedule.from_metadata(metadata(), "openai/gpt-6-luna")
    assert price.input_rate == Decimal("0.0000002")
    assert price.output_rate == Decimal("0.00000075")
    assert price.upper_bound("abc", 12000) > Decimal(".009")


def test_cache_write_price_cannot_escape_the_input_upper_bound():
    doc = metadata();doc["pricing"]["overrides"][0]["input_cache_write"] = "0.00000025"
    price = PriceSchedule.from_metadata(doc, "openai/gpt-6-luna")
    assert price.input_rate == Decimal("0.00000025")


def test_live_sample_settles_cost_without_cache(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    solver = Solver({"cost": .001, "prompt_tokens": 10, "completion_tokens": 10})
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger)
    answer = judge(JudgeRequest("one", 1, "p"))
    assert answer.cached is False and answer.raw == "{}" and solver.calls == 1
    assert Decimal(ledger.snapshot()["committed_usd"]) == Decimal(".001")


def test_budget_rejected_before_network(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=".00001")
    solver = Solver({"cost": .001})
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger)
    with pytest.raises(BudgetExceeded):
        judge(JudgeRequest("one", 1, "p"))
    assert solver.calls == 0


def test_missing_billed_cost_is_counted_at_bound_and_the_next_call_proceeds(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    solver = Solver({"prompt_tokens": 10})
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger)
    assert judge(JudgeRequest("one", 1, "p")).raw == "{}"      # the vote is valid
    record = ledger.snapshot()["requests"]["one"]
    assert record["status"] == "unknown_capped" and record["actual_usd"] is None
    judge(JudgeRequest("two", 2, "p"))
    assert solver.calls == 2 and ledger.snapshot()["halt_reason"] is None


@pytest.mark.parametrize("field,value", [("model", "other-model"), ("reasoning_effort", "low"),
                                        ("pool", object())])
def test_model_effort_and_no_hidden_retry_enforced(tmp_path, field, value):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    solver = Solver({"cost": .001})
    setattr(solver, field, value)
    with pytest.raises(ValueError):
        PricedJudge(solver, PriceSchedule.from_metadata(metadata(), "openai/gpt-6-luna"), ledger)


def test_missing_or_wrong_model_pricing_does_not_guess():
    with pytest.raises(ValueError):
        PriceSchedule.from_metadata({"id": "other"}, "openai/gpt-6-luna")
    doc = metadata(); del doc["pricing"]["prompt"]
    with pytest.raises(ValueError):
        PriceSchedule.from_metadata(doc, "openai/gpt-6-luna")


def test_durable_receipt_recovers_same_vote_without_another_charge(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    solver = Solver({"cost": .001})
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger,
                        receipts=tmp_path / "receipts")
    request = JudgeRequest("same", 1, "p")
    assert judge(request) == judge(request)
    assert solver.calls == 1 and len(ledger.snapshot()["requests"]) == 1


def test_crash_without_receipt_never_retries_a_reserved_vote(tmp_path):
    from haenv.semantic_transport import ReconciledCallFailure
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    ledger.reserve("same", ".1")
    solver = Solver({"cost": .001})
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger)
    with pytest.raises(ReconciledCallFailure):
        judge(JudgeRequest("same", 1, "p"))
    assert solver.calls == 0
    assert ledger.snapshot()["requests"]["same"]["status"] == "unknown_capped"


def test_invalid_response_keeps_provider_receipt_and_exact_charge(tmp_path):
    import json
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="100")
    solver = Solver({"cost": .001})
    solver._post = lambda prompt: {"id": "provider-id", "usage": {"cost": .001},
        "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}]}
    judge = PricedJudge(solver, PriceSchedule.from_metadata(metadata(), solver.model), ledger,
                        receipts=tmp_path / "receipts")
    with pytest.raises(ValueError):
        judge(JudgeRequest("one", 1, "p"))
    receipt = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert receipt["response"]["id"] == "provider-id"
    assert receipt["response"]["choices"][0]["message"]["content"] == "truncated"
    assert ledger.snapshot()["committed_usd"] == "0.001"
