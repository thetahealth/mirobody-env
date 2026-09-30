"""The common paid-call adapter preserves raw receipts and enforces one shared cap."""
from decimal import Decimal

import pytest

from haenv.paid_completion import AccountedCompletion
from haenv.semantic_budget import BudgetLedger, BudgetExceeded
from haenv.semantic_transport import PriceSchedule


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
    from haenv.paid_completion import ReconciledRequestFailure
    paid, ledger, settings = adapter(tmp_path)
    ledger.reserve("r1", ".1")
    with pytest.raises(ReconciledRequestFailure):
        paid.request("r1", "p", settings, lambda p:pytest.fail("lost request duplicated"))
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"


def test_transport_failure_stores_no_secret_exception_text(tmp_path):
    paid, ledger, settings = adapter(tmp_path)
    def fail(prompt):raise OSError("secret-token-do-not-log")
    from haenv.paid_completion import UnknownCostAttemptFailure
    with pytest.raises(UnknownCostAttemptFailure) as error:
        paid.request("r1", "p", settings, fail)
    assert "secret-token" not in str(error.value)
    assert "secret-token" not in (tmp_path / "ledger.json").read_text()
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
