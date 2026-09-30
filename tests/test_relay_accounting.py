"""Relay charges are matched to exact gateway request receipts, never guessed."""
from decimal import Decimal
import json

import pytest

from haenv.paid_completion import AccountedCompletion
from haenv.relay_accounting import RelayPrices, RelayBilling
from haenv.semantic_budget import BudgetLedger, BudgetExceeded


def metadata():
    return {"model": {"model_name": "kimi-k3", "quota_type": 0, "model_ratio": 1.5,
                      "completion_ratio": 5, "cache_ratio": .1},
            "group_ratio": {"default": 1, "second": 1}, "billing_rule_summaries": {},
            "quota_per_unit": 500000, "quota_display_type": "USD", "pricing_version": "test"}


def response():
    return {"_gateway_request_id": "request-one", "usage": {"cost": 0},
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}


def logrow():
    return {"request_id": "request-one", "model_name": "kimi-k3", "type": 2,
            "quota": 110541, "prompt_tokens": 44794, "completion_tokens": 5780,
            "other": json.dumps({"model_ratio": 1.5, "completion_ratio": 5,
                                 "cache_ratio": .1, "group_ratio": 1}),
            "token_name": "do-not-store", "ip": "private-ip", "username": "private-name"}


def make(tmp_path, logs):
    prices = RelayPrices.from_metadata(metadata(), "kimi-k3")
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="10")
    billing = RelayBilling(prices, lambda: logs)
    paid = AccountedCompletion(prices, ledger, tmp_path / "receipts", backend="relay", cost_reader=billing)
    settings = {"backend": "relay", "model": "kimi-k3", "max_tokens": 72000}
    return paid, ledger, settings


def test_verified_relay_bound_uses_highest_group_and_cache_prices():
    doc = metadata(); doc["group_ratio"]["second"] = 2; doc["model"]["cache_ratio"] = 3
    p = RelayPrices.from_metadata(doc, "kimi-k3")
    assert p.input_rate == Decimal(".000018")
    assert p.output_rate == Decimal(".000030")
    assert p.upper_bound("q", 72000) > Decimal("2.16")


@pytest.mark.parametrize("key,value", [("quota_display_type", "CNY"), ("quota_per_unit", 0),
    ("billing_rule_summaries", {"kimi-*": {"unknown": True}}), ("group_ratio", {})])
def test_unverified_denominations_or_rules_refuse_preflight(key, value):
    doc = metadata(); doc[key] = value
    with pytest.raises(ValueError):
        RelayPrices.from_metadata(doc, "kimi-k3")


def test_only_gateway_billing_receipt_settles_not_fake_usage_cost(tmp_path):
    p, ledger, settings = make(tmp_path, [logrow()])
    p.request("r1", "q", settings, lambda _: response())
    assert ledger.snapshot()["settled_usd"] == "0.221082"
    receipts = [json.loads(x.read_text()) for x in (tmp_path / "receipts").glob("*.json")]
    assert len(receipts) == 2
    serialized = json.dumps(receipts)
    assert "private-ip" not in serialized and "do-not-store" not in serialized
    assert any(x.get("kind") == "relay_quota_receipt" for x in receipts)


@pytest.mark.parametrize("change", ["wrong_id", "wrong_model", "duplicate", "missing_id"])
def test_wrong_or_ambiguous_receipt_keeps_reservation_and_raw_answer(tmp_path, change):
    row = logrow(); resp = response(); logs = [row]
    if change == "wrong_id":row["request_id"] = "other"
    if change == "wrong_model":row["model_name"] = "other"
    if change == "duplicate":logs.append(dict(row))
    if change == "missing_id":resp.pop("_gateway_request_id")
    p, ledger, settings = make(tmp_path, logs)
    assert p.request("r1", "q", settings, lambda _: resp) == resp
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"
    assert next((tmp_path / "receipts").glob("*.json")).is_file()


def test_replay_uses_durable_cost_receipt_without_network_or_log_lookup(tmp_path):
    p, ledger, settings = make(tmp_path, [logrow()])
    p.request("r1", "q", settings, lambda _: response())
    p.cost_reader = lambda _: pytest.fail("settled replay queried billing")
    assert p.request("r1", "q", settings, lambda _: pytest.fail("duplicate generation")) == response()


def test_billing_lookup_failure_does_not_erase_completion(tmp_path):
    p, ledger, settings = make(tmp_path, [])
    def fail(_):raise OSError("secret URL must not be logged")
    p.cost_reader = fail
    assert p.request("r1", "q", settings, lambda _: response()) == response()
    assert "secret" not in (tmp_path / "budget.json").read_text()
    assert ledger.snapshot()["requests"]["r1"]["status"] == "unknown_capped"
    saved = json.loads(next((tmp_path / "receipts").glob("*.json")).read_text())
    assert saved["response"] == response()


def test_budgeted_stream_keeps_cost_even_when_no_answer(tmp_path, monkeypatch):
    from haenv.evaluate import OpenAICompatSolver
    from haenv.solver_accounting import SolverRequestAccountant
    import threading
    p, ledger, _ = make(tmp_path, [logrow()])
    solver = OpenAICompatSolver("s", "kimi-k3", "secret", 10, max_tokens=72000,
                                stream=True, backend="relay")
    solver._accounting = SolverRequestAccountant(p, "cell", threading.Event())
    class Stream:
        headers = {"X-Oneapi-Request-Id": "request-one"}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning":"reason"},"finish_reason":"length"}],"usage":{"completion_tokens":6}}'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a,**k:Stream())
    with pytest.raises(ValueError, match="only reasoning|只有推理"):
        solver._post("q")
    assert ledger.snapshot()["settled_usd"] == "0.221082"


def test_preflight_and_bind_relay_use_its_verified_receipt_reader(tmp_path, monkeypatch):
    from haenv.evaluate import OpenAICompatSolver, _const
    from haenv.solver_accounting import prepare_accounting
    import haenv.relay_accounting as relay
    solver = OpenAICompatSolver("s", "kimi-k3", "secret", 10, max_tokens=72000, backend="relay")
    solver.pool = object()
    monkeypatch.setattr(relay, "fetch_metadata", lambda *a: metadata())
    monkeypatch.setattr(relay.RelayBilling, "for_solver",
                        lambda prices, *a: RelayBilling(prices, lambda:[logrow()]))
    run = prepare_accounting([("s", _const(solver))], {}, tmp_path,
                             ledger_path=tmp_path / "budget.json", limit_usd="10", identity={})
    run.bind(solver, "case", "s")
    monkeypatch.setattr(solver, "_post_wire", lambda p: response())
    assert solver._post("q") == response()
    assert solver.pool is None
    assert run.ledger.snapshot()["settled_usd"] == "0.221082"


def test_receipt_cannot_be_rebound_to_a_changed_raw_completion(tmp_path):
    p, ledger, settings = make(tmp_path, [logrow()])
    p.request("r1", "q", settings, lambda _: response())
    raw = next(x for x in (tmp_path / "receipts").glob("*.json") if not x.name.endswith('.billing.json'))
    saved = json.loads(raw.read_text());saved['response']['choices'][0]['message']['content'] = 'tampered'
    raw.write_text(json.dumps(saved))
    with pytest.raises(BudgetExceeded):
        p.request("r1", "q", settings, lambda _: pytest.fail("duplicate"))
