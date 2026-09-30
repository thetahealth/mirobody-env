"""Paid solver attempts share the judge cap without changing sampling conditions."""
import json
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

from haenv import evaluate as ev
from haenv.semantic_budget import BudgetExceeded, BudgetLedger
from haenv.semantic_transport import PriceSchedule
from haenv.solver_accounting import SolverRequestAccountant, prepare_accounting
from haenv.paid_completion import AccountedCompletion


def bound(tmp_path, *, limit="1", cell="a"):
    solver = ev.OpenAICompatSolver("s", "model", "secret", 10, retries=2,
                                   max_tokens=1000, reasoning_effort="medium")
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=limit)
    prices = PriceSchedule("model", Decimal(".000001"), Decimal(".000002"), Decimal(0), 100000)
    solver._accounting = SolverRequestAccountant(
        AccountedCompletion(prices, ledger, tmp_path / "receipts"), cell, threading.Event())
    return solver, ledger


def test_real_post_reserves_before_wire_and_replay_is_free(tmp_path, monkeypatch):
    s, ledger = bound(tmp_path)
    calls = []
    def wire(self, prompt):
        assert any(r["status"] == "reserved" for r in ledger.snapshot()["requests"].values())
        calls.append((prompt, self.max_tokens, self.reasoning_effort))
        return {"usage": {"cost": .001}, "choices": []}
    monkeypatch.setattr(ev.OpenAICompatSolver, "_post_wire", wire)
    s._post("question")
    replay, _ = bound(tmp_path)
    replay._post("question")
    assert calls == [("question", 1000, "medium")]
    assert ledger.snapshot()["settled_usd"] == "0.001"


def test_each_step_and_each_cell_has_separate_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.OpenAICompatSolver, "_post_wire",
                        lambda self, p: {"usage": {"cost": .001}, "choices": []})
    s, ledger = bound(tmp_path)
    s._post("same")
    s._post("same")
    other, _ = bound(tmp_path, cell="b")
    other._post("same")
    assert len(ledger.snapshot()["requests"]) == 3


@pytest.mark.parametrize("kind", ["compat", "google"])
def test_solve_does_not_swallow_budget_stop(tmp_path, monkeypatch, kind):
    s = (ev.GoogleSolver("s", "m", "secret", 10, 2) if kind == "google"
         else bound(tmp_path)[0])
    calls = []
    def stop(prompt):
        calls.append(prompt)
        raise BudgetExceeded("stop")
    monkeypatch.setattr(s, "_post", stop)
    monkeypatch.setattr(ev, "render_for", lambda *a: ("question", "probe"))
    monkeypatch.setattr(ev.time, "sleep", lambda *a: pytest.fail("retry after budget stop"))
    with pytest.raises(BudgetExceeded):
        s.solve(object())
    assert calls == ["question"]


def test_budget_denial_never_reaches_wire(tmp_path, monkeypatch):
    s, _ = bound(tmp_path, limit=".000001")
    monkeypatch.setattr(ev.OpenAICompatSolver, "_post_wire", lambda *a: pytest.fail("unreserved"))
    with pytest.raises(BudgetExceeded):
        s._post("q")
    assert s._accounting.stop.is_set()


def test_http_200_error_is_costed_before_answer_retry(tmp_path, monkeypatch):
    s, ledger = bound(tmp_path)
    monkeypatch.setattr(ev.OpenAICompatSolver, "_post_wire", lambda *a:
                        {"usage": {"cost": .001}, "error": {"message": "bad"}})
    with pytest.raises(RuntimeError, match="error body"):
        s._post("q")
    assert ledger.snapshot()["settled_usd"] == "0.001"
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1


def test_unsupported_route_refuses_entire_preflight(tmp_path, monkeypatch):
    native = ev.OpenAICompatSolver("unsupported", "m", "secret", 10, backend="unverified")
    with pytest.raises(ValueError, match="unverified.*unsupported"):
        prepare_accounting([("unsupported", lambda: native)], {}, tmp_path,
                           ledger_path=tmp_path / "budget.json", limit_usd="1", identity={})
    assert not (tmp_path / "solver-accounting" / "receipts").exists()


def test_binding_disables_key_rotation_and_pins_run(tmp_path, monkeypatch):
    import haenv.solver_accounting as accounting
    s = ev.OpenAICompatSolver("s", "m", "secret", 10, max_tokens=1000)
    s.pool = object()
    metadata = {"id": "m", "pricing": {"prompt": ".000001", "completion": ".000002"},
                "context_length": 100000, "supported_parameters": [],
                "top_provider": {"max_completion_tokens": 2000}}
    monkeypatch.setattr(accounting, "fetch_catalog", lambda cfg: [metadata])
    monkeypatch.setattr(accounting, "fetch_endpoints",
                        lambda cfg, model: [{"provider_name": "only", "pricing": metadata["pricing"]}])
    args = {"ledger_path": tmp_path / "budget.json", "limit_usd": "1", "identity": {"world": "w"}}
    run = prepare_accounting([("s", ev._const(s))], {}, tmp_path, **args)
    a, b = ev._const(s)(), ev._const(s)()
    run.bind(a, "c1", "s")
    run.bind(b, "c2", "s")
    assert a.pool is None and b.pool is None and s.pool is not None
    assert a._accounting is not b._accounting
    args["identity"] = {"world": "changed"}
    with pytest.raises(ValueError, match="identity"):
        prepare_accounting([("s", ev._const(s))], {}, tmp_path, **args)


def test_solver_prices_do_not_require_judge_reasoning_capability():
    metadata = {"id": "m", "pricing": {"prompt": ".000001", "completion": ".000002"},
                "context_length": 100000, "supported_parameters": []}
    with pytest.raises(ValueError, match="reasoning"):
        PriceSchedule.from_metadata(metadata, "m")
    assert PriceSchedule.from_metadata(metadata, "m", require_reasoning=False).model == "m"


def test_parallel_budget_stop_persists_inflight_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.RUN, "workers", 2)
    monkeypatch.setattr(ev, "_backend_of", lambda n: "openrouter")
    started = threading.Barrier(2)
    def work(i):
        started.wait(timeout=2)
        if i == 0:
            raise BudgetExceeded("cap")
        import time
        time.sleep(.05)  # The paid stop reaches the consumer before this valid result.
        return {"case": "survived", "solver": "s"}
    out = tmp_path / "eval.jsonl"
    with pytest.raises(BudgetExceeded):
        ev._run_tasks(work, [0, 1], out, key_of=lambda t: "s")
    assert json.loads(out.read_text())["case"] == "survived"


def test_run_eval_binds_real_solver_to_shared_ledger(tmp_path, monkeypatch):
    import haenv.solver_accounting as accounting
    import haenv.batch as batch
    s = ev.OpenAICompatSolver("s", "m", "secret", 10, max_tokens=1000)
    metadata = {"id": "m", "pricing": {"prompt": ".000001", "completion": ".000002"},
                "context_length": 100000, "top_provider": {"max_completion_tokens": 2000}}
    monkeypatch.setattr(accounting, "fetch_catalog", lambda cfg: [metadata])
    monkeypatch.setattr(accounting, "fetch_endpoints",
                        lambda cfg, model: [{"provider_name": "only", "pricing": metadata["pricing"]}])
    monkeypatch.setattr(batch, "provenance_fields", lambda *a, **k: {"world_sha": "w"})
    monkeypatch.setattr(batch, "record_usage", lambda *a: None)
    monkeypatch.setattr(ev, "build_solvers", lambda *a: [("s", ev._const(s))])
    monkeypatch.setattr(ev, "_preflight_quota", lambda *a, **k: None)
    monkeypatch.setattr(ev, "load_probes", lambda *a: {})
    monkeypatch.setattr(ev.RUN, "resp_path", None)
    monkeypatch.setattr(ev.RUN, "workers", 1)
    def single(cid, name, raw, t, solver):
        assert solver._accounting.paid.ledger.path == (tmp_path / "shared.json").resolve()
        solver._post("question")
        return {"case": cid, "solver": name, "overall": "SCORED"}
    monkeypatch.setattr(ev, "_row_single", single)
    monkeypatch.setattr(ev.OpenAICompatSolver, "_post_wire", lambda *a:
                        {"usage": {"cost": .001}, "choices": []})
    job = SimpleNamespace(job_id="account-test", results_file=tmp_path / "eval.jsonl",
                          root=tmp_path, geometry="single", slices="", tool_mode="")
    rows = ev.run_eval(job, {}, {"c": SimpleNamespace(prediction_context={"prediction_time_T": 1})},
                       budget_ledger=tmp_path / "shared.json", budget_usd="1")
    assert rows[0]["overall"] == "SCORED"
    assert BudgetLedger(tmp_path / "shared.json", limit_usd="1").snapshot()["settled_usd"] == "0.001"


def test_output_cap_is_rejected_not_clamped(tmp_path, monkeypatch):
    import haenv.solver_accounting as accounting
    s = ev.OpenAICompatSolver("s", "m", "secret", 10, max_tokens=72000)
    metadata = {"id": "m", "pricing": {"prompt": ".000001", "completion": ".000002"},
                "context_length": 100000, "top_provider": {"max_completion_tokens": 65536}}
    monkeypatch.setattr(accounting, "fetch_catalog", lambda cfg: [metadata])
    monkeypatch.setattr(accounting, "fetch_endpoints",
                        lambda cfg, model: [{"provider_name": "only", "pricing": metadata["pricing"]}])
    with pytest.raises(ValueError, match="exceeds provider limit"):
        prepare_accounting([("s", lambda: s)], {}, tmp_path,
                           ledger_path=tmp_path / "budget.json", limit_usd="1", identity={})
    assert s.max_tokens == 72000
