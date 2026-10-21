"""A job that adds cases to a batch resumes it in paid mode: answered cells replay, only new
cases are bought; a job that changes a case the batch already holds is refused."""
import json

import pytest

import haenv.runtime.solver_accounting as accounting
import haenv.app.evaluate as ev

METADATA = {"id": "m", "pricing": {"prompt": ".000001", "completion": ".000002"},
            "context_length": 100000, "supported_parameters": [],
            "top_provider": {"max_completion_tokens": 2000}}


def _job(batch, n, job_sha, changed=None):
    rows = [{"case_id": f"case-{i}", "text": f"case {i}" + (" (edited)" if i == changed else "")}
            for i in range(n)]
    (batch / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"world_sha": "w", "job_sha256": job_sha}


def _prepare(batch, monkeypatch, source):
    monkeypatch.setattr(accounting, "fetch_catalog", lambda cfg: [METADATA])
    monkeypatch.setattr(accounting, "fetch_endpoints",
                        lambda cfg, model: [{"provider_name": "only", "pricing": METADATA["pricing"]}])
    solver = ev.OpenAICompatSolver("s", "m", "secret", 10, max_tokens=1000)
    run = accounting.prepare_accounting([("s", ev._const(solver))], {}, batch,
                                        ledger_path=batch / "budget.json", limit_usd="10",
                                        identity=source)
    return run, solver


def _ask(run, solver, case):
    run.bind(solver, case, "s")
    return solver._post(f"question for {case}")


def test_superset_job_resumes_and_buys_only_the_new_cases(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(ev.OpenAICompatSolver, "_send_wire",
                        lambda self, body, prompt, api_key=None: sent.append(prompt) or {
                            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"cost": 0.001}})
    run, solver = _prepare(tmp_path, monkeypatch, _job(tmp_path, 50, "job50"))
    first = _ask(run, solver, "case-7")
    assert len(sent) == 1
    run2, solver2 = _prepare(tmp_path, monkeypatch, _job(tmp_path, 100, "job100"))
    assert run2.run_id == run.run_id
    assert _ask(run2, solver2, "case-7") == first and len(sent) == 1      # replayed, not re-bought
    _ask(run2, solver2, "case-70")
    assert len(sent) == 2                                                  # the new case runs
    saved = json.loads((tmp_path / "solver-accounting" / "manifest.json").read_text())
    (extension,) = saved["case_extensions"]
    assert extension["added_cases"] == sorted(f"case-{i}" for i in range(50, 100))


def test_a_job_that_changes_a_held_case_is_refused(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch, _job(tmp_path, 50, "job50"))
    with pytest.raises(ValueError, match="run identity changed"):
        _prepare(tmp_path, monkeypatch, _job(tmp_path, 100, "job100", changed=3))
    with pytest.raises(ValueError, match="run identity changed"):
        _prepare(tmp_path, monkeypatch, _job(tmp_path, 40, "job40"))       # drops held cases
