"""The default semantic path cannot silently skip judgments or drift mid-run."""
import json
from pathlib import Path

import pytest

from haenv import semantic_pipeline as pipeline


def test_default_online_cli_refuses_before_eval_without_budget(monkeypatch, capsys):
    from haenv import cli
    from haenv import evaluate
    monkeypatch.setattr(evaluate, "run_eval", lambda *a, **k: pytest.fail("Paid evaluation ran without budget"))
    assert cli.main(["run", "inputs/example-ew.job.yaml", "--models", "gpt-6-luna"]) == 2
    assert "default LLM semantic judging" in capsys.readouterr().out


def test_offline_default_never_prepares_or_calls_models(monkeypatch):
    monkeypatch.setattr(pipeline, "prepare_batches", lambda *a, **k: pytest.fail("offline prepared paid work"))
    monkeypatch.setattr(pipeline, "execute_run", lambda *a, **k: pytest.fail("offline called model"))
    result = pipeline.default_after_run(Path("unused"), {}, offline=True,
                                       budget_ledger=None, budget_usd=None)
    assert result == {"status": "not_run_offline", "semantic_scores_available": False}


def test_online_default_requires_explicit_budget_before_dispatch():
    with pytest.raises(ValueError, match="budget"):
        pipeline.default_after_run(Path("unused"), {}, offline=False,
                                   budget_ledger=None, budget_usd=None)


def test_online_default_prepares_seals_and_executes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, "load_policy", lambda: {"version": "v1"})
    monkeypatch.setattr(pipeline, "prepare_batches", lambda *a, **k: calls.append("prepare"))
    monkeypatch.setattr(pipeline, "seal_run", lambda *a: calls.append("seal"))
    def execute(*a, **k):
        calls.append("execute")
        assert k["limit_usd"] == "100"
        return {"budget": {"halt_reason": None}, "states": {"resolved": 2}}
    monkeypatch.setattr(pipeline, "execute_run", execute)
    result = pipeline.default_after_run(tmp_path / "batch", {}, offline=False,
                                       budget_ledger=tmp_path / "budget.json", budget_usd="100")
    assert calls == ["prepare", "seal", "execute"]
    assert result["status"] == "completed"


@pytest.mark.parametrize("state", ["unresolved", "unparseable_response", "missing_response", "empty_response"])
def test_unresolved_judgments_cannot_report_complete(monkeypatch, tmp_path, state):
    monkeypatch.setattr(pipeline, "load_policy", lambda: {"version": "v1"})
    monkeypatch.setattr(pipeline, "prepare_batches", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "seal_run", lambda *a: None)
    monkeypatch.setattr(pipeline, "execute_run", lambda *a, **k: {
        "budget": {"halt_reason": None}, "states": {state: 1}})
    result = pipeline.default_after_run(tmp_path / "batch", {}, offline=False,
                                       budget_ledger=tmp_path / "budget.json", budget_usd="100")
    assert result["status"] == "incomplete"


def test_unsealed_or_modified_inputs_refuse_to_run(monkeypatch, tmp_path):
    tasks = tmp_path / "tasks.jsonl"; tasks.write_text('{}\n')
    manifest = {"sealed": False, "code": {"v": 1},
                "tasks_sha256": pipeline.digest(tasks), "sources": []}
    file = tmp_path / "manifest.json";file.write_text(json.dumps(manifest))
    monkeypatch.setattr(pipeline, "code_state", lambda: {"v": 1})
    with pytest.raises(ValueError, match="sealed"):
        pipeline.validate_run(tmp_path)
    manifest["sealed"] = True;file.write_text(json.dumps(manifest))
    assert pipeline.validate_run(tmp_path)["sealed"] is True
    tasks.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="drifted"):
        pipeline.validate_run(tmp_path)


def fake_run(monkeypatch, tmp_path):
    from haenv import evaluate
    from haenv.semantic_judge import JudgeTask
    from test_semantic_transport import Solver, metadata
    monkeypatch.setattr(pipeline, "validate_run", lambda out: {
        "run_id": "test", "policy": {"judge": {"model_key": "judge", "model_id": Solver.model,
                                               "max_tokens": 12000, "max_concurrency": 1}}})
    monkeypatch.setattr(pipeline, "fetch_prices", lambda *a: metadata())
    monkeypatch.setattr(evaluate, "load_env_file", lambda *a: {})
    solver = Solver({"cost": .001})
    calls = []
    def post(prompt):
        calls.append(prompt)
        raw = json.dumps({"verdicts": {"dx": {"label": "yes", "reason": "matches", "evidence": []}}})
        return {"choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"cost": .001}}
    solver._post = post
    monkeypatch.setattr(evaluate, "solver_for_spec", lambda *a, **k: solver)
    from dataclasses import asdict
    tasks = []
    for i in range(3):
        task = JudgeTask("test", ("dx",), f"prompt {i}", "A", Solver.model, "v1")
        tasks.append({"key": str(i), "status": "ready", "task": asdict(task), "source": {},
                      "rubric": {"groups": {"dx_hit": ["dx"]}, "not_applicable": {}, "version": "v1"}})
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    return {"models": {"judge": {}}}, calls


def test_max_cells_applies_to_new_work_and_partial_is_explicit(monkeypatch, tmp_path):
    cfg, calls = fake_run(monkeypatch, tmp_path)
    first = pipeline.execute_run(tmp_path, cfg, tmp_path / "budget.json", limit_usd="100", max_cells=1)
    assert first["status"] == "incomplete"
    assert first["states"] == {"resolved": 1, "pending": 2}
    second = pipeline.execute_run(tmp_path, cfg, tmp_path / "budget.json", limit_usd="100", max_cells=1)
    assert second["states"] == {"resolved": 2, "pending": 1}
    assert len(calls) == 4
    last = pipeline.execute_run(tmp_path, cfg, tmp_path / "budget.json", limit_usd="100")
    assert last["status"] == "completed" and last["states"] == {"resolved": 3}
    assert len(calls) == 6


def test_budget_stop_leaves_pending_cell_without_fake_result(monkeypatch, tmp_path):
    cfg, calls = fake_run(monkeypatch, tmp_path)
    report = pipeline.execute_run(tmp_path, cfg, tmp_path / "budget.json", limit_usd=".00001")
    assert report["status"] == "incomplete" and report["stop_reason"]
    assert report["states"] == {"pending": 3} and not calls
    assert not list((tmp_path / "results").glob("*.json"))


def test_registry_concurrency_controls_actual_pipeline_requests(monkeypatch, tmp_path):
    import threading
    from haenv import evaluate
    from test_semantic_transport import Solver
    cfg, _ = fake_run(monkeypatch, tmp_path)
    manifest = pipeline.validate_run(tmp_path)
    manifest["policy"]["judge"]["max_concurrency"] = 3
    barrier = threading.Barrier(3);seen = [];lock = threading.Lock()
    def make_solver(*a, **kw):
        solver = Solver({"cost": .001})
        def post(prompt):
            with lock:
                first = prompt not in seen
                seen.append(prompt)
            if first:barrier.wait(timeout=2)
            return {"choices": [{"finish_reason": "stop", "message": {"content":
                '{"verdicts":{"dx":{"label":"yes","reason":"A","evidence":[]}}}'}}],
                "usage": {"cost": .001}}
        solver._post = post
        return solver
    monkeypatch.setattr(pipeline, "validate_run", lambda out:manifest)
    monkeypatch.setattr(evaluate, "solver_for_spec", make_solver)
    report = pipeline.execute_run(tmp_path, cfg, tmp_path / "budget.json", limit_usd="100")
    assert report["states"] == {"resolved": 3}
    assert report["execution"]["concurrency"] == 3
    assert len(seen) == 6
