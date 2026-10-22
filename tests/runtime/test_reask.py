"""Re-asking preserves archived payloads, selection, scoring and runtime caps offline."""
from __future__ import annotations
from _paths import ROOT as _TEST_ROOT

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from haenv.interaction.reask import (
    ARMS, Batch, build_unit, candidates, cover, digest, final_split, half,
    injected, placebo, read_rows,
)
from haenv.interaction.query.loop import FORCE_FINAL, Turn
from haenv.reask import Unit, dry_rows, prepare, run, score
from haenv.runtime.transport import make_envelope

FIXTURE = (_TEST_ROOT / 'tests') / "fixtures/reask"


@pytest.fixture(scope="module", autouse=True)
def expand_archive(tmp_path_factory):
    import gzip
    global FIXTURE
    original = FIXTURE
    files = json.loads(gzip.decompress((FIXTURE / "archive.json.gz").read_bytes()))
    dest = tmp_path_factory.mktemp("reask-archive")
    for name, content in files.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    FIXTURE = dest
    try:
        yield dest
    finally:
        FIXTURE = original


def fixture_unit():
    path = FIXTURE / "unit0"
    expected = json.loads((path / "expected.json").read_text())
    batch = Batch(path)
    plan = build_unit(batch, expected["question"], expected["model"], expected["repeat"],
                      read_rows(path / "candidates.jsonl"))
    return batch, expected, Unit(plan, *batch.payloads[expected["question"]])


def test_archived_items_order_wrapper_and_request_fingerprints():
    paths = sorted(FIXTURE.glob("unit*"))
    assert paths, "archived equivalence fixture is required"
    for path in paths:
        expected = json.loads((path / "expected.json").read_text())
        batch = Batch(path)
        plan = build_unit(batch, expected["question"], expected["model"], expected["repeat"],
                          read_rows(path / "candidates.jsonl"))
        assert plan["prefix_sha"] == expected["prefix_sha"]
        assert plan["missed"] == expected["missed"]
        assert plan["requests"].keys() == expected["arms"].keys()
        for arm, req in plan["requests"].items():
            assert req["items"] == expected["arms"][arm]["items"]
            choice = "auto" if expected["model"] == "glm-5.3-flash" else "none"
            assert digest(make_envelope(req["system"], req["messages"], req["tools"], choice)) == expected["arms"][arm]["envelope_sha"]
        assert len(plan["requests"]["placebo"]["items"]) == len(plan["requests"]["decisive"]["items"])
        if "half" in plan["requests"]:
            assert len(plan["requests"]["half"]["items"]) == len(plan["requests"]["decisive"]["items"]) // 2
        else:
            assert len(plan["requests"]["decisive"]["items"]) < 2


def item(target, keys=(), *, name="order_test", ordered=True, found=True):
    return {"name": name, "arguments": {"target": target}, "keys": list(keys),
            "content": {"target": target, "ordered": ordered, "found": found}}


def test_wrapper_has_no_evidence_labels_or_arm_specific_cues():
    x = item("a", ["hidden_key"])
    y = {**x, "keys": [], "src": "another source"}
    assert injected([x]) == injected([y])
    assert injected([x]) == [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "inj0", "name": "order_test", "arguments": {"target": "a"}}]},
        {"role": "tool", "tool_call_id": "inj0", "name": "order_test", "content": x["content"]}]
    assert injected([]) == []


def test_placebo_excludes_every_evidence_key_and_attempted_purchases():
    # A denied attempt is still excluded; a non-missed evidence item is not a placebo.
    trace = {"messages": [], "calls": [{"name": "order_test", "arguments": {"target": "denied"}, "error": "budget"}]}
    pool = [item("denied"), item("already_seen_key", ["K"]), item("no_result", found=False),
            item("not_ordered", ordered=False), item("read", name="get_labs"), item("eligible")]
    assert placebo(pool, [item("decisive", ["M"])], trace, "m", "q", 1) == [pool[-1]]
    with pytest.raises(ValueError, match="placebo shortfall"):
        placebo(pool[:-1], [item("decisive", ["M"])], trace, "m", "q", 1)


def test_half_uses_floor_and_retains_decisive_order():
    values = [item(str(i)) for i in range(5)]
    selected = half(values, "m", "q", 1)
    assert len(selected) == 2
    assert selected == [v for v in values if v in selected]
    assert selected == half(values, "m", "q", 1)
    with pytest.raises(ValueError, match="at least two"):
        half(values[:1], "m", "q", 1)


def test_greedy_ties_are_stable_and_purchases_precede_reads():
    a = item("a", ["K"])
    b = {**a, "src": "later"}
    read = item("a", ["K"], name="get_labs")
    assert cover([read, a, b], ["K"]) == ([a], [])
    assert cover([b, a], ["K"]) == ([b], [])
    assert cover([], ["K"]) == ([], ["K"])


def test_prefix_drops_forced_final_once_and_rejects_pending_tools():
    trace = {"messages": [{"role": "user", "content": "question"},
                           {"role": "user", "content": FORCE_FINAL}, {"role": "assistant", "content": "{}"}]}
    prefix, answer, forced = final_split(trace)
    assert prefix == trace["messages"][:1] and answer == trace["messages"][-1] and forced
    with pytest.raises(ValueError, match="final assistant answer"):
        final_split({"messages": [{"role": "assistant", "tool_calls": [{"id": "pending"}]}]})


def test_prefix_fingerprint_is_enforced():
    batch, expected, _ = fixture_unit()
    batch.traces[expected["question"], expected["model"]]["system"] += " changed"
    with pytest.raises(ValueError, match="does not match stored fingerprint"):
        batch.prefix_check(expected["question"], expected["model"])


def test_receipt_fallback_matches_the_same_request(tmp_path):
    batch, expected, _ = fixture_unit()
    batch.path = tmp_path
    batch.responses = {}
    # Store state has already been materialized by fixture_unit.
    receipts = tmp_path / "solver-accounting/receipts"
    receipts.mkdir(parents=True)
    (receipts / "r.json").write_text(json.dumps({"cell": {"case_id": expected["question"], "solver": expected["model"]},
                                                "request": {"prompt_sha256": expected["prefix_sha"]}}))
    assert batch.prefix_check(expected["question"], expected["model"])["basis"] == "receipts"


def test_dry_run_and_selection_never_initialize_a_provider(monkeypatch, capsys):
    import haenv.runtime.solvers as solvers
    monkeypatch.setattr(solvers, "build_solvers", lambda *a, **k: pytest.fail("provider initialized in dry-run"))
    from haenv.app.cli import main
    batch, expected, unit = fixture_unit()
    assert main(["reask", str(batch.path), "--models", expected["model"], "--questions", expected["question"],
                 "--repeats", str(expected["repeat"]), "--arms", "resample", "--dry-run"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["arm"] == "resample" and row["injected"] == []
    assert row["prefix_sha"] == expected["prefix_sha"]
    assert len(dry_rows([unit])) == 4
    assert prepare(batch.path, models=["absent"]) == []


def test_shipped_scorer_uses_aliases_and_rejects_malformed_answers():
    batch, expected, unit = fixture_unit()
    assert score("not json", unit.solver_payload, unit.verifier_payload)["D"] == 0
    original = batch.traces[expected["question"], expected["model"]]["messages"][-1]["content"]
    assert score(original, unit.solver_payload, unit.verifier_payload)["D"] == 0
    ddx = unit.verifier_payload.adjudication["ddx"]
    names = ddx.get("unified_diagnosis") or ddx.get("diagnosis")
    assert names, ddx
    diagnosis = names.get("name") if isinstance(names, dict) else names
    correct = json.dumps({"differential": [{"rank": 1, "diagnosis": diagnosis}]})
    assert score(correct, unit.solver_payload, unit.verifier_payload)["D"] == 1
    threads = ddx.get("threads") or []
    aliases = [t["aliases"][0] for t in threads] if threads else (ddx.get("aliases") or [diagnosis])[:1]
    alias_answer = json.dumps({"differential": [{"rank": i + 1, "diagnosis": a} for i, a in enumerate(aliases)]})
    assert score(alias_answer, unit.solver_payload, unit.verifier_payload)["D"] == 1


@pytest.mark.parametrize("mode,expected_status,expected_d", [("answer", "answered", 0), ("tool", "unanswered", 0),
                                                           ("error", "error", None), ("empty", "empty", 0)])
def test_runtime_accounting_scheduler_output_and_resume(tmp_path, monkeypatch, mode, expected_status, expected_d):
    from haenv.runtime import scheduler, solver_accounting, solvers
    from haenv.interaction.query import solver as query_solver
    _, _, unit = fixture_unit()
    bound, scheduled = [], []
    class Accounting:
        def enable_content_keys(self, ids):
            assert len(set(ids)) == 4
        def bind(self, inner, case, model):
            bound.append((case, model))
    monkeypatch.setattr(solver_accounting, "prepare_accounting", lambda *a, **k: Accounting())
    inner = solvers.OpenAICompatSolver(unit.plan["model"], "test-model", "fake", 10,
                                      url="http://127.0.0.1:1/v1/chat/completions", backend="test-local")
    unit.plan["source_answer_settings"] = solver_accounting.answer_settings(inner)
    monkeypatch.setattr(solvers, "build_solvers", lambda *a: [(unit.plan["model"], lambda: inner)])
    class Solver:
        _failed_turns = [{"error": "TimeoutError: transport"}] if mode == "error" else []
        def chat(self, system, messages, specs, choice):
            assert messages[-1] == {"role": "user", "content": FORCE_FINAL}
            return Turn(content="{}" if mode == "answer" else "", tool_calls=[{"name": "order_test"}] if mode == "tool" else [], finish="stop")
    monkeypatch.setattr(query_solver, "as_query_solver", lambda *a: Solver())
    def schedule(fn, tasks, emit, **kwargs):
        overrides, minimum = scheduler._CAP_OVERRIDES.get()
        assert overrides == {unit.plan["model"]: 11} and minimum == 64
        assert kwargs["per_model"] == {} and kwargs["workers"] == 64
        scheduled.extend(tasks)
        return [emit(fn(t)) for t in tasks]
    monkeypatch.setattr(scheduler, "run", schedule)
    cfg = {"models": {unit.plan["model"]: {"backend": "openrouter"}}, "query": {}}
    kwargs = dict(cfg=cfg, model_caps={unit.plan["model"]: 11}, ledger_path=tmp_path / "ledger", budget_usd="1")
    rows = run([unit], tmp_path / "out", **kwargs)
    assert len(rows) == len(bound) == len(scheduled) == 4
    assert {r["status"] for r in rows} == {expected_status}
    assert {r["D"] for r in rows} == {expected_d}
    assert set(rows[0]) == {"model", "source_model", "answer_settings", "settings_replay",
                          "question", "repeat", "kind", "arm", "status", "D", "n_added"}
    assert read_rows(tmp_path / "out/injection-rows.jsonl") == rows
    if mode != "error":
        assert run([unit], tmp_path / "out", **kwargs) == []
        assert len(bound) == 4
    unit.plan["missed"] += ["changed"]
    with pytest.raises(ValueError, match="plan changed"):
        run([unit], tmp_path / "out", **kwargs)


def test_scheduler_raises_small_configured_caps_and_honors_explicit_override():
    from haenv.runtime.scheduler import CellCaps, run_caps
    cfg = {"paid": {}, "eval": {}}
    original = CellCaps({"api": 100}, {"m": 4}, {}, 100, 1)
    caps, _ = run_caps(["m"], cfg=cfg, model_of=lambda t: t, backend_for=lambda m: "api",
                       caps=original, minimum=64)
    assert caps.model("m") == 64 and original.model("m") == 4
    caps, _ = run_caps(["m"], cfg=cfg, model_of=lambda t: t, backend_for=lambda m: "api",
                       caps=original, minimum=64, overrides={"m": 9})
    assert caps.model("m") == 9


def test_prepare_skips_aborted_conversations_before_splitting(monkeypatch):
    batch, _, _ = fixture_unit()
    trace = next(iter(batch.traces.values()))
    trace["stop"] = "unanswered"
    trace["messages"] = [{"role": "user", "content": "no final answer"}]
    monkeypatch.setattr("haenv.reask.Batch", lambda path: batch)
    assert prepare(batch.path) == []


def test_portable_fixture_contains_no_machine_paths():
    import gzip
    import re
    original = (_TEST_ROOT / 'tests') / "fixtures/reask/archive.json.gz"
    data = gzip.decompress(original.read_bytes()).decode()
    assert not re.search(r"/(?:home|root|Users)/", data)
    for member in json.loads(data):
        assert not Path(member).is_absolute() and ".." not in Path(member).parts


def test_unrecorded_decisive_result_uses_the_environment_purchase_path(monkeypatch):
    batch, expected, _ = fixture_unit()
    path = FIXTURE / "unit0"
    sources = read_rows(path / "candidates.jsonl")
    target = expected["arms"]["decisive"]["items"][0]
    from haenv.m2.query import ADAPTER
    lookups = ADAPTER.key_lookups(batch.payloads[expected["question"]][1], None)
    sources = [c for c in candidates(sources, lookups) if not set(target["keys"]) & set(c["keys"])]
    supplied = []
    def environment(question, model, targets):
        supplied.append((question, model, targets))
        assert target["arguments"]["target"] in targets
        return [{**target, "src": "env_purchase"}]
    monkeypatch.setattr(batch, "environment_items", environment)
    plan = build_unit(batch, expected["question"], expected["model"], expected["repeat"], sources, arms=("decisive",))
    assert len(supplied) == 1
    assert any(c["src"] == "env_purchase" and c["content"] == target["content"] for c in plan["requests"]["decisive"]["items"])


def test_run_refuses_writing_to_its_source_directory(tmp_path):
    _, _, unit = fixture_unit()
    unit.source = tmp_path
    with pytest.raises(ValueError, match="output must be separate"):
        run([unit], tmp_path, cfg={}, ledger_path=tmp_path / "ledger", budget_usd="1")


def test_complete_coverage_controls_only_receive_resample(monkeypatch):
    batch, expected, unit = fixture_unit()
    check = batch.prefix_check(expected["question"], expected["model"])
    trace = batch.traces[expected["question"], expected["model"]]
    prefix, answer, _ = final_split(trace)
    trace["messages"] = prefix + injected(unit.plan["requests"]["decisive"]["items"]) + [answer]
    monkeypatch.setattr(batch, "prefix_check", lambda *a: check)
    plan = build_unit(batch, expected["question"], expected["model"], expected["repeat"],
                      read_rows(FIXTURE / "unit0/candidates.jsonl"))
    assert plan["kind"] == "control" and plan["missed"] == []
    assert list(plan["requests"]) == ["resample"]


@pytest.mark.parametrize("change", ["time", "payload_time", "patient", "gold", "case_fingerprint", "accounting_fingerprint"])
def test_sources_reject_same_question_in_another_world(tmp_path, change):
    import shutil
    batch, expected, _ = fixture_unit()
    target = tmp_path / "target"
    source = tmp_path / "source"
    shutil.copytree(batch.path, target)
    shutil.copytree(batch.path, source)
    question = expected["question"]
    if change in ("case_fingerprint", "accounting_fingerprint"):
        for path, fingerprint in ((target, "original"), (source, "different")):
            if change == "case_fingerprint":
                (path / "batch.json").write_text(json.dumps({"cases_sha256": {question: fingerprint}}))
            else:
                (path / "solver-accounting").mkdir()
                (path / "solver-accounting/manifest.json").write_text(json.dumps({"case_digests": {question: fingerprint}}))
    else:
        payload = json.loads((source / "payloads.jsonl").read_text())
        if change == "time":
            payload["sp"]["prediction_context"]["prediction_time_T"] += 1000
        elif change == "payload_time":
            payload["T"] = payload["sp"]["prediction_context"]["prediction_time_T"] + 1000
        elif change == "patient":
            payload["sp"]["user_profile"]["sex"] = "different"
        else:
            payload["vp"]["adjudication"]["ddx"] = {"diagnosis": "different"}
        (source / "payloads.jsonl").write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="source world mismatch"):
        prepare(target, sources=[source], arms=("resample",))


def test_sources_accept_identical_world_and_reject_conflicting_returns(tmp_path):
    import shutil
    batch, expected, _ = fixture_unit()
    source = tmp_path / "source"
    shutil.copytree(batch.path, source)
    units = prepare(batch.path, sources=[source], arms=("resample",))
    assert len(units) == 1
    trace = read_rows(source / "query_trace.jsonl")[0]
    result = next(m for m in trace["messages"] if m["role"] == "tool")
    result["content"] = {**result["content"], "conflicting_value": 999}
    (source / "query_trace.jsonl").write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="conflicting tool returns"):
        prepare(batch.path, sources=[source], arms=("resample",))


def pinned_run(tmp_path, record="manifest"):
    """Build the real production factory, recording its settings without HTTP."""
    import shutil
    from haenv.app.config import load_cfg
    from haenv.runtime.solvers import build_solvers
    from haenv.runtime.solver_accounting import answer_settings, wire_settings
    batch, expected, _ = fixture_unit()
    source = tmp_path / "source"
    shutil.copytree(batch.path, source)
    model = expected["model"]
    cfg = load_cfg()
    cfg["env_file"] = None
    cfg["models"][model] = {"backend": "openrouter", "model": "openai/gpt-6-luna", "max_tokens": 72000,
                            "sampling": {"temperature": 0.2}, "reasoning_effort": "low", "upstream": "provider-a"}
    solver = dict(build_solvers(SimpleNamespace(models=[model], include_baseline=False), cfg))[model]()
    settings = answer_settings(solver)
    root = source / "solver-accounting"
    root.mkdir()
    if record == "manifest":
        (root / "manifest.json").write_text(json.dumps({"identity": {"answer_settings": {model: settings}}}))
    elif record == "legacy":
        (root / "manifest.json").write_text(json.dumps({"identity": {"settings": {model: wire_settings(solver)}}}))
    elif record == "receipt":
        (root / "receipts").mkdir()
        (root / "receipts/final.json").write_text(json.dumps({
            "cell": {"case_id": expected["question"], "solver": model},
            "request": {"prompt_sha256": expected["prefix_sha"], "settings": wire_settings(solver)}}))
    return source, cfg, model, settings


@pytest.mark.parametrize("change,value", [("model", "openai/gpt-6-sol"), ("max_tokens", 73000),
                                         ("sampling", {"temperature": 0.9}),
                                         ("reasoning_effort", "high"), ("upstream", "provider-b")])
def test_original_settings_mismatch_refused_before_accounting(tmp_path, monkeypatch, change, value):
    from haenv.runtime import solver_accounting
    source, cfg, model, _ = pinned_run(tmp_path)
    cfg["models"][model][change] = value
    units = prepare(source, arms=("resample",))
    monkeypatch.setattr(solver_accounting, "prepare_accounting", lambda *a, **k: pytest.fail("accounting before settings check"))
    with pytest.raises(ValueError, match="answer settings mismatch"):
        run(units, tmp_path / "out", cfg=cfg, ledger_path=tmp_path / "ledger", budget_usd="1")


@pytest.mark.parametrize("record", ["manifest", "legacy", "receipt", "missing"])
def test_settings_provenance_and_explicit_override_labels(tmp_path, monkeypatch, record):
    from haenv.runtime import scheduler, solver_accounting
    from haenv.interaction.query import solver as query_solver
    source, cfg, model, settings = pinned_run(tmp_path, record)
    units = prepare(source, arms=("resample",))
    monkeypatch.setattr(solver_accounting, "prepare_accounting", lambda *a, **k: SimpleNamespace(
        enable_content_keys=lambda ids: None, bind=lambda *a: None))
    monkeypatch.setattr(query_solver, "as_query_solver", lambda *a: SimpleNamespace(
        chat=lambda *a: Turn(content="{}", finish="stop")))
    monkeypatch.setattr(scheduler, "run", lambda fn, tasks, emit, **k: [emit(fn(t)) for t in tasks])
    kwargs = dict(cfg=cfg, ledger_path=tmp_path / "ledger", budget_usd="1")
    if record == "missing":
        with pytest.raises(ValueError, match="source answer settings unavailable"):
            run(units, tmp_path / "out", **kwargs)
    else:
        rows = run(units, tmp_path / "out", **kwargs)
        assert rows[0]["model"] == "openai/gpt-6-luna"
        assert rows[0]["source_model"] == model and rows[0]["answer_settings"] == settings
        assert rows[0]["settings_replay"] == "verified"
    cfg["models"][model]["model"] = "openai/gpt-6-sol"
    rows = run(units, tmp_path / "override", allow_settings_override=True, **kwargs)
    assert rows[0]["model"] == "openai/gpt-6-sol" and rows[0]["source_model"] == model
    assert rows[0]["answer_settings"]["model_canonical"] == "gpt-6-sol"
    assert rows[0]["settings_replay"] == ("unavailable" if record == "missing" else "mismatch")
    assert run(units, tmp_path / "override", allow_settings_override=True, **kwargs) == []
    cfg["models"][model]["max_tokens"] += 100
    with pytest.raises(ValueError, match="plan changed"):
        run(units, tmp_path / "override", allow_settings_override=True, **kwargs)



def test_bad_numeric_answer_is_persisted_without_aborting_local_batch(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from haenv.app.config import load_cfg
    from haenv.runtime import scheduler, solver_accounting, solvers
    from haenv.runtime.budget import BudgetLedger
    _, _, unit = fixture_unit()
    requests = []
    bad = '{"differential":[],"forecast":{"risk":"low"}}'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            response = {"id": f"local-{len(requests)}", "choices": [{"message": {
                "role": "assistant", "content": bad if len(requests) == 1 else '{"differential":[]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = load_cfg()
    model = unit.plan["model"]
    cfg["models"][model] = {"backend": "test-local", "model": "local-model", "max_tokens": 72000,
                            "proxy": "none", "price": {"input_per_million_usd": 1, "output_per_million_usd": 1}}
    cfg["eval"]["saturation_report_s"] = 0

    def make():
        solver = solvers.OpenAICompatSolver(model, "local-model", "fake-local-key", 10, retries=0,
            max_tokens=72000, url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions", backend="test-local")
        solver.proxy = "none"
        return solver

    unit.plan["source_answer_settings"] = solver_accounting.answer_settings(make())
    monkeypatch.setattr(solvers, "build_solvers", lambda *a: [(model, make)])
    monkeypatch.setattr(scheduler, "_cfg", lambda: cfg)
    kwargs = dict(cfg=cfg, workers=1, ledger_path=tmp_path / "ledger", budget_usd="1")
    try:
        rows = run([unit], tmp_path / "out", **kwargs)
        assert len(rows) == len(requests) == 4
        assert rows[0]["status"] == "error" and rows[0]["D"] is None
        assert rows[0]["error_stage"] == "scoring" and "ValueError" in rows[0]["error"]
        assert all(r["status"] == "answered" and r["D"] == 0 for r in rows[1:])
        answers = read_rows(tmp_path / "out/answers.jsonl")
        assert len(answers) == 4 and answers[0]["answer"] == bad
        assert answers[0]["score"]["D"] is None
        assert float(BudgetLedger(tmp_path / "ledger").summary()["committed_usd"]) > 0
        assert run([unit], tmp_path / "out", **kwargs) == []
        assert len(requests) == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join()



@pytest.mark.parametrize("backend_limit,paid_limit,expected", [(2, 44, 2), (100, 7, 4)])
def test_run_respects_configured_backend_and_shared_paid_caps(tmp_path, monkeypatch, backend_limit, paid_limit, expected):
    from haenv.runtime import scheduler, solver_accounting
    from haenv.interaction.query import solver as query_solver
    source, cfg, model, _ = pinned_run(tmp_path)
    cfg["backend_limits"]["openrouter"] = backend_limit
    cfg["paid"]["backend_request_limits"] = {}
    cfg["paid"]["global_request_limit"] = paid_limit
    cfg["paid"]["judge_reserved_lanes"] = 3
    cfg["eval"]["model_caps"] = {model: 1}
    units = prepare(source, arms=("resample",))
    monkeypatch.setattr(solver_accounting, "prepare_accounting", lambda *a, **k: SimpleNamespace(
        enable_content_keys=lambda ids: None, bind=lambda *a: None))
    monkeypatch.setattr(query_solver, "as_query_solver", lambda *a: SimpleNamespace(
        chat=lambda *a: Turn(content="{}", finish="stop")))

    def schedule(fn, tasks, emit, **kwargs):
        overrides, minimum = scheduler._CAP_OVERRIDES.get()
        caps, _ = scheduler.run_caps(tasks, cfg=cfg, model_of=kwargs["key_of"], backend_for=kwargs["backend_of"],
            backend_limits=kwargs.get("backend_limits"), minimum=minimum, overrides=overrides)
        assert caps.backend("openrouter") == expected
        assert caps.model(model) == expected
        return [emit(fn(t)) for t in tasks]

    monkeypatch.setattr(scheduler, "run", schedule)
    run(units, tmp_path / "out", cfg=cfg, workers=64, ledger_path=tmp_path / "ledger", budget_usd="1")



def test_help_and_guide_explain_per_unit_and_pooled_reproduction():
    from haenv.app.reask import parser
    help_text = " ".join(parser().format_help().split())
    guide = " ".join(((_TEST_ROOT / "docs/reask.md")).read_text().split())
    for text in (help_text, guide):
        assert "per unit" in text and "pooled" in text
        assert "inclusion list" in text and "reask_pooled.py --check" in text
        assert "not shipped" not in text
    for doc in ("README.md", "README.zh-CN.md"):
        assert "docs/reask.md" in (_TEST_ROOT / doc).read_text()
    assert parser().parse_args(["batch", "--allow-settings-override"]).allow_settings_override
