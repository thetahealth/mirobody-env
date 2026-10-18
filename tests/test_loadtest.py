"""Offline rehearsal (`haenv.loadtest`): the redirect, the fake provider and the responder.

The fake provider must speak each provider's protocol well enough that the real client stack
accepts it: a generation id and a cost on every OpenRouter completion, a generation record
that matches it, Google usage metadata, a 429 before generation, and judge answers that fit
the requested schema.
"""
from __future__ import annotations
from haenv.locations import source_path as _haenv_source_path

import json
import urllib.error
import urllib.request

import pytest

import haenv.runtime.transport as transport
from haenv.runtime.loadtest.profile import ModelProfile
from haenv.runtime.loadtest.profile import Profile
from haenv.runtime.loadtest.profile import answer_keys
from haenv.runtime.loadtest.responder import Responder
from haenv.runtime.loadtest.responder import schema_instance
from haenv.runtime.loadtest.server import FakeProvider

WIRE = "vendor/model-a"
PRICING = {"prompt": "0.000001", "completion": "0.000002"}
LIBRARY = [
    {"model": WIRE, "kind": "triage", "keys": ["disposition", "reason"], "weight": 1.0,
     "text": '{"disposition": "ed_now", "reason": "r"}',
     "usage": {"prompt": 100, "completion": 10, "cached": 0, "reasoning": 0}},
    {"model": WIRE, "kind": "meds", "keys": ["med_decision"], "weight": 1.0,
     "text": '{"med_decision": "hold"}',
     "usage": {"prompt": 100, "completion": 10, "cached": 0, "reasoning": 0}},
]


def _profile(**errors) -> Profile:
    return Profile(models={WIRE: ModelProfile(service_s={"*": [0.0]}, errors=errors,
                                              served_by="Upstream")},
                   catalog={WIRE: {"id": WIRE, "pricing": PRICING, "context_length": 1000}},
                   max_concurrency={})


@pytest.fixture
def fake(tmp_path):
    started = []

    def start(profile=None):
        provider = FakeProvider(profile or _profile(), Responder(LIBRARY), tmp_path / "log.jsonl").start()
        started.append(provider)
        return provider
    yield start
    for provider in started:
        provider.stop()


@pytest.fixture
def rehearse_to(monkeypatch):
    def point(url):
        monkeypatch.setattr(transport, "_SETTINGS", [{"connect_per_s": 100.0, "connect_burst": 100,
                                                      "connect_retries": 0, "rehearsal_url": url}])
        monkeypatch.setattr(transport, "_BUCKET", [None])
    return point


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers, json.loads(r.read())


def test_rehearsal_url_sends_https_requests_to_the_fake_provider(fake, rehearse_to):
    provider = fake()
    rehearse_to(provider.url)
    headers, doc = _post("https://openrouter.ai/api/v1/chat/completions",
                         {"model": WIRE, "messages": [{"role": "user", "content": 'answer {"disposition": ...}'}]})
    assert doc["choices"][0]["message"]["content"].startswith('{"disposition"')
    assert headers["X-Generation-Id"] == doc["id"]
    log = [json.loads(line) for line in provider.log_path.read_text().splitlines()]
    assert [r["backend"] for r in log] == ["openrouter"]


def test_openrouter_completion_carries_cost_and_a_matching_generation_record(fake, rehearse_to):
    provider = fake()
    rehearse_to(provider.url)
    _, doc = _post("https://openrouter.ai/api/v1/chat/completions",
                   {"model": WIRE, "provider": {"only": ["Upstream"]},
                    "messages": [{"role": "user", "content": '"med_decision" ' + "x" * 400}]})
    usage = doc["usage"]
    assert doc["provider"] == "Upstream"
    assert usage["cost"] == pytest.approx(usage["prompt_tokens"] * 1e-6 + usage["completion_tokens"] * 2e-6)
    with urllib.request.urlopen(urllib.request.Request(
            f"https://openrouter.ai/api/v1/generation?id={doc['id']}"), timeout=10) as r:
        record = json.loads(r.read())["data"]
    assert record["id"] == doc["id"] and record["total_cost"] == usage["cost"]


def test_google_completion_has_usage_metadata(fake, rehearse_to):
    provider = fake(Profile(models={"gem": ModelProfile(service_s={"*": [0.0]})}))
    rehearse_to(provider.url)
    _, doc = _post("https://generativelanguage.googleapis.com/v1beta/models/gem:generateContent?key=k",
                   {"contents": [{"parts": [{"text": '"disposition" and "reason"'}]}]})
    assert answer_keys(doc["candidates"][0]["content"]["parts"][0]["text"]) == ["disposition", "reason"]
    assert doc["usageMetadata"]["totalTokenCount"] > 0


def test_rate_limit_is_a_429_before_generation(fake, rehearse_to):
    provider = fake(_profile(rate_limited=1.0))
    rehearse_to(provider.url)
    with pytest.raises(urllib.error.HTTPError) as error:
        _post("https://openrouter.ai/api/v1/chat/completions",
              {"model": WIRE, "messages": [{"role": "user", "content": "x"}]})
    assert error.value.code == 429
    assert provider.generations == {}


def test_extra_data_appends_a_second_object(fake, rehearse_to):
    provider = fake(_profile(extra_data=1.0))
    rehearse_to(provider.url)
    _, doc = _post("https://openrouter.ai/api/v1/chat/completions",
                   {"model": WIRE, "messages": [{"role": "user", "content": '"med_decision"'}]})
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        json.loads(doc["choices"][0]["message"]["content"])


def test_judge_answer_fits_the_requested_schema():
    atom = {"type": "object", "required": ["label", "reason", "evidence_ids"],
            "properties": {"label": {"type": "string", "enum": ["no", "yes"]},
                           "reason": {"type": "string"},
                           "evidence_ids": {"type": "array", "items": {"$ref": "#/$defs/e"}}}}
    schema = {"type": "object", "required": ["verdicts"], "$defs": {"e": {"type": "string"}, "a": atom},
              "properties": {"verdicts": {"type": "object", "required": ["dx", "t0"],
                                          "properties": {"dx": {"$ref": "#/$defs/a"},
                                                         "t0": {"$ref": "#/$defs/a"}}}}}
    doc = schema_instance(schema)
    assert set(doc["verdicts"]) == {"dx", "t0"}
    assert doc["verdicts"]["dx"] == {"label": "no", "reason": "rehearsal", "evidence_ids": []}


def test_responder_picks_the_answer_kind_the_prompt_asks_for():
    responder = Responder(LIBRARY)
    assert responder.answer(WIRE, 'reply with {"med_decision": ...}', None).kind == "meds"
    assert responder.answer(WIRE, 'fields "disposition", "reason"', None).kind == "triage"
    assert responder.answer(WIRE, "no schema named here", None).text == "{}"


def test_probes_time_plain_and_context_manager_functions(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / "probed.py").write_text(
        "import contextlib, time\n"
        "@contextlib.contextmanager\n"
        "def slot():\n    time.sleep(0.05)\n    yield\n"
        "def cell(case, solver):\n    return solver\n")
    site = _haenv_source_path('haenv/runtime/loadtest/probe_site', root=Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(tmp_path)]),
           "HAENV_LOADTEST_PROBES": json.dumps({"slot": ["probed:slot"], "cell": ["probed:cell@1"]}),
           "HAENV_LOADTEST_PROBE_LOG": str(tmp_path / 'probes')}
    code = "import probed\nwith probed.slot():\n    pass\nprobed.cell('c1', 'model-a')\n"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    events = [json.loads(line) for f in (tmp_path / 'probes').glob("*.jsonl")
              for line in f.read_text().splitlines()]
    by = {e["p"]: e for e in events}
    assert by["slot"]["dt"] >= 0.05
    assert by["cell"]["label"] == "model-a"


def _finished_run(tmp_path, rcs, judging_states):
    """A minimal run directory: no requests, one sample whose semantic judging left `states`."""
    tmp_path.mkdir()
    (tmp_path / "run.json").write_text(json.dumps({
        "spec": {"time_scale": 1.0}, "started": 0.0, "ended": 10.0, "models": ["m"],
        "expected_cells": 0, "answered": 0, "attempts": [{"ended": 10.0, "rcs": rcs}]}))
    summary = tmp_path / "root" / "results" / "t" / "job" / "b-pk1" / "semantic" / "v" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"states": judging_states}))
    return tmp_path


def test_unresolved_judging_is_not_a_crash_but_a_nonzero_unknown_exit_is(tmp_path):
    from haenv.runtime.loadtest.report import SpeedReport
    done = SpeedReport.read(_finished_run(tmp_path / "a", [0, 6], {"resolved": 3, "unresolved": 2}))
    assert done.checks["no_crash"]["pass"]
    assert done.numbers["judging"] == {"resolved": 3, "unresolved": 2}
    crashed = SpeedReport.read(_finished_run(tmp_path / "b", [0, 1], {}))
    assert not crashed.checks["no_crash"]["pass"]


def test_a_run_cut_at_max_wall_is_incomplete_but_not_a_crash(tmp_path):
    from haenv.runtime.loadtest.report import SpeedReport
    run = _finished_run(tmp_path / "r", [-9, -9], {})
    doc = json.loads((run / "run.json").read_text())
    doc.update(stopped="max_wall_s=10", expected_cells=5, answered=3)
    (run / "run.json").write_text(json.dumps(doc))
    report = SpeedReport.read(run)
    assert report.checks["no_crash"]["pass"] and not report.checks["complete"]["pass"]


def test_backend_load_is_mean_in_flight_against_the_limit():
    from haenv.runtime.loadtest.report import SpeedReport
    requests = [{"backend": "openrouter", "t0": 0.0, "t1": 10.0}, {"backend": "openrouter", "t0": 0.0, "t1": 10.0},
                {"backend": "google", "t0": 5.0, "t1": 10.0}]
    load = SpeedReport._backend_load(requests, {"openrouter": 4})
    assert load["openrouter"] == {"requests": 2, "mean_in_flight": 2.0, "limit": 4, "utilization": 0.5}
    assert load["google"]["limit"] is None


def test_probe_records_thread_cpu_time_apart_from_wall_time(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    (tmp_path / "probed2.py").write_text("import time\ndef wait():\n    time.sleep(0.1)\n")
    site = _haenv_source_path('haenv/runtime/loadtest/probe_site', root=Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(tmp_path)]),
           "HAENV_LOADTEST_PROBES": json.dumps({"ledger": ["probed2:wait"]}),
           "HAENV_LOADTEST_PROBE_LOG": str(tmp_path / 'probes')}
    subprocess.run([sys.executable, "-c", "import probed2; probed2.wait()"], env=env, check=True)
    event = [json.loads(line) for f in (tmp_path / 'probes').glob("*.jsonl") for line in f.read_text().splitlines()][0]
    assert event["dt"] >= 0.1 and event["cpu"] < 0.05


def test_ledger_latency_is_reported_for_the_start_up_burst_and_after_it(tmp_path):
    from haenv.runtime.loadtest.report import SpeedReport
    run = _finished_run(tmp_path / "r", [0], {})
    doc = json.loads((run / "run.json").read_text())
    doc["spec"]["burst_s"] = 5
    (run / "run.json").write_text(json.dumps(doc))
    events = [{"p": "ledger", "f": "m:BudgetLedger.reserve", "t": 1.0, "dt": 2.0, "tags": []},
              {"p": "ledger", "f": "m:BudgetLedger.reserve", "t": 9.0, "dt": 0.001, "tags": []}]
    (run / 'probes').mkdir()
    (run / 'probes' / 'probe-1.jsonl').write_text("".join(json.dumps(e) + "\n" for e in events))
    report = SpeedReport.read(run)
    assert report.numbers["ledger_burst"]["p99_ms"] == 2000.0
    assert report.numbers["ledger_steady"]["p99_ms"] == 1.0
    assert report.checks["ledger_p99_steady"]["pass"] and not report.checks["ledger_p99"]["pass"]


def test_a_rehearsal_can_pin_the_day_tariff_checks_see(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    site = _haenv_source_path('haenv/runtime/loadtest/probe_site', root=root)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(root)]),
           "HAENV_LOADTEST_TODAY": "2026-10-05"}
    code = "from haenv.runtime import native_accounting as n; print(n.utc_today().isoformat())"
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "2026-10-05"
