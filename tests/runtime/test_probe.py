"""`haenv probe concurrency`: the ramp stops where a provider starts refusing, every request
is reserved on the ledger before it is sent, and the table turns into cap config.

A local HTTP server stands in for the provider: it answers 200 with a cost while at most
`ACCEPT` requests are in flight and 429 beyond that.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from decimal import Decimal

import pytest

import haenv.runtime.probe as pr
from haenv.scoring.semantic.budget import BudgetLedger

ACCEPT = 6
COST = "0.0001"


class _Provider(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.lock, self.in_flight, self.peak = threading.Lock(), 0, 0


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        srv = self.server
        with srv.lock:
            srv.in_flight += 1
            srv.peak = max(srv.peak, srv.in_flight)
            over = srv.in_flight > ACCEPT
        try:
            time.sleep(0.1)
            status, body = ((429, {"error": {"message": "rate limited"}}) if over else
                            (200, {"choices": [{"message": {"content": "OK"}}],
                                   "usage": {"cost": float(COST)}}))
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            with srv.lock:
                srv.in_flight -= 1


@pytest.fixture
def provider(monkeypatch):
    for k in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(k, "127.0.0.1,localhost")
    import haenv.runtime.transport as transport
    monkeypatch.setattr(transport, "_BUCKET", [None])      # restored after the test
    transport.use_connect_pacing(1000.0, 64)               # as `run_probe` does
    srv = _Provider()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _route(srv, bound="0.001"):
    def cost_of(resp):
        c = (resp.get("usage") or {}).get("cost")
        return Decimal(str(c)) if c is not None else None
    return pr.Route("m", "fake", f"http://127.0.0.1:{srv.server_address[1]}/", {}, b"{}",
                    Decimal(bound), cost_of)


def _probe(**over):
    return {"steps": [2, 4, 32, 64], "step_s": 0.8, "request_timeout_s": 5,
            "max_rejection_rate": 0.05, "max_latency_growth": 50.0, "safety_factor": 0.7,
            "lease_requests": 2, **over}


def test_ramp_stops_at_the_first_refusing_step_and_derates_the_last_pass(provider, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="5")
    wallet = pr.Wallet(ledger, _route(provider), share_usd=Decimal(5), lease_requests=2)
    out = pr.probe_model(_route(provider), wallet, _probe())
    verdicts = [s["verdict"] for s in out["steps"]]
    assert verdicts[:2] == ["pass", "pass"] and verdicts[2].startswith("rejections")
    assert len(out["steps"]) == 3                          # no step after the first failure
    assert out["last_pass"] == 4 and out["safe_concurrency"] == 2
    assert out["steps"][2]["outcomes"]["rate_limited"] > 0


def test_every_request_is_paid_on_the_ledger(provider, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="5")
    wallet = pr.Wallet(ledger, _route(provider), share_usd=Decimal(5), lease_requests=2)
    step = pr.run_step(_route(provider), wallet, 3, _probe(step_s=0.3))
    snap = ledger.snapshot()
    ok, refused = step["outcomes"].get("ok", 0), step["requests"] - step["outcomes"].get("ok", 0)
    # ok requests at their reported cost, refused ones at their bound; nothing left reserved
    assert Decimal(snap["committed_usd"]) == ok * Decimal(COST) + refused * Decimal("0.001")
    assert Decimal(snap["in_flight_usd"]) == 0
    assert wallet.spent == Decimal(snap["committed_usd"])


def test_a_model_stops_when_its_share_runs_out(provider, tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", limit_usd="5")
    route = _route(provider, bound=COST)
    wallet = pr.Wallet(ledger, route, share_usd=Decimal(COST) * 10, lease_requests=2)
    step = pr.run_step(route, wallet, 2, _probe(step_s=5))
    assert step["budget_stop"] and step["wall_s"] < 5
    assert Decimal(ledger.snapshot()["committed_usd"]) <= Decimal(COST) * 10


@pytest.mark.parametrize("status,body,error,want", [
    (200, {"choices": []}, "", "ok"),
    (429, None, "", "rate_limited"),
    (200, {"error": {"code": 429, "message": "x"}}, "", "rate_limited"),
    (200, {"error": {"message": "Provider returned error: too many requests"}}, "", "rate_limited"),
    (503, None, "", "overloaded"),
    (400, {"error": {"message": "bad field"}}, "", "error"),
    (None, None, "TimeoutError: timed out", "timeout"),
    (None, None, "URLError: refused", "transport"),
])
def test_outcomes_are_classified(status, body, error, want):
    assert pr.classify(status, body, error) == want


def test_latency_growth_fails_a_step():
    first = {"p95_s": 1.0}
    step = {"requests": 10, "outcomes": {"ok": 10}, "rejection_rate": 0.0, "p95_s": 3.5,
            "budget_stop": False}
    assert pr.verdict(step, first, _probe(max_latency_growth=3.0)).startswith("p95")
    assert pr.verdict(step, first, _probe(max_latency_growth=4.0)) is None


def test_caps_from_probe_feed_model_and_backend_limits():
    rows = [{"model": "a", "backend": "X", "safe_concurrency": 5},
            {"model": "b", "backend": "X", "safe_concurrency": 3},
            {"model": "c", "backend": "Y", "safe_concurrency": 0}]
    caps = pr.caps_from_probe(rows)
    from haenv.runtime.scheduler import CellCaps
    from haenv.app.settings import validate
    configured = validate(caps)
    effective = CellCaps.from_config(configured)
    assert effective.model("a") == 5 and effective.model("b") == 3
    assert effective.backend("X") == effective.model("a") + effective.model("b")
    assert caps["eval"]["model_workers"] == caps["eval"]["model_caps"]
    assert caps["eval"]["elastic_max"] >= max(caps["eval"]["model_caps"].values())
    assert "Y" not in caps["backend_limits"]


def test_a_rerun_keeps_finished_models_and_refuses_other_settings(tmp_path):
    probe = _probe()
    pr._write(tmp_path, [{"model": "a", "backend": "X", "steps": [], "last_pass": 8,
                          "stopped_by": None, "ramp_complete": True, "safe_concurrency": 5,
                          "spent_usd": "0", "requests_without_cost": 0}], probe)
    assert [r["model"] for r in pr.finished(tmp_path, probe)] == ["a"]
    with pytest.raises(ValueError):
        pr.finished(tmp_path, _probe(step_s=9))


def test_probe_profile_supports_aggregate_capacity_above_single_model_ramp(tmp_path):
    rows = [{"model": "a", "backend": "X", "safe_concurrency": 200},
            {"model": "b", "backend": "X", "safe_concurrency": 200}]
    caps = pr.caps_from_probe(rows)
    assert caps["paid"]["max_capacity"] >= caps["paid"]["backend_request_limits"]["X"]
    # The machine-readable YAML agrees with the caps receipt and loads without guessing.
    full = [{**row, "steps": [], "last_pass": row["safe_concurrency"], "stopped_by": None,
             "ramp_complete": True, "spent_usd": "0", "requests_without_cost": 0} for row in rows]
    pr._write(tmp_path, full, _probe())
    import yaml
    assert yaml.safe_load((tmp_path / "concurrency.yaml").read_text()) == caps
