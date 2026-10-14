"""Model routing against a local provider and forwarding proxy; no external requests."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import http.client
import http.server
import json
import socket
import threading
import urllib.request
from urllib.parse import urlsplit

import pytest
import yaml

from haenv import config, spend, transport
from haenv.solvers import OpenAICompatSolver, GoogleSolver


class Provider(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply()

    def reply(self):
        self.server.seen.append(self.path)
        data = json.dumps({"choices": [{"message": {"content": "{}"}}],
                           "usage": {"cost": 0.001},
                           "data": {"total_credits": 10, "total_usage": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class Proxy(Provider):
    def reply(self):
        self.server.seen.append(self.path)
        target = urlsplit(self.path)
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
        try:
            conn.request("GET", target.path)
            response = conn.getresponse()
            data = response.read()
        finally:
            conn.close()
        self.send_response(response.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def network(monkeypatch):
    from haenv import call_log
    monkeypatch.setattr(call_log, "_SAMPLING", {"by_model": {}, "n_requests": 0, "n_echo": 0})
    servers = []
    for handler in (Provider, Proxy):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        srv.seen = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    origin, proxy = servers
    url = f"http://127.0.0.1:{origin.server_port}/chat/completions"
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(key, proxy_url)
    for key in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(urllib.request, "_opener", urllib.request._opener)
    monkeypatch.setattr(transport, "_WIRE_ROUTE", threading.local())
    monkeypatch.setattr(transport, "_SETTINGS", [{"connect_retries": 3, "rehearsal_url": None}])
    monkeypatch.setattr(transport, "_BUCKET", [transport.ConnectBucket(10000, 10000)])
    monkeypatch.setattr(transport, "_SLEEP", [lambda _: None])
    transport.install()
    cfg = {"models": {"alpha": {"model": "shared", "backend": "local", "proxy": "direct"},
                      "beta": {"model": "shared", "backend": "local", "proxy": "env"}}}
    monkeypatch.setattr(config, "load_cfg", lambda: cfg)
    yield url, origin, proxy, cfg
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def solver(name, url):
    return OpenAICompatSolver(name, "shared", "local-key", 5, retries=0,
                              max_tokens=100, url=url, backend="local")


def test_frozen_solver_send_sites_select_routes_concurrently(network):
    url, origin, proxy, cfg = network
    solvers = [solver(n, url) for n in cfg["models"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda n: solvers[n % 2]._send_wire(
            solvers[n % 2]._wire_body("hello"), "hello"), range(20)))
    assert len(responses) == 20
    assert len(proxy.seen) == 10 and len(origin.seen) == 20


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_connect_phase_retry_for_both_routes(network, monkeypatch, name):
    url, origin, proxy, _ = network
    real, calls = socket.create_connection, []

    def connect(*args, **kwargs):
        # Only connections from this requesting thread, excluding proxy forwarding.
        if threading.current_thread() is threading.main_thread():
            calls.append(args[0])
            if len(calls) <= 2:
                raise ConnectionRefusedError("temporary local failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", connect)
    s = solver(name, url)
    s._send_wire(s._wire_body("hello"), "hello")
    assert len(calls) == 3 and transport.connect_retries_taken() == 2
    assert len(origin.seen) == 1
    assert len(proxy.seen) == (name == "beta")
    assert calls[-1][1] == (proxy.server_port if name == "beta" else origin.server_port)


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_balances_share_the_selector(network, name):
    url, origin, proxy, cfg = network
    balances = spend.fetch_balances({"OPENROUTER_API_KEY": "local-key"},
                                   or_base=url.rsplit("/chat/completions", 1)[0], model=name, cfg=cfg)
    assert balances["openrouter"]["account"]["left_usd"] == "8"
    assert len(origin.seen) == 2
    assert len(proxy.seen) == (2 if name == "beta" else 0)


def test_verified_wire_binding_does_not_route_unrelated_gets(network):
    url, _, proxy, _ = network
    s = solver("alpha", url)
    transport.verify_wire(s, s._wire_body("hello"), "hello")
    with urllib.request.urlopen(url, timeout=5) as response:
        response.read()
    assert len(proxy.seen) == 1


def test_google_send_site_uses_model_route(network):
    url, origin, proxy, cfg = network
    cfg["models"]["alpha"]["backend"] = "google"
    s = GoogleSolver("alpha", "shared", "local-key", 5, max_tokens=100)
    s.URL = url.rsplit("/chat/completions", 1)[0]
    s._send_wire(s._wire_body("hello"), "hello")
    assert len(origin.seen) == 1 and proxy.seen == []


def test_configuration_overlay_and_backend_default(tmp_path, monkeypatch):
    # Exercise the actual loader and strict validation rather than repeating resolver logic.
    raw = {"models": {"alpha": {"backend": "local", "model": "shared"},
                      "beta": {"backend": "local", "model": "shared", "proxy": "env"}},
           "backends": {"local": {"proxy": "direct"}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    (tmp_path / "config.local.yaml").write_text("models:\n  alpha:\n    proxy: env\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("models:\n  alpha:\n    proxy: direct\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv(config.CONFIG_OVERLAY_ENV, str(overlay))
    cfg = config.load_cfg()
    assert transport.proxy_for_model("alpha", cfg=cfg) == "direct"
    assert transport.proxy_for_model("beta", cfg=cfg) == "env"
    assert transport.proxy_for_model(backend="local", cfg=cfg) == "direct"
    assert transport.proxy_for_model(cfg={}) == "env"
    monkeypatch.setenv("HAENV_CONFIG_SET", '["models.alpha.proxy=env"]')
    assert transport.proxy_for_model("alpha", cfg=config.load_cfg()) == "env"


@pytest.mark.parametrize("section", ["models", "backends"])
@pytest.mark.parametrize("value", [None, True, "typo", {}, []])
def test_invalid_routes_are_refused(section, value):
    from haenv.settings import validate, ConfigError
    raw = {section: {"alpha": {"proxy": value, **({"model": "shared"} if section == "models" else {})}}}
    with pytest.raises(ConfigError, match="proxy must be"):
        transport.validate_proxy_config(raw, validate)


def test_route_receipt_field_does_not_change_answer_identity(network):
    from haenv.solver_accounting import wire_settings
    url, _, _, _ = network
    a, b = (wire_settings(solver(name, url)) for name in ("alpha", "beta"))
    assert transport.answer_view(a) == transport.answer_view(b)
    assert transport.route_view(a)["proxy"] == "direct"
    assert transport.route_view(b)["proxy"] == "env"


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_semantic_receipts_record_the_sent_route(network, tmp_path, name):
    from haenv.semantic_budget import BudgetLedger
    from haenv.semantic_judge import JudgeRequest
    from haenv.semantic_transport import PricedJudge, PriceSchedule
    url, _, proxy, _ = network
    s = solver(name, url)
    s.reasoning_effort = "high"
    receipts = tmp_path / "receipts"
    judge = PricedJudge(s, PriceSchedule("shared", Decimal("0.0001"), Decimal("0.0001"),
                                        Decimal(0), 10000),
                        BudgetLedger(tmp_path / "ledger.json", limit_usd="10"), receipts=receipts)
    judge(JudgeRequest("sample", 1, "hello"))
    saved = json.loads(next(receipts.glob("*.json")).read_text())
    assert saved["route"]["proxy"] == ("direct" if name == "alpha" else "env")
    assert len(proxy.seen) == (name == "beta")


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_retries_remain_bounded_for_both_routes(network, monkeypatch, name):
    import urllib.error
    url, origin, proxy, _ = network
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ConnectionRefusedError("local failure")

    monkeypatch.setattr(socket, "create_connection", fail)
    s = solver(name, url)
    with pytest.raises(urllib.error.URLError) as caught:
        s._send_wire(s._wire_body("hello"), "hello")
    assert isinstance(caught.value.reason, transport.PreSendFailure)
    assert len(calls) == 4 and origin.seen == [] and proxy.seen == []


def test_startup_log_groups_effective_routes(network, caplog):
    import logging
    _, _, _, cfg = network
    with caplog.at_level(logging.INFO):
        transport.log_proxy_routes(cfg["models"], cfg, logging.getLogger("haenv"))
    assert "代理路由：direct=['alpha'] env=['beta']" in caplog.text


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_paid_receipts_record_route_with_an_attempt_route(network, tmp_path, name):
    from haenv.paid_completion import AccountedCompletion
    from haenv.semantic_budget import BudgetLedger
    from haenv.semantic_transport import PriceSchedule
    from haenv.solver_accounting import wire_settings
    url, _, proxy, _ = network
    s = solver(name, url)
    s.backend = "openrouter"
    receipts = tmp_path / "receipts"
    paid = AccountedCompletion(
        PriceSchedule("shared", Decimal("0.0001"), Decimal("0.0001"), Decimal(0), 10000),
        BudgetLedger(tmp_path / "ledger.json", limit_usd="10"), receipts)
    paid.request("local-request", "hello", wire_settings(s), s._post_wire,
                 attempt={"route": "primary"})
    saved = json.loads(next(receipts.glob("*.json")).read_text())
    assert saved["route"]["proxy"] == ("direct" if name == "alpha" else "env")
    assert saved["attempt"]["route"] == "primary"
    assert len(proxy.seen) == (name == "beta")

