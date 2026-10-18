"""`haenv probe concurrency`: measure how many requests in flight each model's provider accepts.

For every model the probe sends tiny requests (`probe.prompt`, `probe.max_tokens` output
tokens, least reasoning the route allows) at each concurrency in `probe.steps`, for
`probe.step_s` seconds per step, through the model's own route (backend, key, upstream pin).
Each step records requests sent, rejections (HTTP 429, rate-limit error bodies, 5xx
overload, timeouts) and latency percentiles. A model stops ramping at the first step whose
rejection rate exceeds `probe.max_rejection_rate` or whose latency p95 exceeds
`probe.max_latency_growth` times the first step's p95. Its safe concurrency is
`floor(probe.safety_factor x the last step that passed)`.

Every request is paid for on a budget ledger: requests are reserved in leases of
`probe.lease_requests` upper bounds before they are sent, and each lease settles at the
summed cost the endpoints report (a request with no usable cost counts at its bound). The
probe stops a model when its share of the ledger's remaining cap runs out.

The probe's own connections are paced at `probe.connect_per_s` instead of
`transport.connect_per_s`, so the process's pacing does not cap what it measures.

Tiny requests test request concurrency and request rate, not token throughput: a board
cell's requests carry long prompts and run for minutes, so the token-per-minute quota can
bind before the request concurrency measured here.

The output directory receives `probe.json` (every step of every model) and `probe.md` (the
table). `caps_from_probe` turns the table into `eval.model_caps` and
`paid.backend_request_limits` entries.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

log = logging.getLogger("haenv.probe")

#: Outcome classes; the first four are rejections.
REJECTIONS = ("rate_limited", "overloaded", "timeout", "transport")
_RATE_WORDS = ("rate", "quota", "too many", "limit", "capacity", "overload", "busy")


# ---------------------------------------------------------------- one route

@dataclass
class Route:
    """How the probe reaches one model: URL, headers, request body and cost rules."""
    name: str
    backend: str
    url: str
    headers: dict
    body: bytes
    bound_usd: Decimal
    cost_of: object              # response dict -> Decimal | None
    secret: str = field(repr=False, default="")
    proxy: str = "env"

    def scrub(self, text: str) -> str:
        return text.replace(self.secret, "<key>") if self.secret else text


def build_route(name: str, cfg: dict, env: dict, probe: dict) -> Route:
    """The probe route of config model `name`, priced before any request is sent."""
    from haenv.runtime.solvers import BACKENDS
    from haenv.runtime.solvers import _ensure_backends_registered
    from haenv.runtime.solvers import _ensure_no_proxy
    _ensure_backends_registered(cfg)
    spec = cfg["models"][name]
    backend = spec.get("backend", "openrouter")
    bdef = BACKENDS[backend]
    key = env.get(bdef["key_env"], "")
    if not key:
        raise ValueError(f"{name}: no {bdef['key_env']} in the env file")
    if bdef.get("no_proxy_host"):
        _ensure_no_proxy({bdef["no_proxy_host"]})
    extra = dict((probe.get("extra_body") or {}).get(backend) or {})
    if bdef.get("kind") == "google":
        route = _google_route(name, spec, key, bdef, probe, extra)
    else:
        route = _openai_route(name, spec, key, bdef, backend, cfg, probe, extra)
    from haenv.runtime.transport import proxy_for_model
    route.proxy = proxy_for_model(name, backend=backend, cfg=cfg)
    return route


def _google_route(name, spec, key, bdef, probe, extra) -> Route:
    from haenv.runtime.native_accounting import NativePrices
    prices = NativePrices.verified(spec["model"], "google")
    gen = {"maxOutputTokens": probe["max_tokens"], **extra}
    body = {"contents": [{"parts": [{"text": probe["prompt"]}]}], "generationConfig": gen}
    return Route(name, "google", f"{bdef['url']}/{spec['model']}:generateContent",
                 {"Content-Type": "application/json", "x-goog-api-key": key},
                 json.dumps(body).encode(), _native_bound(prices, probe), _native_cost(prices), key)


def _openai_route(name, spec, key, bdef, backend, cfg, probe, extra) -> Route:
    body = {"model": spec["model"], bdef.get("max_tokens_field", "max_tokens"): probe["max_tokens"],
            "messages": [{"role": "user", "content": probe["prompt"]}], **extra}
    if backend == "openrouter":
        body["reasoning"] = {"effort": probe["reasoning_effort"]}
        body["usage"] = {"include": True}
        if spec.get("provider"):
            body["provider"] = dict(spec["provider"])
        bound, cost_of = _openrouter_price(cfg, spec, probe)
    else:
        from haenv.runtime.native_accounting import NativePrices
        prices = NativePrices.verified(spec["model"], backend)
        bound, cost_of = _native_bound(prices, probe), _native_cost(prices)
    return Route(name, backend, bdef["url"],
                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                 json.dumps(body).encode(), bound, cost_of, key)


def _openrouter_price(cfg: dict, spec: dict, probe: dict):
    """Upper bound from the highest listed price among the pinned upstream's endpoints (all
    endpoints when unpinned); the cost of a response is the `usage.cost` OpenRouter reports."""
    from haenv.runtime.solver_accounting import fetch_endpoints
    pin = set(((spec.get("provider") or {}).get("only")) or [])
    endpoints = [e for e in fetch_endpoints(cfg, spec["model"])
                 if not pin or e["provider_name"] in pin]
    if not endpoints:
        raise ValueError(f"{spec['model']}: no endpoint for the pinned upstream {sorted(pin)}")
    top = lambda k: max(Decimal(str(e["pricing"].get(k) or 0)) for e in endpoints)  # noqa: E731
    bound = (_prompt_tokens(probe) * top("prompt") + probe["max_tokens"] * top("completion")
             + top("request"))

    def cost_of(resp: dict):
        cost = (resp.get("usage") or {}).get("cost")
        return Decimal(str(cost)) if isinstance(cost, (int, float)) else None
    return bound, cost_of


def _native_bound(prices, probe: dict) -> Decimal:
    return _tokens_usd(prices, _prompt_tokens(probe), probe["max_tokens"])


def _tokens_usd(prices, prompt: int, output: int) -> Decimal:
    """Cost at the route's verified top-tier rates, converted at the bounding exchange rate."""
    native = (prompt * Decimal(prices.input_per_million)
              + output * Decimal(prices.output_per_million)) / 1000000
    return native * Decimal(prices.usd_per_currency_bound)


def _native_cost(prices):
    """Cost of a Google or dashscope response from the token counts it reports."""
    def cost_of(resp: dict):
        meta, usage = resp.get("usageMetadata"), resp.get("usage")
        if isinstance(meta, dict):
            prompt, total = meta.get("promptTokenCount"), meta.get("totalTokenCount")
            output = total - prompt if isinstance(prompt, int) and isinstance(total, int) else None
        elif isinstance(usage, dict):
            prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
        else:
            return None
        if not (isinstance(prompt, int) and isinstance(output, int) and min(prompt, output) >= 0):
            return None
        return _tokens_usd(prices, prompt, output)
    return cost_of


def _prompt_tokens(probe: dict) -> int:
    """Upper bound on the prompt's tokens: one per byte plus the chat-template overhead."""
    return len(probe["prompt"].encode()) + int(probe["prompt_overhead_tokens"])


# ---------------------------------------------------------------- one request

def classify(status: int | None, resp: dict | None, error: str = "") -> str:
    """`ok`, a rejection class (`REJECTIONS`) or `error` (a refusal that is not about load)."""
    if status == 429:
        return "rate_limited"
    if status is not None and (status >= 500 or status == 408):
        return "overloaded"
    if status is None:
        return "timeout" if "timed out" in error.lower() else "transport"
    if status != 200:
        return "error"
    err = (resp or {}).get("error")
    if not err:
        return "ok"
    text = json.dumps(err).lower()
    code = err.get("code") if isinstance(err, dict) else None
    if code == 429 or any(w in text for w in _RATE_WORDS):
        return "rate_limited"
    return "overloaded" if isinstance(code, int) and code >= 500 else "error"


def send(route: Route, timeout_s: float) -> dict:
    """One request: outcome class, HTTP status, latency (HTTP only) and response body."""
    req = urllib.request.Request(route.url, data=route.body, headers=route.headers)
    t0 = time.monotonic()
    status, resp, error = None, None, ""
    try:
        from haenv.runtime.transport import model_urlopen
        with model_urlopen(req, proxy=route.proxy, timeout=timeout_s) as r:
            status, resp = r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            resp = json.loads(e.read() or b"{}")
        except ValueError:
            resp = None
    except Exception as e:                                   # noqa: BLE001
        error = route.scrub(f"{type(e).__name__}: {e}")[:200]
    latency = time.monotonic() - t0
    outcome = classify(status, resp if isinstance(resp, dict) else None, error)
    return {"outcome": outcome, "status": status, "latency_s": latency,
            "resp": resp if isinstance(resp, dict) else None, "error": error}


# ---------------------------------------------------------------- ledger leases

class Wallet:
    """Reservations for one model on the budget ledger, in leases of several request bounds."""

    def __init__(self, ledger, route: Route, *, share_usd: Decimal, lease_requests: int):
        self.ledger, self.route, self.share = ledger, route, share_usd
        self.lease_requests = lease_requests
        self.lock = threading.Lock()
        self.spent = Decimal(0)          # settled cost of finished leases
        self.held = Decimal(0)           # reserved by open leases
        self.unknown = 0

    def open(self):
        """A new lease, or None when the model's share cannot cover one."""
        amount = self.route.bound_usd * self.lease_requests
        with self.lock:
            if self.spent + self.held + amount > self.share:
                return None
            self.held += amount
        rid = f"probe:{self.route.name}:{uuid.uuid4().hex[:12]}"
        try:
            self.ledger.reserve(rid, str(amount))
        except Exception:
            with self.lock:
                self.held -= amount
            raise
        return {"id": rid, "amount": amount, "left": self.lease_requests, "cost": Decimal(0)}

    def charge(self, lease: dict, resp: dict | None) -> None:
        cost = self.route.cost_of(resp) if resp else None
        if cost is None or cost > self.route.bound_usd:
            with self.lock:
                self.unknown += cost is None
            cost = self.route.bound_usd if cost is None else cost
        lease["cost"] += cost
        lease["left"] -= 1

    def close(self, lease: dict) -> None:
        self.ledger.settle(lease["id"], str(lease["cost"]))
        with self.lock:
            self.held -= lease["amount"]
            self.spent += lease["cost"]


# ---------------------------------------------------------------- one step

def run_step(route: Route, wallet: Wallet, concurrency: int, probe: dict) -> dict:
    """`concurrency` workers send requests back to back for `probe.step_s` seconds."""
    results: list[dict] = []
    lock = threading.Lock()
    deadline = time.monotonic() + float(probe["step_s"])
    out_of_budget = threading.Event()

    def worker():
        lease = None
        try:
            while time.monotonic() < deadline and not out_of_budget.is_set():
                if lease is None or lease["left"] == 0:
                    if lease is not None:
                        wallet.close(lease)
                    lease = wallet.open()
                    if lease is None:
                        out_of_budget.set()
                        return
                r = send(route, float(probe["request_timeout_s"]))
                wallet.charge(lease, r["resp"] if r["status"] == 200 else None)
                r.pop("resp")
                with lock:
                    results.append(r)
        finally:
            if lease is not None:
                wallet.close(lease)

    started = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return summarize(results, concurrency, time.monotonic() - started,
                     budget_stop=out_of_budget.is_set())


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))], 3)


def summarize(results: list[dict], concurrency: int, wall_s: float, *, budget_stop=False) -> dict:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    n = len(results)
    ok = [r["latency_s"] for r in results if r["outcome"] == "ok"]
    rejected = sum(counts.get(k, 0) for k in REJECTIONS)
    errors = sorted({r["error"] or f"HTTP {r['status']}" for r in results
                     if r["outcome"] != "ok"})[:5]
    return {"concurrency": concurrency, "requests": n, "outcomes": counts,
            "rejection_rate": round(rejected / n, 4) if n else None,
            "p50_s": _pct(ok, 0.5), "p95_s": _pct(ok, 0.95),
            "mean_in_flight": round(sum(r["latency_s"] for r in results) / wall_s, 1) if wall_s else 0,
            "requests_per_min": round(60 * n / wall_s, 1) if wall_s else 0,
            "wall_s": round(wall_s, 1), "budget_stop": budget_stop, "sample_errors": errors}


def verdict(step: dict, first: dict | None, probe: dict) -> str | None:
    """Why `step` fails the ramp (None = it passes)."""
    if step["budget_stop"] and not step["requests"]:
        return "budget"
    if not step["requests"]:
        return "no requests"
    if (step["outcomes"].get("ok", 0) == 0):
        return "no successful request"
    if step["rejection_rate"] > float(probe["max_rejection_rate"]):
        return f"rejections {step['rejection_rate']:.1%}"
    base = (first or step)["p95_s"]
    if base and step["p95_s"] and step["p95_s"] > float(probe["max_latency_growth"]) * base:
        return f"p95 {step['p95_s']}s > {probe['max_latency_growth']}x {base}s"
    return None


# ---------------------------------------------------------------- one model, all models

def probe_model(route: Route, wallet: Wallet, probe: dict) -> dict:
    steps, first, passed, stop = [], None, 0, None
    for c in probe["steps"]:
        step = run_step(route, wallet, int(c), probe)
        why = verdict(step, first, probe)
        step["verdict"] = why or "pass"
        steps.append(step)
        log.info("[probe] %s c=%d n=%d rej=%s p50=%s p95=%s in-flight=%s → %s", route.name, c,
                 step["requests"], step["rejection_rate"], step["p50_s"], step["p95_s"],
                 step["mean_in_flight"], step["verdict"])
        if first is None and why is None:
            first = step
        if why is not None:
            stop = why
            break
        passed = int(c)
        if step["budget_stop"]:
            stop = "budget"
            break
    safe = math.floor(float(probe["safety_factor"]) * passed)
    return {"model": route.name, "backend": route.backend, "proxy": route.proxy, "steps": steps,
            "last_pass": passed, "stopped_by": stop, "ramp_complete": stop is None,
            "safe_concurrency": safe, "spent_usd": str(wallet.spent),
            "requests_without_cost": wallet.unknown}


def finished(out_dir: Path, probe: dict) -> list[dict]:
    """Models already probed into `out_dir` under the same probe settings (a rerun keeps
    them and probes only the rest)."""
    path = out_dir / "probe.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if json.dumps(doc.get("probe"), sort_keys=True) != json.dumps(probe, sort_keys=True, default=str):
        raise ValueError(f"{path} was written under other probe settings; use another --out")
    return list(doc.get("models") or [])


def run_probe(models: list[str], cfg: dict, env: dict, ledger, out_dir: Path) -> list[dict]:
    """Probe every model not yet in `out_dir`; models on different backends run side by
    side, one model at a time per backend so their latencies do not share a queue."""
    from haenv.runtime.transport import use_connect_pacing, log_proxy_routes, proxy_for_model
    log_proxy_routes(models, cfg, log)
    probe = dict(cfg["probe"])
    use_connect_pacing(float(probe["connect_per_s"]), max(int(c) for c in probe["steps"]))
    results = finished(out_dir, probe)
    if any(r.get("proxy", "env") != proxy_for_model(r["model"], cfg=cfg) for r in results):
        raise ValueError("Probe proxy route changed; use another --out")
    done = {r["model"] for r in results}
    if done:
        log.info("[probe] kept from %s: %s", out_dir, sorted(done))
    routes = [build_route(m, cfg, env, probe) for m in models if m not in done]
    by_backend: dict[str, list[Route]] = {}
    for r in routes:
        by_backend.setdefault(r.backend, []).append(r)
    lock = threading.Lock()
    remaining = [len(routes)]

    def lane(rs: list[Route]):
        for route in rs:
            with lock:
                share = _remaining_usd(ledger) / max(1, remaining[0])
            wallet = Wallet(ledger, route, share_usd=share,
                            lease_requests=int(probe["lease_requests"]))
            row = probe_model(route, wallet, probe)
            with lock:
                remaining[0] -= 1
                results.append(row)
                _write(out_dir, results, probe)

    threads = [threading.Thread(target=lane, args=(rs,)) for rs in by_backend.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    order = {m: k for k, m in enumerate(models)}
    results.sort(key=lambda r: order.get(r["model"], len(order)))
    _write(out_dir, results, probe)
    return results


def _remaining_usd(ledger) -> Decimal:
    snap = ledger.snapshot()
    return Decimal(str(ledger.limit)) - Decimal(str(snap.get("committed_usd") or 0))


# ---------------------------------------------------------------- output

def caps_from_probe(results: list[dict]) -> dict:
    """A complete deployment profile: measured model caps, their backend sum, pools,
    and paid capacity including judge reserves."""
    model_caps = {r["model"]: r["safe_concurrency"] for r in results if r["safe_concurrency"] > 0}
    backends: dict[str, int] = {}
    for r in results:
        if r["model"] in model_caps:
            backends[r["backend"]] = backends.get(r["backend"], 0) + model_caps[r["model"]]
    from haenv.app.settings import PaidSection
    from haenv.runtime.ops.runtime import DEFAULT_MAX_CAPACITY
    reserved = PaidSection().judge_reserved_lanes
    requests = {b: n + reserved for b, n in backends.items()}
    return {"backend_limits": backends,
            "eval": {"model_caps": model_caps, "model_workers": model_caps,
                     "elastic_max": max(model_caps.values(), default=1)},
            "paid": {"judge_reserved_lanes": reserved,
                     "backend_request_limits": requests,
                     "max_capacity": max(DEFAULT_MAX_CAPACITY, max(requests.values(), default=1))}}


def table(results: list[dict], probe: dict) -> str:
    L = [f"Steps {probe['steps']} x {probe['step_s']}s; stop at rejections > "
         f"{probe['max_rejection_rate']:.0%} or p95 > {probe['max_latency_growth']}x the first "
         f"step; safe = floor({probe['safety_factor']} x last passing step).", "",
         "| model | backend | proxy | last pass | stopped by | safe | rejections seen | p50/p95 at last pass (s) | peak req/min | spent USD |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        last = next((s for s in reversed(r["steps"]) if s["verdict"] == "pass"), None)
        seen = {}
        for s in r["steps"]:
            for k in REJECTIONS + ("error",):
                if s["outcomes"].get(k):
                    seen[k] = seen.get(k, 0) + s["outcomes"][k]
        rpm = max((s["requests_per_min"] for s in r["steps"]), default=0)
        lat = f"{last['p50_s']}/{last['p95_s']}" if last else "-"
        safe = f"{'>=' if r['ramp_complete'] else ''}{r['safe_concurrency']}"
        L.append(f"| {r['model']} | {r['backend']} | {r.get('proxy', 'env')} | {r['last_pass']} | {r['stopped_by'] or 'ramp complete'} "
                 f"| {safe} | {seen or '-'} | {lat} | {rpm} | {Decimal(r['spent_usd']):.4f} |")
    return "\n".join(L) + "\n"


def _write(out_dir: Path, results: list[dict], probe: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"probe": probe, "models": results, "caps": caps_from_probe(results)}
    (out_dir / "probe.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    (out_dir / "probe.md").write_text(table(results, probe), encoding="utf-8")
    import yaml
    (out_dir / "concurrency.yaml").write_text(yaml.safe_dump(doc["caps"], sort_keys=False),
                                             encoding="utf-8")


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="haenv probe concurrency",
        description="Ramp tiny paid requests per model to find the concurrency each provider "
                    "accepts. Settings: config `probe:` (steps, step_s, stop rule, safety "
                    "factor); every request is reserved on --ledger first.")
    ap.add_argument("--models", default=None,
                    help="comma-separated config model names (default: default_models)")
    ap.add_argument("--ledger", type=Path, required=True,
                    help="budget ledger for the probe's own spend")
    ap.add_argument("--limit-usd", default=None,
                    help="cap for a new ledger (an existing ledger keeps its cap; a smaller "
                         "value only tightens this run)")
    ap.add_argument("--out", type=Path, required=True, help="directory for probe.json / probe.md")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config value, e.g. probe.steps=[8,16]")
    a = ap.parse_args(argv)
    import haenv.app.settings as settings
    from haenv.app.config import load_cfg
    from haenv.runtime.budget import BudgetLedger
    from haenv.runtime.solvers import load_env_file
    if a.set:
        settings.export_set(a.set)
    cfg = load_cfg()
    models = a.models.split(",") if a.models else list(cfg.get("default_models") or [])
    unknown = [m for m in models if m not in (cfg.get("models") or {})]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")
    env = load_env_file(cfg.get("env_file"))
    if not env:
        ap.error("no keys: set HAENV_ENV_FILE to the KEY=VALUE file (config.env_file)")
    ledger = BudgetLedger(a.ledger, limit_usd=a.limit_usd)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    results = run_probe(models, cfg, env, ledger, a.out)
    print(table(results, cfg["probe"]))
    print(json.dumps(caps_from_probe(results), indent=1))
    return 0
