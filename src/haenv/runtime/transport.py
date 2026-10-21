"""What a paid solver request is, as the model sees it, and the pre-send wire check.

Two views of one request:

* the **answer view** (`answer_view`): canonical model, upstream provider, `max_tokens`,
  `reasoning_effort`, `response_format`, sampling. It is the batch identity and the part
  of a receipt a replay must match;
* the **route** (`route_view`): backend, endpoint, `max_tokens_field`, stream, retries,
  provider routing options, key ordinal. Recorded on every receipt, never part of the
  identity.

`verify_wire` re-reads the final request body right before it goes out and refuses to
send (`WireMismatch`) when its messages or answer fields differ from what was accounted.
It judges bytes, not file lists: code that changes what the model is sent is stopped at
its first request, before a reservation.

Importing this module makes `urllib.request.urlopen` follow two connection rules
(`config.transport`):

* new connections are paced by one token bucket per process (`connect_per_s`,
  `connect_burst`), so a burst of cells does not open dozens of proxy tunnels at once;
* a failure inside the connect phase (TCP connect, proxy `CONNECT` tunnel, TLS
  handshake) is retried up to `connect_retries` times with backoff. No request byte has
  left the process at that point, so nothing can have been billed. A failure after the
  request was written (a disconnect before the response) is never retried here;
* with `rehearsal_url` set, every HTTPS request goes to that local fake provider instead
  (`haenv loadtest`); nothing leaves the machine.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from haenv.runtime.budget import BudgetExceeded
from haenv.runtime.config_contracts import check_proxy
from haenv.runtime.config_contracts import validate_proxy_config  # noqa: F401

#: Answer-view keys (the batch identity of one solver).
ANSWER_KEYS = ("model_canonical", "upstream", "max_tokens", "reasoning_effort",
               "response_format", "sampling")
#: Route keys (recorded per receipt, never part of the identity).
ROUTE_KEYS = ("backend", "model", "proxy", "endpoint_sha256", "max_tokens_field", "stream", "retries",
              "provider")
#: Upstream provider of the direct (non-routing) backends.
DIRECT_UPSTREAM = {"google": "google", "dashscope": "alibaba-dashscope"}
#: Body keys of an OpenAI-compatible request that only route or transport it.
_OPENAI_ROUTE_BODY_KEYS = {"provider", "stream", "stream_options"}
#: Keys of a model's `sampling` block (`config.models.<m>.sampling`, part of the answer view).
#: `temperature` is a number sent as is, or `"unsupported:<why>"` for a model whose provider
#: takes no temperature (then nothing is sent and the reason is the record). `thinking_budget`
#: is Google's `generationConfig.thinkingConfig.thinkingBudget`.
SAMPLING_KEYS = ("temperature", "thinking_budget")
UNSUPPORTED = "unsupported:"


def check_sampling(name: str, backend: str, sampling) -> dict | None:
    """The validated `sampling` block of model `name` (None when it declares none)."""
    if sampling is None:
        return None
    if not isinstance(sampling, dict) or not sampling:
        raise ValueError(f"config.models[{name}].sampling must be a non-empty mapping")
    bad = sorted(set(sampling) - set(SAMPLING_KEYS))
    if bad:
        raise ValueError(f"config.models[{name}].sampling has unknown key(s) {bad}")
    t = sampling.get("temperature")
    if isinstance(t, str):
        if not t.startswith(UNSUPPORTED) or not t[len(UNSUPPORTED):].strip():
            raise ValueError(f"config.models[{name}].sampling.temperature: a number, or "
                             f"'{UNSUPPORTED}<why>' for a provider that takes none")
    elif t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 2):
        raise ValueError(f"config.models[{name}].sampling.temperature must be in [0, 2]")
    tb = sampling.get("thinking_budget")
    if tb is not None and (backend != "google" or type(tb) is not int or tb < 0):
        raise ValueError(f"config.models[{name}].sampling.thinking_budget is a Google "
                         f"thinking budget (backend google, integer >= 0)")
    return dict(sampling)


#: Model-entry keys that are the model's conditions *as a solver under test* (upstream pin,
#: sampling). A judge built from the same `config.models` entry does not inherit them: its
#: request is defined by the judging policy.
SOLVER_ONLY_KEYS = ("provider", "sampling")


def judge_spec(spec: dict) -> dict:
    """`spec` without the solver-only conditions (see `SOLVER_ONLY_KEYS`)."""
    return {k: v for k, v in dict(spec).items() if k not in SOLVER_ONLY_KEYS}


def sent_temperature(sampling) -> float | None:
    """The temperature a request carries (None: not sent)."""
    t = (sampling or {}).get("temperature")
    return None if t is None or isinstance(t, str) else t


class UpstreamMismatch(RuntimeError):
    """A pinned OpenRouter request was served by another upstream than the pin names."""


def pinned_upstream(provider) -> str | None:
    """The one upstream an OpenRouter `provider` object pins with fallbacks off, else None."""
    only = (provider or {}).get("only")
    if isinstance(only, list) and len(only) == 1 and (provider or {}).get("allow_fallbacks") is False:
        return str(only[0])
    return None


def check_pin(name: str, backend: str, provider) -> None:
    """A `provider` object pins exactly one upstream with fallbacks off, or is refused."""
    if provider is None:
        return
    if backend != "openrouter" or not isinstance(provider, dict):
        raise ValueError(f"config.models[{name}].provider is an OpenRouter routing object "
                         f"(needs backend: openrouter and a mapping); got backend={backend!r}")
    if pinned_upstream(provider) is None:
        raise ValueError(f"config.models[{name}].provider must pin one upstream with fallbacks "
                         f"off (only: [<upstream>], allow_fallbacks: false); got {provider!r}")


def served_of(response) -> dict:
    """Who served a completion, as the response says (absent fields are None)."""
    r = response if isinstance(response, dict) else {}
    return {"provider": r.get("provider") if isinstance(r.get("provider"), str) else None,
            "model": r.get("model") or r.get("modelVersion"),
            "generation_id": r.get("_generation_id") or r.get("id") or r.get("responseId"),
            "gateway_request_id": r.get("_gateway_request_id")}


def check_served(solver, response) -> None:
    """Refuse a completion a pinned route says another upstream served (never silent)."""
    pin = pinned_upstream(getattr(solver, "provider", None))
    got = served_of(response)["provider"]
    if pin is not None and got is not None and got != pin:
        raise UpstreamMismatch(f"pinned upstream {pin!r} but served by {got!r}")


class WireMismatch(BudgetExceeded):
    """The request about to be sent differs from the accounted one; nothing was sent.

    A code defect, not a failed attempt: it stops the run the way a budget stop does
    (no row is written, no retry is made), before any reservation.
    """


def canonical_model(backend: str, model: str) -> str:
    """Model name without the router's vendor prefix (`moonshotai/kimi-k3` -> `kimi-k3`)."""
    if backend == "openrouter" and "/" in str(model):
        return str(model).split("/", 1)[1]
    return str(model)


def upstream_of(backend: str, provider: dict | None, declared: str | None = None) -> str:
    """Who serves the weights. Declared (config `upstream`) wins; an OpenRouter pin to a
    single provider with fallbacks off names it; otherwise the route says it is unpinned."""
    if declared:
        return str(declared)
    if backend in DIRECT_UPSTREAM:
        return DIRECT_UPSTREAM[backend]
    only = (provider or {}).get("only")
    if (backend == "openrouter" and isinstance(only, list) and len(only) == 1
            and (provider or {}).get("allow_fallbacks") is False):
        return str(only[0])
    return f"{backend}:unpinned"


def answer_view(settings: dict) -> dict:
    """Answer view of a settings record: v1 wire settings, v2 answer view (idempotent)."""
    # `query` (how a query-mode solver is asked: tool surface, round and call limits) is part of
    # the answer view only when set, so the identity of every single-prompt solver is unchanged.
    query = {"query": dict(settings["query"])} if settings.get("query") else {}
    if "model_canonical" in settings:
        return {**{k: settings.get(k) for k in ANSWER_KEYS}, **query}
    backend = settings.get("backend")
    return {"model_canonical": canonical_model(backend, settings.get("model")),
            "upstream": upstream_of(backend, settings.get("provider"), settings.get("upstream")),
            "max_tokens": settings.get("max_tokens"),
            "reasoning_effort": settings.get("reasoning_effort"),
            "response_format": settings.get("response_format"),
            "sampling": dict(settings.get("sampling") or {}), **query}


def route_view(settings: dict, *, key_ordinal=None) -> dict:
    out = {k: settings.get(k) for k in ROUTE_KEYS if k in settings}
    out["upstream"] = answer_view(settings)["upstream"]
    out["key_ordinal"] = key_ordinal
    return out


def messages_sha256(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def expected_messages(prompt: str) -> list:
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------- query-mode envelope

#: A query-mode request (`haenv/query`) is accounted as one envelope: a JSON document of the whole
#: conversation (system text, messages, tools, tool choice) in provider-neutral form. The request
#: body's conversation keys are derived from it by `envelope_content`; `verify_wire` compares the
#: body against that derivation, so a tool request is held to the same pre-send check as a
#: single-prompt request.
ENVELOPE = "haenv-query/1"
_ENVELOPE_HEAD = '{"envelope":"' + ENVELOPE + '"'
#: Body keys that carry the conversation of an envelope request (OpenAI-compatible and Google).
ENVELOPE_KEYS = ("messages", "tools", "tool_choice", "contents", "systemInstruction", "toolConfig")


def is_envelope(prompt) -> bool:
    return isinstance(prompt, str) and prompt.startswith(_ENVELOPE_HEAD)


def make_envelope(system: str, messages: list, tools: list | None, tool_choice: str) -> str:
    """The accounted prompt of one query-mode turn. `messages` are neutral: `user` {content},
    `assistant` {content, tool_calls [{id, name, arguments}], native}, `tool` {tool_call_id,
    name, content (a JSON object)}; `native` holds what the provider returned and must get back
    verbatim (OpenAI-compatible `tool_calls` / `reasoning_details`, Google `parts` with their
    `thoughtSignature`). `tools` are `{name, description, parameters}`; `tool_choice` is
    `auto` or `none` (the forced final turn keeps the tools declared and closes them)."""
    if tool_choice not in ("auto", "none"):
        raise ValueError("tool_choice must be 'auto' or 'none'")
    return json.dumps({"envelope": ENVELOPE, "system": system, "messages": messages,
                       "tools": tools or None, "tool_choice": tool_choice},
                      ensure_ascii=False, separators=(",", ":"))


def _openai_message(m: dict, backend: str) -> dict:
    role = m.get("role")
    if role == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"],
                "content": json.dumps(m.get("content"), ensure_ascii=False, separators=(",", ":"))}
    if role != "assistant":
        return {"role": role, "content": m.get("content") or ""}
    native = m.get("native") or {}
    out = {"role": "assistant", "content": m.get("content") or ""}
    calls = native.get("tool_calls") or [
        {"id": c["id"], "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
        for c in m.get("tool_calls") or ()]
    if calls:
        out["tool_calls"] = calls
    if backend == "openrouter" and native.get("reasoning_details"):
        out["reasoning_details"] = native["reasoning_details"]
    return out


def _google_contents(messages: list) -> list:
    """Neutral messages -> Gemini `contents`. A model turn is the provider's own `parts`, sent back
    verbatim (a function call's `thoughtSignature` must return unchanged); tool results and any
    user text that follows them share one user turn, so the roles alternate."""
    contents: list[dict] = []

    def user(part):
        if contents and contents[-1]["role"] == "user":
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": "user", "parts": [part]})
    for m in messages:
        role = m.get("role")
        if role == "tool":
            part = {"functionResponse": {"name": m.get("name"), "response": m.get("content") or {}}}
            if str(m.get("tool_call_id") or "").startswith("g:"):
                part["functionResponse"]["id"] = m["tool_call_id"][2:]
            user(part)
        elif role == "assistant":
            parts = (m.get("native") or {}).get("parts")
            if parts is None:
                parts = ([{"text": m["content"]}] if m.get("content") else []) + [
                    {"functionCall": {"name": c["name"], "args": c.get("arguments") or {}}}
                    for c in m.get("tool_calls") or ()]
            contents.append({"role": "model", "parts": parts})
        else:
            user({"text": m.get("content") or ""})
    return contents


def envelope_content(backend: str, prompt: str) -> dict:
    """The conversation keys of the request body for an envelope prompt on `backend`."""
    env = json.loads(prompt)
    if env.get("envelope") != ENVELOPE:
        raise ValueError("not a query-mode envelope")
    tools, choice, system = env.get("tools"), env["tool_choice"], env.get("system") or ""
    if backend == "google":
        out: dict = {"contents": _google_contents(env["messages"])}
        if system:
            out["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            decl = []
            for t in tools:
                d = {"name": t["name"], "description": t["description"]}
                if (t.get("parameters") or {}).get("properties"):     # Gemini refuses an empty object schema
                    d["parameters"] = t["parameters"]
                decl.append(d)
            out["tools"] = [{"functionDeclarations": decl}]
            out["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO" if choice == "auto" else "NONE"}}
        return out
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        _openai_message(m, backend) for m in env["messages"]]
    out = {"messages": msgs}
    if tools:
        out["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                          "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}
                        for t in tools]
        out["tool_choice"] = choice
    return out


def solver_envelope_content(solver, prompt: str) -> dict:
    """Keep MiniMax/Novita's implicit automatic tool selection wire-compatible.

    The router reports this route supports tools but no explicit tool_choice values.
    With tools present the API default is auto; with tools absent they are closed.
    All conversation messages, sampling and the pinned upstream remain unchanged.
    """
    content = envelope_content(solver.backend, prompt)
    if (solver.backend == "openrouter" and solver.model == "minimax/minimax-m3"
            and pinned_upstream(getattr(solver, "provider", None)) == "Novita"):
        choice = content.pop("tool_choice", None)
        if choice == "none":
            content.pop("tools", None)
    return content


def google_response_format(response_format: dict) -> dict:
    """Google's JSON Schema wire fields; OpenAI name/strict are not sent."""
    if response_format.get("type") != "json_schema":
        raise ValueError("Google structured output requires a JSON schema")
    return {"responseMimeType": "application/json",
            "responseJsonSchema": response_format["json_schema"]["schema"]}


def wire_view(backend: str, body: bytes, max_tokens_field: str = "max_tokens") -> dict:
    """Messages and answer fields as they stand in the final request body."""
    data = json.loads(body)
    if backend == "google":
        contents = data.get("contents") or []
        parts = [(p or {}).get("text") for c in contents for p in (c or {}).get("parts") or []]
        cfg = dict(data.get("generationConfig") or {})
        thinking = cfg.get("thinkingConfig") if isinstance(cfg.get("thinkingConfig"), dict) else {}
        extra = sorted((set(data) - {"contents", "generationConfig"})
                       | (set(cfg) - {"maxOutputTokens", "temperature", "thinkingConfig",
                                      "responseMimeType", "responseJsonSchema"})
                       | {f"thinkingConfig.{k}" for k in set(thinking) - {"thinkingBudget"}})
        roles = [(c or {}).get("role", "user") for c in contents]
        messages = ([{"role": "user", "content": parts[0]}]
                    if len(parts) == 1 and roles == ["user"] and isinstance(parts[0], str)
                    else {"contents": contents})
        return {"messages": messages, "model": None, "max_tokens": cfg.get("maxOutputTokens"),
                "reasoning_effort": None,
                "response_format": ({k: cfg[k] for k in ("responseMimeType", "responseJsonSchema") if k in cfg}
                                    or None),
                "temperature": cfg.get("temperature"),
                "thinking_budget": thinking.get("thinkingBudget"), "unexpected_keys": extra}
    effort = data.get("reasoning_effort")
    if isinstance(data.get("reasoning"), dict):
        effort = data["reasoning"].get("effort")
    known = {"model", "messages", max_tokens_field, "reasoning", "reasoning_effort",
             "response_format", "temperature"} | _OPENAI_ROUTE_BODY_KEYS
    return {"messages": data.get("messages"), "model": data.get("model"),
            "max_tokens": data.get(max_tokens_field),
            "reasoning_effort": effort, "response_format": data.get("response_format"),
            "temperature": data.get("temperature"), "thinking_budget": None,
            "unexpected_keys": sorted(set(data) - known)}


def verify_wire(solver, body: bytes, prompt: str) -> None:
    """Refuse to send `body` unless it carries exactly the accounted prompt and answer fields.

    Compares the final body's messages with the prompt the request was reserved for, its
    `max_tokens` / effort / response format / model with the solver's answer settings, and
    refuses any body key it does not know (a sampling key would change the answer).
    """
    original_body = body
    backend = solver.backend
    problems = []
    if is_envelope(prompt):
        # The conversation keys must be exactly the envelope's derivation; the rest of the body
        # is checked below as for any request (no conversation key may appear outside one).
        data = json.loads(body)
        if {k: data[k] for k in ENVELOPE_KEYS if k in data} != solver_envelope_content(solver, prompt):
            problems.append("messages")
        body = json.dumps({k: v for k, v in data.items() if k not in ENVELOPE_KEYS}).encode()
    seen = wire_view(backend, body, getattr(solver, "max_tokens_field", "max_tokens"))
    if not is_envelope(prompt) and messages_sha256(seen["messages"]) != messages_sha256(expected_messages(prompt)):
        problems.append("messages")
    if seen["max_tokens"] != solver.max_tokens:
        problems.append("max_tokens")
    if seen["reasoning_effort"] != getattr(solver, "reasoning_effort", None):
        problems.append("reasoning_effort")
    expected_format = getattr(solver, "response_format", None)
    if backend == "google" and expected_format is not None:
        expected_format = google_response_format(expected_format)
    if seen["response_format"] != expected_format:
        problems.append("response_format")
    if backend != "google" and seen["model"] != solver.model:
        problems.append("model")
    sampling = getattr(solver, "sampling", None)
    if seen["temperature"] != sent_temperature(sampling):
        problems.append("temperature")
    if seen["thinking_budget"] != (sampling or {}).get("thinking_budget"):
        problems.append("thinking_budget")
    if seen["unexpected_keys"]:
        problems.append("unexpected body keys " + ",".join(seen["unexpected_keys"]))
    if problems:
        raise WireMismatch("request body differs from the accounted request ("
                           + "; ".join(problems) + "); not sent")
    # The frozen solver send sites already call this immediately before urlopen. Bind
    # routing to these exact bytes and endpoint, never to a host or a thread alone.
    _WIRE_ROUTE.pending = (solver.URL, original_body, proxy_for_solver(solver))


def proxy_for_model(name=None, *, backend=None, cfg=None, spec=None) -> str:
    """Model override, then config.backends.<backend>.proxy, then env."""
    if cfg is None:
        from haenv.app.config import load_cfg
        cfg = load_cfg()
    models = cfg.get("models") or {}
    entry = spec if spec is not None else (models.get(name, {}) if isinstance(models, dict) else {})
    backend = backend or entry.get("backend", "openrouter")
    return check_proxy(entry.get("proxy", ((cfg.get("backends") or {}).get(backend) or {})
                                 .get("proxy", (cfg.get("transport") or {}).get("backend_proxies", {}).get(backend, "env"))))


def proxy_for_solver(solver, cfg=None) -> str:
    declared = getattr(solver, "proxy", None)
    return (check_proxy(declared) if declared is not None else
            proxy_for_model(getattr(solver, "name", None),
                            backend=getattr(solver, "backend", "openrouter"), cfg=cfg))


def model_urlopen(request, *, name=None, backend=None, cfg=None, spec=None,
                  proxy=None, proxy_url=None, timeout=30):
    """Open a model or backend request through the same selector as solver sends."""
    request._haenv_proxy = (check_proxy(proxy) if proxy is not None else
                            proxy_for_model(name, backend=backend, cfg=cfg, spec=spec))
    request._haenv_proxy_url = proxy_url
    return urllib.request.urlopen(request, timeout=timeout)


def log_proxy_routes(models, cfg, logger) -> None:
    routes = {"direct": [], "env": []}
    for name in models:
        routes.setdefault(proxy_for_model(name, cfg=cfg), []).append(name)
    logger.info("Proxy routes: direct=%s env=%s", sorted(routes["direct"]), sorted(routes["env"]))
    for proxy, names in routes.items():
        if proxy not in {"direct", "env"}:
            logger.info("Proxy routes: %s=%s", proxy, sorted(names))


_WIRE_ROUTE = threading.local()


class ExplicitProxyHandler(urllib.request.ProxyHandler):
    """A route URL wins over process NO_PROXY, without mutating the environment."""
    def proxy_open(self, request, proxy, protocol):
        target = urllib.parse.urlsplit(proxy)
        original = request.type
        request.set_proxy(target.netloc, target.scheme)
        if original == target.scheme or original == "https":
            return None
        return self.parent.open(request, timeout=request.timeout)


class ModelOpener:
    """Keep the env opener and dispatch verified model sends before proxy handling."""
    def __init__(self):
        self.env = urllib.request.build_opener(PacedHTTPHandler(), PacedHTTPSHandler())
        self.direct = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                  PacedHTTPHandler(), PacedHTTPSHandler())

    def open(self, request, data=None, timeout=None):
        # Provider catalogs are free GETs. A ledger scope can serve the identical
        # catalog to the frozen judge preflight without changing its price checks.
        if (data is None and isinstance(request, urllib.request.Request)
                and request.get_method() == "GET" and request.has_header("Authorization")
                and urllib.parse.urlsplit(request.full_url).path.rstrip('/').endswith('/models')):
            from haenv.runtime.solver_accounting import catalog_response
            response = catalog_response(request)
            if response is not None:
                return response
        route = getattr(request, "_haenv_proxy", None)
        pending = getattr(_WIRE_ROUTE, "pending", None)
        if route is None and pending and isinstance(request, urllib.request.Request):
            endpoint, body, proxy = pending
            if (request.data == body and (request.full_url == endpoint or
                    request.full_url.startswith(endpoint.rstrip("/") + "/"))):
                route = proxy
        opener = self.direct if route == "direct" else self.env
        proxy_url = getattr(request, "_haenv_proxy_url", None) or (route if route not in (None, "env", "direct") else None)
        if route != "direct" and proxy_url:
            opener = urllib.request.build_opener(
                ExplicitProxyHandler({"http": proxy_url, "https": proxy_url}),
                PacedHTTPHandler(), PacedHTTPSHandler())
        if timeout is None:
            return opener.open(request, data)
        return opener.open(request, data, timeout)


# ---------------------------------------------------------------- connections

#: Defaults when `config.transport` is not set.
CONNECT_PER_S, CONNECT_BURST, CONNECT_RETRIES = 8.0, 8, 3


class PreSendFailure(OSError):
    """The connection failed before any request byte was written."""

    def __init__(self, cause: BaseException):
        super().__init__(f"{type(cause).__name__} while connecting; the request was not sent")
        self.cause = cause


class ConnectBucket:
    """Token bucket for new connections; `take` blocks until a token is free."""

    def __init__(self, rate: float, burst: int, clock=time.monotonic, sleep=time.sleep):
        if rate <= 0 or burst < 1:
            raise ValueError("config.transport.connect_per_s and connect_burst must be positive")
        self.rate, self.burst, self.clock, self.sleep = float(rate), int(burst), clock, sleep
        self.tokens, self.at = float(burst), clock()
        self.lock = threading.Lock()

    def take(self) -> float:
        with self.lock:
            now = self.clock()
            self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
            self.at = now
            self.tokens -= 1.0
            wait = max(0.0, -self.tokens / self.rate)
        if wait:
            self.sleep(wait)
        return wait


_SETTINGS: list = [None]


def _settings() -> dict:
    from haenv.app.config import load_cfg
    if _SETTINGS[0] is not None:
        return _SETTINGS[0]
    try:
        got = (load_cfg() or {}).get("transport") or {}
    except Exception:                                        # noqa: BLE001
        got = {}
    _SETTINGS[0] = {"connect_per_s": float(got.get("connect_per_s", CONNECT_PER_S)),
                    "connect_burst": int(got.get("connect_burst", CONNECT_BURST)),
                    "connect_retries": int(got.get("connect_retries", CONNECT_RETRIES)),
                    "rehearsal_url": got.get("rehearsal_url")}
    return _SETTINGS[0]


_BUCKET: list = [None]
_BUCKET_LOCK = threading.Lock()


def connect_bucket() -> ConnectBucket:
    with _BUCKET_LOCK:
        if _BUCKET[0] is None:
            cfg = _settings()
            _BUCKET[0] = ConnectBucket(cfg["connect_per_s"], cfg["connect_burst"])
        return _BUCKET[0]


def use_connect_pacing(per_s: float, burst: int) -> None:
    """Pace this process's new connections at `per_s` with `burst`, replacing the
    `config.transport` bucket (for a tool that measures the provider's own limits)."""
    with _BUCKET_LOCK:
        _BUCKET[0] = ConnectBucket(per_s, burst)


def _paced(base):
    class Paced(base):
        def connect(self):
            connect_bucket().take()
            try:
                super().connect()
            except Exception as error:                       # noqa: BLE001
                self.close()
                raise PreSendFailure(error) from error
    Paced.__name__ = f"Paced{base.__name__}"
    return Paced


_HTTPConnection = _paced(http.client.HTTPConnection)
_HTTPSConnection = _paced(http.client.HTTPSConnection)


_RETRIES = threading.local()


def connect_retries_taken() -> int:
    """Connect-phase retries of the calling thread's last request."""
    return getattr(_RETRIES, "n", 0)


def _open_retrying(handler, conn, req, **kw):
    retries = _settings()["connect_retries"]
    _RETRIES.n = 0
    for attempt in range(retries + 1):
        try:
            return handler.do_open(conn, req, **kw)
        except urllib.error.URLError as error:
            if (isinstance(error, urllib.error.HTTPError)
                    or not isinstance(error.reason, PreSendFailure) or attempt == retries):
                raise
            _RETRIES.n = attempt + 1
            _SLEEP[0](min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))


_SLEEP = [time.sleep]


class PacedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return _open_retrying(self, _HTTPConnection, req)


class PacedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_request(self, req):
        req = super().https_request(req)
        target = _settings()["rehearsal_url"]
        if target:
            rehearse(req, target)
        return req

    def https_open(self, req):
        extra = {"check_hostname": self._check_hostname} if hasattr(self, "_check_hostname") else {}
        return _open_retrying(self, _HTTPSConnection, req, context=self._context, **extra)


def rehearse(req, target: str) -> None:
    """Point an HTTPS request at the fake provider `target` over HTTP.

    The original host travels in the `Host` header, so the fake provider answers as that
    provider; the path and query are kept.
    """
    original = urllib.parse.urlsplit(req.full_url)
    req.add_unredirected_header("Host", original.netloc)
    path = original.path + (f"?{original.query}" if original.query else "")
    req.full_url = target.rstrip("/") + path


def install() -> None:
    """Make `urllib.request.urlopen` pace connections and retry connect-phase failures.

    Environment proxies apply as before (the opener keeps urllib's default handlers).
    Runs on import; a caller passing its own `context` to `urlopen` bypasses it.
    """
    urllib.request.install_opener(ModelOpener())


install()
