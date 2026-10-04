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
  request was written (a disconnect before the response) is never retried here.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request

from .semantic_budget import BudgetExceeded

#: Answer-view keys (the batch identity of one solver).
ANSWER_KEYS = ("model_canonical", "upstream", "max_tokens", "reasoning_effort",
               "response_format", "sampling")
#: Route keys (recorded per receipt, never part of the identity).
ROUTE_KEYS = ("backend", "model", "endpoint_sha256", "max_tokens_field", "stream", "retries",
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
    if "model_canonical" in settings:
        return {k: settings.get(k) for k in ANSWER_KEYS}
    backend = settings.get("backend")
    return {"model_canonical": canonical_model(backend, settings.get("model")),
            "upstream": upstream_of(backend, settings.get("provider"), settings.get("upstream")),
            "max_tokens": settings.get("max_tokens"),
            "reasoning_effort": settings.get("reasoning_effort"),
            "response_format": settings.get("response_format"),
            "sampling": dict(settings.get("sampling") or {})}


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


def wire_view(backend: str, body: bytes, max_tokens_field: str = "max_tokens") -> dict:
    """Messages and answer fields as they stand in the final request body."""
    data = json.loads(body)
    if backend == "google":
        contents = data.get("contents") or []
        parts = [(p or {}).get("text") for c in contents for p in (c or {}).get("parts") or []]
        cfg = dict(data.get("generationConfig") or {})
        thinking = cfg.get("thinkingConfig") if isinstance(cfg.get("thinkingConfig"), dict) else {}
        extra = sorted((set(data) - {"contents", "generationConfig"})
                       | (set(cfg) - {"maxOutputTokens", "temperature", "thinkingConfig"})
                       | {f"thinkingConfig.{k}" for k in set(thinking) - {"thinkingBudget"}})
        roles = [(c or {}).get("role", "user") for c in contents]
        messages = ([{"role": "user", "content": parts[0]}]
                    if len(parts) == 1 and roles == ["user"] and isinstance(parts[0], str)
                    else {"contents": contents})
        return {"messages": messages, "model": None, "max_tokens": cfg.get("maxOutputTokens"),
                "reasoning_effort": None, "response_format": None,
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
    backend = solver.backend
    seen = wire_view(backend, body, getattr(solver, "max_tokens_field", "max_tokens"))
    problems = []
    if messages_sha256(seen["messages"]) != messages_sha256(expected_messages(prompt)):
        problems.append("messages")
    if seen["max_tokens"] != solver.max_tokens:
        problems.append("max_tokens")
    if seen["reasoning_effort"] != getattr(solver, "reasoning_effort", None):
        problems.append("reasoning_effort")
    if seen["response_format"] != getattr(solver, "response_format", None):
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
    from .config import load_cfg
    if _SETTINGS[0] is not None:
        return _SETTINGS[0]
    try:
        got = (load_cfg() or {}).get("transport") or {}
    except Exception:                                        # noqa: BLE001
        got = {}
    _SETTINGS[0] = {"connect_per_s": float(got.get("connect_per_s", CONNECT_PER_S)),
                    "connect_burst": int(got.get("connect_burst", CONNECT_BURST)),
                    "connect_retries": int(got.get("connect_retries", CONNECT_RETRIES))}
    return _SETTINGS[0]


_BUCKET: list = [None]
_BUCKET_LOCK = threading.Lock()


def connect_bucket() -> ConnectBucket:
    with _BUCKET_LOCK:
        if _BUCKET[0] is None:
            cfg = _settings()
            _BUCKET[0] = ConnectBucket(cfg["connect_per_s"], cfg["connect_burst"])
        return _BUCKET[0]


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
    def https_open(self, req):
        extra = {"check_hostname": self._check_hostname} if hasattr(self, "_check_hostname") else {}
        return _open_retrying(self, _HTTPSConnection, req, context=self._context, **extra)


def install() -> None:
    """Make `urllib.request.urlopen` pace connections and retry connect-phase failures.

    Environment proxies apply as before (the opener keeps urllib's default handlers).
    Runs on import; a caller passing its own `context` to `urlopen` bypasses it.
    """
    urllib.request.install_opener(urllib.request.build_opener(PacedHTTPHandler(), PacedHTTPSHandler()))


install()
