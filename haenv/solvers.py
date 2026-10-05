"""Solvers and backends: the CLI, OpenAI-compatible and Google solvers, backend registration,
key pools and route fallback.

Split out of `haenv/evaluate.py`; `evaluate` re-exports every name defined here.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from .run_scheduler import stall_guard as _stall_guard
import hashlib
import json
import os
import subprocess
import threading as _threading
import time
from pathlib import Path
from haenv_kernel.schema import SolverOutput
from haenv_kernel.solver import Solver, BaselineSolver, RobustSolver, _extract_json
from . import call_log as _llm
from .run_state import (  # noqa: F401
    log,
)
from . import qside as _qside   # `render_for` is a plugin hook point: read it live
from .qside import (  # noqa: F401
    render_for,
)
from .call_log import billed_tokens
from .baselines import ConstantDdxSolver
from .baselines import FlipFlopSolver
from .baselines import GatedProbeSolver
from .baselines import GatedShotgunSolver
from .baselines import HumbleSolver
from .baselines import OracleProbeSolver
from .baselines import OracleTestsSolver
from .baselines import TestOrderingSolver
from .baselines import OracleReviewSolver
from .baselines import OneLongTestSolver
from .baselines import NoReviewFlagSolver
from .baselines import LateConvergeSolver
from .baselines import GateTripSolver
from .baselines import TraceJunkSolver
from .baselines import BlindConfidentSolver
from .baselines import GatedSignalFloodSolver
from .baselines import GatedMinimalSolver
from .transport import verify_wire
from .semantic_budget import BudgetExceeded
from .call_log import (  # noqa: F401
    _EXHAUSTED_MARKERS,
    _is_exhausted,
)
from .solve_guard import (  # noqa: F401
    json_unextractable,
)


def _dq_hashable(dq):
    """Coerce `data_quality.signal_quality` values to strings.

    The kernel gate `acted_on_unverified_signal` tests `sq.get(sig) in {...}`, which raises
    on a dict value. Non-string values become their JSON text, which never matches, so
    verdicts are unchanged.
    """
    if not isinstance(dq, dict):
        # `None`, not `"insufficient_data"`: a non-object `data_quality` is not a declaration of
        # insufficiency, which would exempt the answer from `acted_on_unverified_signal`.
        return {"data_sufficiency": None, "signal_quality": {}}
    sq = dq.get("signal_quality")
    if not isinstance(sq, dict):
        return dq
    import json as _json
    fixed = {}
    for k, v in sq.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            fixed[k] = v
        else:                                    # dict / list => coerce to text (still never matches a flag word)
            try:
                fixed[k] = _json.dumps(v, ensure_ascii=False)
            except Exception:                    # noqa: BLE001
                fixed[k] = str(v)
    out = dict(dq)
    out["signal_quality"] = fixed
    return out


def _to_output(data: dict, payload) -> SolverOutput:
    """Collapse the model's JSON into a `SolverOutput`; never crashes, never fabricates.

    Fields outside the kernel signature (differential, join_type, tests_to_order, ...) are kept
    on `_raw`, which the judges read. When the model answered but omitted `action` or
    `data_quality`, `clinician_review_required` and `data_sufficiency` stay `None`: a canned
    `True` / `"insufficient_data"` would assert a safety flag or grant a gate exemption the
    model never gave. Only a parse failure (`data == {}`) gets the full canned abstention.
    """
    # A field of the wrong JSON type (e.g. `"action": "refer to endocrinology"`) is treated as
    # omitted, so the rest of the answer is still graded instead of the cell crashing.
    if isinstance(data, dict):
        data = dict(data)
        for _k in ("forecast", "action", "data_quality"):
            if _k in data and not isinstance(data[_k], dict):
                data.pop(_k)
        if "drivers" in data:
            _dr = data["drivers"]
            data["drivers"] = ([d for d in _dr if isinstance(d, dict)]
                               if isinstance(_dr, list) else [])
    f = data.get("forecast", {}) or {}
    cited = data.get("cited_evidence") or f.get("key_predictive_evidence") or []
    _answered = isinstance(data, dict) and bool(data)
    out = SolverOutput(
        forecast={"target_event": f.get("target_event", payload.prediction_context.get("target_event_type")),
                  "risk": float(f.get("risk", 0.5) or 0.5),
                  "risk_category": f.get("risk_category", "indeterminate"),
                  "confidence": float(f.get("confidence", 0.3) or 0.3),
                  "key_predictive_evidence": f.get("key_predictive_evidence", [])},
        drivers=data.get("drivers", []),
        action=data.get("action", {"selected_action_class": "A1",
                                   "specific_action": "collect more data (parse-fail abstain)",
                                   "what_not_to_do": [],
                                   "clinician_review_required": (None if _answered else True),
                                   "followup_interval": "7d"}),
        data_quality=_dq_hashable(data.get(
            "data_quality", {"data_sufficiency": (None if _answered else "insufficient_data"),
                             "signal_quality": {}})),
        cited_evidence=cited)
    out._raw = data if isinstance(data, dict) else {}     # noqa: SLF001 -- see docstring
    # `SolverOutput` has no `notes` field, but `runner.run_multiround` reads `out.notes` and
    # `_premise_suffix` asks the model to use it; normalized to `str`.
    _n = data.get("notes") if isinstance(data, dict) else None
    out.notes = (_n.strip() if isinstance(_n, str)
                 else " ".join(str(x).strip() for x in _n if str(x).strip()) if isinstance(_n, list)
                 else "")
    return out


class CLISolverBlocked(RuntimeError):
    """The CLI subprocess path refuses to start by default; see `CLISolver`."""


_CLI_SANDBOX_CWD: str | None = None


def _cli_cwd() -> str:
    global _CLI_SANDBOX_CWD
    if _CLI_SANDBOX_CWD is None:
        import tempfile
        _CLI_SANDBOX_CWD = tempfile.mkdtemp(prefix="haenv-cli-cwd-")
    return _CLI_SANDBOX_CWD


def secret_env_names(env: dict) -> list[str]:
    """Environment variable names that look like credentials. Returns names
    only, never values."""
    import re
    pat = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL", re.I)
    return sorted(k for k in (env or {}) if pat.search(str(k)))


class CLISolver(Solver):
    """Dispatches to a real model via `~/.local/bin/ai`; a parse failure degrades to abstention.

    Refuses to start unless `allow_cli_solver: true`: an agentic CLI with file tools could
    read ground-truth files. When allowed, the subprocess runs from an empty directory outside
    the repo, but env and `$HOME` (needed for credentials) are not contained; the names of
    credential-shaped variables are logged.
    """

    def __init__(self, name: str, ai_args: list[str], ai: str, timeout: int, retries: int = 2,
                 env: dict[str, str] | None = None, allow_unsandboxed: bool = False):
        if not allow_unsandboxed:
            raise CLISolverBlocked(
                f"model {name!r} takes the CLI-subprocess path, which refuses to start by "
                f"default: an agentic CLI with file tools given the repo root as cwd and the "
                f"full env is a leak risk, and containment of env/HOME is still unresolved. "
                f"To allow it anyway: set `allow_cli_solver: true` at the "
                f"top level of config.yaml, knowing that the residual exposure will be logged "
                f"(see evaluate.CLISolver's docstring).")
        self.name, self.ai_args, self.ai, self.timeout, self.retries = name, ai_args, ai, timeout, retries
        self.env = {**os.environ, **(env or {})}
        _names = secret_env_names(self.env)
        log.warning("[solver %s] CLI path allowed: cwd=%s (outside the repo); "
                    "residual exposure: %d credential-shaped env var(s) %s; "
                    "$HOME is not isolated (transcripts contain ground truth)",
                    name, _cli_cwd(), len(_names), _names)

    def solve(self, payload):
        prompt, pid = _qside.render_for(self, payload)
        data, raw_text = {}, ""
        for attempt in range(1, self.retries + 2):
            try:
                proc = subprocess.run([self.ai, *self.ai_args], input=prompt,
                                      capture_output=True, text=True, timeout=self.timeout,
                                      env=self.env, cwd=_cli_cwd())
                raw_text = proc.stdout
                data = _extract_json(raw_text)
                break
            except Exception as e:
                log.warning("[solver %s] attempt %d failed: %s", self.name, attempt, e)
                time.sleep(2 * attempt)
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return out


# ============================================================ backend registry (default-deny)
#
# The relay host must be in `no_proxy`. A liveness probe must send a real-scale request: the
# relay accepts a tiny request on a nearly exhausted key.
BACKENDS: dict[str, dict] = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "key_env": "OPENROUTER_API_KEY", "kind": "openai_compat"},
    "openai":     {"url": "https://api.openai.com/v1/chat/completions",
                   "key_env": "OPENAI_API_KEY", "kind": "openai_compat",
                   # OpenAI's newer models only accept max_completion_tokens,
                   # not max_tokens
                   "max_tokens_field": "max_completion_tokens"},
    # An OpenAI-compatible relay. Its url and key names are deployment-specific and come from
    # `config.backends.relay` (see `register_backends`); until a url is configured, its models are
    # skipped. The relay reports an exhausted key as 401, so keys are pooled by the
    # `key_env_pool` prefix and rotated (see `_KeyPool`).
    "relay":      {"url": None, "key_env": "RELAY_API_KEY", "key_env_pool": "RELAY_KEY_",
                   "kind": "openai_compat"},
    "dashscope":  {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                   "key_env": "DASHSCOPE_API_KEY", "kind": "openai_compat",
                   "no_proxy_host": "dashscope.aliyuncs.com"},
    "google":     {"url": "https://generativelanguage.googleapis.com/v1beta/models",
                   "key_env": "GOOGLE_GENERATIVE_AI_API_KEY", "kind": "google"},
}


#: Fields a config entry may set on a backend; anything else is rejected.
_BACKEND_FIELDS = ("url", "key_env", "key_env_pool", "kind", "no_proxy_host",
                   "max_tokens_field", "quota_url")


def register_backends(cfg: dict | None) -> None:
    """Apply `config.backends.<name>` over the built-in table.

    Used for deployment-specific endpoints (the relay's url, key names, billing
    endpoint). Entries override field by field; a new name registers a new
    `openai_compat` backend. Unknown fields raise.
    """
    for name, over in ((cfg or {}).get("backends") or {}).items():
        if not isinstance(over, dict):
            raise ValueError(f"config.backends.{name} must be a mapping")
        unknown = sorted(set(over) - set(_BACKEND_FIELDS))
        if unknown:
            raise ValueError(f"config.backends.{name}: unknown field(s) {unknown}; "
                             f"allowed: {list(_BACKEND_FIELDS)}")
        base = dict(BACKENDS.get(name) or {"kind": "openai_compat"})
        base.update({k: v for k, v in over.items() if v is not None})
        BACKENDS[str(name)] = base


_BACKENDS_FROM_CFG = False


def _ensure_backends_registered(cfg: dict | None = None) -> None:
    from .config import load_cfg
    global _BACKENDS_FROM_CFG
    if _BACKENDS_FROM_CFG:
        return
    if cfg is None:
        try:
            cfg = load_cfg()
        except FileNotFoundError:
            cfg = {}
    register_backends(cfg)
    _BACKENDS_FROM_CFG = True


class _KeyPool:

    def __init__(self, keys: list[str], backend: str, ordinals: list | None = None):
        self.backend = backend
        # Key ordinals (the numeric suffix of the env name); request accounting logs and
        # disables keys by ordinal, never by value.
        pairs = [(k, o) for k, o in zip(keys, ordinals or [None] * len(keys)) if k]
        self._keys = [k for k, _ in pairs]
        self.ordinals = [o if o is not None else i for i, (_, o) in enumerate(pairs)]
        self._dead: set[str] = set()
        self._rr = 0          # round-robin cursor
        self._lock = _threading.Lock()

    def current(self) -> str:
        """Return the next usable key, round-robin, so concurrent requests use distinct keys and
        avoid per-key rate limits.
        """
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                k = self._keys[self._rr % n]
                self._rr += 1
                if k not in self._dead:
                    return k
        return ""

    def mark_dead(self, key: str) -> bool:
        with self._lock:
            if key and key not in self._dead:
                self._dead.add(key)
                log.warning("[keypool %s] a key ran out of quota (%d/%d dead) -> rotating",
                            self.backend, len(self._dead), len(self._keys))
            return any(k not in self._dead for k in self._keys)

    @property
    def n_alive(self) -> int:
        return sum(1 for k in self._keys if k not in self._dead)


_KEY_POOLS: dict[str, _KeyPool] = {}


def _load_key_pool(bdef: dict, env: dict, backend: str) -> _KeyPool:
    """Collect keys by the `key_env_pool` prefix, sorted by numeric suffix; falls back to the
    single `key_env`.
    """
    prefix = bdef.get("key_env_pool")
    if not prefix:
        return _KeyPool([env.get(bdef["key_env"], "")], backend)

    def _idx(name: str) -> int:
        tail = name[len(prefix):]
        return int(tail) if tail.isdigit() else 10 ** 6

    names = sorted((n for n in env if n.startswith(prefix) and _idx(n) < 10 ** 6), key=_idx)
    keys = [env[n] for n in names if str(env[n]).startswith("sk-")]
    if not keys:
        keys = [env.get(bdef["key_env"], "")]
    log.info("[keypool %s] loaded %d key(s) (%s)", backend, len(keys),
             ", ".join(names) or bdef["key_env"])
    ordinals = [_idx(n) for n in names if str(env[n]).startswith("sk-")]
    return _KeyPool(keys, backend, ordinals if len(ordinals) == len(keys) else None)


class OpenAICompatSolver(Solver):
    """Connects directly to any OpenAI-compatible endpoint (OpenRouter / OpenAI / a relay /
    dashscope). The prompt and payload are identical across backends.
    """

    def __init__(self, name: str, model: str, api_key: str, timeout: int, retries: int = 2,
                 max_tokens: int = 6000, url: str = "", backend: str = "openrouter",
                 max_tokens_field: str = "max_tokens", stream: bool = False,
                 reasoning_effort: str | None = None, response_format: dict | None = None,
                 provider: dict | None = None, sampling: dict | None = None):
        self.name, self.model, self.timeout, self.retries = name, model, timeout, retries
        # Declared sampling (`config.models.<m>.sampling`, see `transport.check_sampling`).
        self.sampling = dict(sampling) if sampling else None
        # OpenRouter upstream routing (`provider` request field), e.g. {"only": ["Moonshot AI"],
        # "allow_fallbacks": False}. None = the router picks the upstream (the default).
        self.provider = dict(provider) if provider else None
        self.api_key, self.max_tokens = api_key, max_tokens
        self.pool: _KeyPool | None = None        # injected by build_solvers; None = the single-key path
        self.backend = backend
        self.URL = url or BACKENDS["openrouter"]["url"]
        self.max_tokens_field = max_tokens_field
        # `stream` changes only the transport (payload and max_tokens are identical); used for
        # models that hit the gateway's 300 s idle timeout.
        self.stream = bool(stream)
        if reasoning_effort is not None and reasoning_effort not in {
                "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unrecognized reasoning_effort")
        self.reasoning_effort = reasoning_effort
        self.response_format = response_format
        self._last_reasoning_chars = 0
        if not api_key:
            log.error("[solver %s] backend=%s is missing %s (check config.yaml:env_file)",
                      name, backend, BACKENDS.get(backend, {}).get("key_env", "?"))

    def _post(self, prompt: str) -> dict:
        from .transport import check_served
        accounting = getattr(self, "_accounting", None)
        resp = (accounting.request(self, prompt, self._post_wire) if accounting is not None
                else self._post_wire(prompt))
        # The receipt already holds this response: a pin the router did not honor fails the
        # attempt here, recorded, never taken as an answer.
        check_served(self, resp)
        if self.stream and accounting is not None:
            self._last_reasoning_chars = resp.get("_reasoning_chars", 0)
            self._last_reasoning_text = resp.get("_reasoning_text")
            choice = (resp.get("choices") or [{}])[0]
            answer = (choice.get("message") or {}).get("content") or ""
            self._validate_stream(answer, self._last_reasoning_chars, resp.get("usage"),
                                  choice.get("finish_reason"))
        # Cost-bearing error responses must be persisted and settled before the
        # answer retry loop sees them. No error-body token/cost is fabricated.
        if isinstance(resp, dict) and resp.get("error") and not resp.get("choices"):
            error = resp["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"error body with HTTP 200: {str(message)[:160]}")
        return resp

    def _post_wire(self, prompt: str, *, api_key: str | None = None) -> dict:
        """Send one request. Only a recognized "quota exhausted" body (both it and a bad key are 401)
        switches keys and resends, without consuming the cell's retry budget. `api_key` (set by
        request accounting) pins the key for this single attempt; the pool is then not used.
        """
        return self._send_wire(self._wire_body(prompt), prompt, api_key=api_key)

    def _wire_body(self, prompt: str) -> bytes:
        """The request body for `prompt` (what the model is sent and how it is asked)."""
        payload_d: dict = {"model": self.model, self.max_tokens_field: self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}
        if self.reasoning_effort is not None:
            if self.backend == "openrouter":
                payload_d["reasoning"] = {"effort": self.reasoning_effort}
            else:
                payload_d["reasoning_effort"] = self.reasoning_effort
        from .transport import sent_temperature
        temperature = sent_temperature(self.sampling)
        if temperature is not None:
            payload_d["temperature"] = temperature
        if self.provider is not None:
            payload_d["provider"] = dict(self.provider)
        if temperature is not None and self.backend == "openrouter":
            # The router drops a parameter its upstream does not take unless told not to.
            payload_d["provider"] = {**payload_d.get("provider", {}), "require_parameters": True}
        if self.response_format is not None:
            payload_d["response_format"] = self.response_format
            if self.backend == "openrouter":
                payload_d["provider"] = {**payload_d.get("provider", {}), "require_parameters": True}
        if self.stream:
            payload_d["stream"] = True
            payload_d["stream_options"] = {"include_usage": True}
        _llm.note_payload(self.name, payload_d)
        return json.dumps(payload_d).encode()

    def _send_wire(self, body: bytes, prompt: str, *, api_key: str | None = None) -> dict:
        """Send `body` (transport only). Refuses a body that differs from the accounted one.
        `api_key` pins the key for this attempt (see `_post_wire`)."""
        import urllib.request
        import urllib.error
        verify_wire(self, body, prompt)
        while True:
            key = api_key or (self.pool.current() if self.pool else self.api_key)
            req = urllib.request.Request(
                self.URL, data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            # Transport progress, attached to any failure for the accounting receipt.
            wire = {"http_status": None, "response_started": False, "bytes_received": 0,
                    "generation_id": None}
            try:
                with _stall_guard(self) as _stall, \
                        urllib.request.urlopen(req, timeout=self.timeout) as r:
                    wire["http_status"], wire["response_started"] = getattr(r, "status", None), True
                    # OpenRouter names the generation in a response header, before any body byte:
                    # a request cut at any later point can still be settled from its record.
                    _gid = getattr(getattr(r, "headers", None), "get", lambda _k: None)("X-Generation-Id")
                    if isinstance(_gid, str) and _gid:
                        from .paid_completion import note_generation
                        wire["generation_id"] = _gid
                        note_generation(_gid)
                    r = _stall.attach(r)
                    accounted = getattr(self, "_accounting", None) is not None
                    if self.stream:
                        _resp = self._read_sse(r, defer_validation=accounted, wire=wire)
                    else:
                        _raw = r.read()
                        wire["bytes_received"] = len(_raw)
                        _resp = json.loads(_raw)
                    if accounted and self.backend == "relay" and isinstance(_resp, dict):
                        _resp["_gateway_request_id"] = r.headers.get("X-Oneapi-Request-Id")
                # Recorded inline rather than via `llm.observe_post`, whose monkey-patched `_post` would be
                # shared by the shallow copies `_const` makes.
                _llm.note_call(self.name, self.backend, _resp if isinstance(_resp, dict) else None)
                # HTTP-200 error bodies are checked by _post after the raw
                # cost-bearing response has crossed the accounting boundary.
                return _resp
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")
                except Exception:
                    pass
                # Kept for the failed-attempt receipt (the body can be read only once).
                e.haenv_error_body = detail.replace(key, "[redacted]") if key else detail
                e.haenv_wire = {**wire, "http_status": e.code, "response_started": False}
                if not self.pool:
                    raise
                if _is_exhausted(detail) and self.pool.mark_dead(key):
                    continue                      # switch to the next key, resend the same request
                if _is_exhausted(detail):
                    raise RuntimeError(
                        f"relay key pool exhausted ({self.pool.n_alive} usable)") from e
                raise
            except Exception as e:
                try:
                    e.haenv_wire = dict(wire)
                except Exception:  # noqa: BLE001 -- an exception type without attributes
                    pass
                raise

    #: Only `content` is spliced into the answer; reasoning fields (e.g. `reasoning_content`) are
    #: kept apart, matching the non-streaming path, which returns `message.content` only.
    _SSE_ANSWER_FIELD = "content"
    _SSE_THINK_FIELDS = ("reasoning_content", "reasoning")

    def _read_sse(self, r, *, defer_validation: bool = False, wire: dict | None = None) -> dict:
        """Reconstruct an SSE stream into a response equivalent to the non-streaming one.

        Streaming keeps the connection active under the relay's 300 s idle timeout without
        changing the payload or `max_tokens`.
        """
        from .paid_completion import note_generation
        txt, think_n, usage, fin = [], 0, None, None
        generation_id = None                             # provider id; exact cost lookup if usage is lost
        served, served_model = None, None                # OpenRouter names the upstream on each chunk
        think_txt: list[str] = []                        # raw reasoning text, never added to txt
        wire = wire if wire is not None else {}
        for line in r:
            wire["bytes_received"] = wire.get("bytes_received", 0) + len(line)
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            s = s[5:].strip()
            if not s or s == "[DONE]":
                continue
            try:
                ch = json.loads(s)
            except Exception:                             # noqa: BLE001
                continue                                  # heartbeat/comment frame, skip
            if generation_id is None and isinstance(ch.get("id"), str):
                generation_id = ch["id"]
                wire["generation_id"] = generation_id
                note_generation(generation_id)          # durable before the stream can be cut
            if served is None and isinstance(ch.get("provider"), str):
                served = ch["provider"]
            if served_model is None and isinstance(ch.get("model"), str):
                served_model = ch["model"]
            if ch.get("usage"):
                usage = ch["usage"]                       # final frame (include_usage)
            c0 = (ch.get("choices") or [{}])[0]
            if c0.get("finish_reason"):
                fin = c0["finish_reason"]
            delta = c0.get("delta") or {}
            piece = delta.get(self._SSE_ANSWER_FIELD)
            if isinstance(piece, str):
                txt.append(piece)
            for f in self._SSE_THINK_FIELDS:              # collected into a separate column, never into the answer
                if isinstance(delta.get(f), str):
                    think_n += len(delta[f])
                    think_txt.append(delta[f])
        # The raw reasoning text goes to `trace.jsonl` only, never into the answer or
        # `responses.jsonl`, which the judges read.
        self._last_reasoning_chars = think_n
        self._last_reasoning_text = "".join(think_txt) or None
        answer = "".join(txt)
        if not defer_validation:
            self._validate_stream(answer, think_n, usage, fin)
        result = {"choices": [{"message": {"content": answer}, "finish_reason": fin}],
                  "usage": usage, "_reasoning_chars": think_n,
                  **({"provider": served} if served is not None else {}),
                  **({"model": served_model} if served_model is not None else {}),
                  **({"id": generation_id} if generation_id is not None else {})}
        if defer_validation:
            result["_reasoning_text"] = self._last_reasoning_text
            result["_generation_id"] = generation_id
        return result

    @staticmethod
    def _validate_stream(answer, think_n, usage, fin):
        # A truncated stream ends without an exception; no terminating frame means no answer.
        if fin is None:
            raise ValueError(
                f"流式响应未收到终止帧(finish_reason 缺失):作答 {len(answer)} 字符、"
                f"推理 {think_n} 字符、usage={'有' if usage else '无'} —— 判为流被截断")
        if not answer.strip() and think_n > 0:
            # `tools/attribute_failures.py` matches this text; change both together.
            raise ValueError(
                f"流式响应只有推理没有作答(推理 {think_n} 字符、作答 0 字符)——"
                f"推理不进作答,所以这一次视为没答上")

    def solve(self, payload):
        prompt, pid = _qside.render_for(self, payload)
        data, raw_text, usage, fin = {}, "", None, None
        # Failed attempts are persisted with raw text and elapsed time; the successful call's
        # latency is recorded as `latency_s`.
        attempts: list[dict] = []
        for attempt in range(1, self.retries + 2):
            text = ""
            resp = None
            _t0 = time.time()
            try:
                resp = self._post(prompt)
                choice = (resp.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage, fin = resp.get("usage"), choice.get("finish_reason")
                if fin == "length" and not text.strip():
                    raise ValueError("响应被 max_tokens 截断且无内容")
                raw_text = text
                data = _extract_json(text)
                latency = round(time.time() - _t0, 2)
                break
            except BudgetExceeded:
                raise
            except Exception as e:                    # rate limit/timeout/truncation/non-JSON -> back off and retry
                _dt = round(time.time() - _t0, 2)
                log.warning("[solver %s] attempt %d failed after %.1fs: %s: %s",
                            self.name, attempt, _dt, type(e).__name__, str(e)[:160])
                # Usage of a failed attempt is recorded here or nowhere (`out._usage` covers only the
                # successful attempt). A network failure has no usage: `None`, not 0.
                _fu = resp.get("usage") if isinstance(resp, dict) else None
                attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:200]}",
                                 "elapsed_s": _dt,
                                 "finish_reason": fin, "n_chars": len(text),
                                 "usage": _fu,
                                 "billed": billed_tokens(_fu) if _fu else None,
                                 "raw": text,
                                 "head": text[:400], "tail": text[-400:] if len(text) > 400 else ""})
                time.sleep(3 * attempt)
        else:
            latency = None                            # every attempt failed => no successful latency to record
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # Reasoning tokens count against max_tokens; `usage` and `finish_reason` separate a
        # truncated answer from a wrong one.
        out._usage, out._finish = usage, fin
        out._max_tokens = self.max_tokens
        out._failed_attempts = attempts
        out._rate_limit_retries = _drain_backoff(self)
        out._requests = _drain_requests(self)
        out._prompt_sha_full = hashlib.sha256(prompt.encode()).hexdigest()
        out._latency_s = latency
        out._reasoning_chars = getattr(self, "_last_reasoning_chars", None) if self.stream else None
        out._reasoning_text = getattr(self, "_last_reasoning_text", None) if self.stream else None
        return out


def _drain_backoff(solver) -> list | None:
    """Rate-limit retries the request accountant made for this solve (None when there were none)."""
    drain = getattr(getattr(solver, "_accounting", None), "drain_backoff", None)
    return (drain() or None) if callable(drain) else None


def _drain_requests(solver) -> list | None:
    """Accounted requests of this solve, one record each: request id, full prompt sha256,
    route, outcome and who served it (None without request accounting)."""
    drain = getattr(getattr(solver, "_accounting", None), "drain_requests", None)
    return (drain() or None) if callable(drain) else None


def _google_usage(resp: dict) -> dict | None:
    """Google `usageMetadata` -> usage shaped like the OpenAI-compatible backends.

    `thoughtsTokenCount` is kept separately and the unaccounted residual is recorded, so
    thinking tokens are not dropped from cost estimates.
    """
    um = (resp or {}).get("usageMetadata") or {}
    if not um:
        return None
    _thoughts = um.get("thoughtsTokenCount")
    _tot = um.get("totalTokenCount")
    _pt, _ct = um.get("promptTokenCount"), um.get("candidatesTokenCount")
    return {"prompt_tokens": _pt,
            "prompt_tokens_details": {"cached_tokens": um.get("cachedContentTokenCount")},
            "completion_tokens": _ct,
            "total_tokens": _tot,
            "thoughts_tokens": _thoughts,
            "unaccounted_tokens": (
                _tot - (_pt or 0) - (_ct or 0) - (_thoughts or 0)
                if isinstance(_tot, int) else None)}


class GoogleSolver(Solver):
    """Connects directly to the Google Generative Language API (`generateContent`; the key goes
    in the query string).
    """

    def __init__(self, name: str, model: str, api_key: str, timeout: int, retries: int = 2,
                 max_tokens: int = 6000):
        self.name, self.model, self.timeout, self.retries = name, model, timeout, retries
        self.api_key, self.max_tokens = api_key, max_tokens
        self.backend, self.URL = "google", BACKENDS["google"]["url"]
        self.pool, self.stream = None, False
        self.reasoning_effort, self.response_format = None, None
        self.sampling = None                        # set by `solver_for_spec` from config
        self.max_tokens_field = "maxOutputTokens"
        if not api_key:
            log.error("[solver %s] missing GOOGLE_GENERATIVE_AI_API_KEY", name)

    def _post(self, prompt: str) -> dict:
        accounting = getattr(self, "_accounting", None)
        return (accounting.request(self, prompt, self._post_wire) if accounting is not None
                else self._post_wire(prompt))

    def _post_wire(self, prompt: str) -> dict:
        return self._send_wire(self._wire_body(prompt), prompt)

    def _wire_body(self, prompt: str) -> bytes:
        from .transport import sent_temperature
        gen = {"maxOutputTokens": self.max_tokens}
        if sent_temperature(self.sampling) is not None:
            gen["temperature"] = sent_temperature(self.sampling)
        if (self.sampling or {}).get("thinking_budget") is not None:
            gen["thinkingConfig"] = {"thinkingBudget": self.sampling["thinking_budget"]}
        payload_d = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}
        _llm.note_payload(self.name, payload_d)     # same as `OpenAICompatSolver._post`
        return json.dumps(payload_d).encode()

    def _send_wire(self, body: bytes, prompt: str) -> dict:
        import urllib.request
        verify_wire(self, body, prompt)
        url = (f"{self.URL}/{self.model}:generateContent?key={self.api_key}")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with _stall_guard(self) as _stall, urllib.request.urlopen(req, timeout=self.timeout) as r:
            _stall.attach(r)
            _resp = json.loads(r.read())
        _llm.note_call(self.name, "google", _resp if isinstance(_resp, dict) else None)
        return _resp

    def solve(self, payload):
        prompt, pid = _qside.render_for(self, payload)
        data, raw_text, usage, fin = {}, "", None, None
        attempts: list[dict] = []
        for attempt in range(1, self.retries + 2):
            text = ""
            try:
                resp = self._post(prompt)
                cand = (resp.get("candidates") or [{}])[0]
                parts = ((cand.get("content") or {}).get("parts") or [{}])
                text = "".join(p.get("text", "") for p in parts)
                fin = cand.get("finishReason")
                usage = _google_usage(resp)
                if fin == "MAX_TOKENS" and not text.strip():
                    raise ValueError("响应被 maxOutputTokens 截断且无内容")
                raw_text = text
                data = _extract_json(text)
                break
            except BudgetExceeded:
                raise
            except Exception as e:
                log.warning("[solver %s] attempt %d failed: %s: %s",
                            self.name, attempt, type(e).__name__, str(e)[:160])
                attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {str(e)[:200]}",
                                 "finish_reason": fin, "n_chars": len(text),
                                 "head": text[:400], "tail": text[-400:] if len(text) > 400 else ""})
                time.sleep(3 * attempt)
        out = _to_output(data, payload)
        out._raw_text, out._prompt_mode = raw_text, getattr(self, "prompt_mode", "default")
        out._probe_id, out._prompt_sha = pid, hashlib.sha256(prompt.encode()).hexdigest()[:16]
        out._usage, out._finish = usage, fin
        out._max_tokens = self.max_tokens          # same as `OpenAICompatSolver.solve`
        out._failed_attempts = attempts
        out._rate_limit_retries = _drain_backoff(self)
        out._requests = _drain_requests(self)
        out._prompt_sha_full = hashlib.sha256(prompt.encode()).hexdigest()
        return out


def _const(inst):
    """Factory for a stateless solver: one shallow copy per case.

    `_one` and `run_gated` write per-case attributes (`probe_id`, `gated_context`, ...), so a
    shared instance would leak state between cases under `--workers > 1`. Solvers hold only
    scalars, and key pools live in `_KEY_POOLS`, so a shallow copy is safe. Stateful stubs are
    constructed per case in `build_solvers` instead.
    """
    import copy as _copy
    return lambda: _copy.copy(inst)


def _ensure_no_proxy(hosts: set[str]) -> None:
    """Append `hosts` to this process's `no_proxy` (the relay and dashscope fail through a local
    proxy). Never overwrites the user's setting or writes to disk.
    """
    if not hosts:
        return
    for key in ("no_proxy", "NO_PROXY"):
        cur = {h.strip() for h in (os.environ.get(key, "") or "").split(",") if h.strip()}
        missing = hosts - cur
        if missing:
            os.environ[key] = ",".join(sorted(cur | hosts))
    if hosts:
        log.info("[eval] no_proxy already includes %s", sorted(hosts))


def solver_for_spec(name: str, spec: dict, env: dict, timeout: int, retries: int):
    """Build a solver from the mapped form of `config.models[name]`; the one constructor shared
    by batch runs and tooling. An unregistered backend returns None.
    """
    backend = spec.get("backend", "openrouter")
    # Google-family models (by `vendor`, not by name) must connect directly: an
    # OpenAI-compatible relay drops `thought_signature`, and requests with tools then fail with 400.
    if str(spec.get("vendor") or "").lower() == "google" and backend != "google":
        raise ValueError(
            f"config.models[{name}]: vendor=google but backend={backend!r}. "
            f"Google-family models must connect directly (`backend: google`): an "
            f"OpenAI-compatible relay drops `thought_signature`, and requests with tools "
            f"fail with HTTP 400. Matched on `vendor`, not on the model name.")
    _ensure_backends_registered()
    bdef = BACKENDS.get(backend)
    if bdef is None:
        log.warning("[eval] model %s's backend=%s is not registered (default-deny, see "
                    "evaluate.BACKENDS), skipping",
                    name, backend)
        return None
    if bdef.get("kind") != "google" and not bdef.get("url"):
        log.warning("[eval] model %s: backend %s has no url configured "
                    "(set config.backends.%s.url, e.g. in config.local.yaml); skipped",
                    name, backend, backend)
        return None
    from .transport import check_pin, check_sampling
    check_pin(name, backend, spec.get("provider"))
    sampling = check_sampling(name, backend, spec.get("sampling"))
    mt = int(spec.get("max_tokens", 6000))
    # Below `llm.recommended_budgets()` a score is a budget-limited lower bound, not a capability
    # reading, so this raises. `--offline` never reaches here.
    _bv = _llm.note_budget(name, backend, mt, bool(spec.get("stream", False)))
    if _bv["below_recommended"]:
        raise ValueError(
            f"config.models[{name}].max_tokens = {mt} is below the recommended "
            f"{_bv['recommended']}; a run at this budget gives a budget-limited lower bound, "
            f"not a capability reading. Raise it to >={_bv['recommended']}, or update "
            f"`llm.MEASURED_BUDGET_BASIS` and record the source of the new value.")
    if backend not in _KEY_POOLS:
        _KEY_POOLS[backend] = _load_key_pool(bdef, env, backend)
    pool = _KEY_POOLS[backend]                   # shared within a backend: an exhausted key is skipped by every model
    key = pool.current() or env.get(bdef["key_env"], "")
    if bdef["kind"] == "google":
        g = GoogleSolver(name, str(spec["model"]), key, timeout, retries, mt)
        g.sampling = sampling
        if spec.get("upstream"):
            g.upstream = str(spec["upstream"])
        return g
    s = OpenAICompatSolver(
        name, str(spec["model"]), key, timeout,
        int(spec.get("retries", retries)), mt,
        url=bdef["url"], backend=backend,
        max_tokens_field=bdef.get("max_tokens_field", "max_tokens"),
        # `stream` is the computed value (see `llm.note_budget`); config can only turn it on.
        stream=bool(_bv["stream"]), reasoning_effort=spec.get("reasoning_effort"),
        provider=spec.get("provider"), sampling=sampling)
    # Declared upstream (who serves the weights): part of the answer view, so two routes
    # that declare the same upstream answer as one solver (`transport.upstream_of`).
    if spec.get("upstream"):
        s.upstream = str(spec["upstream"])
    s.pool = pool
    return s


#: Keys of a route entry (`config.models.<m>.fallback_routes[]`, runtime.yaml `routes.<m>[]`):
#: each overrides the model's own spec for that route.
ROUTE_ENTRY_KEYS = frozenset({"backend", "model", "upstream", "provider", "stream", "retries",
                              "max_tokens_field"})


def route_name(spec: dict) -> str:
    only = ((spec.get("provider") or {}).get("only") or [None])[0]
    return f"{spec.get('backend', 'openrouter')}/{spec.get('upstream') or only or spec.get('model')}"


def route_chain(cfg: dict, name: str, ledger_path=None) -> list[dict]:
    """Fallback route specs for model `name`, in order, after its primary route.

    The chain is runtime.yaml `routes.<name>` when that file sets one (read when the run's
    accounting is prepared), else `config.models.<name>.fallback_routes`. Each entry
    overrides the model's spec; an entry that names the primary's own backend and model
    is the primary and is skipped. A route may change only how the model is reached: its
    answer view (model, upstream, max_tokens, effort, format) must equal the primary's,
    which `prepare_accounting` checks before any request.
    """
    base = {k: v for k, v in (cfg.get("models", {}).get(name) or {}).items() if k != "fallback_routes"}
    chain = None
    if ledger_path is not None:
        from .ops.runtime import for_ledger
        chain = (for_ledger(ledger_path).current().get("routes") or {}).get(name)
    if chain is None:
        chain = (cfg.get("models", {}).get(name) or {}).get("fallback_routes") or []
    out = []
    for entry in chain:
        bad = set(entry) - ROUTE_ENTRY_KEYS
        if bad:
            raise ValueError(f"route entry for {name} has unknown key(s) {sorted(bad)}")
        spec = {**base, **entry}
        if "provider" not in entry and spec.get("backend") != base.get("backend"):
            spec.pop("provider", None)            # a pin belongs to the route that declared it
        if (spec.get("backend", "openrouter"), spec.get("model")) == (
                base.get("backend", "openrouter"), base.get("model")):
            continue
        out.append(spec)
    return out


def fallback_solver_factories(cfg: dict, name: str, primary, ledger_path=None) -> list:
    """(route name, factory, spec) for each fallback route of `name` (see `route_chain`)."""
    env = load_env_file(cfg.get("env_file"))
    timeout = int(getattr(primary, "timeout", cfg.get("timeout_s", 900)))
    retries = int(cfg.get("max_retries", 2))
    out = []
    for spec in route_chain(cfg, name, ledger_path):
        if (spec.get("backend", "openrouter"), spec.get("model")) == (
                getattr(primary, "backend", None), getattr(primary, "model", None)):
            continue                                   # the route this run already takes
        _ensure_backends_registered(cfg)
        if spec.get("backend", "openrouter") not in BACKENDS:
            raise ValueError(f"fallback route {route_name(spec)} for {name}: backend not registered")
        out.append((route_name(spec),
                    (lambda _spec=spec: solver_for_spec(name, _spec, env, timeout, retries)), spec))
    return out


def raw_complete_with_usage(solver, prompt: str) -> tuple[str, dict | None]:
    """Fire one request and return (answer text, usage), normalizing the two backends' shapes.

    Only the answer channel is read, never reasoning. `usage` is `None` when unavailable, so
    `billed_tokens` reports it as absent rather than zero.
    """
    resp = solver._post(prompt)
    if "candidates" in resp:
        cand = (resp.get("candidates") or [{}])[0]
        return ("".join(p.get("text", "")
                        for p in ((cand.get("content") or {}).get("parts") or [])),
                _google_usage(resp))
    ch = (resp.get("choices") or [{}])[0]
    return (str(((ch.get("message") or {}).get("content")) or ""),
            resp.get("usage") or None)


def raw_complete(solver, prompt: str) -> str:
    return raw_complete_with_usage(solver, prompt)[0]


def build_solvers(job, cfg) -> list[tuple[str, object]]:
    """Offline reference solvers first (free), then real models. Returns (name, factory) pairs;
    the factory is called once per case, because some stubs (`StubbornSolver`,
    `FlipFlopSolver`) keep per-case state.
    """
    _ensure_backends_registered(cfg)
    out: list[tuple[str, object]] = []
    if job.include_baseline:
        from haenv_kernel.solver import StubbornSolver                          # kernel: anchored / never recants
        out += [("baseline_slope", _const(BaselineSolver())), ("robust_ref", _const(RobustSolver())),
                ("no_revision", StubbornSolver), ("flip_flop", FlipFlopSolver),
                # Constant baseline (unified + A3 without reading the question): the floor for the join /
                # urgency dimensions.
                ("const_ddx", _const(ConstantDdxSolver())),
                # Test-ordering stubs that ignore the question: the whole menu (shotgun ceiling) and a fixed
                # routine panel.
                ("shotgun_tests", _const(TestOrderingSolver("shotgun"))),
                ("common_panel", _const(TestOrderingSolver("panel"))),
                ("gated_probe", GatedProbeSolver),
                ("gated_shotgun", GatedShotgunSolver),
                # Orders every available signal: a score near 1.0 means `sd_coverage` rewards blanket
                # ordering. Stateful, so a factory.
                ("gated_flood", GatedSignalFloodSolver),
                # Queries exactly once, which maximizes the tool track's ratio metrics (see `GatedMinimalSolver`).
                ("gated_minimal", GatedMinimalSolver),
                # Always declares insufficient information: must score full marks on that tier and be penalized
                # on ordinary questions.
                ("humble_ref", _const(HumbleSolver())),
                # Oracle: orders exactly the gold tests (a ceiling, labeled oracle in reports).
                ("oracle_tests", _const(OracleTestsSolver())),
                # Every name in `BASELINE_NAMES` must be constructed here. `onelong` is the reference for the
                # degenerate ceilings of `disc_recall` / `tests_precision`.
                ("onelong", _const(OneLongTestSolver())),
                # Right/wrong oracle answers for the Q-side probes, so the checks are shown to pass and to fail.
                ("oracle_probe_right", _const(OracleProbeSolver("right"))),
                ("oracle_probe_wrong", _const(OracleProbeSolver("wrong"))),
                # Both ends of `review_macro` (specificity; "always says so" scores 0.000).
                ("oracle_review_right", _const(OracleReviewSolver("right"))),
                ("oracle_review_wrong", _const(OracleReviewSolver("wrong"))),
                # Positive controls that trip the two action gates by construction.
                ("gatetrip_treatment", _const(GateTripSolver("treatment"))),
                ("gatetrip_invasive", _const(GateTripSolver("invasive"))),
                # Positive control for escalating to A2 without requesting clinician review.
                ("noreview", _const(NoReviewFlagSolver())),
                # Wavers on the first two slices and settles on the third (range for `converged_at`); stateful.
                ("late_converge", LateConvergeSolver),
                # Process-track floor: shape-valid traces with filler content (see `TraceJunkSolver`).
                ("trace_junk", _const(TraceJunkSolver())),
                # `self_discovery` floor: no queries plus a claim that the data suffices (see `BlindConfidentSolver`).
                ("blind_confident", _const(BlindConfidentSolver()))]
    ai = os.path.expanduser(cfg.get("ai_dispatcher", "~/.local/bin/ai"))
    env = load_env_file(cfg.get("env_file"))
    timeout, retries = int(cfg.get("timeout_s", 900)), int(cfg.get("max_retries", 2))
    _wanted_backends = {(cfg.get("models", {}).get(n) or {}).get("backend", "openrouter")
                        for n in (job.models or cfg.get("default_models", []))
                        if isinstance(cfg.get("models", {}).get(n), dict)}
    _ensure_no_proxy({BACKENDS[b]["no_proxy_host"] for b in _wanted_backends
                      if b in BACKENDS and BACKENDS[b].get("no_proxy_host")})
    # An explicitly requested unknown model fails rather than running zero cells and exiting 0.
    # Unknown names in `default_models` are skipped with a warning.
    _explicit = bool(job.models)
    _unknown = [n for n in (job.models or ()) if not cfg.get("models", {}).get(n)]
    if _explicit and _unknown:
        raise ValueError(f"--models has name(s) not defined in config.models: {_unknown} "
                         f"(registered: {sorted(cfg.get('models', {}))}); "
                         f"an explicitly requested model that does not exist must fail")
    # Retired models stay registered so historical rows have a known solver name; naming one
    # explicitly is refused, since it would send billed requests.
    _retired = [n for n in (job.models or ())
                if (cfg.get("models", {}).get(n) or {}).get("retired")]
    if _explicit and _retired:
        raise ValueError(
            f"--models includes retired model(s): "
            f"{[(n, cfg['models'][n]['retired']) for n in _retired]}. They stay registered "
            f"so historical batches resolve; remove `retired` to run one.")
    for name in (job.models or cfg.get("default_models", [])):
        spec = cfg.get("models", {}).get(name)
        if not spec:
            log.warning("[eval] unknown model %s (not defined in config.models), skipping", name)
            continue
        if isinstance(spec, dict):
            _sv = solver_for_spec(name, spec, env, timeout, retries)
            if _sv is None:
                if _explicit:
                    raise ValueError(
                        f"--models {name}: no solver could be built "
                        f"(backend {spec.get('backend', 'openrouter')!r}; see the warning above)")
                continue
            backend = spec.get("backend", "openrouter")
            bdef = BACKENDS[backend]
            mt = int(spec.get("max_tokens", 6000))
            pool = _KEY_POOLS[backend]
            key = pool.current() or env.get(bdef["key_env"], "")
            if bdef["kind"] == "google":
                out.append((name, _const(_sv)))
            else:
                _s = _sv
                _s.pool = pool
                out.append((name, _const(_s)))
        else:
            # List form = CLI subprocess via `~/.local/bin/ai`, refused unless the top-level
            # `allow_cli_solver` is set (see `CLISolver`).
            out.append((name, _const(CLISolver(
                name, [str(x) for x in spec], ai, timeout, retries, env=env,
                allow_unsandboxed=bool(cfg.get("allow_cli_solver", False))))))
    return out


def load_env_file(path: str | None) -> dict[str, str]:
    """Read `config.env_file` (`export KEY=...` lines) into a dict for dispatched subprocesses.

    Values are never logged; a missing file returns `{}`. `$VAR` is expanded against variables
    assigned earlier in the same file (shell order); no other shell syntax is supported.
    """
    if not path:
        return {}
    path = os.path.expandvars(str(path))
    if not path.strip() or "$" in path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        log.warning("[eval] env_file not found: %s (skipped; the backend's own "
                    "configuration takes over)", p)
        return {}
    import re
    _VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    out: dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s[7:].lstrip() if s.startswith("export ") else s
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            v = _VAR_RE.sub(lambda m: out.get(m.group(1), m.group(0)), v)
            if k.strip():
                out[k.strip()] = v
    except OSError as e:
        log.warning("[eval] failed to read env_file %s: %s", p, e)
        return {}
    log.info("[eval] loaded from %s: %d environment variable(s) (values are not logged)", p, len(out))
    return out


