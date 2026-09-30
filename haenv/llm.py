"""llm.py -- LLM dispatch for the generation side, with an on-disk cache.

Calls `~/.local/bin/ai` (or a direct backend) per `config.models[<key>]`.
Responses are cached under `cases/_llm_cache/` keyed by sha256(model + prompt),
storing prompt and response; `--regen` forces a fresh call. Generation prompts
contain verifier-side information and are never shown to the solver.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Isolated working directory for generation subprocesses, so they cannot read the repo.
_GEN_CWD: str | None = None


def _gen_cwd() -> str:
    global _GEN_CWD
    if _GEN_CWD is None:
        _GEN_CWD = tempfile.mkdtemp(prefix="haenv-gen-cwd-")
    return _GEN_CWD

log = logging.getLogger("haenv.llm")

# ============================================================ Generation-side usage ledger
# Process-scoped; read by `batch.register`. Cache hits are counted separately
# from calls whose backend returned no usage.
_GEN_USAGE: dict = {"n_calls": 0, "n_hits": 0, "n_with_usage": 0, "n_without_usage": 0,
                    "in_fresh": 0, "in_cached": 0, "out": 0, "by_model": {}}


def _note_gen_usage(model: str, usage: dict | None, path: str) -> None:
    """Record one generation call that actually went out. `usage=None`
    is recorded under `n_without_usage`, not collapsed to 0."""
    from .evaluate import billed_tokens
    b = billed_tokens(usage)
    m = _GEN_USAGE["by_model"].setdefault(
        str(model), {"path": path, "n_calls": 0, "n_with_usage": 0,
                     "in_total": 0, "out": 0})
    _GEN_USAGE["n_calls"] += 1
    m["n_calls"] += 1
    if b.get("status") == "measured":
        _GEN_USAGE["n_with_usage"] += 1
        m["n_with_usage"] += 1
        for f in ("in_fresh", "in_cached", "out"):
            _GEN_USAGE[f] += int(b.get(f) or 0)
        m["in_total"] += int(b.get("in_total") or 0)
        m["out"] += int(b.get("out") or 0)
    else:
        _GEN_USAGE["n_without_usage"] += 1


def note_gen_cache_hit(model: str) -> None:
    """Record one cache hit (zero cost, tracked in its own field)."""
    _GEN_USAGE["n_hits"] += 1
    m = _GEN_USAGE["by_model"].setdefault(
        str(model), {"path": "cache", "n_calls": 0, "n_with_usage": 0,
                     "in_total": 0, "out": 0})
    m["n_hits"] = m.get("n_hits", 0) + 1


def gen_usage_snapshot() -> dict:
    """This process's generation-side usage snapshot; `batch.register` reads it.

    `status`: `absent:no_llm_generation` (no LLM generation),
    `absent:cache_only` (every call hit the cache), `missing:no_usage` (no call
    returned usage), or `measured`.
    """
    g = dict(_GEN_USAGE)
    if not g["n_calls"] and not g["n_hits"]:
        return {"status": "absent:no_llm_generation"}
    if not g["n_calls"]:
        return {"status": "absent:cache_only", "n_hits": g["n_hits"],
                "by_model": g["by_model"]}
    g["in_total"] = g["in_fresh"] + g["in_cached"]
    g["total_tokens"] = g["in_total"] + g["out"]
    g["status"] = "measured" if g["n_with_usage"] else "missing:no_usage"
    return g


# ============================================================ Sampling parameters: not pinned, but recorded
# Temperature is left unpinned (temperature 0 degrades long generations and
# reasoning models often ignore it); the conditions actually sent and echoed
# are recorded instead. "Not specified" is never recorded as a value.

#: Names of the sampling knobs (used by `batch.sampling_fields`). `max_tokens`
#: is excluded: the generation budget is tiered separately below.
SAMPLING_PARAM_NAMES: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p", "top_a", "typical_p",
    "seed", "n", "stop", "logit_bias", "logprobs",
    "frequency_penalty", "presence_penalty", "repetition_penalty",
    "reasoning_effort", "thinking",
)

#: Sampling knobs Google nests inside `generationConfig`.
GOOGLE_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature", "topP", "topK", "candidateCount", "stopSequences",
    "seed", "thinkingConfig",
)

#: Call conditions a provider echoes back. `provider` matters because one model
#: id can be served by several providers with different defaults.
ECHO_FIELDS: tuple[str, ...] = ("temperature", "top_p", "seed",
                                "system_fingerprint", "provider", "model")

#: Response-side aliases (Google returns `modelVersion`).
_ECHO_ALIASES: dict[str, tuple[str, ...]] = {"model": ("model", "modelVersion")}

_ECHO_KEEP = 8

_SAMPLING: dict = {"by_model": {}, "n_requests": 0, "n_echo": 0}


def payload_sampling(payload: dict | None) -> dict:
    """Sampling parameters in the payload actually sent. `{}` means none were
    sent (provider defaults apply), not zero values."""
    p = payload or {}
    out = {k: p[k] for k in SAMPLING_PARAM_NAMES if k in p}
    gc = p.get("generationConfig")
    if isinstance(gc, dict):
        out.update({f"generationConfig.{k}": gc[k]
                    for k in GOOGLE_SAMPLING_KEYS if k in gc})
    return out


def response_echo(resp: dict | None) -> dict:
    """Call conditions echoed in a raw response; missing keys are `"absent"`."""
    r = resp or {}
    out: dict = {}
    for f in ECHO_FIELDS:
        keys = _ECHO_ALIASES.get(f, (f,))
        hit = next((k for k in keys if k in r), None)
        out[f] = r[hit] if hit is not None else "absent"
    return out


def note_call(model: str, backend: str, resp: dict | None) -> None:
    """Record the call conditions echoed for one call. `resp=None` (CLI
    dispatch has no response object) is recorded as `absent:no_response_object`."""
    e = _SAMPLING["by_model"].setdefault(
        str(model), {"backend": str(backend), "n_calls": 0,
                     "echo_status": "absent:not_observed", "echo_distinct": [],
                     "n_echo_distinct_dropped": 0})
    e["n_calls"] += 1
    _SAMPLING["n_requests"] += 1
    if resp is None:
        e["echo_status"] = "absent:no_response_object"
        return
    ech = response_echo(resp)
    measured = {k: v for k, v in ech.items() if v != "absent"}
    e["echo_status"] = "measured" if measured else "absent:provider_returned_none"
    if measured:
        _SAMPLING["n_echo"] += 1
    if ech not in e["echo_distinct"]:
        # Keep every distinct echo (up to `_ECHO_KEEP`) to detect provider re-routing.
        if len(e["echo_distinct"]) < _ECHO_KEEP:
            e["echo_distinct"].append(ech)
        else:
            e["n_echo_distinct_dropped"] += 1


def sampling_snapshot() -> dict:
    """Sampling conditions observed by this process, for `batch.register` /
    `batch.record_usage`. `status`: `absent:no_real_calls_this_process`,
    `absent:provider_returned_none`, or `measured`."""
    if not _SAMPLING["n_requests"]:
        return {"status": "absent:no_real_calls_this_process", "by_model": {}}
    return {"status": "measured" if _SAMPLING["n_echo"] else "absent:provider_returned_none",
            "n_requests": _SAMPLING["n_requests"], "n_echo": _SAMPLING["n_echo"],
            "by_model": _SAMPLING["by_model"]}


def observe_post(solver, model: str, backend: str):
    """Wrap `solver._post` to record echoed call conditions; inputs and return
    value are unchanged."""
    _orig = solver._post

    def _observed(prompt, *a, **kw):
        resp = _orig(prompt, *a, **kw)
        try:
            note_call(model, backend, resp if isinstance(resp, dict) else None)
        except Exception as _e:                                 # noqa: BLE001
            # Recording must not break the call; failures are logged.
            log.warning("[llm] failed to record sampling condition (%s): %s", model, type(_e).__name__)
        return resp

    solver._post = _observed
    return solver


# ============================================================ Generation budget (`max_tokens`) tiers
# gen = `evaluate.billed_tokens(usage)["out"]` (completion + reasoning);
# base = observed cap if the cap censored the distribution, else gen's p99:
#   max_tokens = ceil_4000( max( base × TAIL_FACTOR,  base + ANSWER_RESERVE_TOKENS ) )

#: Answer reserve (tokens): the longest scoreable answer body (8,392 chars,
#: `<think>` stripped) / `CHARS_PER_TOKEN`, rounded up to 500.
ANSWER_RESERVE_TOKENS = 4500
#: Upper bound of `gen_max / gen_p99` over models not near their cap (2.18),
#: rounded up.
TAIL_FACTOR = 2.25
#: Tier granularity for `max_tokens`.
BUDGET_STEP = 4000
#: Lower bound of chars/token (visible-text models measure 2.07–2.77); the
#: lower bound gives the larger reserve.
CHARS_PER_TOKEN = 2.0
#: The relay's hard timeout for non-streaming requests (seconds).
RELAY_NONSTREAM_WALL_S = 300
#: Distinct 40-char-segment ratio below which a truncated response is a
#: repetition loop (loops ~0.003, normal answers ~1.0).
DEGENERATE_UNIQ_RATIO = 0.5


def ceil_to_step(x: float, step: int = BUDGET_STEP) -> int:
    """Round up to the nearest multiple of `step`."""
    return int(-(-int(x) // step) * step)


def budget_for(gen_p99: float, observed_cap: float, *, censored: bool = False,
               current: int | None = None) -> int:
    """Recommended `max_tokens` under the rule above.

    `censored=True` uses `observed_cap` as base (p99 is then only a lower bound,
    and so is the result). With `current`, returns `max(recommended, current)`.
    """
    base = float(observed_cap if censored else gen_p99)
    want = ceil_to_step(max(base * TAIL_FACTOR, base + ANSWER_RESERVE_TOKENS))
    return max(want, int(current)) if current else want


def budget_risk_line(max_tokens: int) -> int:
    """Budget minus answer reserve; a diagnostic, not a ceiling criterion."""
    return int(max_tokens) - ANSWER_RESERVE_TOKENS


def looks_degenerate(text: str) -> bool:
    """Whether text looks like a runaway repetition loop (not evidence of a
    too-small budget)."""
    t = str(text or "")
    if len(t) < 4000:            # This ratio has no discriminative power on short text; always "not degenerate"
        return False
    seg = [t[i:i + 40] for i in range(0, len(t) - 40, 40)]
    return bool(seg) and (len(set(seg)) / len(seg)) < DEGENERATE_UNIQ_RATIO


def ceiling_flags(gen_tokens: int | None, finish_reason, max_tokens: int,
                  raw_text: str = "") -> dict:
    """Ceiling-hit readout for one response.

    - `ceil_by_finish` — the provider's `finish_reason` is length/MAX_TOKENS.
    - `ceil_by_ratio` — `gen >= max_tokens` (`None` when usage is missing).
    - `at_risk` — `gen >= risk line`; diagnostic only.
    - `degenerate` — the truncated response is a repetition loop.
    """
    fin = str(finish_reason or "").lower()
    by_fin = fin in ("length", "max_tokens")
    g = None if gen_tokens is None else int(gen_tokens)
    return {"gen_tokens": g,
            "budget": int(max_tokens),
            "ceil_by_finish": by_fin,
            "ceil_by_ratio": (None if g is None else g >= int(max_tokens)),
            "at_risk": (None if g is None else g >= budget_risk_line(max_tokens)),
            "degenerate": (looks_degenerate(raw_text) if by_fin else False)}


#: Generation-length readings behind the budget tiers, from recorded
#: `responses.jsonl` without offline stubs, deduplicated on `(solver, case,
#: slice_t, prompt_sha256, sha256(raw))`. `cap` is the budget those rows ran
#: under, taken from the trace of each response's actual cap.
#: p99 = `sorted(gen)[floor(0.99×(n−1))]`; n counts measured-usage rows.
MEASURED_BUDGET_BASIS: dict[str, dict] = {
    #                       gen_p99  gen_max  cap     trunc  exhaust  n
    "glm-5.3":          dict(p99=31702, mx=31942, cap=48000, trunc=13, exhaust=0, n=268),
    "minimax-m3":       dict(p99=20520, mx=30368, cap=72000, trunc=0,  exhaust=0, n=699),
    "kimi-k3":          dict(p99=15913, mx=21257, cap=72000, trunc=0,  exhaust=0, n=724),
    "gemini-3.8-flash": dict(p99=12571, mx=19082, cap=20000, trunc=0,  exhaust=0, n=1207),
    "gemini-3.1-pro":   dict(p99=14191, mx=17096, cap=72000, trunc=0,  exhaust=0, n=706),
    "deepseek-v4-pro":  dict(p99=31215, mx=36964, cap=72000, trunc=0, exhaust=0, n=729),
    "deepseek-v4-flash": dict(p99=26468, mx=50690, cap=72000, trunc=0, exhaust=0, n=735),
    "gemini-3.7-flash": dict(p99=4909, mx=7324, cap=72000, trunc=0, exhaust=0, n=674),
    "glm-5.3-flash":    dict(p99=28435, mx=42635, cap=72000, trunc=0, exhaust=0, n=758),
    "gpt-6-luna":       dict(p99=4918, mx=5178, cap=72000, trunc=0, exhaust=0, n=666),
    "gpt-6-sol":        dict(p99=3358, mx=3829, cap=72000, trunc=0, exhaust=0, n=664),
    "claude-opus-4-8":  dict(p99=3939, mx=4004, cap=16000, trunc=0, exhaust=0, n=21),
    "terra":            dict(p99=3832,  mx=4746,  cap=16000, trunc=0,  exhaust=0, n=3692),
    "luna":             dict(p99=3440,  mx=3823,  cap=16000, trunc=0,  exhaust=0, n=799),
    "gpt-5.4":          dict(p99=1552,  mx=2103,  cap=16000, trunc=0,  exhaust=0, n=2791),
    "qwen3.8-max":      dict(p99=1335, mx=1702, cap=16000, trunc=0, exhaust=0, n=1796),
    "glm-5.2":          dict(p99=1347,  mx=1431,  cap=16000, trunc=0,  exhaust=0, n=199),
    "qwen3.7-flash":    dict(p99=10592, mx=12980, cap=72000, trunc=0, exhaust=0, n=697),
}

#: Models whose readings were censored by their cap at some point (a record;
#: `is_censored` decides tiering).
CENSORED_MODELS: frozenset[str] = frozenset({"glm-5.3", "minimax-m3", "kimi-k3"})


def is_censored(basis: dict) -> bool:
    """Whether these readings were censored by their own cap (`mx >= cap`), in
    which case p99 is only a lower bound."""
    try:
        return int(basis["mx"]) >= int(basis["cap"])
    except (KeyError, TypeError, ValueError):
        return False            # incomplete reading ⇒ treated as "not censored" (the conservative side is not to auto-raise)


def recommended_budgets() -> dict[str, int]:
    """Recommended `max_tokens` per model from `MEASURED_BUDGET_BASIS`; every
    configured value must be at least this."""
    return {name: budget_for(b["p99"], b["cap"], censored=is_censored(b))
            for name, b in MEASURED_BUDGET_BASIS.items()}


#: Slowest observed throughput (tokens/s) on near-ceiling rows, for models
#: that reach their cap; used to decide whether streaming is needed.
NEAR_CAP_MIN_TPS: dict[str, float] = {
    "glm-5.3": 26.9, "kimi-k3": 29.3, "minimax-m3": 66.6, "gemini-3.8-flash": 78.9,
}


def needs_stream(name: str, backend: str, max_tokens: int) -> bool:
    """Whether this budget cannot finish within the relay's non-streaming
    timeout on this route, so `stream: true` is required."""
    tps = NEAR_CAP_MIN_TPS.get(name)
    if tps is None or str(backend) != "relay":
        return False
    return (int(max_tokens) / tps) >= RELAY_NONSTREAM_WALL_S


# ------------------------------------------------------------------ Budget decision + recording
_BUDGET_AUDIT: dict[str, dict] = {}


def effective_transport(name: str, backend: str, max_tokens: int, stream_cfg: bool) -> dict:
    """The effective budget and transport for this model and route (pure; also
    used by `batch.sampling_fields`, which feeds `solving_sha16`)."""
    rec = recommended_budgets().get(str(name))
    need = needs_stream(str(name), str(backend), int(max_tokens))
    return {"backend": str(backend),
            "max_tokens": int(max_tokens),
            "recommended": rec,
            "basis": ("measured" if rec is not None else "absent:no_measured_basis"),
            "below_recommended": (None if rec is None else int(max_tokens) < rec),
            "needs_stream": bool(need),
            "stream_cfg": bool(stream_cfg),
            "stream": bool(stream_cfg) or bool(need),
            "stream_from": ("rule" if need else ("config" if stream_cfg else "neither"))}


def note_budget(name: str, backend: str, max_tokens: int, stream_cfg: bool) -> dict:
    """Decide the budget/transport for a solver being constructed and record it.

    `recommended` / `below_recommended` are `None` when the model has no
    measured basis. Streaming required by the rule cannot be turned off by config.
    """
    v = effective_transport(name, backend, max_tokens, stream_cfg)
    _BUDGET_AUDIT[str(name)] = v
    return v


def budget_audit_snapshot() -> dict:
    """Budget decisions for solvers this process constructed (see `status`)."""
    if not _BUDGET_AUDIT:
        return {"status": "absent:no_solvers_built", "by_model": {}}
    _no_basis = sorted(n for n, v in _BUDGET_AUDIT.items() if v["recommended"] is None)
    return {"status": "measured",
            "n_models": len(_BUDGET_AUDIT),
            # Models never checked against a measured basis.
            "without_basis": _no_basis,
            "by_model": dict(sorted(_BUDGET_AUDIT.items()))}


def note_payload(name: str, payload: dict | None) -> dict:
    """Record the sampling parameters in one payload actually sent (paired with
    `note_call`, since a relay can rewrite the payload)."""
    got = payload_sampling(payload)
    e = _SAMPLING["by_model"].setdefault(
        str(name), {"backend": "?", "n_calls": 0,
                    "echo_status": "absent:not_observed", "echo_distinct": [],
                    "n_echo_distinct_dropped": 0})
    e["payload_sampling"] = got
    return got


#: Generation `max_tokens` the existing cache (and the checked-in extraction replay fixtures) were
#: keyed under: 72000 unless the model is listed. Only a different budget enters the cache key.
GEN_CACHE_DEFAULT_MAX_TOKENS = 72000
GEN_CACHE_DEFAULT_MAX_TOKENS_BY_MODEL: dict[str, int] = {
    "gemini-3.1-pro": 65536, "gemini-3.7-flash": 65536,
    # Capped at the official 65536 output maximum; cache entries were keyed without a budget,
    # so 65536 reads as this model's default.
    "gemini-3.8-flash": 65536,
}
#: Knobs of a mapped model spec that change a reply (`max_tokens` is handled apart).
_KEYED_SPEC_KNOBS: tuple[str, ...] = tuple(SAMPLING_PARAM_NAMES)


def keyed_sampling(spec: dict, name: str = "") -> dict:
    """The sampling conditions of a mapped generation spec that are not the defaults
    (budget, thinking tier, temperature, ...); `{}` when all are default."""
    out: dict = {}
    mt = int(spec.get("max_tokens", 6000))            # 6000 = the solver's own default
    if mt != GEN_CACHE_DEFAULT_MAX_TOKENS_BY_MODEL.get(name, GEN_CACHE_DEFAULT_MAX_TOKENS):
        out["max_tokens"] = mt
    for k in _KEYED_SPEC_KNOBS:
        if spec.get(k) is not None:
            out[k] = spec[k]
    return out


class Dispatcher:
    """A prompt→text caller for one model key (e.g. `terra`)."""

    def __init__(self, name: str, argv: list[str], ai: str, timeout: int, retries: int,
                 env: dict[str, str] | None = None, cache_dir: Path | None = None,
                 use_cache: bool = True):
        self.name, self.argv, self.ai = name, argv, ai
        self.timeout, self.retries = timeout, retries
        self.env = {**os.environ, **(env or {})}
        self.cache_dir, self.use_cache = cache_dir, use_cache
        self.n_calls = self.n_hits = 0
        self.sampling: dict = {}

    # ------------------------------------------------------------ cache
    def _key(self, prompt: str) -> str:
        # `self.sampling` holds only the knobs that differ from the defaults the cache was
        # built under, so a default run keeps the historical key and a changed knob gets its own.
        tag = ("" if not self.sampling else
               "#sampling=" + json.dumps(self.sampling, sort_keys=True, default=str) + "\x00")
        h = hashlib.sha256(f"{self.name}\x00{' '.join(self.argv)}\x00{tag}{prompt}".encode())
        return h.hexdigest()[:32]

    def unusable(self, out: str | None) -> bool:
        """Whether a response's JSON cannot be extracted (via
        `evaluate.json_unextractable`, the evaluation side's extractor). Only
        applied at call sites that pass `require_json=True`; the LLM judge
        returns free text."""
        if out is None or not str(out).strip():
            return True
        from .evaluate import json_unextractable
        return bool(json_unextractable(out))

    def _load(self, key: str, require_json: bool = False) -> str | None:
        if not (self.use_cache and self.cache_dir):
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.is_file():
            return None
        try:
            resp = json.loads(p.read_text(encoding="utf-8")).get("response")
        except (OSError, json.JSONDecodeError):
            return None
        # An unusable cached response is treated as a miss and overwritten.
        if require_json and self.unusable(resp):
            log.warning("[llm] %s cache entry %s: response could not be parsed as JSON; "
                        "treating as a miss and retrying", self.name, key[:8])
            return None
        return resp

    def _save(self, key: str, prompt: str, response: str, usage: dict | None = None) -> None:
        if not (self.use_cache and self.cache_dir):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # `usage` is stored for auditing; it is not part of the key.
        (self.cache_dir / f"{key}.json").write_text(json.dumps(
            {"model": self.name, "argv": self.argv, "prompt": prompt, "response": response,
             "usage": usage, **({"sampling": self.sampling} if self.sampling else {}),
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    # ------------------------------------------------------------ call
    def __call__(self, prompt: str, require_json: bool = False) -> str:
        key = self._key(prompt)
        hit = self._load(key, require_json)
        if hit is not None:
            self.n_hits += 1
            note_gen_cache_hit(self.name)
            log.info("[llm] %s cache hit %s (cumulative hits %d)", self.name, key[:8], self.n_hits)
            return hit
        last = ""
        for attempt in range(1, self.retries + 2):
            self.n_calls += 1
            try:
                # Isolated cwd: an agentic CLI must not read gold files by relative path.
                proc = subprocess.run([self.ai, *self.argv], input=prompt, capture_output=True,
                                      text=True, timeout=self.timeout, env=self.env,
                                      cwd=_gen_cwd())
                out = proc.stdout or ""
                if out.strip() and not (require_json and self.unusable(out)):
                    # The CLI does not expose usage or the response object;
                    # recorded as unknown, not 0.
                    _note_gen_usage(self.name, None, path="cli")
                    note_call(self.name, "cli", None)
                    self._save(key, prompt, out, usage=None)
                    return out
                last = (f"could not extract JSON (len={len(out)}): {out.strip()[:120]}"
                        if out.strip() else
                        f"empty output (rc={proc.returncode}) {(proc.stderr or '')[-200:]}")
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("[llm] %s call attempt %d failed: %s", self.name, attempt, last)
            time.sleep(3 * attempt)
        raise RuntimeError(f"LLM dispatch failed ({self.name}): {last}")


def make_dispatcher(cfg: dict, model_key: str, root: Path, use_cache: bool = True,
                    cache_subdir: str = "_llm_cache") -> Dispatcher:
    """Build a dispatcher from `config.models[model_key]`.

    `cache_subdir`: keep prose-returning callers (e.g. `record_render`) out of
    the generation cache, whose entries must contain extractable JSON.
    """
    from .evaluate import load_env_file

    spec = (cfg.get("models") or {}).get(model_key)
    if spec is None:
        raise ValueError(f"config.models has no entry for {model_key!r}")
    if isinstance(spec, dict):
        # Mapping form: reuse the evaluation side's direct backend.
        return _MappingDispatcher(
            name=model_key, spec=spec,
            timeout=int(cfg.get("synth", {}).get("gen_timeout_s",
                                                 cfg.get("timeout_s", 900))),
            retries=int(cfg.get("max_retries", 2)),
            env=load_env_file(cfg.get("env_file")),
            cache_dir=root / "cases" / cache_subdir, use_cache=use_cache)
    return Dispatcher(
        name=model_key, argv=[str(x) for x in spec],
        ai=os.path.expanduser(cfg.get("ai_dispatcher", "~/.local/bin/ai")),
        timeout=int(cfg.get("synth", {}).get("gen_timeout_s", cfg.get("timeout_s", 900))),
        retries=int(cfg.get("max_retries", 2)),
        env=load_env_file(cfg.get("env_file")),
        cache_dir=root / "cases" / cache_subdir, use_cache=use_cache)


class _MappingDispatcher(Dispatcher):
    """Generation-side dispatcher for a direct backend; same cache format as
    `Dispatcher`, with the backend in the key so the two paths never cross-hit."""

    def __init__(self, name: str, spec: dict, timeout: int, retries: int,
                 env: dict[str, str] | None = None, cache_dir: Path | None = None,
                 use_cache: bool = True):
        _be = str(spec.get("backend", "openrouter"))
        super().__init__(name=name,
                         argv=[f"direct:{_be}", str(spec.get("model") or name)],
                         ai="", timeout=timeout, retries=retries,
                         env=env, cache_dir=cache_dir, use_cache=use_cache)
        self.spec = dict(spec)
        self.sampling = keyed_sampling(self.spec, name)

    def __call__(self, prompt: str, require_json: bool = False) -> str:
        key = self._key(prompt)
        hit = self._load(key, require_json)
        if hit is not None:
            self.n_hits += 1
            note_gen_cache_hit(self.name)
            log.info("[llm] %s cache hit %s (cumulative hits %d)", self.name, key[:8], self.n_hits)
            return hit
        from .evaluate import raw_complete_with_usage, solver_for_spec
        sv = solver_for_spec(self.name, self.spec, self.env, self.timeout, self.retries)
        if sv is None:
            raise RuntimeError(f"generation model {self.name} could not construct a solver (spec={self.spec})")
        observe_post(sv, self.name, str(self.spec.get("backend", "openrouter")))
        last = ""
        for attempt in range(1, self.retries + 2):
            self.n_calls += 1
            try:
                out, _usage = raw_complete_with_usage(sv, prompt)
                if out and out.strip() and not (require_json and self.unusable(out)):
                    _note_gen_usage(self.name, _usage,
                                    path=f"direct:{self.spec.get('backend', 'openrouter')}")
                    self._save(key, prompt, out, usage=_usage)
                    return out
                last = (f"could not extract JSON (len={len(out)}): {out.strip()[:120]}"
                        if (out or "").strip() else "(empty response)")
            except Exception as e:                                # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
            log.warning("[llm] %s attempt %d failed: %s", self.name, attempt, last)
        raise RuntimeError(f"generation model {self.name} failed all {self.retries + 1} attempts: {last}")
