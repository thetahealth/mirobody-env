"""Model-call bookkeeping shared by every caller: per-call usage and sampling echo, the
token-budget policy, and the budget audit. Imports nothing above the standard library, so
the solver layer and the generation dispatcher can both depend on it.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import logging


log = logging.getLogger("haenv.llm")


# ============================================================ Generation-side usage ledger
# Process-scoped; read by `batch.register`. Cache hits are counted separately
# from calls whose backend returned no usage.
_GEN_USAGE: dict = {"n_calls": 0, "n_hits": 0, "n_with_usage": 0, "n_without_usage": 0,
                    "in_fresh": 0, "in_cached": 0, "out": 0, "by_model": {}}


def _note_gen_usage(model: str, usage: dict | None, path: str) -> None:
    """Record one generation call that actually went out. `usage=None`
    is recorded under `n_without_usage`, not collapsed to 0."""
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


def billed_tokens(usage: dict | None) -> dict:
    """Normalize each backend's usage into fresh input / cached input / output.

    With `billing_usage.claude_usage` (some relays), `input + cache_read == prompt_tokens`.
    Otherwise `cached = prompt_tokens_details.cached_tokens or 0` and
    `fresh = prompt_tokens - cached`.
    """
    u = usage or {}
    out = int(u.get("completion_tokens") or 0)
    cu = (u.get("billing_usage") or {}).get("claude_usage") or {}
    fresh_c, cached_c = cu.get("input_tokens"), cu.get("cache_read_input_tokens")
    if isinstance(fresh_c, int) and isinstance(cached_c, int):
        fresh, cached = fresh_c, cached_c
    else:
        total = int(u.get("prompt_tokens") or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        fresh = max(0, total - cached)
    # Thinking tokens (Google `thoughtsTokenCount`) count as output; the unaccounted
    # residual is kept.
    _th = int(u.get("thoughts_tokens") or 0)
    _tot = u.get("total_tokens")
    return {"in_fresh": fresh, "in_cached": cached,
            "out": out + _th, "out_thoughts": _th or None,
            "in_total": fresh + cached,
            "unaccounted": (int(_tot) - (fresh + cached) - out - _th
                            if isinstance(_tot, int) else None),
            "status": "measured" if u else "absent"}


# ---------------------------------------------------------------- key pool
#
# Rotation happens only when "quota exhausted" is recognized (never on timeouts, rate limits
# or server errors), does not consume the cell's retry budget, and marks the key dead for
# the process lifetime. An empty pool falls through to ABORT(no_response).
_EXHAUSTED_MARKERS = ("quota is exhausted", "TokenStatusExhausted", "quota is not enough")


def _is_exhausted(err_body: str) -> bool:
    return any(m.lower() in (err_body or "").lower() for m in _EXHAUSTED_MARKERS)
