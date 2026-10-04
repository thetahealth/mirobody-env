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
import os
import subprocess
import tempfile
import time
from pathlib import Path
from .call_log import (  # noqa: F401
    ANSWER_RESERVE_TOKENS,
    BUDGET_STEP,
    CENSORED_MODELS,
    CHARS_PER_TOKEN,
    DEGENERATE_UNIQ_RATIO,
    ECHO_FIELDS,
    GOOGLE_SAMPLING_KEYS,
    MEASURED_BUDGET_BASIS,
    NEAR_CAP_MIN_TPS,
    RELAY_NONSTREAM_WALL_S,
    SAMPLING_PARAM_NAMES,
    TAIL_FACTOR,
    _BUDGET_AUDIT,
    _ECHO_ALIASES,
    _ECHO_KEEP,
    _GEN_USAGE,
    _SAMPLING,
    _note_gen_usage,
    budget_audit_snapshot,
    budget_for,
    budget_risk_line,
    ceil_to_step,
    ceiling_flags,
    effective_transport,
    gen_usage_snapshot,
    is_censored,
    log,
    looks_degenerate,
    needs_stream,
    note_budget,
    note_call,
    note_gen_cache_hit,
    note_payload,
    observe_post,
    payload_sampling,
    recommended_budgets,
    response_echo,
    sampling_snapshot,
)
from .solve_guard import json_unextractable

# Isolated working directory for generation subprocesses, so they cannot read the repo.
_GEN_CWD: str | None = None


def _gen_cwd() -> str:
    global _GEN_CWD
    if _GEN_CWD is None:
        _GEN_CWD = tempfile.mkdtemp(prefix="haenv-gen-cwd-")
    return _GEN_CWD

# ============================================================ Sampling parameters: not pinned, but recorded
# Temperature is left unpinned (temperature 0 degrades long generations and
# reasoning models often ignore it); the conditions actually sent and echoed
# are recorded instead. "Not specified" is never recorded as a value.

# ============================================================ Generation budget (`max_tokens`) tiers
# gen = `evaluate.billed_tokens(usage)["out"]` (completion + reasoning);
# base = observed cap if the cap censored the distribution, else gen's p99:
#   max_tokens = ceil_4000( max( base × TAIL_FACTOR,  base + ANSWER_RESERVE_TOKENS ) )

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
    from .solvers import load_env_file

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
        from .solvers import raw_complete_with_usage, solver_for_spec
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
