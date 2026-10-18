"""Typed schema of `config.yaml` and the role of every field.

`load_cfg` (`haenv.cli`) validates the merged configuration here before returning it: an
unknown key inside a known section raises, a missing key takes the one default declared
below. Unknown top-level sections are reported, not refused, so a plugin may keep its own
section (`rare:`); they are listed in the run's configuration record.

Every field carries a role (`role_of`), the claim it makes about batch identity:

* `answer`     -- part of what a model is asked and answers with; in the answer view
                  (`transport.answer_view`), so changing it makes a different solver;
* `route`      -- how a request reaches the model; recorded per receipt, never identity;
* `execution`  -- how a run is executed (threads, lanes, timeouts, pacing, logging);
* `generation` -- how cases are generated; covered by the world stamp;
* `selection`  -- which models a run covers;
* `report`     -- how reports render;
* `provenance` -- recorded facts with no effect on answers;
* `location`   -- where code, files or tools are found;
* `judging`    -- part of a judge policy's identity (`semantic_rubric.judging_policy`).

The roles are checked against the identity functions themselves: perturbing an `answer`
field of a model moves its answer view, perturbing any other field does not.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from haenv.runtime.config_contracts import ConfigError

log = logging.getLogger("haenv.settings")

Role = Literal["answer", "route", "execution", "generation", "selection", "report",
               "provenance", "location", "judging"]


def _f(role: Role, default: Any = None, **kw):
    return Field(default, json_schema_extra={"role": role}, **kw)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteSpec(_Section):
    """One `fallback_routes` entry: overrides the model's own spec on that route."""
    backend: str | None = _f("route")
    model: str | None = _f("route")
    upstream: str | None = _f("route")
    provider: dict | None = _f("route")
    stream: bool | None = _f("route")
    retries: int | None = _f("route")
    max_tokens_field: str | None = _f("route")


class PriceSpec(_Section):
    """`config.models.<name>.price`: what the model costs on a backend with no verified price
    data, in USD per million tokens (0 for a free endpoint). Billed requests on that backend
    are counted at this price; on `openrouter`, `relay`, `google` and `dashscope` it is refused."""
    input_per_million_usd: float = _f("route", ..., ge=0, allow_inf_nan=False)
    output_per_million_usd: float = _f("route", ..., ge=0, allow_inf_nan=False)


class JudgeIdentity(_Section):
    """Execution's declaration of the snapshot behind a sealed judge alias."""
    canonical_model: str = _f("provenance", ..., min_length=1)
    snapshot: date = _f("provenance", ...)


class JudgeTokenRates(PriceSpec):
    cached_input_per_million_usd: float | None = _f("route", None, ge=0, allow_inf_nan=False)
    cache_creation_per_million_usd: float | None = _f("route", None, ge=0, allow_inf_nan=False)
    reasoning_per_million_usd: float | None = _f("route", None, ge=0, allow_inf_nan=False)


class JudgePriceTier(JudgeTokenRates):
    prompt_tokens_above: int = _f("route", ..., gt=0, strict=True)


class JudgePriceSpec(JudgeTokenRates):
    source: str = _f("provenance", ..., min_length=1)
    tariff_checked_on: date = _f("provenance", ...)
    completion_includes_reasoning: bool = _f("route", True)
    prompt_tiers: list[JudgePriceTier] = _f("route", [])

    @model_validator(mode="after")
    def ordered_tiers(self):
        limits = [tier.prompt_tokens_above for tier in self.prompt_tiers]
        if limits != sorted(set(limits)):
            raise ValueError("prompt_tiers thresholds must be unique and increasing")
        return self


class JudgeRouteSpec(_Section):
    """One equivalent judge route; credentials are environment variable names only."""
    backend: Literal["openrouter", "relay", "google"] = _f("route", ...)
    model: str = _f("route", ..., min_length=1)
    canonical_model: str = _f("provenance", ..., min_length=1)
    snapshot: date = _f("provenance", ...)
    url: str | None = _f("route")
    base_url_env: str | None = _f("route")
    env_file: str | None = _f("location")
    key_env: str | None = _f("route")
    key_env_pool: str | None = _f("route")
    proxy: str = _f("route", "env")
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = _f("route", "max_tokens")
    concurrency: int = _f("execution", 1, ge=1, le=64, strict=True)
    provider: dict | None = _f("route")
    upstream: str | None = _f("route")
    price: JudgePriceSpec | None = _f("route")
    failure_backoff_s: float = _f("execution", 2.0, gt=0, le=300)
    max_backoff_s: float = _f("execution", 60.0, gt=0, le=3600)

    @field_validator("proxy")
    @classmethod
    def valid_proxy(cls, value):
        from haenv.runtime.config_contracts import check_proxy
        return check_proxy(value)

    @field_validator("url")
    @classmethod
    def valid_url(cls, value):
        if value is not None:
            from urllib.parse import urlsplit
            parsed = urlsplit(value)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("url must be an HTTP(S) endpoint without credentials or query")
        return value

    @field_validator("key_env", "key_env_pool", "base_url_env")
    @classmethod
    def env_name(cls, value):
        import re
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
            raise ValueError("expected an environment variable name")
        return value

    @model_validator(mode="after")
    def route_requirements(self):
        if self.url and self.base_url_env:
            raise ValueError("use url or base_url_env, not both")
        if self.backend == "relay" and self.price is None:
            raise ValueError("relay judge routes require a dated price with a source")
        if self.backend != "relay" and self.price is not None:
            raise ValueError("declared prices are only for relay judge routes")
        return self


class ModelSpec(_Section):
    """`config.models.<name>`."""
    backend: str = _f("answer", "openrouter")
    model: str = _f("answer", ...)
    max_tokens: int = _f("answer", 6000)
    reasoning_effort: str | None = _f("answer")
    upstream: str | None = _f("answer")
    provider: dict | None = _f("answer")
    #: temperature (number or "unsupported:<why>") and Google thinking_budget
    #: (`transport.check_sampling`).
    sampling: dict | None = _f("answer")
    stream: bool = _f("route", False)
    retries: int | None = _f("route")
    fallback_routes: list[RouteSpec] = _f("route", [])
    price: PriceSpec | None = _f("route")
    vendor: str | None = _f("provenance")
    retired: str | None = _f("provenance")


class BackendSpec(_Section):
    """`config.backends.<name>`: applied over `evaluate.BACKENDS` field by field."""
    url: str | None = _f("route")
    key_env: str | None = _f("route")
    key_env_pool: str | None = _f("route")
    kind: str | None = _f("route")
    no_proxy_host: str | None = _f("route")
    max_tokens_field: str | None = _f("route")
    quota_url: str | None = _f("route")


class EvalSection(_Section):
    workers: int = _f("execution", 8)
    model_workers: dict[str, int] = _f("execution", {})
    model_wall_budget_s: dict[str, float] = _f("execution", {})
    pooling: Literal["model", "backend", "elastic"] = _f("execution", "elastic")
    elastic_max: int = _f("execution", 16)
    #: model -> most cells of that model in flight, whatever the pooling (a provider's
    #: per-model concurrency limit; `haenv probe concurrency` measures it). Unnamed models
    #: are bounded by their backend only.
    model_caps: dict[str, int] = _f("execution", {})


class PaidSection(_Section):
    global_request_limit: int = _f("execution", 48)
    judge_reserved_lanes: int = _f("execution", 3)
    #: backend -> size of its own paid-request pool (`paid_slots`); unnamed backends share
    #: `global_request_limit`.
    backend_request_limits: dict[str, int] = _f("execution", {})
    max_capacity: int | None = _f("execution")
    #: How often the head of a queue of requests waiting for a lane looks for a free one
    #: (`paid_slots.request_slot`); the other waiters sleep until it is their turn.
    slot_poll_s: float = _f("execution", 0.01, gt=0)
    #: Requests one process may hold a ledger reservation for while they wait for a lane. A
    #: request reserves before it queues, so that ledger latency never holds a lane; without
    #: this bound every cell thread of every process reserves at start-up and the ledger's
    #: write queue is hundreds deep. 0: no bound.
    reserve_ahead: int = _f("execution", 8, ge=0)
    #: Refuse a billed run whose OpenRouter solver has no provider pin (checked before any
    #: request; the pin itself is part of the answer view through `upstream`).
    require_upstream_pin: bool = _f("execution", False)
    #: Day you compared the built-in `google` / `dashscope` prices (`haenv.runtime.native_accounting`)
    #: with the vendors' pricing pages. A built-in check is good for 7 days; past that, a
    #: request warns once per backend unless this date (YYYY-MM-DD) is within 7 days of today.
    #: Recorded in the run's price metadata as `tariff_check`.
    tariff_checked_on: date | None = _f("provenance")
    #: What a request does when the tariff check is older than 7 days: `warn` sends it and warns
    #: once per backend, `refuse` stops it with "Tariff verification expired". Official boards
    #: run with `refuse`.
    tariff_expiry: Literal["warn", "refuse"] = _f("execution", "warn")


class BudgetSection(_Section):
    """`config.budget`: how the paid-request ledger (`semantic_budget.BudgetLedger`) reserves,
    waits at its cap and shares its store between processes."""
    # The historic schema description is stable; implementation: haenv.runtime.budget.
    #: A request reserves this quantile of the settled cost of its cost key (role, backend,
    #: model, request geometry) once the key has `reservation_min_observations` settled
    #: requests; before that, and whenever the quantile is larger, it reserves its worst case.
    reservation_quantile: float = _f("execution", 0.99, gt=0, le=1)
    reservation_min_observations: int = _f("execution", 30, ge=1)
    #: Admission keeps `overrun_slots` x the largest (worst case - reservation) of the
    #: in-flight requests free under the cap, so that many requests overrunning their
    #: reservation at the same time still settle within it. 0 disables quantile reservations.
    overrun_slots: int = _f("execution", 1, ge=0)
    #: At the cap a request (solver or judge) waits for room, re-reading the ledger every
    #: `pause_poll_s`, and gives up after `pause_timeout_s` (0: refuse at once, the default;
    #: a run that should wait for a raised cap sets it). The runtime document's `budget`
    #: block overrides both while a run is live.
    pause_poll_s: float = _f("execution", 30.0, ge=0)
    pause_timeout_s: float = _f("execution", 0.0, ge=0)
    #: How long one ledger transaction waits for another process's write lock.
    busy_timeout_s: float = _f("execution", 60.0, gt=0)
    #: fsync every ledger commit. Off, a commit survives the process being killed and is
    #: lost only if the host itself goes down before the next checkpoint (SQLite WAL with
    #: `synchronous=NORMAL`); on, each commit waits for the disk.
    durable_commits: bool = _f("execution", False)


class TransportSection(_Section):
    connect_per_s: float = _f("execution", 8.0)
    connect_burst: int = _f("execution", 8)
    connect_retries: int = _f("execution", 3)
    #: Base URL of a local fake provider (`haenv loadtest`). When set, every HTTPS request
    #: goes to it over plain HTTP instead, keeping its path, query and `Host` header.
    rehearsal_url: str | None = _f("route", None)


class ProbeSection(_Section):
    """`haenv probe concurrency` (`haenv/runtime/probe.py`): ramp tiny requests per model."""
    #: concurrency of each step, in order
    steps: list[int] = _f("execution", [8, 16, 32, 64, 128, 192])
    #: seconds each step keeps sending
    step_s: float = _f("execution", 15.0)
    prompt: str = _f("execution", "Reply with the single word OK.")
    max_tokens: int = _f("execution", 16)
    #: tokens a chat template adds to the prompt, for the per-request cost bound
    prompt_overhead_tokens: int = _f("execution", 64)
    #: OpenRouter `reasoning.effort` sent with every probe request
    reasoning_effort: str = _f("execution", "minimal")
    #: backend -> fields merged into the request body (Google: into `generationConfig`)
    extra_body: dict[str, dict] = _f("execution", {})
    request_timeout_s: float = _f("execution", 60.0)
    #: a step fails above this share of rejected requests (429, rate-limit bodies, 5xx, timeouts)
    max_rejection_rate: float = _f("execution", 0.05)
    #: a step fails when its latency p95 exceeds this multiple of the first step's p95
    max_latency_growth: float = _f("execution", 3.0)
    #: safe concurrency = floor(safety_factor x the last step that passed)
    safety_factor: float = _f("execution", 0.7)
    #: new connections per second for the probe's own requests (burst: the largest step);
    #: replaces `transport.connect_per_s`, which would otherwise cap the measured rate
    connect_per_s: float = _f("execution", 1000.0)
    #: request bounds reserved on the ledger per reservation
    lease_requests: int = _f("execution", 2)


class SynthSection(_Section):
    #: Hospital rules: unset follows the world (long = library, short = printed).
    rules_delivery: Literal["printed", "library"] | None = _f("generation")
    max_rounds: int = _f("generation", 6)
    generator: Literal["llm", "deterministic"] = _f("generation", "deterministic")
    gen_model: str = _f("generation", "gemini-3.8-flash")
    gen_workers: int = _f("execution", 8)
    gen_timeout_s: int = _f("execution", 900)
    gen_cache: bool = _f("execution", True)
    #: pack 3's background layer (`haenv/p3`: conspicuous factors that do not act on the decision); false
    #: builds the pack as it was before the layer existed, byte for byte
    background: bool = _f("generation", True)
    #: task-pack worlds: `long` = years of record before day 0 (sparse labs, daily wearable streams,
    #: medication record), `short` = the course only (the world as before)
    world_history: Literal["long", "short"] = _f("generation", "long")
    #: task-pack worlds: background medicines start, stop and change dose at real rates (`true`) or keep the
    #: course-only up-titration (`false`, the world as before); a bool, since YAML reads on/off as one
    med_dynamics: bool = _f("generation", True)


class WorldSection(_Section):
    """Optional canonical world controls; unset fields use the legacy synth defaults."""
    history: Literal["long", "short"] | None = _f("generation")
    med_dynamics: bool | None = _f("generation")


class QuerySection(_Section):
    """`config.query`: how a pack case is put to a model (`haenv/query`). Under `geometry: query`
    a case of a pack with a query adapter is asked without its data inlined; the model reads the
    record through read-only tools. Every field is part of the solver's answer view
    (`solver_accounting.wire_settings`), so a batch records the limits it ran under."""
    #: `query`: tools over the record (default); `single`: the whole record in one prompt.
    geometry: Literal["query", "single"] = _f("answer", "query")
    #: tool turns before the forced final turn (the tools stay declared and are closed)
    max_rounds: int = _f("answer", 32, ge=1)
    #: tool calls answered in a cell; a call past it gets an exhausted-budget result
    max_calls: int = _f("answer", 256, ge=1)
    #: points or rows one call returns at most; a longer window says it was cut and how to page
    max_points: int = _f("answer", 400, ge=10)


class ReportSection(_Section):
    sample_cases: int = _f("report", 5)
    timeline_points: int = _f("report", 6)


class Config(BaseModel):
    """The whole of `config.yaml` after `config.local.yaml`, the overlay and `--set`."""
    model_config = ConfigDict(extra="allow")

    background: bool | None = _f("generation")
    world: WorldSection = _f("generation", WorldSection())

    kernel_path: str = _f("location", "haenv_kernel")
    models: dict[str, ModelSpec] | list[str] = _f("selection", {})
    default_models: list[str] = _f("selection", [])
    backends: dict[str, BackendSpec] = _f("route", {})
    judge_identity: JudgeIdentity | None = _f("provenance")
    judge_routes: dict[str, JudgeRouteSpec] = _f("route", {})
    backend_limits: dict[str, int] = _f("execution", {})
    eval: EvalSection = _f("execution", EvalSection())
    paid: PaidSection = _f("execution", PaidSection())
    budget: BudgetSection = _f("execution", BudgetSection())
    ai_dispatcher: str = _f("location", "~/.local/bin/ai")
    env_file: str | None = _f("location")
    timeout_s: int = _f("execution", 900)
    #: Seconds a thread keeps the interpreter lock before another may take it (Python's own
    #: default is 0.005). A run has hundreds of threads; with the default, a thread that has to
    #: retake the lock in the middle of a ledger transaction waits behind every runnable thread
    #: for 5 ms each time, and everything queued on the transaction's write lock waits with it.
    thread_switch_interval_s: float = _f("execution", 0.0005, gt=0)
    max_retries: int = _f("route", 2)
    transport: TransportSection = _f("execution", TransportSection())
    probe: ProbeSection = _f("execution", ProbeSection())
    verbose: int = _f("execution", 1)
    synth: SynthSection = _f("generation", SynthSection())
    query: QuerySection = _f("answer", QuerySection())
    report: ReportSection = _f("report", ReportSection())


class JudgeSpec(_Section):
    """The `judge:` block of `registry/semantic_judging*.yaml`. Only `max_concurrency` says
    how a run executes; every other field is part of what the judge is."""
    model_key: str = _f("judging", ...)
    model_id: str = _f("judging", ...)
    backend: str = _f("judging", ...)
    reasoning_effort: str = _f("judging", ...)
    max_tokens: int = _f("judging", ...)
    consensus: str = _f("judging", ...)
    failure_policy: str = _f("judging", ...)
    evidence_mode: str = _f("judging", ...)
    max_concurrency: int | None = _f("execution")


def role_of(model: type[BaseModel], name: str) -> str:
    extra = model.model_fields[name].json_schema_extra or {}
    return extra["role"]


def fields_with_role(model: type[BaseModel], *roles: str) -> tuple[str, ...]:
    return tuple(n for n in model.model_fields if role_of(model, n) in roles)


def _fill(raw: dict, model: type[BaseModel]) -> dict:
    """`raw` with each absent declared field set to its default, recursively into sections.
    Present values are kept as written (no coercion), so validation never changes a value."""
    out = dict(raw)
    for name, field in model.model_fields.items():
        sub = field.annotation if isinstance(field.annotation, type) else None
        if name not in out:
            if not field.is_required():
                default = field.get_default(call_default_factory=True)
                out[name] = default.model_dump() if isinstance(default, BaseModel) else copy.deepcopy(default)
        elif sub is not None and issubclass(sub, BaseModel) and isinstance(out[name], dict):
            out[name] = _fill(out[name], sub)
    return out


def validate(cfg: dict) -> dict:
    """Validate the merged configuration and return it with defaults filled in.

    Values present in `cfg` are returned unchanged; only absent keys are added.
    """
    try:
        Config.model_validate(cfg)
    except ValidationError as error:
        # `models` is a mapping or a list: report the branch the file actually uses, and
        # drop pydantic's union tags (`dict[str,ModelSpec]`) from the location.
        branch = "list[str]" if isinstance(cfg.get("models"), list) else "dict[str,ModelSpec]"
        errors = [e for e in error.errors()
                  if not (e["loc"][:1] == ("models",) and len(e["loc"]) > 1 and e["loc"][1] != branch)]
        lines = [f"  {'.'.join(str(p) for p in e['loc'] if p != branch)}: {e['msg']}" for e in errors]
        raise ConfigError("configuration does not match haenv/app/settings.py:\n" + "\n".join(lines)) from None
    unknown = unknown_sections(cfg)
    if unknown:
        log.debug("[config] sections not in the schema (plugin-owned?): %s", unknown)
    return _fill(cfg, Config)


def unknown_sections(cfg: dict) -> list[str]:
    return sorted(set(cfg) - set(Config.model_fields))


# ---------------------------------------------------------------- `--set a.b=value`

SET_ENV = "HAENV_CONFIG_SET"


def parse_set(items: list[str]) -> list[tuple[str, Any]]:
    """`["eval.workers=4", ...]` -> `[("eval.workers", 4), ...]`; values are read as YAML."""
    import yaml
    out = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        out.append((key.strip(), yaml.safe_load(value) if value.strip() else None))
    return out


def export_set(items: list[str]) -> None:
    """Record `--set` items in the environment, so subprocesses apply the same overrides. Items this
    process inherited (a parent's `--set`) stay in front; the process's own come after and win."""
    parse_set(items)                                     # refuse a malformed item now
    inherited = json.loads(os.environ.get(SET_ENV) or "[]")
    os.environ[SET_ENV] = json.dumps(list(inherited) + [i for i in items if i not in inherited])


def apply_set(cfg: dict) -> dict:
    """Apply `$HAENV_CONFIG_SET` over `cfg`, in place."""
    raw = os.environ.get(SET_ENV, "").strip()
    if not raw:
        return cfg
    for key, value in parse_set(json.loads(raw)):
        node, parts = cfg, key.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if nxt is None:
                nxt = node[part] = {}
            if not isinstance(nxt, dict):
                raise ConfigError(f"--set {key}: {part} is not a section")
            node = nxt
        node[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------- the run's configuration record

_SECRET_WORDS = ("key", "token", "secret", "password")


def _redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            low = str(k).lower()
            secret = (any(w in low for w in _SECRET_WORDS)
                      and not low.endswith(("_env", "_env_pool")))
            out[k] = "<redacted>" if secret and isinstance(v, str) else _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def record(cfg: dict, layers: list[dict]) -> dict:
    """One entry of `<batch>/config_runs.jsonl`: every layer and the resolved configuration."""
    from datetime import datetime, timezone
    resolved = _redact(cfg)
    text = json.dumps(resolved, sort_keys=True, ensure_ascii=False, default=str)
    return {"at": datetime.now(timezone.utc).isoformat(), "layers": layers,
            "set": json.loads(os.environ.get(SET_ENV) or "[]"),
            "unknown_sections": unknown_sections(cfg),
            "resolved_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "resolved": resolved}


def append_record(batch_dir, cfg: dict, layers: list[dict]) -> None:
    from pathlib import Path
    path = Path(batch_dir) / "config_runs.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record(cfg, layers), ensure_ascii=False, default=str) + "\n")
