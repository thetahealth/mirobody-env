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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class ModelSpec(_Section):
    """`config.models.<name>`."""
    backend: str = _f("answer", "openrouter")
    model: str = _f("answer", ...)
    max_tokens: int = _f("answer", 6000)
    reasoning_effort: str | None = _f("answer")
    upstream: str | None = _f("answer")
    provider: dict | None = _f("answer")
    stream: bool = _f("route", False)
    retries: int | None = _f("route")
    fallback_routes: list[RouteSpec] = _f("route", [])
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
    pooling: Literal["model", "backend"] = _f("execution", "model")


class PaidSection(_Section):
    global_request_limit: int = _f("execution", 48)
    judge_reserved_lanes: int = _f("execution", 3)
    max_capacity: int | None = _f("execution")


class TransportSection(_Section):
    connect_per_s: float = _f("execution", 8.0)
    connect_burst: int = _f("execution", 8)
    connect_retries: int = _f("execution", 3)


class SynthSection(_Section):
    max_rounds: int = _f("generation", 6)
    generator: Literal["llm", "deterministic"] = _f("generation", "deterministic")
    gen_model: str = _f("generation", "gemini-3.8-flash")
    gen_workers: int = _f("execution", 8)
    gen_timeout_s: int = _f("execution", 900)
    gen_cache: bool = _f("execution", True)


class ReportSection(_Section):
    sample_cases: int = _f("report", 5)
    timeline_points: int = _f("report", 6)


class Config(BaseModel):
    """The whole of `config.yaml` after `config.local.yaml`, the overlay and `--set`."""
    model_config = ConfigDict(extra="allow")

    kernel_path: str = _f("location", "core")
    models: dict[str, ModelSpec] | list[str] = _f("selection", {})
    default_models: list[str] = _f("selection", [])
    backends: dict[str, BackendSpec] = _f("route", {})
    backend_limits: dict[str, int] = _f("execution", {})
    eval: EvalSection = _f("execution", EvalSection())
    paid: PaidSection = _f("execution", PaidSection())
    ai_dispatcher: str = _f("location", "~/.local/bin/ai")
    env_file: str | None = _f("location")
    timeout_s: int = _f("execution", 900)
    max_retries: int = _f("route", 2)
    transport: TransportSection = _f("execution", TransportSection())
    verbose: int = _f("execution", 1)
    synth: SynthSection = _f("generation", SynthSection())
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


class ConfigError(ValueError):
    """The merged configuration does not match the schema."""


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
        raise ConfigError("configuration does not match haenv/settings.py:\n" + "\n".join(lines)) from None
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
    """Record `--set` items in the environment, so subprocesses apply the same overrides."""
    parse_set(items)                                     # refuse a malformed item now
    os.environ[SET_ENV] = json.dumps(items)


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
