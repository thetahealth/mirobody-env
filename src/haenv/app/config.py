"""Configuration loading: base, legacy local, deployment profile, experiment overlay,
then `--set`, with typed validation.

Below every caller: the solver layer, the schedulers and the judges read configuration here
without importing the command line.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from haenv import data_root as _data_root
import os
from pathlib import Path
from pydantic import ValidationError
import haenv.app.settings as settings
from haenv.resources.yamlcache import load_yaml as _cached

ROOT = _data_root()

DEFAULT_BACKEND_LIMIT = 2
DEFAULT_BACKEND_LIMITS = {"relay": 6, "openai": 3, "google": 3,
                          "openrouter": 2, "_cli": 2, "_offline": 8}


def _deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` into `base` in place: mappings recurse, everything else replaces."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg() -> dict:
    """Merge deployment and experiment layers, then validate the resolved configuration."""
    # Parsed once per process and returned as deep copies (callers mutate them).
    cfg = _cached(ROOT / "config.yaml")
    local = ROOT / "config.local.yaml"
    if local.is_file():
        legacy = _cached(local) or {}
        for path in CONCURRENCY_PATHS:
            parent = legacy if len(path) == 1 else legacy.get(path[0], {})
            if isinstance(parent, dict) and path[-1] in parent:
                raise ConfigError(f"config.local.yaml cannot set {'.'.join(path)}; "
                                  "move concurrency to HAENV_DEPLOY_CONFIG")
        _deep_merge(cfg, legacy)
    deploy = deployment_path()
    if deploy is not None:
        if not deploy.is_file():
            raise FileNotFoundError(f"{DEPLOY_CONFIG_ENV} names a file that does not exist: {deploy}")
        profile = _cached(deploy) or {}
        if not isinstance(profile, dict):
            raise ConfigError(f"{DEPLOY_CONFIG_ENV} must name a YAML mapping: {deploy}")
        _deep_merge(cfg, profile)
    cfg = apply_config_overlay(cfg)
    overrides = settings.apply_set({})
    if overrides:
        check_concurrency_overlay(cfg, overrides)
    return validate(_deep_merge(cfg, overrides))


def config_layers() -> list[dict]:
    """Every file merged into `load_cfg`, in order, with its SHA-256 (`--set` is recorded
    separately, see `settings.record`)."""
    import hashlib
    paths = [ROOT / "config.yaml", ROOT / "config.local.yaml"]
    deploy = deployment_path()
    if deploy is not None:
        paths.append(deploy)
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if named:
        paths.append(Path(named))
    return [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths if p.is_file()]


def value_source(*path: str) -> dict:
    """Origin of a resolved leaf, including equal-valued overlay copies."""
    def contains(mapping):
        for key in path:
            if not isinstance(mapping, dict) or key not in mapping:
                return False
            mapping = mapping[key]
        return True

    source = {"kind": "default", "path": None}
    files = [("config.yaml", ROOT / "config.yaml"),
             ("config.local.yaml", ROOT / "config.local.yaml"),
             ("deployment", deployment_path())]
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if named:
        files.append(("overlay", Path(named)))
    for kind, file in files:
        if file is not None and file.is_file() and contains(_cached(file) or {}):
            source = {"kind": kind, "path": str(file)}
    if contains(settings.apply_set({})):
        source = {"kind": "--set", "path": None}
    return source


CONFIG_OVERLAY_ENV = "HAENV_CONFIG_OVERLAY"
DEPLOY_CONFIG_ENV = "HAENV_DEPLOY_CONFIG"


def deployment_path() -> Path | None:
    named = os.environ.get(DEPLOY_CONFIG_ENV, "").strip()
    return Path(named).expanduser() if named else None


def paid_experiment_limits(cfg: dict, backend: str) -> tuple[int | None, int | None]:
    """Explicit experiment narrowing, independent of a live ledger's shared ceiling."""
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    over = (_cached(Path(named)) or {}) if named else {}
    _deep_merge(over, settings.apply_set({}))
    paid = over.get("paid") or {}
    own = (paid.get("backend_request_limits") or {}).get(backend)
    if own is None and backend not in cfg["paid"]["backend_request_limits"]:
        own = paid.get("global_request_limit")
    return own, paid.get("judge_reserved_lanes")


def apply_config_overlay(cfg: dict) -> dict:
    """Merge the per-run overlay named by `$HAENV_CONFIG_OVERLAY` over `cfg`, in place.

    The overlay is a YAML file in the shape of `config.yaml`; it is applied after
    `config.local.yaml`, so one process (one `haenv run`) can route a model to a different
    backend without touching either file. What was routed is recorded in `batch.json`
    (`sampling.per_model.<model>.backend` and `solving_sha16`). A named overlay that does not
    exist is an error: a silent fallback would run the cells on the wrong provider.
    """
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if not named:
        return cfg
    path = Path(named)
    if not path.is_file():
        raise FileNotFoundError(f"{CONFIG_OVERLAY_ENV} names a file that does not exist: {path}")
    overlay = _cached(path) or {}
    if not isinstance(overlay, dict):
        raise ConfigError(f"{CONFIG_OVERLAY_ENV} must name a YAML mapping: {path}")
    check_concurrency_overlay(cfg, overlay)
    return _deep_merge(cfg, overlay)


# Execution-only extensions stay here because settings.py is currently a whole-file
# dependency of pack WORLD stamps. See the concurrency configuration decision.
ConfigError = settings.ConfigError


class PaidConcurrencySettings(settings.PaidSection):
    max_resends: int = settings._f("execution", 2, ge=0, strict=True)
    catalog_cache_ttl_s: float = settings._f("execution", 3600.0, ge=0, allow_inf_nan=False)
    starvation_threshold: float = settings._f("execution", 0.3, ge=0, le=1, strict=True)
    starvation_units: int = settings._f("execution", 5, ge=1, strict=True)


class GateSettings(settings._Section):
    n: int = settings._f("execution", 50, ge=1)
    k: int = settings._f("execution", 3, ge=1)
    models: list[str] = settings._f("execution", [])
    pack_models: dict[str, list[str]] = settings._f("execution", {})
    bootstrap: int = settings._f("execution", 2000, ge=10)
    seed: int = settings._f("execution", 20261012)
    strongest_raw_max: float = settings._f("execution", 0.90, ge=0, le=1)
    all_right_max_fraction: float = settings._f("execution", 0.5, ge=0, le=1)


class DiversitySettings(settings._Section):
    max_category_share: float = settings._f("execution", 0.5, gt=0, le=1)


class RuntimeBudgetSettings(settings.BudgetSection):
    counted_backends: list[str] | None = settings._f("execution")


class BoardSettings(settings._Section):
    official: bool = settings._f("execution", False, strict=True)


class RuntimeTransportSettings(settings.TransportSection):
    backend_proxies: dict[str, str] = settings._f("route", {})


class RuntimeEvalSettings(settings.EvalSection):
    saturation_report_s: float = settings._f("execution", 300.0, ge=0, strict=True, allow_inf_nan=False)


class RuntimeConfig(settings.Config):
    eval: RuntimeEvalSettings = settings._f("execution", RuntimeEvalSettings())
    board: BoardSettings = settings._f("execution", BoardSettings())
    transport: RuntimeTransportSettings = settings._f("route", RuntimeTransportSettings())
    paid: PaidConcurrencySettings = settings._f("execution", PaidConcurrencySettings())
    gate: GateSettings = settings._f("execution", GateSettings())
    diversity: DiversitySettings = settings._f("execution", DiversitySettings())
    budget: RuntimeBudgetSettings = settings._f("execution", RuntimeBudgetSettings())
    concurrency_override_reason: str | None = settings._f("provenance")


def validate(cfg: dict) -> dict:
    from haenv.runtime.config_contracts import validate_proxy_config
    return validate_proxy_config(cfg, _validate_runtime)


def _validate_runtime(cfg: dict) -> dict:
    """Validate the original schema and the execution-only concurrency extension."""
    original = dict(cfg)
    original.pop("concurrency_override_reason", None)
    original.pop("board", None)
    original.pop("gate", None)
    original.pop("diversity", None)
    if isinstance(original.get("eval"), dict):
        original["eval"] = {k: v for k, v in original["eval"].items() if k != "saturation_report_s"}
    if isinstance(original.get("budget"), dict):
        original["budget"] = {k: v for k, v in original["budget"].items() if k != "counted_backends"}
    if isinstance(original.get("paid"), dict):
        original["paid"] = {k: v for k, v in original["paid"].items()
                            if k not in {"starvation_threshold", "starvation_units", "max_resends", "catalog_cache_ttl_s"}}
    if isinstance(original.get("transport"), dict):
        original["transport"] = {k: v for k, v in original["transport"].items() if k != "backend_proxies"}
    settings.validate(original)
    try:
        RuntimeConfig.model_validate(cfg)
    except ValidationError as error:
        lines = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()]
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from error
    return settings._fill(cfg, RuntimeConfig)


def append_record(batch_dir, cfg: dict, layers: list[dict]) -> None:
    """Keep the existing configuration receipt, with the extended schema's known fields."""
    import json
    entry = settings.record(cfg, layers)
    entry["unknown_sections"] = sorted(set(cfg) - set(RuntimeConfig.model_fields))
    with (Path(batch_dir) / "config_runs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# Experiment layers may narrow these execution limits, with a recorded reason.
CONCURRENCY_PATHS = (("backend_limits",), ("eval", "model_caps"),
                     ("eval", "elastic_max"), ("eval", "model_workers"),
                     ("paid", "backend_request_limits"), ("paid", "global_request_limit"),
                     ("paid", "judge_reserved_lanes"))


def check_concurrency_overlay(base: dict, overlay: dict) -> None:
    """Refuse unreasoned concurrency copies and any experiment-side increase."""
    base = validate(base)
    backend_limits = {**DEFAULT_BACKEND_LIMITS, **base["backend_limits"]}
    for path in CONCURRENCY_PATHS:
        parent = overlay if len(path) == 1 else overlay.get(path[0], {})
        if not isinstance(parent, dict):
            raise ConfigError(f"{path[0]} must be a mapping; change HAENV_DEPLOY_CONFIG")
        if path[-1] not in parent:
            continue
        reason = overlay.get("concurrency_override_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"Experiment overlay cannot set {'.'.join(path)}; change "
                              "HAENV_DEPLOY_CONFIG, or only lower it with an explicit "
                              "concurrency_override_reason")
        old = base if len(path) == 1 else base[path[0]]
        old = old[path[-1]]
        value = parent[path[-1]]
        values = value.items() if isinstance(old, dict) and isinstance(value, dict) else [(None, value)]
        for name, new in values:
            bound = old.get(name) if isinstance(old, dict) else old
            if bound is None:
                paid, ev = base["paid"], base["eval"]
                if path == ("paid", "backend_request_limits"):
                    raise ConfigError("Experiment overlay may only lower existing backend request pools; "
                                      "adding a pool changes global sharing. Change HAENV_DEPLOY_CONFIG")
                elif path == ("backend_limits",):
                    bound = backend_limits.get(name, DEFAULT_BACKEND_LIMIT)
                else:
                    specs = base.get("models") or {}
                    spec = specs.get(name, {}) if isinstance(specs, dict) else {}
                    backend = spec.get("backend", "_offline")
                    lanes = paid["backend_request_limits"].get(backend, paid["global_request_limit"])
                    bound = min(backend_limits.get(backend, DEFAULT_BACKEND_LIMIT),
                                max(1, lanes - paid["judge_reserved_lanes"]),
                                ev["model_caps"].get(name, lanes),
                                ev["model_workers"].get(name, ev["elastic_max"]
                                    if ev["pooling"] == "elastic" else ev["workers"]))
            # Increasing reserved judge lanes lowers solver capacity; decreasing raises it.
            valid = (type(new) is int and new >= 0 and new >= bound
                     if path == ("paid", "judge_reserved_lanes") else
                     type(new) is int and 1 <= new <= bound)
            if not valid:
                raise ConfigError(f"Experiment overlay may only lower {'.'.join(path)}"
                                  f"{'.' + str(name) if name else ''} concurrency (bound {bound}); "
                                  "change HAENV_DEPLOY_CONFIG")
