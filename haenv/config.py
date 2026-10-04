"""Configuration loading: `config.yaml`, the untracked `config.local.yaml`, the overlay named by
`HAENV_CONFIG_OVERLAY`, then `--set` and validation (`settings`).

Below every caller: the solver layer, the schedulers and the judges read configuration here
without importing the command line.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

from haenv import data_root as _data_root
import os
from pathlib import Path
from .yamlcache import load_yaml as _cached

ROOT = _data_root()


def _deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` into `base` in place: mappings recurse, everything else replaces."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg() -> dict:
    """`config.yaml`, with the untracked `config.local.yaml` (deployment settings) merged over it."""
    # Parsed once per process and returned as deep copies (callers mutate them).
    cfg = _cached(ROOT / "config.yaml")
    local = ROOT / "config.local.yaml"
    if local.is_file():
        _deep_merge(cfg, _cached(local) or {})
    from . import settings                    # pydantic; imported on first load only
    return settings.validate(settings.apply_set(apply_config_overlay(cfg)))


def config_layers() -> list[dict]:
    """Every file merged into `load_cfg`, in order, with its SHA-256 (`--set` is recorded
    separately, see `settings.record`)."""
    import hashlib
    paths = [ROOT / "config.yaml", ROOT / "config.local.yaml"]
    named = os.environ.get(CONFIG_OVERLAY_ENV, "").strip()
    if named:
        paths.append(Path(named))
    return [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths if p.is_file()]


CONFIG_OVERLAY_ENV = "HAENV_CONFIG_OVERLAY"


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
    return _deep_merge(cfg, _cached(path) or {})
