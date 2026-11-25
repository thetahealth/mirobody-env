"""Opt-in, case-owned observation truth. SYNTHETIC, evaluation only."""
from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=8)
def _enabled(key: str) -> bool:
    from haenv.app.config import load_cfg
    cfg = load_cfg()
    explicit = (cfg.get("world") or {}).get("course_coupling")
    return bool((cfg.get("synth") or {}).get("course_coupling", False) if explicit is None else explicit)


def coupling_enabled() -> bool:
    """Read synth.course_coupling, with the canonical world override."""
    from haenv.app.settings import SET_ENV
    from haenv.app.config import CONFIG_OVERLAY_ENV
    return _enabled(os.environ.get(SET_ENV, "") + "|" + os.environ.get(CONFIG_OVERLAY_ENV, ""))


def plan_of(raw) -> dict | None:
    """Queries use the saved plan, independent of the current build configuration."""
    return (getattr(raw, "latent_premise", None) or {}).get("truth")
