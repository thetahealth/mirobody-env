"""Switches of the generated world that a job turns on (`job.findings`, `job.physio`).

One-element lists, so every module that imported them sees the value a job set.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=16)
def _background_enabled(key: str) -> bool:
    from haenv.app.config import load_cfg
    cfg = load_cfg()
    explicit = cfg.get("background")
    return bool(cfg["synth"]["background"] if explicit is None else explicit)


def background_enabled() -> bool:
    """The shared configuration switch; each pack implements its own background mechanism."""
    from haenv.app.settings import SET_ENV
    from haenv.app.config import CONFIG_OVERLAY_ENV
    return _background_enabled(os.environ.get(SET_ENV, "") + "|" + os.environ.get(CONFIG_OVERLAY_ENV, ""))


# ---------------------------------------------------------------- Findings-layer hook
FINDINGS_ENABLED: list = [False]      # set by run/build from job.findings; off by default


#: Physiology-layer switch (`job.yaml: physio`), off by default. When on,
#: rendered streams go through `haenv.physio.apply_physio`.
PHYSIO_ENABLED: list = [False]


STEP, END = 7, 365
