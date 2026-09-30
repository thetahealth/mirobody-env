"""Single resolution point for registry tables, and the hook for world-side plugin overlays.

World-layer call sites read `registry/` through this module, so an out-of-repo plugin
overlay attaches in exactly one place per table. The module belongs only to the
`generation` segment: judging-side readers (`gated.py`, `gates.py`, `llm_rubric.py`)
build their own paths and never see an overlay, so the judging fingerprint does not
depend on which plugins are installed. A plugin stream with no registry entry visible
to the judging side gets the strictest default (fail-closed).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import contextlib
import pathlib

from .yamlcache import clear as _cache_clear
from .yamlcache import load_yaml as _load_yaml

#: Temporary redirect for the registry directory, for negative controls only. It is read in
#: `registry_dir()`, the same place resolution happens, so an override always takes effect.
_OVERRIDE: pathlib.Path | None = None


def registry_dir() -> pathlib.Path:
    """Location of `registry/`, the single resolution point across the repo."""
    if _OVERRIDE is not None:
        return pathlib.Path(_OVERRIDE)
    from haenv import data_root
    return data_root() / "registry"


def registry_path(name: str) -> pathlib.Path:
    """Path to a single registry table, for callers that need a path
    (e.g. `load_coupling_rules(path)`)."""
    return registry_dir() / name


def load_registry(name: str):
    """Load a registry table as a private deep copy, merged with any world-side plugin overlay.

    A missing file raises `OSError`. With no plugin installed the result is unchanged.
    """
    from . import world_plugins
    return world_plugins.apply_overlay(name, _load_yaml(registry_path(name)))


def overlaid_yaml(path):
    """Parse a registry yaml by path and merge the plugin overlay when `path` is the in-repo table.

    For readers that take a path (stream and kernel loaders, event pools): a path that is not the
    repository table of that name (a negative-control copy) is returned without the overlay.
    """
    from . import world_plugins
    p = pathlib.Path(path)
    doc = _load_yaml(p)
    if p.resolve() == registry_path(p.name).resolve():
        return world_plugins.apply_overlay(p.name, doc)
    return doc


def registry_cached(*tables: str):
    """`lru_cache(maxsize=1)` for a zero-argument reader of registry tables, keyed on the
    registry directory and on the overlay of each of `tables`, and emptied by
    `registry_dir_override`."""
    from . import world_plugins
    return world_plugins.overlay_cached(*tables, key=lambda: str(registry_dir()))


@contextlib.contextmanager
def registry_dir_override(path):
    """Temporarily point the registry directory at `path` (negative controls only).

    Clears the yaml cache and every `registry_cached` layer (`build._clinical_cv` and the like)
    on entry and exit, so readings taken during the override do not leak.
    """
    global _OVERRIDE
    old = _OVERRIDE
    from . import world_plugins
    _OVERRIDE = pathlib.Path(path)
    _cache_clear()
    world_plugins.clear_overlay_caches()
    try:
        yield
    finally:
        _OVERRIDE = old
        _cache_clear()
        world_plugins.clear_overlay_caches()
