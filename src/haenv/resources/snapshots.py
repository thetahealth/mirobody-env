"""Immutable YAML snapshots for generation-side readers.

Mutable consumers keep using yamlcache.load_yaml and receive independent copies.
SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import os
import pathlib
from types import MappingProxyType
from haenv.resources.yamlcache import load_yaml

_READONLY: dict[tuple[str, int, int], object] = {}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def load_yaml_readonly(path: str | pathlib.Path):
    """Share an immutable snapshot; edits invalidate it just like `load_yaml`.

    Readers opt in. Callers that mutate data continue using the private-copy API.
    """
    p = pathlib.Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if os.environ.get("HAENV_NO_YAML_CACHE") == "1":
        return _freeze(load_yaml(p))
    if key not in _READONLY:
        _READONLY[key] = _freeze(load_yaml(p))
    return _READONLY[key]
