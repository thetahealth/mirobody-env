"""Cache YAML parse results by (path, mtime, size), returning a fresh deep copy
every time.

Callers mutate what `load_*()` returns, so each call gets a private deep copy;
the result is semantically identical to re-parsing. Set `HAENV_NO_YAML_CACHE=1`
to disable the cache.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import os
import pathlib

#: (path, mtime_ns, size, wrap_text) -> parsed doc. Append-only, never evicted.
_CACHE: dict[tuple[str, int, int, bool], object] = {}

STATS = {"hit": 0, "miss": 0, "off": 0}


def load_yaml(path: str | pathlib.Path, *, wrap_text: bool = True):
    """Read and parse a YAML file, returning a private deep copy.

    A missing file raises `OSError`; it is never folded into `None`.
    """
    import yaml

    p = pathlib.Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size, wrap_text)
    disabled = os.environ.get("HAENV_NO_YAML_CACHE") == "1"
    if disabled:
        STATS["off"] += 1
    elif key in _CACHE:
        STATS["hit"] += 1
        return copy.deepcopy(_CACHE[key])
    else:
        STATS["miss"] += 1
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    # Catalogs need raw text: wrapping a lexicon while loading it recurses into itself.
    if wrap_text:
        from haenv.i18n.text import wrap
        doc = wrap(doc)
    if disabled:
        return doc
    _CACHE[key] = doc
    return copy.deepcopy(doc)


def clear() -> None:
    """Clear the cache (for tests). Counters are not cleared."""
    _CACHE.clear()
