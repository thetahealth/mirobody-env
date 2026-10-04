"""Pack 2 shared helpers: registry-table loading (with a stat-keyed cache) and the case-keyed
hash draws every pack-2 module uses. PACK2_WORLD segment.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import hashlib
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]



def _registry_dir() -> pathlib.Path:
    try:
        from haenv.regpath import registry_dir
        return pathlib.Path(registry_dir())
    except Exception:                              # noqa: BLE001 -- import-light callers
        return ROOT / "registry"


@functools.lru_cache(maxsize=8)
def _load(name: str, _key: tuple) -> dict:
    return yaml.safe_load((_registry_dir() / name).read_text(encoding="utf-8"))


def registry_table(name: str) -> dict:
    """A pack-2 registry table, re-read when the file changes."""
    p = _registry_dir() / name
    st = p.stat()
    return _load(name, (str(p), st.st_mtime_ns, st.st_size))


def u01(*parts) -> float:
    s = hashlib.sha256(("pack2|" + "|".join(str(p) for p in parts)).encode("utf-8")).hexdigest()
    return int(s[:13], 16) / float(16 ** 13)


def pick(lo, hi, u: float, dp: int):
    """A value on the printed grid of [lo, hi] (inclusive), chosen by `u`."""
    step = 10 ** (-dp)
    n = int(round((hi - lo) / step))
    k = min(n, int(math.floor(u * (n + 1))))
    v = lo + k * step
    return int(round(v)) if dp == 0 else round(v, dp)

