"""Lab world: follow-up analytes, per-point observation records, pre-analytic and method
perturbations, and visible-cause truth effects (GLP-1 dose line and weight on fasting glucose,
ACEI on creatinine).

Shared by the question packs that read a lab value against its truth: pack 4 (follow-up
interpretation) uses all of it; pack 3 (chronic medication adjustment) is meant to reuse the
drug-effect terms (`truth`) and the observation records (`observe`). Nothing here runs unless a
job declares a plugin that calls it, so existing jobs build byte-identically.

Tables: `registry/labworld.yaml` (provisional, design doc K5). Segment:
`tools/make_freeze.py:P4_WORLD`.

Modules:
  truth    -- noise-free set points and the visible-cause effect terms
  observe  -- measurement layer (EFLM total CV, clip +-3 sigma), perturbations, printed values,
              per-point records and their consistency check
  perturb  -- the factor catalogue (kind, analytes acted on, intrinsic sign, effect ranges)

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import functools
import hashlib
import math
import statistics

TABLE = "labworld.yaml"


@functools.lru_cache(maxsize=1)
def _tables_cached(path: str, mtime: int) -> dict:
    from ..yamlcache import load_yaml
    return load_yaml(path) or {}


def tables() -> dict:
    """`registry/labworld.yaml` (read through the registry resolution point)."""
    from ..regpath import registry_path
    p = registry_path(TABLE)
    return _tables_cached(str(p), p.stat().st_mtime_ns)


def tables_sha16() -> str:
    """Semantic fingerprint of the table (comments do not move it)."""
    from ..anchor import semantic_bytes
    from ..regpath import registry_path
    return hashlib.sha256(semantic_bytes(registry_path(TABLE))).hexdigest()[:16]


def analyte(name: str) -> dict:
    a = (tables().get("analytes") or {}).get(name)
    if not isinstance(a, dict):
        raise KeyError(f"labworld: unknown analyte {name!r}")
    return a


def cv_total(name: str) -> float:
    """Per-draw total CV. Glucose reads the world's own value (`physio_streams.yaml`), so the
    follow-up draw and every other glucose draw share one noise model."""
    a = analyte(name)
    ref = a.get("cv_total_from")
    if ref:
        from ..regpath import registry_path
        from ..yamlcache import load_yaml
        fname, path = ref.split("#")
        node = load_yaml(registry_path(fname))
        for k in path.split("."):
            node = node[k]
        return float(node)
    return float(a["cv_total"])


def rcv95(cv: float) -> float:
    """Two-sided 95% reference change value on the log scale: 1.96 x sqrt(2) x CV."""
    return 1.96 * math.sqrt(2.0) * float(cv)


def u01(*parts) -> float:
    """Deterministic uniform in [0, 1) from the path (`haenv.rng.unit`)."""
    from ..rng import unit
    return unit("labworld", *parts)


def z(*parts, clip: float = 3.0) -> float:
    """Deterministic standard normal from the path, clipped to +-clip."""
    u = min(max(u01(*parts), 1e-12), 1 - 1e-12)
    return max(-clip, min(clip, statistics.NormalDist().inv_cdf(u)))
