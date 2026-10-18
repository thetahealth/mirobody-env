"""Lab world: follow-up analytes, per-point observation records, pre-analytic and method
perturbations, and visible-cause truth effects (GLP-1 dose line and weight on fasting glucose,
ACEI on creatinine).

Shared by the question packs that read a lab value against its truth: pack 4 (follow-up
interpretation) uses all of it; pack 3 (chronic medication adjustment) is meant to reuse the
drug-effect terms (`truth`) and the observation records (`observe`). Nothing here runs unless a
job declares a plugin that calls it, so existing jobs build byte-identically.

Tables: `registry/labworld.yaml` (provisional, design doc K5). Segment:
`tools/release/make_freeze.py:P4_WORLD`.

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
    from haenv.resources.yamlcache import load_yaml
    return load_yaml(path) or {}


def tables() -> dict:
    """`registry/labworld.yaml` (read through the registry resolution point)."""
    from haenv.resources.regpath import registry_path
    p = registry_path(TABLE)
    return _tables_cached(str(p), p.stat().st_mtime_ns)


def tables_sha16() -> str:
    """Semantic fingerprint of the table (comments do not move it)."""
    from haenv.provenance.anchor import semantic_bytes
    from haenv.resources.regpath import registry_path
    return hashlib.sha256(semantic_bytes(registry_path(TABLE))).hexdigest()[:16]


def analyte(name: str) -> dict:
    a = (tables().get("analytes") or {}).get(name)
    if not isinstance(a, dict):
        raise KeyError(f"labworld: unknown analyte {name!r}")
    return a


def cv_total(name: str, method: str | None = None) -> float:
    """Per-draw total CV. Glucose reads the world's own value (`physio_streams.yaml`), so the
    follow-up draw and every other glucose draw share one noise model. An analyte with `cva`
    (analytical CV by method) has sqrt(cvi^2 + cva[method]^2); `method` None is its default method."""
    a = analyte(name)
    ref = a.get("cv_total_from")
    if ref:
        from haenv.resources.regpath import registry_path
        from haenv.resources.yamlcache import load_yaml
        fname, path = ref.split("#")
        node = load_yaml(registry_path(fname))
        for k in path.split("."):
            node = node[k]
        return float(node)
    cva = a.get("cva")
    if cva:
        return math.sqrt(float(a["cvi"]) ** 2 + float(cva[method or a["cva_default"]]) ** 2)
    return float(a["cv_total"])


def rcv95(cv: float, cv1: float | None = None) -> float:
    """Two-sided 95% reference change value on the log scale: 1.96 x sqrt(cv0^2 + cv1^2), the two
    draws' total CVs (equal when `cv1` is omitted: 1.96 x sqrt(2) x CV)."""
    return 1.96 * math.sqrt(float(cv) ** 2 + float(cv if cv1 is None else cv1) ** 2)


def u01(*parts) -> float:
    """Deterministic uniform in [0, 1) from the path (`haenv.rng.unit`)."""
    from haenv.world.generation.rng import unit
    return unit("labworld", *parts)


def z(*parts, clip: float = 3.0) -> float:
    """Deterministic standard normal from the path, clipped to +-clip."""
    u = min(max(u01(*parts), 1e-12), 1 - 1e-12)
    return max(-clip, min(clip, statistics.NormalDist().inv_cdf(u)))
