"""Opt-in aggregate EMR refits, independent of course coupling.

SYNTHETIC world generation only; evaluation, not medical advice. Saved cases
already contain the resulting observations; queries do not read these values.
"""
from __future__ import annotations

import functools
import math
import os
import re

from haenv.resources.regpath import load_registry, registry_cached


@functools.lru_cache(maxsize=16)
def _variant(config_key: tuple[str, ...]) -> str:
    from haenv.app.config import load_cfg
    value = (load_cfg().get("world") or {}).get("emr_params", "v14")
    if value not in ("v14", "calibration15"):
        raise ValueError(f"unknown world.emr_params: {value!r}")
    return value


def variant() -> str:
    """The selected generation parameter set (the shipped default is v14)."""
    from haenv.app.config import CONFIG_OVERLAY_ENV, DEPLOY_CONFIG_ENV
    from haenv.app.settings import SET_ENV
    return _variant(tuple(os.environ.get(k, "") for k in
                          (SET_ENV, CONFIG_OVERLAY_ENV, DEPLOY_CONFIG_ENV)))


def validate(doc: dict) -> dict:
    """Refuse unsourced, suppressed, duplicate or nonfinite aggregate cells."""
    seen = set()
    for section in ("parameters", "statistics", "check"):
        for row in doc.get(section, []):
            path = row.get("path")
            if not isinstance(path, str) or not path or path in seen:
                raise ValueError(f"missing or duplicate parameter path: {path!r}")
            seen.add(path)
            n = row.get("n_patients")
            value = row.get("value")
            if isinstance(n, bool) or not isinstance(n, int) or n < 20:
                raise ValueError(f"{path}: fewer than 20 contributing patients")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: a finite numeric value is required")
            if (row.get("split") != "calibration world14-split-v1"
                    or row.get("script") != "tools/calibration/emr_refit_15.py"
                    or row.get("script_commit") != "521faf846"
                    or not row.get("source_id")
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("input_sha256", "")))):
                raise ValueError(f"{path}: missing verified aggregate provenance")
            if row.get("status", "reproduced") != "reproduced" or "suppressed" in row:
                raise ValueError(f"{path}: producer not reproduced or cell suppressed")
    if not doc.get("parameters"):
        raise ValueError("no refit parameters registered")
    return doc


@registry_cached("emr_refit_15.yaml")
def _parameters() -> dict:
    doc = validate(load_registry("emr_refit_15.yaml"))
    return {row["path"]: row for row in doc["parameters"]}


def provenance(path: str) -> dict:
    """The aggregate source receipt, returned as a private copy."""
    import copy
    return copy.deepcopy(_parameters()[path])


def emr(path: str, default):
    """Return the existing constant under v14; calibration paths fail closed.

    Checks and aggregate statistics without equivalent generation parameters
    deliberately have no read site and cannot be requested as overrides.
    """
    if variant() == "v14":
        return default
    return _parameters()[path]["value"]
