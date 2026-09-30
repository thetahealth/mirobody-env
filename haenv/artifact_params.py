"""artifact_params.py — the single read point for `registry/artifact_rates.yaml`,
the artifact-injection magnitudes consumed by `haenv/build.py`.

Each entry records its value, the kernel's signature default (`kernel_default`)
and its evidence. `assert_matches_kernel` raises when a default in `core/noise.py`
and the registered `kernel_default` disagree, since which value applies would
otherwise depend on whether `**kw` is passed.
"""
from __future__ import annotations

from .yamlcache import load_yaml as _cached_yaml

import inspect
from pathlib import Path
from typing import Any

import yaml

# resource root: repo root in a checkout, `haenv/_data` in a wheel
from haenv import data_root as _data_root
from haenv.regpath import registry_path as _rp
ROOT = _data_root()
REGISTRY = _rp("artifact_rates.yaml")


class ArtifactParamsUnregistered(KeyError):
    """This noise class has no entry in the registry (no fallback to the kernel
    default)."""


class ArtifactParamsError(ValueError):
    """The registry is malformed or disagrees with the kernel signature."""


_CACHE: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    if _CACHE:
        return _CACHE
    if not REGISTRY.is_file():
        raise ArtifactParamsError(
            f"{REGISTRY} is missing: artifact magnitudes have no source; there is no fallback to kernel defaults.")
    doc = _cached_yaml(REGISTRY) or {}
    inj = doc.get("injectors")
    if not isinstance(inj, dict) or not inj:
        raise ArtifactParamsError(f"{REGISTRY}: `injectors:` is missing or empty")
    for name, spec in inj.items():
        if not isinstance(spec, dict):
            raise ArtifactParamsError(f"{REGISTRY}: {name} is not a mapping")
        for field in ("param", "value", "kernel_default", "calibration",
                      "evidence", "review"):
            if field not in spec:
                raise ArtifactParamsError(
                    f"{REGISTRY}: {name} is missing `{field}:`; "
                    "all six fields are required, and `calibration` is one of measured / no_corpus / "
                    "corpus_too_thin / by_design")
        if spec["calibration"] not in ("measured", "corpus_too_thin",
                                       "no_corpus", "by_design"):
            raise ArtifactParamsError(
                f"{REGISTRY}: {name}.calibration={spec['calibration']!r} is not an allowed value")
    _CACHE.update(doc)
    return _CACHE


def injectors() -> dict[str, dict[str, Any]]:
    """All registered entries."""
    return dict(_load()["injectors"])


def spec_of(noise_class: str) -> dict[str, Any]:
    """Full declaration for one noise class; raises `ArtifactParamsUnregistered`."""
    inj = injectors()
    if noise_class not in inj:
        raise ArtifactParamsUnregistered(
            f"{noise_class!r} has no magnitude declaration in registry/artifact_rates.yaml. "
            "Register a new noise class first: param, value, kernel_default, calibration, evidence, review")
    return inj[noise_class]


def kwargs_for(noise_class: str) -> dict[str, float]:
    """Keyword arguments for the kernel's `noise.inject(..., **kw)`; `{}` for a
    class registered with `param: null` (e.g. `mnar_missing`)."""
    spec = spec_of(noise_class)
    if spec.get("param") is None:
        return {}
    return {str(spec["param"]): float(spec["value"])}


def assert_matches_kernel(noise_classes: dict[str, Any]) -> None:
    """Raise unless each registered `kernel_default:` equals the kernel signature's
    default. `noise_classes` is the kernel's `NOISE_CLASSES`, passed by the caller.
    """
    bad = []
    for name, spec in injectors().items():
        fn = noise_classes.get(name)
        if fn is None:
            bad.append(f"{name}: registered but absent from the kernel's NOISE_CLASSES")
            continue
        param = spec.get("param")
        if param is None:
            sig = inspect.signature(fn)
            extra = [p for p in list(sig.parameters)[3:]
                     if sig.parameters[p].default is not inspect.Parameter.empty]
            if extra:
                bad.append(f"{name}: registered with param=null, but the kernel signature has defaulted parameters {extra}")
            continue
        sig = inspect.signature(fn)
        if param not in sig.parameters:
            bad.append(f"{name}: registered param={param!r} is not a parameter of the kernel signature")
            continue
        kd = sig.parameters[param].default
        if kd is inspect.Parameter.empty:
            bad.append(f"{name}.{param}: the kernel signature has no default, but the registry declares kernel_default")
        elif float(kd) != float(spec["kernel_default"]):
            bad.append(f"{name}.{param}: kernel default {kd} != registered kernel_default "
                       f"{spec['kernel_default']}; one side changed without the other")
    if bad:
        raise ArtifactParamsError(
            "artifact magnitude registry does not match the core/ kernel defaults:\n  " + "\n  ".join(bad)
            + "\nUpdate `kernel_default:`; decide separately whether `value:` follows (changing it changes the cases).")


def missing_calibration() -> dict[str, str]:
    """Magnitudes without measured calibration, with the reason."""
    return {n: s["calibration"] for n, s in injectors().items()
            if s["calibration"] != "measured"}
