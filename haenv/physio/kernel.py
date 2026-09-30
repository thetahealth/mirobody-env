"""kernel.py -- the event impulse-response kernel: logistic onset, plateau, exponential
decay. With `d = t - t_start`:

    d <= 0                       g = 0
    0 < d <= duration            g = sigma(k (d - tau_rise/2))
    0 < d - duration <= tau_fade g = g(t_end) * exp(-alpha (d - duration))
    d - duration > tau_fade      g = 0

with `k = 6 / tau_rise` and `alpha = 3 / tau_fade` (sigma reaches ~0.95 at +-3 and
exp(-3) ~= 0.05, so onset and decay complete within their timescales). Decay starts from
the actual value at `t_end`, so short events stay continuous. Each event contributes at
most `beta` (signed, in the indicator's unit); `superpose.py` stacks and saturates.

Parameters come only from `registry/physio_kernels.yaml` (per-entry `source:`); missing or
invalid values raise `KernelParamError`, never default.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from ..yamlcache import load_yaml as _cached_yaml

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# See the module docstring for these two shape constants.
_RISE_SHAPE = 6.0
_FADE_SHAPE = 3.0

# Clamp exp arguments to avoid underflow warnings.
_EXP_FLOOR = -60.0


class KernelParamError(ValueError):
    """A kernel parameter is missing or invalid (there are no defaults)."""


@dataclass(frozen=True)
class KernelParams:
    """One event's influence parameters on one indicator.

    `beta` is signed, in the same unit as the indicator's registered `unit` --
    no conversion is done here.
    """

    beta: float          # signed magnitude, in the indicator's native unit
    tau_rise: float      # onset time in days, > 0
    tau_fade: float      # decay time in days, > 0
    duration: float      # event duration in days, >= 0
    source: str          # provenance string, must not be empty
    review: str          # review status, required: pending / done. Propagates all the way to the report.
    #: Generator event topic that triggers this kernel (`events.event_topics()`), or
    #: `NO_GENERATOR` to declare the generator cannot produce it.
    generator_source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta):
            raise KernelParamError(f"beta must be finite: {self.beta!r}")
        for name in ("tau_rise", "tau_fade"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise KernelParamError(f"{name} must be finite and > 0: {v!r}")
        if not (math.isfinite(self.duration) and self.duration >= 0):
            raise KernelParamError(f"duration must be finite and >= 0: {self.duration!r}")
        if not str(self.source).strip():
            raise KernelParamError("source must not be empty: every kernel parameter needs a citation")
        if str(self.review).strip() not in ("pending", "done"):
            raise KernelParamError(
                f"review must be 'pending' or 'done', got {self.review!r}"
            )

    @property
    def is_pending(self) -> bool:
        return str(self.review).strip() == "pending"

    @property
    def is_emitted(self) -> bool:
        """Does the generator actually trigger it? `False` = declared but never
        triggered, and must be countable."""
        return str(self.generator_source).strip() != NO_GENERATOR


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(max(_EXP_FLOOR, -x)))
    e = math.exp(max(_EXP_FLOOR, x))
    return e / (1.0 + e)


def kernel_value(p: KernelParams, days_since_start: float) -> float:
    """Kernel value in [0, 1] at `days_since_start` (see the module docstring)."""
    d = float(days_since_start)
    if not math.isfinite(d):
        raise KernelParamError(f"days_since_start must be finite: {days_since_start!r}")
    if d <= 0:
        return 0.0

    k = _RISE_SHAPE / p.tau_rise
    if d <= p.duration:
        return _sigmoid(k * (d - p.tau_rise / 2.0))

    at_end = _sigmoid(k * (p.duration - p.tau_rise / 2.0))
    since_end = d - p.duration
    if since_end > p.tau_fade:
        return 0.0
    alpha = _FADE_SHAPE / p.tau_fade
    return at_end * math.exp(max(_EXP_FLOOR, -alpha * since_end))


def contribution(p: KernelParams, days_since_start: float) -> float:
    """Signed contribution = `beta * kernel_value`. Superposition is
    `superpose.saturate`'s job."""
    return p.beta * kernel_value(p, days_since_start)


# -- Registry ────────────────────────────────────────────────────────────────

_REQUIRED = ("beta", "tau_rise", "tau_fade", "duration", "source", "review",
             "generator_source")

#: Explicit "no generator" value for `generator_source` (distinct from a blank).
NO_GENERATOR = "none"


def _generator_vocabulary() -> frozenset[str]:
    """Event topics the generator can inject; raises if unavailable or empty."""
    try:
        from ..events import event_topics
    except Exception as e:                                    # noqa: BLE001
        raise KernelParamError(
            f"cannot load the generator event vocabulary ({type(e).__name__}: {e}); "
            f"`generator_source` cannot be validated") from e
    v = event_topics()
    if not v:
        raise KernelParamError("generator event vocabulary is empty; refusing to load the registry")
    return v


def _parse_entry(event: str, indicator: str, raw: Any) -> KernelParams:
    if not isinstance(raw, Mapping):
        raise KernelParamError(f"{event}/{indicator}: entry must be a mapping, got {type(raw).__name__}")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise KernelParamError(f"{event}/{indicator}: missing fields {missing}; no defaults are provided")
    gs = str(raw["generator_source"]).strip()
    if gs != NO_GENERATOR and gs not in _generator_vocabulary():
        raise KernelParamError(
            f"{event}/{indicator}: `generator_source: {gs!r}` is not in the generator event vocabulary.\n"
            f"Allowed: a member of `events.event_topics()`, or {NO_GENERATOR!r} "
            f"(the generator cannot produce it yet; say in `source` why the entry is kept).")
    return KernelParams(
        beta=float(raw["beta"]),
        tau_rise=float(raw["tau_rise"]),
        tau_fade=float(raw["tau_fade"]),
        duration=float(raw["duration"]),
        source=str(raw["source"]),
        review=str(raw["review"]),
        generator_source=gs,
    )


@dataclass(frozen=True)
class PhysioRegistry:
    """Kernel parameters and saturation caps, loaded together and cross-validated: every
    kernel indicator has a cap, and every cap is used."""

    kernels: dict[tuple[str, str], KernelParams]
    saturation: dict[str, float]

    def m_for(self, indicator: str) -> float:
        """The saturation cap for this indicator. Missing is impossible (checked
        at load time); if it is truly missing, that is an internal bug."""
        if indicator not in self.saturation:
            raise KernelParamError(f"{indicator}: no saturation cap (load-time validation should have caught this)")
        return self.saturation[indicator]

    def pending(self) -> list[tuple[str, str]]:
        """All entries with `review: pending`. For reports -- pending status
        must propagate all the way out."""
        return sorted(k for k, p in self.kernels.items() if p.is_pending)

    def unemitted(self) -> list[tuple[str, str]]:
        """Entries the generator cannot produce (`generator_source: none`), for reports."""
        return sorted(k for k, p in self.kernels.items() if not p.is_emitted)

    def emitted_topics(self) -> frozenset[str]:
        """The set of topics this table actually attaches to generator events."""
        return frozenset(p.generator_source for p in self.kernels.values() if p.is_emitted)


def load_physio_registry(path: str | Path) -> PhysioRegistry:
    """Load `registry/physio_kernels.yaml` (`kernels` + `saturation`) and cross-validate;
    any bad entry raises."""
    import yaml  # deferred import: this module's pure-computation part should not depend on yaml

    p = Path(path)
    if not p.is_file():
        raise KernelParamError(f"kernel registry not found: {p}")
    from ..regpath import overlaid_yaml as _overlaid_yaml
    doc = _overlaid_yaml(p)          # the in-repo table carries the world-plugin overlay
    if not isinstance(doc, Mapping):
        raise KernelParamError(f"{p}: top level must be a mapping")
    for seg in ("kernels", "saturation"):
        if seg not in doc:
            raise KernelParamError(f"{p}: missing `{seg}` section; no defaults are provided")

    kernels: dict[tuple[str, str], KernelParams] = {}
    for event, per_ind in (doc["kernels"] or {}).items():
        if not isinstance(per_ind, Mapping):
            raise KernelParamError(f"{event}: value must be a mapping of indicator -> parameters")
        for indicator, raw in per_ind.items():
            key = (str(event), str(indicator))
            if key in kernels:
                raise KernelParamError(f"{key}: duplicate entry")
            kernels[key] = _parse_entry(str(event), str(indicator), raw)
    if not kernels:
        raise KernelParamError(f"{p}: `kernels` is empty")

    saturation: dict[str, float] = {}
    for indicator, raw in (doc["saturation"] or {}).items():
        if not isinstance(raw, Mapping) or "m" not in raw:
            raise KernelParamError(f"saturation/{indicator}: must be a mapping with `m`")
        m = float(raw["m"])
        if not (math.isfinite(m) and m > 0):
            raise KernelParamError(f"saturation/{indicator}: m must be > 0, got {m}")
        if not str(raw.get("source", "")).strip():
            raise KernelParamError(f"saturation/{indicator}: source must not be empty")
        saturation[str(indicator)] = m

    used = {ind for _, ind in kernels}
    missing = sorted(used - set(saturation))
    if missing:
        raise KernelParamError(
            f"{p}: indicators with kernel parameters but no saturation cap: {missing}. "
            "Without a cap the indicator is unbounded and will hit the hard clamp"
        )
    unused = sorted(set(saturation) - used)
    if unused:
        raise KernelParamError(
            f"{p}: saturation caps not used by any kernel parameter: {unused}. "
            "Remove them or add the kernel entries"
        )
    return PhysioRegistry(kernels=kernels, saturation=saturation)


def load_kernel_table(path: str | Path) -> dict[tuple[str, str], KernelParams]:
    """Just the kernel-parameter half. Still routes through
    `load_physio_registry` internally, so cross-validation still applies."""
    return load_physio_registry(path).kernels
