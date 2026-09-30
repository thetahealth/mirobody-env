"""physio -- the world model's physiology layer: event kernel, superposition, correlated
noise, physiological bounds, comorbidity coupling and distribution audit.

Kernel parameters come only from `registry/physio_kernels.yaml`; missing or invalid
parameters raise. Randomness goes through the counter-based RNG in `haenv/rng.py`, so
values are pure functions of `(case_id, indicator, t)`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from .bounds import (
    BoundSpec,
    ProjectionReport,
    project,
    to_transform,
    from_transform,
)
from .coupling import (
    Blackboard,
    CouplingRule,
    CouplingError,
    apply_couplings,
    load_coupling_rules,
    pending_rules,
)
from .kernel import (
    KernelParams,
    KernelParamError,
    kernel_value,
    load_kernel_table,
    load_physio_registry,
    PhysioRegistry,
)
from .noise import (
    NoiseSpec,
    correlated_noise,
    normal_dev,
)
from .superpose import SaturationError, saturate

__all__ = [
    "BoundSpec", "ProjectionReport", "project", "to_transform", "from_transform",
    "Blackboard", "CouplingRule", "CouplingError", "apply_couplings", "load_coupling_rules", "pending_rules",
    "KernelParams", "KernelParamError", "kernel_value", "load_kernel_table", "load_physio_registry", "PhysioRegistry",
    "NoiseSpec", "correlated_noise", "normal_dev",
    "SaturationError", "saturate",
]
