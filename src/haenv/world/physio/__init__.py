"""physio -- the world model's physiology layer: event kernel, superposition, correlated
noise, physiological bounds, comorbidity coupling and distribution audit.

Kernel parameters come only from `registry/physio_kernels.yaml`; missing or invalid
parameters raise. Randomness goes through the counter-based RNG in `haenv/world/generation/rng.py`, so
values are pure functions of `(case_id, indicator, t)`.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.world.physio.bounds import BoundSpec
from haenv.world.physio.bounds import ProjectionReport
from haenv.world.physio.bounds import project
from haenv.world.physio.bounds import to_transform
from haenv.world.physio.bounds import from_transform
from haenv.world.physio.coupling import Blackboard
from haenv.world.physio.coupling import CouplingRule
from haenv.world.physio.coupling import CouplingError
from haenv.world.physio.coupling import apply_couplings
from haenv.world.physio.coupling import load_coupling_rules
from haenv.world.physio.coupling import pending_rules
from haenv.world.physio.kernel import KernelParams
from haenv.world.physio.kernel import KernelParamError
from haenv.world.physio.kernel import kernel_value
from haenv.world.physio.kernel import load_kernel_table
from haenv.world.physio.kernel import load_physio_registry
from haenv.world.physio.kernel import PhysioRegistry
from haenv.world.physio.noise import NoiseSpec
from haenv.world.physio.noise import correlated_noise
from haenv.world.physio.noise import normal_dev
from haenv.world.physio.superpose import SaturationError
from haenv.world.physio.superpose import saturate

__all__ = [
    "BoundSpec", "ProjectionReport", "project", "to_transform", "from_transform",
    "Blackboard", "CouplingRule", "CouplingError", "apply_couplings", "load_coupling_rules", "pending_rules",
    "KernelParams", "KernelParamError", "kernel_value", "load_kernel_table", "load_physio_registry", "PhysioRegistry",
    "NoiseSpec", "correlated_noise", "normal_dev",
    "SaturationError", "saturate",
]
