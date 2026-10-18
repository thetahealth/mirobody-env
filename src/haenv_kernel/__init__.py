"""The L0 kernel: generation, isolation and judging primitives.

Forked into this repository on 2026-09-23 (see `FORKED_FROM.md`); it was a bare
directory of top-level modules until this package was introduced, and was put on
`sys.path` by `haenv.__init__._mount_kernel`. It is now an ordinary package and is
imported as one: `from haenv_kernel import build_instance`.

Changing a module here reprices a frozen segment -- `haenv_kernel/verifier.py`,
`gatekeeper.py`, `schema.py`, `runner.py` and `solver.py` are in the JUDGING segment;
`build.py`, `latent.py`, `synth.py`, `noise.py` and `joint_scenarios.py` are in
GENERATION. See `tools/release/make_freeze.py` and `FORKED_FROM.md`.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

from haenv_kernel.build import FORBIDDEN_TOKENS
from haenv_kernel.build import build_instance
from haenv_kernel.build import leakage_probe
from haenv_kernel.latent import DISEASE_SIGNAL_DOMAIN
from haenv_kernel.latent import DRUG_PKPD
from haenv_kernel.latent import KNOWN_DEVICES
from haenv_kernel.latent import KNOWN_MISSINGNESS
from haenv_kernel.latent import LatentPremise
from haenv_kernel.latent import drug_pkpd
from haenv_kernel.latent import make_premise
from haenv_kernel.latent import validate_premise
from haenv_kernel.noise import NOISE_CLASSES
from haenv_kernel.noise import POINT_ARTIFACTS
from haenv_kernel.noise import PRIMARY
from haenv_kernel.noise import device_switch
from haenv_kernel.noise import inject
from haenv_kernel.noise import context_confound
from haenv_kernel.noise import trap_evidence_days
from haenv_kernel.noise import transient_spike
from haenv_kernel.noise import unit_error
from haenv_kernel.schema import GradeReport
from haenv_kernel.schema import RawCase
from haenv_kernel.schema import SolverOutput
from haenv_kernel.schema import SolverPayload
from haenv_kernel.schema import VerifierPayload
from haenv_kernel.solver import BaselineSolver
from haenv_kernel.solver import LLMSolver
from haenv_kernel.solver import Solver
from haenv_kernel.solver import UnsafeSolver
from haenv_kernel.synth import AUX_SIGNALS
from haenv_kernel.synth import TemplateGenerator
from haenv_kernel.synth import premise_conflicts
from haenv_kernel.synth import register_aux_signals
from haenv_kernel.synth import synthesize
from haenv_kernel.verifier import grade

__all__ = [
    "AUX_SIGNALS",
    "BaselineSolver",
    "DISEASE_SIGNAL_DOMAIN",
    "DRUG_PKPD",
    "FORBIDDEN_TOKENS",
    "GradeReport",
    "KNOWN_DEVICES",
    "KNOWN_MISSINGNESS",
    "LLMSolver",
    "LatentPremise",
    "NOISE_CLASSES",
    "POINT_ARTIFACTS",
    "PRIMARY",
    "RawCase",
    "Solver",
    "SolverOutput",
    "SolverPayload",
    "TemplateGenerator",
    "UnsafeSolver",
    "VerifierPayload",
    "build_instance",
    "context_confound",
    "device_switch",
    "drug_pkpd",
    "grade",
    "inject",
    "leakage_probe",
    "make_premise",
    "premise_conflicts",
    "register_aux_signals",
    "synthesize",
    "trap_evidence_days",
    "transient_spike",
    "unit_error",
    "validate_premise",
]
