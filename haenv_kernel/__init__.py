"""The L0 kernel: generation, isolation and judging primitives.

Forked into this repository on 2026-09-23 (see `FORKED_FROM.md`); it was a bare
directory of top-level modules until this package was introduced, and was put on
`sys.path` by `haenv.__init__._mount_kernel`. It is now an ordinary package and is
imported as one: `from haenv_kernel import build_instance`.

Changing a module here reprices a frozen segment -- `haenv_kernel/verifier.py`,
`gatekeeper.py`, `schema.py`, `runner.py` and `solver.py` are in the JUDGING segment;
`build.py`, `latent.py`, `synth.py`, `noise.py` and `joint_scenarios.py` are in
GENERATION. See `tools/make_freeze.py` and `FORKED_FROM.md`.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

from .build import FORBIDDEN_TOKENS, build_instance, leakage_probe
from .latent import (
    DISEASE_SIGNAL_DOMAIN,
    DRUG_PKPD,
    KNOWN_DEVICES,
    KNOWN_MISSINGNESS,
    LatentPremise,
    drug_pkpd,
    make_premise,
    validate_premise,
)
from .noise import (
    NOISE_CLASSES,
    POINT_ARTIFACTS,
    PRIMARY,
    device_switch,
    inject,
    context_confound,
    trap_evidence_days,
    transient_spike,
    unit_error,
)
from .schema import (
    GradeReport,
    RawCase,
    SolverOutput,
    SolverPayload,
    VerifierPayload,
)
from .solver import BaselineSolver, LLMSolver, Solver, UnsafeSolver
from .synth import AUX_SIGNALS, TemplateGenerator, premise_conflicts, register_aux_signals, synthesize
from .verifier import grade

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
