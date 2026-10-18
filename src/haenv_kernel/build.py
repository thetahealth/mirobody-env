"""build_instance + leakage_probe -- the enforcement layer for temporal isolation and answer
isolation.

This is the only entry point that splits a RawCase into two physically separate payloads:
    solver_payload, verifier_payload = build_instance(raw, until_T)
solver_payload contains only ts<=T and no answer field whatsoever; leakage_probe enforces this.
"""
from __future__ import annotations

import json
import logging

from haenv_kernel.schema import RawCase
from haenv_kernel.schema import SolverPayload
from haenv_kernel.schema import VerifierPayload

log = logging.getLogger("harness.build")

# Any field/substring whose appearance in the solver payload is itself judged a leak (answer-side
# vocabulary)
FORBIDDEN_TOKENS = (
    "outcome_label", "adjudication", "gold_driver", "future_data",
    "verifier", "risk_truth", "tau_star", "d_star", "reversal_points",
    # noise-robustness latent-variable fields (the truth behind latent-variable-controlled
    # noise, verifier-side only)
    "artifact_flags", "noise_class",
    # the synthesis latent premise (the pre-generation constraint, verifier-side only)
    "latent_premise", "premise",
)


def _filter_le_T(series: dict[str, list[dict]], T: int) -> dict[str, list[dict]]:
    return {sig: [p for p in pts if p["ts"] <= T] for sig, pts in series.items()}


def _filter_gt_T(series: dict[str, list[dict]], T: int) -> dict[str, list[dict]]:
    return {sig: [p for p in pts if p["ts"] > T] for sig, pts in series.items()}


def solver_case_id(raw: RawCase) -> str:
    """The id the solver sees: the opaque `latent_premise.meta.solver_case_id` recorded at
    build time (`haenv.opaque_ids`), else the real id (cases built before it). The real id
    can carry the variant number, which correlates with gold; the verifier keeps it."""
    meta = (getattr(raw, "latent_premise", None) or {}).get("meta") or {}
    return str(meta.get("solver_case_id") or raw.case_id)


def build_instance(raw: RawCase, until_T: int) -> tuple[SolverPayload, VerifierPayload]:
    """Splits by index-time T. The solver side keeps only the timeline and EVs with ts<=T."""
    solver = SolverPayload(
        case_id=solver_case_id(raw),
        user_profile=raw.user_profile,
        prediction_context={**raw.prediction_context, "prediction_time_T": until_T},
        longitudinal_data=_filter_le_T(raw.longitudinal_data, until_T),
        evidence_ledger=[e for e in raw.evidence_ledger
                         if e.get("source_timestamp", 10**9) <= until_T],
    )
    verifier = VerifierPayload(
        case_id=raw.case_id,
        T=until_T,
        future_data=_filter_gt_T(raw.longitudinal_data, until_T),
        outcome_label=raw.outcome_label,
        label_rule=raw.label_rule,
        gold_drivers=raw.gold_drivers,
        adjudication=raw.adjudication,
        reversal_points=raw.reversal_points,
    )
    log.info("[build] %s: T=%d, solver signals=%d, verifier future signals=%d",
             raw.case_id, until_T, len(solver.longitudinal_data), len(verifier.future_data))
    return solver, verifier


def leakage_probe(solver: SolverPayload, T: int) -> tuple[bool, list[str]]:
    """CI gate: if the solver payload contains any point with ts>T, or any answer vocabulary
    -> leaked."""
    violations: list[str] = []

    # (1) Temporal leakage
    for sig, pts in solver.longitudinal_data.items():
        bad = [p["ts"] for p in pts if p["ts"] > T]
        if bad:
            violations.append(f"future_timepoint:{sig}@{bad[:3]}")

    # (2) EV timestamp leakage
    for e in solver.evidence_ledger:
        if e.get("source_timestamp", 0) > T:
            violations.append(f"future_evidence:{e.get('evidence_id')}")

    # (3) Answer-vocabulary leakage (scanned after serialization)
    blob = json.dumps(solver.__dict__, ensure_ascii=False, default=lambda o: o.__dict__).lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in blob:
            violations.append(f"forbidden_token:{tok}")

    ok = not violations
    (log.info if ok else log.error)("[leakage_probe] %s -> %s%s",
                                     solver.case_id, "CLEAN" if ok else "LEAK",
                                     "" if ok else f" {violations}")
    return ok, violations
