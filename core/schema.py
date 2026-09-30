"""Typed contracts for the metabolic longitudinal RBE harness.

The first layer of isolation is the type boundary:
- The solver can only ever receive a `SolverPayload` (<=T).
- The verifier can only ever receive `SolverOutput` + `VerifierPayload` (future data + labels +
  adjudication).
- The two never appear together in a single function signature -- see build.py / runner.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


Point = dict  # {"ts": int(day), "value": float}


# ---------------------------------------------------------------- raw case
@dataclass
class RawCase:
    """The complete case (including hidden state S) -- only build.py may touch this; it is never
    handed to the solver as a whole."""
    case_id: str
    user_profile: dict
    prediction_context: dict            # {prediction_time_T, target_event_type, prediction_window}
    # full-timeline signal -> [{ts, value}] (day 0..365)
    longitudinal_data: dict[str, list[Point]]
    evidence_ledger: list[dict]         # each entry includes source_timestamp
    # ---- hidden state (verifier-only) ----
    outcome_label: str                  # event_occurred / event_not_occurred / ...
    label_rule: dict                    # minimum_change_magnitude / persistence / baseline_window
    gold_drivers: list[str]
    adjudication: dict                  # ground-truth driver, etc.
    reversal_points: list[dict] = field(default_factory=list)  # for reference: real/false reversal annotations
    # synthesis premise (verifier-only): the four-dimensional latent premise; build never
    # cuts this into SolverPayload, and leakage_probe backstops it
    latent_premise: dict = field(default_factory=dict)


# ---------------------------------------------------------------- payloads
@dataclass
class SolverPayload:
    """Everything handed to the solver -- strictly <=T, with no answer field whatsoever."""
    case_id: str
    user_profile: dict
    prediction_context: dict
    longitudinal_data: dict[str, list[Point]]
    evidence_ledger: list[dict]

    def dumps(self) -> str:
        import json
        # Compact separators: the payload is mostly long numeric series, and indentation
        # roughly doubles its token count without adding information.
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


@dataclass
class VerifierPayload:
    """Visible only to the verifier -- future data + gold standard + adjudication. Never reaches
    the solver."""
    case_id: str
    T: int
    future_data: dict[str, list[Point]]
    outcome_label: str
    label_rule: dict
    gold_drivers: list[str]
    adjudication: dict
    reversal_points: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- solver I/O
@dataclass
class SolverOutput:
    forecast: dict          # {target_event, risk:float, risk_category, confidence, key_predictive_evidence:[EV]}
    drivers: list[dict]     # [{rank, driver, causal_status, evidence_for:[EV]}]
    action: dict            # {selected_action_class, specific_action, what_not_to_do:[...], clinician_review_required, followup_interval}
    data_quality: dict      # {data_sufficiency, signal_quality:{...}}
    cited_evidence: list[str] = field(default_factory=list)  # all cited EV ids (used by the no_claim_without_source gate)


# ---------------------------------------------------------------- grading
@dataclass
class GradeReport:
    case_id: str
    hard_gate_failures: list[str]
    tracks: dict[str, float]        # {A,B,C,D,E: score in [0,1] or None}
    overall: str                    # FAIL(gate) / SCORED (grade never returns PASS)
    notes: list[str] = field(default_factory=list)

    def dumps(self) -> str:
        import json
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
