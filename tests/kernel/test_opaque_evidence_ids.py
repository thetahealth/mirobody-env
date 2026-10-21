"""Ledger-grounded EV shorthand controls; synthetic fixtures, no clinical claims."""
from _paths import ROOT as _TEST_ROOT
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from haenv.world.generation.build import anonymize_evidence_ids
from haenv_kernel.build import build_instance
from haenv_kernel.schema import RawCase, SolverOutput, VerifierPayload
from haenv_kernel.verifier import _resolves_in_ledger, grade

REAL = "JD-04v15"
OPAQUE = "C0123456789"
OTHER = "C9876543210"
LEDGER = [f"EV-{OPAQUE}-01", f"EV-{OPAQUE}-05", f"EV-{OPAQUE}-23"]


def _payload():
    return VerifierPayload(case_id=REAL, T=10, future_data={},
                           outcome_label="event_not_occurred", label_rule={},
                           gold_drivers=[], adjudication={})


def _answer(cited):
    return SolverOutput(forecast={"risk": 0, "risk_category": "low"}, drivers=[],
                        action={"selected_action_class": "A0", "what_not_to_do": ["x"]},
                        data_quality={"data_sufficiency": "sufficient"}, cited_evidence=cited)


@pytest.mark.parametrize("citation", ["EV-1", "EV-01", "EV-0001", "EV-5", "EV-05", "EV-23"])
def test_opaque_shorthand_uses_ledger_namespace(citation):
    report = grade(_answer([citation]), _payload(), LEDGER)
    assert report.hard_gate_failures == []
    assert report.overall == "SCORED"


@pytest.mark.parametrize("citation", [
    "EV-99", "EV-0", "EV-230", f"EV-{REAL}-01", f"EV-{OTHER}-01",
    f"EV-{OPAQUE}-1", f"EV-{OPAQUE}-001", f"EV-{OPAQUE}-99",
    "EV-+1", "EV--1", "ev-1", "EV-1x", "EV-1.0", " EV-1", "EV-1 ",
])
def test_missing_wrong_namespace_and_malformed_refs_fail(citation):
    report = grade(_answer([citation]), _payload(), LEDGER, evidence_names={"ev99"})
    assert report.hard_gate_failures == [f"hallucinated_clinical_fact:{citation}"]
    assert report.overall == "FAIL(gate)"


@pytest.mark.parametrize("ledger", [
    [f"EV-{OPAQUE}-01", f"EV-{OPAQUE}-001"],
    [f"EV-{OPAQUE}-01", f"EV-{OTHER}-01"],
    [f"EV-{OPAQUE}-01", f"EV-{OTHER}-02"],
    [f"EV-{REAL}-01", "EV-OTHER-02"],
    [f"EV-{REAL}-01", f"EV-{OPAQUE}-02"],
    ["EV-OTHER-01"], [], ["not-an-evidence-id"],
])
def test_shorthand_requires_one_supported_unambiguous_namespace(ledger):
    report = grade(_answer(["EV-1"]), _payload(), ledger)
    assert report.hard_gate_failures == ["hallucinated_clinical_fact:EV-1"]


@pytest.mark.parametrize("ledger", [LEDGER, [f"EV-{OPAQUE}-01", f"EV-{OTHER}-01"],
                                    [f"EV-{OPAQUE}-01", f"EV-{OPAQUE}-001"],
                                    ["EV-01"], [f"EV-{REAL}-01"]])
def test_complete_ids_keep_exact_membership_semantics(ledger):
    assert grade(_answer(ledger), _payload(), ledger).hard_gate_failures == []


@pytest.mark.parametrize("citation", ["EV-1", "EV-01", "EV-0001"])
def test_legacy_real_case_namespace_still_resolves(citation):
    assert _resolves_in_ledger(citation, REAL, {f"EV-{REAL}-01"})


@pytest.mark.parametrize("case_id", [None, ""])
def test_missing_case_context_does_not_enable_shorthand(case_id):
    assert not _resolves_in_ledger("EV-1", case_id, set(LEDGER))


def test_mixed_citations_only_flag_the_absent_evidence():
    report = grade(_answer(["EV-1", LEDGER[1], "EV-99"]), _payload(), LEDGER)
    assert report.hard_gate_failures == ["hallucinated_clinical_fact:EV-99"]


def test_anonymized_build_resolves_only_solver_visible_ledger():
    raw = RawCase(case_id=REAL, user_profile={}, prediction_context={},
                  longitudinal_data={},
                  evidence_ledger=[{"evidence_id": f"EV-{REAL}-S1", "source_timestamp": 1},
                                   {"evidence_id": f"EV-{REAL}-B1", "source_timestamp": 11}],
                  outcome_label="event_not_occurred", label_rule={}, gold_drivers=[],
                  adjudication={}, latent_premise={"meta": {"solver_case_id": OPAQUE}})
    mapping = anonymize_evidence_ids(raw)
    assert mapping[f"EV-{REAL}-S1"] == f"EV-{OPAQUE}-01"
    sp, vp = build_instance(raw, 10)
    assert sp.case_id == OPAQUE and vp.case_id == REAL
    ids = [e["evidence_id"] for e in sp.evidence_ledger]
    assert ids == [f"EV-{OPAQUE}-01"]
    assert grade(_answer(["EV-1"]), vp, ids).hard_gate_failures == []
    assert grade(_answer(["EV-2"]), vp, ids).hard_gate_failures == [
        "hallucinated_clinical_fact:EV-2"]


def test_standalone_verifier_needs_no_process_local_opaque_mapping():
    root = _TEST_ROOT
    # The checkout uses src/: do not fall back to another worktree's editable
    # install when this verifier runs in a fresh interpreter.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))}
    data = {"solver_output": asdict(_answer(["EV-1", "EV-99"])),
            "verifier_payload": asdict(_payload()), "ledger_ids": LEDGER}
    proc = subprocess.run([sys.executable, "-m", "haenv_kernel.verifier"],
                          cwd=root, env=env, input=json.dumps(data), text=True,
                          capture_output=True, check=True)
    report = json.loads(proc.stdout)
    assert report["hard_gate_failures"] == ["hallucinated_clinical_fact:EV-99"]
