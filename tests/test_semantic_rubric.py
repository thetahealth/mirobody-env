"""Semantic atoms are grounded and applicability is explicit before model calls."""
import json

from haenv.semantic_rubric import make_rubric, summarize_verdicts


def reference():
    return {"kind": "ddx:unified", "diagnosis": "A", "aliases": ["A"],
            "required_tests": ["test-a", "test-b"], "optional_tests": ["test-c"],
            "rivals": [{"name": "B", "discriminator": "test-a"}],
            "noop": {"target": "steps", "window": [0, 10], "truth_present": False}}


def answer():
    return {"differential": [{"diagnosis": "B", "ruled_out_by": "test-a normal"}],
            "tests_to_order": ["test-a", "test-c"]}


def test_atoms_come_from_actual_reference_and_answer_items():
    criteria, meta = make_rubric(reference(), answer())
    assert "diagnosis_hit" in criteria and "data_availability" in criteria
    assert meta["groups"]["tests_recall"] == ["required_test_0", "required_test_1"]
    assert meta["groups"]["tests_precision"] == ["proposed_test_0", "proposed_test_1"]
    assert meta["groups"]["disc_recall"] == ["discriminator_0"]
    assert "exclusion_reason_0" in criteria


def test_insufficient_gold_never_requires_hidden_diagnosis():
    ref = reference();ref["kind"] = "ddx:insufficient"
    criteria, meta = make_rubric(ref, answer())
    assert "diagnosis_hit" not in criteria
    assert meta["not_applicable"]["dx_hit"]


def test_missing_probe_not_guessed_from_global_data_sufficiency():
    ref = reference();ref["noop"] = None
    criteria, meta = make_rubric(ref, answer())
    assert "data_availability" not in criteria
    assert meta["not_applicable"]["noop_ok"]


def test_f1_computed_from_binary_consensus_without_llm_continuous_score():
    criteria, meta = make_rubric(reference(), answer())
    verdicts = {k: "yes" for k in criteria}
    verdicts["required_test_1"] = "no"
    summary = summarize_verdicts(verdicts, meta)
    assert summary["metrics"]["tests_recall"] == .5
    assert summary["metrics"]["tests_precision"] == 1
    assert abs(summary["metrics"]["tests_f1"] - 2 / 3) < 1e-8


def test_unresolved_component_is_not_removed_to_inflate_score():
    criteria, meta = make_rubric(reference(), answer())
    verdicts = {k: "yes" for k in criteria};verdicts["required_test_1"] = None
    summary = summarize_verdicts(verdicts, meta)
    assert summary["metrics"]["tests_recall"] is None
    assert summary["metrics"]["tests_f1"] is None
    assert summary["coverage"]["tests_recall"] == {"expected": 2, "resolved": 1}


def test_no_test_proposals_has_undefined_precision_not_an_imputed_zero():
    a = answer();a["tests_to_order"] = []
    criteria, meta = make_rubric(reference(), a)
    summary = summarize_verdicts({k: "no" for k in criteria}, meta)
    assert summary["metrics"]["tests_precision"] is None
    assert summary["metrics"]["tests_recall"] == 0
    assert summary["metrics"]["tests_f1"] is None
    ref = reference();ref["required_tests"] = [];ref["optional_tests"] = []
    criteria, meta = make_rubric(ref, a)
    summary = summarize_verdicts({k: "no" for k in criteria}, meta)
    assert summary["metrics"]["tests_precision"] is None
