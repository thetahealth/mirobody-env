"""Scoring contracts: ranked evidence, joint hypotheses and applicability retain their meaning."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from haenv.scoring.tracks import alternative_a1, ddx_hit, join_gold_hit

DATA = json.loads(Path(__file__).with_name("diagnosis.json").read_text())


def answer(case):
    return SimpleNamespace(_raw=case.get("raw", {}),
        action=case.get("action", {"specific_action": "collect more data"}),
        drivers=case.get("drivers", []), forecast={})


@pytest.mark.parametrize("case", DATA["rank"], ids=lambda c: c["id"])
def test_diagnosis_rank_and_applicability_reproduce_the_stored_verdict(case):
    got = ddx_hit(answer(case), case.get("gold", DATA["gold"]))
    assert {key: got[key] for key in case["expected"]} == case["expected"]
    assert all(key not in got for key in case.get("absent", []))


@pytest.mark.parametrize("case", DATA["join"], ids=lambda c: c["id"])
def test_joint_hypothesis_agreement_preserves_the_answer_source(case):
    got = join_gold_hit(answer(case), DATA["gold"])
    assert {key: got[key] for key in case["expected"]} == case["expected"]


@pytest.mark.parametrize("case", DATA["candidates"], ids=lambda c: c["id"])
def test_alternative_hypotheses_require_independent_evidence(case):
    assert alternative_a1(answer(case))["a1"] == case["score"]


def test_compound_diagnosis_scores_cover_each_disease_thread():
    gold = {"diagnosis": "心肾代谢综合征", "aliases": ["糖尿病", "慢性肾病", "ckd"],
            "join_gold": "comorbidity", "tests": [], "specialty": []}
    first = {"rank": 1, "diagnosis": "2型糖尿病失控"}
    one = ddx_hit(answer({"raw": {"differential": [first]}}), gold)
    both = ddx_hit(answer({"raw": {"differential": [first, {"rank": 2, "diagnosis": "慢性肾病进展"}]}}), gold)
    assert one["dx_compound"] and one["dx_alias_coverage"] is not None
    assert one["dx_alias_coverage"] < 1.0
    assert both["dx_alias_coverage"] > one["dx_alias_coverage"]
