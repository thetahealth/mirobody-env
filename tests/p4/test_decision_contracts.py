"""Follow-up decisions use the visible draw, the relevant course and the accepted actions."""
import json
from pathlib import Path

import pytest
from haenv.p4 import gold as GD, score as SC

CASES = json.loads(Path(__file__).with_name("decisions.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_followup_decision_and_score_follow_the_printed_record(case):
    gold = case["gold"]
    primary, accepted = GD.next_step(gold)
    assert primary == case["primary"]
    if "acceptable" in case:
        assert list(accepted) == case["acceptable"]
    for action, expected in case["actions"].items():
        atoms = SC.row_atoms({"change_source": gold["class"], "next_step": action}, gold)
        assert bool(atoms["p4_next_ok"]) is expected
