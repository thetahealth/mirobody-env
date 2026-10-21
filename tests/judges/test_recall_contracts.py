"""Recall matching distinguishes specific orders, uncertain proposals and past results."""
import json
from pathlib import Path

import pytest
from haenv.judges import recall_match as RM

CASES = json.loads(Path(__file__).with_name("recall.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_recall_matches_specificity_context_and_the_registered_vocabulary(case, monkeypatch):
    vocab = RM._vocab()
    if any(key in case for key in ("remove_shadow", "clear_rv", "clear_past")):
        vocab = {**vocab, "rv": dict(vocab["rv"]), "shadows": dict(vocab["shadows"])}
        if "remove_shadow" in case:
            vocab["shadows"].pop(case["remove_shadow"])
        for key in case.get("clear_rv", []):
            vocab["rv"][key] = ()
        if case.get("clear_past"):
            vocab["past"] = ()
        monkeypatch.setattr(RM, "_vocab", lambda: vocab)
    result = RM.match_item(case["item"], case["proposals"])
    assert result["verdict"] in case["verdicts"]
    if "rule" in case:
        assert result["rule"] == case["rule"]
    if "hint_stop_absent" in case:
        assert case["hint_stop_absent"] not in vocab["rv"]["hint_stop_zh"]
