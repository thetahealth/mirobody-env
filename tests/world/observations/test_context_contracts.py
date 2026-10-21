"""Observation contract: remove joint-answer cues while retaining clinical context."""
import json
from pathlib import Path
import pytest
from haenv.world.observations.events import sanitize_context

CASES = json.loads(Path(__file__).with_name("context.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_context_removes_answer_structure_and_preserves_clinical_evidence(case):
    text, stripped = sanitize_context(case["text"])
    assert bool(stripped) is case["stripped"]
    assert text == ("" if case["stripped"] else case["text"])
