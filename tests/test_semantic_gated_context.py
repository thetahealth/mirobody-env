"""Rejudging sees purchased observations, never hidden or not-yet-delivered data."""
from types import SimpleNamespace

import pytest

from haenv.semantic_inputs import gated_observations


def events():
    return [
        {"type": "case/start", "step": None, "data": {}},
        {"type": "tool/call", "step": 1, "data": {"call_id": "1", "target": "lab"}},
        {"type": "tool/result", "step": 1, "data": {"call_id": "1", "revealed": True,
            "truncated": False, "is_test": True, "observation": [{"ts": 1, "value": 7}]}},
        {"type": "request/header", "step": 2, "data": {"revealed_targets": ["lab"]}},
        {"type": "assistant/message", "step": 2, "data": {"raw": "answer", "reasoning_text": "PRIVATE"}},
        {"type": "tool/call", "step": 2, "data": {"call_id": "2", "target": "late"}},
        {"type": "tool/result", "step": 2, "data": {"call_id": "2", "revealed": True,
            "truncated": False, "is_test": True, "observation": [{"ts": 2, "value": 99}]}},
    ]


def test_only_delivered_observations_enter_context():
    sp = SimpleNamespace(longitudinal_data={"weight": [1], "hidden": [2]})
    visible, bought = gated_observations(sp, events(), "answer")
    assert visible.longitudinal_data == {"weight": [1], "lab": [{"ts": 1, "value": 7}]}
    assert bought == ["lab", "late"]
    assert "hidden" in sp.longitudinal_data
    assert "PRIVATE" not in str(visible)


def test_wrong_attempt_or_incomplete_trace_refuses_to_invent_context():
    sp = SimpleNamespace(longitudinal_data={"weight": []})
    with pytest.raises(ValueError, match="answer"):
        gated_observations(sp, events(), "different")
    ev = [e for e in events() if e["type"] != "tool/result"]
    with pytest.raises(ValueError, match="observation"):
        gated_observations(sp, ev, "answer")


def test_last_attempt_replaces_prior_trace_in_full():
    sp = SimpleNamespace(longitudinal_data={"weight": []})
    ev = events() + [
        {"type": "case/start", "step": None, "data": {}},
        {"type": "request/header", "step": 1, "data": {"revealed_targets": []}},
        {"type": "assistant/message", "step": 1, "data": {"raw": "retry"}},
    ]
    visible, bought = gated_observations(sp, ev, "retry")
    assert visible.longitudinal_data == {"weight": []} and bought == []


def test_recorded_unavailable_feedback_is_preserved_as_feedback_not_a_lab_value():
    sp = SimpleNamespace(longitudinal_data={"weight": []}, prediction_context={"prediction_time_T": 2})
    ev = events()
    header = next(e for e in ev if e["type"] == "request/header")
    header["data"]["gated_context"] = {"unavailable_results": {"other-test": "unsupported_or_ambiguous_target"}}
    visible, _ = gated_observations(sp, ev, "answer")
    assert visible.prediction_context["tool_unavailable_results"]["other-test"] == "unsupported_or_ambiguous_target"
    assert "other-test" not in visible.longitudinal_data
