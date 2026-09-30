"""A model answer whose `action` / `forecast` / `data_quality` is not an object, or whose
`drivers` holds non-objects, is graded with those fields treated as omitted; it never
crashes the cell."""
from haenv import evaluate as ev
from haenv.tracks import _answer_text


def _payload():
    from schema import SolverPayload
    return SolverPayload(case_id="X-01", user_profile={}, prediction_context={
        "target_event_type": "weight_regain"}, longitudinal_data={}, evidence_ledger=[])


def test_string_action_is_treated_as_omitted():
    data = {"differential": [{"rank": 1, "diagnosis": "hypothyroidism"}],
            "action": "refer to endocrinology", "forecast": "high",
            "data_quality": ["x"], "drivers": ["a", {"driver": "adherence"}]}
    out = ev._to_output(data, _payload())
    assert isinstance(out.action, dict) and isinstance(out.forecast, dict)
    assert out.action.get("clinician_review_required") is None    # answered, field omitted
    assert out.drivers == [{"driver": "adherence"}]
    assert out._raw["differential"][0]["diagnosis"] == "hypothyroidism"
    assert isinstance(_answer_text(out), str)                     # renders without raising


def test_well_formed_answer_is_unchanged():
    data = {"action": {"selected_action_class": "A2", "specific_action": "order TSH"},
            "forecast": {"risk_category": "high"}, "drivers": [{"driver": "adherence"}]}
    out = ev._to_output(dict(data), _payload())
    assert out.action == data["action"] and out.drivers == data["drivers"]
    assert out.forecast["risk_category"] == "high"
