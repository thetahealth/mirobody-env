"""Stored-answer judging must stay anonymous and keep failed or missing inputs explicit."""
import json

import pytest

from haenv.semantic_inputs import anonymous_prompt, select_responses


def test_last_slice_is_selected_even_when_file_order_differs():
    rows = [{"case": "case-a", "solver": "model-a", "geometry": "slices",
             "slice_rows": [{"t": 2}, {"t": 4}]}]
    responses = [{"case": "case-a", "solver": "model-a", "slice_t": 4, "raw": '{"answer":"last"}'},
                 {"case": "case-a", "solver": "model-a", "slice_t": 2, "raw": '{"answer":"early"}'}]
    got = select_responses(rows, responses)
    assert got[0]["response"]["raw"] == '{"answer":"last"}'
    assert got[0]["status"] == "ready"


def test_failed_last_attempt_not_replaced_by_previous_success():
    rows = [{"case": "case-a", "solver": "model-a", "geometry": "gated"}]
    responses = [{"case": "case-a", "solver": "model-a", "slice_t": None, "raw": '{}'},
                 {"case": "case-a", "solver": "model-a", "slice_t": None, "raw": ''}]
    assert select_responses(rows, responses)[0]["status"] == "empty_response"


def test_missing_last_slice_stays_missing():
    rows = [{"case": "case-a", "solver": "model-a", "geometry": "slices",
             "slice_rows": [{"t": 4}]}]
    responses = [{"case": "case-a", "solver": "model-a", "slice_t": 2, "raw": '{}'}]
    assert select_responses(rows, responses)[0]["status"] == "missing_response"


def test_duplicate_eval_identity_is_not_silently_counted_twice():
    row = {"case": "c", "solver": "m", "geometry": "gated"}
    with pytest.raises(ValueError):
        select_responses([row, row], [])


def test_anonymous_prompt_has_no_model_or_prior_scores():
    criteria = {"dx": "Is the asserted diagnosis supported by the reference?"}
    answer = {"differential": [{"diagnosis": "A"}], "solver": "secret-model",
              "score": .99, "trace": {"private": "previous reasoning"}}
    prompt, body = anonymous_prompt(criteria, {"diagnosis": "A"}, answer)
    assert "secret-model" not in prompt and "previous reasoning" not in prompt
    assert "0.99" not in prompt
    assert '"diagnosis": "A"' in body
    assert "untrusted" in prompt.lower()
    assert "uncertain" in prompt and "verdicts" in prompt


def test_prompt_changes_when_rubric_changes():
    first = anonymous_prompt({"dx": "rule-one"}, {"diagnosis": "A"}, {"differential": []})
    second = anonymous_prompt({"dx": "rule-two"}, {"diagnosis": "A"}, {"differential": []})
    assert first[0] != second[0]


def test_empty_rubric_or_empty_answer_is_rejected():
    with pytest.raises(ValueError):
        anonymous_prompt({}, {"diagnosis": "A"}, {"differential": []})
    with pytest.raises(ValueError):
        anonymous_prompt({"dx": "rule"}, {"diagnosis": "A"}, {"score": 1})
