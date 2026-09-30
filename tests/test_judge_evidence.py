import json

import pytest

from haenv.judge_evidence import evidence_catalogue


def test_every_catalogue_entry_is_verbatim_and_has_an_unambiguous_path():
    answer = {"action": {"review": True, "note": 'do not say "normal"\nwithout data'},
              "data_quality": {"a/b~c": None}, "differential": [{"diagnosis": "A"}]}
    text = json.dumps(answer, ensure_ascii=False, sort_keys=True, indent=2)
    quotes, paths = evidence_catalogue(text)
    assert set(quotes) == set(paths) and len(quotes) == 4
    assert all(q in text for q in quotes.values())
    assert "/data_quality/a~1b~0c" in paths.values()
    assert quotes[next(k for k,v in paths.items() if v == "/action/review")] == "true"
    assert evidence_catalogue(text) == (quotes, paths)


def test_no_evidence_is_not_a_fabricated_quote():
    assert evidence_catalogue('{"differential": [], "note": ""}') == ({}, {})


@pytest.mark.parametrize("raw", ['{"risk":NaN}', '[]', '"text"'])
def test_non_json_or_unstructured_answer_is_refused(raw):
    with pytest.raises(ValueError):evidence_catalogue(raw)
