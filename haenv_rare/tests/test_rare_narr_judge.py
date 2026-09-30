"""`rc_narr_*`: coding on the whole case report is matched back to the gold by char-span overlap.
Each judge gets a positive case and a negative control (right term at the wrong place, a span in
a noise sentence, flipped polarity, no report at all)."""
import types

import pytest

import haenv_rare as R

MD_SPANS = [
    {"hpo_id": "HP:0001250", "polarity": "present", "subject": "proband", "start": 10, "end": 30, "idx": 0},
    {"hpo_id": "HP:0001263", "polarity": "absent", "subject": "proband", "start": 40, "end": 60, "idx": 1},
    {"hpo_id": None, "polarity": None, "subject": None, "start": 70, "end": 90, "idx": None},
]
RARE = {"attachments": {"narrative": {"spans": MD_SPANS}}}


class _Ont:
    def canon(self, c):
        return c


@pytest.fixture(autouse=True)
def _no_ontology(monkeypatch):
    monkeypatch.setattr(R, "_ont", lambda: _Ont())


def _a(hpo, span, pol="present"):
    return {"codes": {"hpo": hpo}, "char_span": list(span), "polarity": pol, "subject": "proband"}


def _judge(asserts):
    return R._judge_narrative({"narrative": {"assertions": asserts}}, RARE)


def test_all_right():
    r = _judge([_a("HP:0001250", (12, 28)), _a("HP:0001263", (41, 59), "absent")])
    assert r == {"rc_narr_recall": 1.0, "rc_narr_span_ok": 1.0, "rc_narr_polarity_ok": 1.0,
                 "rc_narr_noise_abstain": 1.0}


def test_right_term_at_the_wrong_place_is_recalled_but_not_located():
    r = _judge([_a("HP:0001250", (41, 59)), _a("HP:0001263", (41, 59), "absent")])
    assert r["rc_narr_recall"] == 1.0 and r["rc_narr_span_ok"] == 0.5


def test_a_span_inside_a_noise_sentence_costs_abstention():
    r = _judge([_a("HP:0001250", (12, 28)), _a("HP:0001263", (41, 59), "absent"), _a("HP:0000001", (72, 80))])
    assert r["rc_narr_noise_abstain"] == 0.0 and r["rc_narr_recall"] == 1.0


def test_flipped_polarity():
    r = _judge([_a("HP:0001250", (12, 28)), _a("HP:0001263", (41, 59), "present")])
    assert r["rc_narr_polarity_ok"] == 0.5


def test_nothing_coded_scores_zero_not_none():
    r = _judge([])
    assert r["rc_narr_recall"] == 0.0 and r["rc_narr_span_ok"] == 0.0 and r["rc_narr_polarity_ok"] == 0


def test_no_report_is_not_applicable():
    r = R._judge_narrative({"narrative": {"assertions": [_a("HP:0001250", (12, 28))]}}, {"attachments": {}})
    assert set(r.values()) == {None}


def test_keys_are_declared_and_reach_the_row():
    for k in ("rc_narr_recall", "rc_narr_span_ok", "rc_narr_noise_abstain", "rc_narr_polarity_ok"):
        assert k in R.KEYS
    assert R.KEYS[-1] == "rc_na_reason"
