"""Contract tests for `tracks.ddx_hit` / `join_gold_hit` / `alternative_a1`.

`tracks.ddx_hit` is not on the production path -- `evaluate._row_single` goes
through `judges.run_judges` -- and is kept for the reporting code that
calls it. These tests are its contract.

Hit determination always delegates to `dx_rank_of` (the single implementation
on the scoring side); `ddx_hit` must not implement a second copy. This file
therefore also pins down that delegation: if `dx_rank_of`'s behavior changes,
the tests below change with it.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv.events import sanitize_context                            # noqa: E402
from haenv.tracks import alternative_a1, ddx_hit, join_gold_hit      # noqa: E402


class FakeOut:
    def __init__(self, raw=None, action=None, drivers=None):
        self._raw = raw or {}
        self.action = action or {"specific_action": ""}
        self.drivers = drivers or []
        self.forecast = {}


GOLD = {"diagnosis": "多囊卵巢综合征(PCOS)", "aliases": ["多囊", "pcos", "polycystic"],
        "join_gold": "unified", "tests": ["OGTT+胰岛素"], "specialty": ["妇科内分泌"]}

TOP1 = FakeOut({"differential": [{"rank": 1, "diagnosis": "多囊卵巢综合征",
                                  "supporting_evidence": ["EV-1"]},
                                 {"rank": 2, "diagnosis": "甲减",
                                  "supporting_evidence": ["EV-2"]}],
                "join_type": "unified"})
BASE = FakeOut(action={"specific_action": "collect more data"})


# ---------------------------------------------------------------- diagnosis hit

def test_top1_hit_reports_rank_one_and_structured_source():
    r = ddx_hit(TOP1, GOLD)
    assert r["dx_hit"] and r["dx_hit_top1"] and r["dx_rank"] == 1
    assert r["dx_source"] == "structured", r


def test_hit_at_rank_three_is_a_hit_but_not_top1():
    rank3 = FakeOut({"differential": [{"rank": 1, "diagnosis": "甲减"},
                                      {"rank": 2, "diagnosis": "库欣"},
                                      {"rank": 3, "diagnosis": "PCOS"}]})
    r = ddx_hit(rank3, GOLD)
    assert r["dx_hit"] and not r["dx_hit_top1"] and r["dx_rank"] == 3, r


def test_no_mention_is_not_a_hit():
    """Negative control: no mention at all of the gold diagnosis => must not count as a hit."""
    miss = FakeOut({"differential": [{"rank": 1, "diagnosis": "甲减"}]})
    assert not ddx_hit(miss, GOLD)["dx_hit"]


def test_missing_differential_falls_back_to_text_scan():
    """The offline baseline has no `differential` => falls back to text scan instead of crashing or silently scoring 0."""
    assert ddx_hit(BASE, GOLD)["dx_source"] == "text_fallback"


# ---------------------------------------------------------------- join type

def test_join_type_structured_hit_and_miss():
    assert join_gold_hit(FakeOut({"join_type": "unified"}), GOLD)["join_hit"]
    assert not join_gold_hit(FakeOut({"join_type": "comorbidity"}), GOLD)["join_hit"]


def test_join_type_falls_back_when_field_absent():
    assert join_gold_hit(BASE, GOLD)["join_source"] == "text_fallback"


# ---------------------------------------------------------------- candidate hypothesis A1

def test_a1_full_credit_for_two_evidence_backed_candidates():
    assert alternative_a1(TOP1)["a1"] == 1.0, alternative_a1(TOP1)


def test_a1_half_credit_when_baseline_reports_one_driver():
    out = FakeOut(drivers=[{"driver": "x", "evidence_for": ["EV-1"]}])
    assert alternative_a1(out)["a1"] == 0.5


# ---------------------------------------------------------------- de-attribution of join-structure phrasing

def test_join_structure_phrasing_is_scrubbed():
    """Join-structure phrasing must be scrubbed: it names no disease and no banned token, so the alias scanner cannot catch it."""
    for txt in ("两个过程叠加,非单一", "彼此独立、互不相关", "可用同一病因统一解释"):
        out_txt, stripped = sanitize_context(txt)
        assert stripped and not out_txt, (txt, out_txt)


def test_ordinary_clinical_context_is_not_scrubbed():
    """Positive control: ordinary clinical context must not be scrubbed by mistake (otherwise de-attribution would eat the prompt itself)."""
    for txt in ("红肿硬结,护肤/外用无明显效果", "无怀孕可能", "血压 220/120"):
        out_txt, stripped = sanitize_context(txt)
        assert (not stripped) and out_txt == txt, (txt, out_txt)


# ---------------------------------------------------------------- compound / independent

def test_compound_diagnosis_scores_by_alias_coverage():
    """A compound diagnosis is judged by alias coverage: naming only one thread should not score the same as catching all of them."""
    gold = {"diagnosis": "心肾代谢综合征", "aliases": ["糖尿病", "慢性肾病", "ckd"],
            "join_gold": "comorbidity", "tests": [], "specialty": []}
    one = FakeOut({"differential": [{"rank": 1, "diagnosis": "2型糖尿病失控"}]})
    both = FakeOut({"differential": [{"rank": 1, "diagnosis": "2型糖尿病失控"},
                                     {"rank": 2, "diagnosis": "慢性肾病进展"}]})
    r1, r2 = ddx_hit(one, gold), ddx_hit(both, gold)
    assert r1["dx_compound"] and r1["dx_alias_coverage"] is not None
    assert r1["dx_alias_coverage"] < 1.0, r1
    assert r2["dx_alias_coverage"] > r1["dx_alias_coverage"], (r1, r2)


def test_independent_question_reports_na_not_zero():
    """An `independent` case has no single diagnosis => the dx column is N/A and must not enter `dx_hit`'s denominator."""
    gold = {"diagnosis": "无统一病理", "aliases": ["独立", "良性", "无统一"],
            "join_gold": "independent"}
    r = ddx_hit(FakeOut({"differential": [{"rank": 1, "diagnosis": "过敏性鼻炎"}]}), gold)
    assert r.get("dx_applicable") is False and "dx_hit" not in r, r
