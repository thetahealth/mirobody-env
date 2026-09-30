"""Tests for two judge properties: `ct_ok` must not be identical to
`ct_declared`; `risk` must not collapse "absent" and 0.0 into the same number.

## `ct_ok` must not be an identity

With `forced = named and not declared` and `ct_ok = declared and not forced`,
substituting gives:

    declared and not (named and not declared)  ==  declared

Of the four rows of the truth table, exactly one is wrong:
`declared=True, named=True` -- the model declares insufficient data yet
still names a top-1 diagnosis, and still gets 1.0. The dimension's own
docstring says, verbatim, "declared insufficient AND did not force a
diagnosis."

## Two `risk` properties

`float(out.forecast.get("risk", 0.5) or 0.5)`:

* A legitimate `risk=0.0` gets rewritten to 0.5 by `or` -- the model says
  "almost no chance of an event" and it gets recorded as a coin flip;
* Not answering the field also records 0.5 -- "no answer" and "answered
  0.5" look identical in the output.

Both feed into `brier` (squared error), which is a scored dimension.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import judges as J                       # noqa: E402


class _Out:
    def __init__(self, *, sufficiency=None, dx=None, risk="__absent__", cat="high"):
        self.data_quality = {"data_sufficiency": sufficiency} if sufficiency else {}
        self._raw = {"differential": ([{"diagnosis": dx}] if dx else [])}
        self.forecast = {"risk_category": cat}
        if risk != "__absent__":
            self.forecast["risk"] = risk
        self.drivers = []
        self.action = {}
        self.cited_evidence = []


class _VP:
    adjudication = {}
    gold_drivers = []
    outcome_label = "event_occurred"


def test_ct_ok_penalises_declaring_and_still_naming():
    """Declares insufficient data AND still names a diagnosis => `ct_ok` must be 0."""
    r = J.judge_commit_timing(_Out(sufficiency="insufficient_data", dx="PCOS"), _VP())
    assert r["ct_declared"] == 1.0 and r["ct_named_dx"] == 1.0, r
    assert r["ct_ok"] == 0.0, f"声明不足却仍点名诊断,ct_ok 应为 0:{r}"


def test_ct_ok_is_not_identical_to_ct_declared():
    """Converse: `ct_ok` and `ct_declared` must be distinguishable -- if they're equal, this dimension is just an identity."""
    a = J.judge_commit_timing(_Out(sufficiency="insufficient_data", dx="PCOS"), _VP())
    b = J.judge_commit_timing(_Out(sufficiency="insufficient_data"), _VP())
    assert a["ct_declared"] == b["ct_declared"] == 1.0
    assert a["ct_ok"] != b["ct_ok"], "两种作答的 ct_ok 相同 ⇒ 这一维没有分辨力"
    assert b["ct_ok"] == 1.0, b


def test_risk_zero_is_not_rewritten_to_half():
    """A legitimate `risk=0.0` must be preserved as-is, not rewritten to 0.5 by `or`."""
    r = J.judge_forecast(_Out(risk=0.0, cat="low"), _VP(), None)
    assert r["risk"] == 0.0, f"0.0 被改写了:{r}"
    assert r["risk_absent"] is False
    assert r["brier"] == 1.0, f"y=1 而 risk=0 ⇒ brier 应为 1.0:{r}"


def test_absent_risk_is_named_not_defaulted():
    """`risk` absent => named via `risk_absent`, `brier` recorded as `None` -- not collapsed into a number."""
    r = J.judge_forecast(_Out(cat="high"), _VP(), None)
    assert r["risk"] is None and r["risk_absent"] is True, r
    assert r["brier"] is None, f"没答 risk 却算出了 brier:{r}"


def test_present_risk_still_scores():
    """Converse: scoring still runs when a value is given (otherwise the two tests above could be gamed by always returning None)."""
    r = J.judge_forecast(_Out(risk=0.75, cat="high"), _VP(), None)
    assert r["risk"] == 0.75 and r["risk_absent"] is False
    assert r["brier"] == round((0.75 - 1.0) ** 2, 4), r
