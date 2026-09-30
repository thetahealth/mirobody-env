"""Contract H: the weight label rule reads the same in every place that applies it.

`gates.derive_outcome` (the build's label guard and `audit.weight_truth_label`) and
`gates.check_outcome_derivable` (GEN13; the build's emission gate and the shortcut gate
in `haenv build` read its `outcome_rule_not_applicable`) must reach the same verdict on
the same case.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import types

import pytest

from haenv import gates
from haenv.latent_rules import LABEL_SMOOTHING, MIN_CHANGE_FRAC, MIN_CHANGE_KG, MIN_PERSIST_DAYS

pytestmark = pytest.mark.world

RULE = {"min_change_frac": MIN_CHANGE_FRAC, "min_change_kg": MIN_CHANGE_KG,
        "smoothing": LABEL_SMOOTHING, "min_persist_days": MIN_PERSIST_DAYS}


def _gain_from_start(rise: float) -> list[dict]:
    """The first reading is the lowest; from day 70 the weight sits `rise` kg higher for
    100 days, read every other day."""
    return [{"ts": d, "value": round(90.0 + (rise if d >= 70 else 0.0), 2)}
            for d in range(0, 200, 2)]


def _raw(series: list[dict], rule: dict, label: str):
    return types.SimpleNamespace(adjudication={}, longitudinal_data={"weight": series},
                                 label_rule=rule, outcome_label=label)


@pytest.mark.parametrize("rise,label", [(MIN_CHANGE_KG + 0.25, "event_occurred"),
                                        (MIN_CHANGE_KG - 0.25, "event_not_occurred")])
def test_h1_gen13_agrees_with_derive_outcome_when_nothing_was_lost(rise, label):
    """With a floor, a course whose first reading is its lowest is read, not declared
    not applicable, and GEN13 accepts exactly the verdict `derive_outcome` gives."""
    series = _gain_from_start(rise)
    got, detail = gates.derive_outcome({"weight": series}, RULE)
    assert got == label and detail["lost"] == 0
    hits = gates.check_outcome_derivable(_raw(series, RULE, label))
    assert [h["kind"] for h in hits] == []
    wrong = "event_not_occurred" if label == "event_occurred" else "event_occurred"
    hits = gates.check_outcome_derivable(_raw(series, RULE, wrong))
    assert [h["kind"] for h in hits] == ["outcome_declared_not_derived"]
    # What the rule measures here is recorded with the verdict.
    assert detail.get("reading") == "gain_from_low_point"


def test_h1_negative_control_rule_without_floor_is_not_applicable():
    """A rule without `min_change_kg` (a batch declared without a floor) reads without one:
    nothing counts as lost, the rule cannot apply, and GEN13 says so."""
    old = {k: v for k, v in RULE.items() if k not in ("min_change_kg", "smoothing")}
    series = _gain_from_start(MIN_CHANGE_KG + 0.25)
    got, _ = gates.derive_outcome({"weight": series}, old)
    assert got == "event_not_occurred"
    hits = gates.check_outcome_derivable(_raw(series, old, "event_occurred"))
    assert [h["kind"] for h in hits] == ["outcome_rule_not_applicable"]


def test_h2_clean_reference_is_read_pointwise():
    """A persistent-offset case reads the clean reference (`weight_ref`, a clinic scale
    every 28 days). Seven of its readings span half a year; smoothing them would halve the
    loss. The rule reads it pointwise: `lost` equals the pointwise loss."""
    ref = [{"ts": d, "value": round(96.0 - 11.0 * min(d, 168) / 168, 2)} for d in range(0, 365, 28)]
    obs = [{"ts": d, "value": 99.0} for d in range(0, 365, 2)]
    got, det = gates.derive_outcome({"weight": obs, "weight_ref": ref}, RULE,
                                    persistent_offset=True)
    assert det["basis"] == "weight_ref"
    assert abs(det["lost"] - 11.0) <= 0.01 and got == "event_not_occurred"
    assert det["smoothing"] == "none"


def test_h2_negative_control_smoothed_reference_loses_the_loss():
    """The same series through the 7-reading median loses more than 2 kg of the 11 kg:
    what the pointwise reading prevents."""
    ref = [{"ts": d, "value": round(96.0 - 11.0 * min(d, 168) / 168, 2)} for d in range(0, 365, 28)]
    sm = gates.label_series(ref, RULE)
    assert sm[0]["value"] - min(q["value"] for q in sm) < 11.0 - 2.0
