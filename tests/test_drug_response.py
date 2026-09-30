"""Drug effect on fasting glucose, per-person response, and the glucose -> HbA1c lag.

Each check is two-sided: one input that must be judged negative, one that must pass.

SYNTHETIC data, evaluation only; not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haenv import drug_effects as D  # noqa: E402

WIRED = ("semaglutide", "tirzepatide", "dulaglutide", "liraglutide")
KW = dict(per_kg=0.06, atten=0.75, cohort="T2D")


@pytest.mark.parametrize("drug", WIRED)
def test_fasting_glucose_effect_has_the_same_sign_as_hba1c(drug):
    a = D.effect_at(drug, "HbA1c", 180, 1.0, **KW)
    g = D.effect_at(drug, "fasting_glucose", 180, 1.0, per_kg=0.05, atten=0.75, cohort="T2D")
    assert a < 0 and g < 0, (a, g)


def test_unregistered_signal_and_unwired_drug_stay_at_zero():
    """The other side: no citable number => no effect, not an invented one."""
    assert D.effect_at("semaglutide", "LDL", 180, 1.0, **KW) == 0.0
    assert D.effect_at("metformin", "fasting_glucose", 180, 1.0, **KW) == 0.0


def test_weight_is_still_forbidden():
    with pytest.raises(D.DrugEffectsError):
        D.effect_at("semaglutide", "weight", 180, 1.0, **KW)


def test_low_response_band_and_responder_band_do_not_overlap():
    (lo, hi), (blo, bhi) = D.response_bounds()
    low = [D.response_for(f"C{i}", D.LOW_RESPONSE_DRIVER, "semaglutide") for i in range(200)]
    rest = [D.response_for(f"C{i}", "poor_medication_adherence", "semaglutide") for i in range(200)]
    assert all(lo <= x <= hi for x in low)
    assert all(blo <= x <= bhi for x in rest)
    assert max(low) < min(rest)


def test_low_response_keeps_the_six_month_fall_under_one_point():
    """NG28 2015 criterion: under 1.0 pp at 6 months is clinical non-response."""
    _, hi = D.response_bounds()[0]
    for drug in WIRED:
        for dose in (None, 5.0, 10.0, 15.0):
            fall = D.effect_at(drug, "HbA1c", 182, 1.0, dose_mg=dose, response=hi, **KW)
            assert fall > -1.0, (drug, dose, fall)


def test_a_full_responder_falls_by_more_than_one_point():
    """The other side: at the trial mean the same window shows a clinical response."""
    assert D.effect_at("semaglutide", "HbA1c", 182, 1.0, response=1.0, **KW) < -1.0


def test_response_is_deterministic_and_varies_across_people():
    a = [D.response_for(f"P{i}", "maintain", "tirzepatide") for i in range(50)]
    b = [D.response_for(f"P{i}", "maintain", "tirzepatide") for i in range(50)]
    assert a == b and len({round(x, 6) for x in a}) > 40


def test_declared_response_must_match_the_driver():
    assert D.check_declared_response(0.2, D.LOW_RESPONSE_DRIVER) is None
    assert D.check_declared_response(0.9, D.LOW_RESPONSE_DRIVER)
    assert D.check_declared_response(0.9, "poor_medication_adherence") is None
    assert D.check_declared_response(0.2, "poor_medication_adherence")
    assert D.check_declared_response(0.42, "maintain")          # the gap fits neither band


def test_glucose_leads_hba1c():
    """Tahara & Shima 1995: fasting glucose half-time 6.3 d, HbA1c 34.6 d."""
    g = D.effect_at("semaglutide", "fasting_glucose", 30, 1.0, per_kg=0.05, atten=0.75,
                    cohort="T2D") / D.effect_at("semaglutide", "fasting_glucose", 365, 1.0,
                                                per_kg=0.05, atten=0.75, cohort="T2D")
    a = D.effect_at("semaglutide", "HbA1c", 30, 1.0, **KW) / \
        D.effect_at("semaglutide", "HbA1c", 365, 1.0, **KW)
    assert g > 0.9 and a < 0.5, (g, a)


def test_the_trial_total_is_still_the_plateau():
    full = D.direct_effect("semaglutide", 0.06, 0.75)
    at_readout = D.effect_at("semaglutide", "HbA1c", 280, 1.0, **KW)
    assert at_readout == pytest.approx(full, rel=0.03)


def test_an_adherence_drop_moves_glucose_before_hba1c():
    """Stop taking the drug at day 200: two weeks later glucose has rebounded most of the
    way, HbA1c only a little -- the cross-scale lag the judges' "adherence down ->
    indicator worsens" relies on."""
    def adh(d):
        return 1.0 if d < 200 else 0.0

    def frac(sig, day, per_kg):
        on = D.effect_at("semaglutide", sig, 199, 1.0, per_kg=per_kg, atten=0.75, cohort="T2D")
        return D.effect_at("semaglutide", sig, day, adh, per_kg=per_kg, atten=0.75,
                           cohort="T2D") / on
    assert frac("fasting_glucose", 214, 0.05) < 0.35
    assert frac("HbA1c", 214, 0.06) > 0.6


def test_job_load_refuses_a_contradictory_declared_response(tmp_path):
    from haenv.job import load_job
    base = (ROOT / "inputs" / "example-ew.job.yaml").read_text(encoding="utf-8")
    import yaml
    doc = yaml.safe_load(base)
    case = doc["cases"][0]
    case["latent"]["driver"] = D.LOW_RESPONSE_DRIVER
    case["latent"]["drug_response"] = 0.9
    bad = tmp_path / "bad.job.yaml"
    bad.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="drug_response"):
        load_job(bad)
    case["latent"]["drug_response"] = 0.2
    ok = tmp_path / "ok.job.yaml"
    ok.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    load_job(ok)
