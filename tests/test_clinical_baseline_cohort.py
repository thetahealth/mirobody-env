"""Clinical baselines depend on the patient's cohort.

## What this guards against

A single constant baseline per signal in `CLINICAL_SPEC`, ignoring disease and
comorbidities. In the kernel's `DISEASE_SIGNAL_DOMAIN`, `HbA1c` belongs to both `obesity`
and `T2D`, so an obesity-only patient would get the 7.4% calibrated on 671 real diabetic
patients.

With that constant baseline, most obesity-only cases carrying an HbA1c stream would start
at HbA1c >= 6.5 (the ADA diabetes threshold) while `known_conditions` is only
`['obesity']`. The comorbidity vocabulary holds no diabetes (the values in use are the
OSA / thyroid family), so this is the normal state of the obesity cohort.

## The load-bearing test

It is not "HbA1c equals 5.0 now" -- that number changes with clinical sign-off. It is
`test_only_hba1c_is_cross_cohort`: whether a cohort split exists is decided by the overlap
structure of `DISEASE_SIGNAL_DOMAIN`, not by anyone's impression. It catches both a signal
that genuinely crosses cohorts without a split, and a split added for symmetry that no case
ever exercises.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import gates  # noqa: E402
from haenv.build import (CLINICAL_SPEC, _clinical_baselines,  # noqa: E402
                         clinical_cohort, clinical_plan)

DEVICES = ["smart_scale", "lab_panel", "wearable", "cgm", "bp_cuff"]


def _raw(streams, known):
    return types.SimpleNamespace(longitudinal_data=streams,
                                 user_profile={"known_conditions": list(known)})


def _premise(disease, comorb=()):
    return types.SimpleNamespace(
        patient_basics={"disease": disease, "comorbidities": list(comorb)})


def _gate_kinds(hits):
    return sorted(h["kind"] for h in hits if h["severity"] == "gate")


# ─────────────────────────────────────────────── the structure decides whether a split exists

def test_only_hba1c_is_cross_cohort():
    """The load-bearing test of this file.

    Whether a signal needs cohort baselines is decided by the overlap structure of the
    kernel's `DISEASE_SIGNAL_DOMAIN`: a clinical signal needs them only when it appears in
    two or more disease domains. Only `HbA1c` does. This fails in both directions:
      - a signal becomes cross-domain and nobody adds its cohorts;
      - a single-domain signal gets cohorts for symmetry, which no case would exercise.
    """
    from latent import DISEASE_SIGNAL_DOMAIN

    seen: dict[str, set[str]] = {}
    for dz, dom in DISEASE_SIGNAL_DOMAIN.items():
        for sig in dom:
            if sig == "weight" or sig not in CLINICAL_SPEC:
                continue
            seen.setdefault(sig, set()).add(dz)
    cross = {s for s, dzs in seen.items() if len(dzs) > 1}
    registered = set((_clinical_baselines().get("signals") or {}))
    assert cross == registered, (
        f"跨病种域的信号 {sorted(cross)} 与登记了队列档的 {sorted(registered)} 对不上\n"
        f"(逐信号所属域:{ {k: sorted(v) for k, v in sorted(seen.items())} })")


def test_every_cohort_declares_its_provenance():
    """Every cohort states `source:` and `review:` -- a fake citation is worse than `pending`."""
    for sig, entry in (_clinical_baselines().get("signals") or {}).items():
        for coh, spec in (entry.get("cohorts") or {}).items():
            assert str(spec.get("source") or "").strip(), f"{sig}/{coh} 没有 source"
            assert str(spec.get("review") or "").strip(), f"{sig}/{coh} 没有 review"
            assert "real_abnormal_rate" in spec, (
                f"{sig}/{coh} 没写 real_abnormal_rate —— "
                "「没量过」要显式写 null,不许省略成「继承上一档」")


def test_thresholds_are_published_criteria_not_our_numbers():
    """Diagnostic thresholds state their source and live apart from reference ranges
    (the two serve different purposes)."""
    thr = _clinical_baselines().get("diagnostic_thresholds") or {}
    assert thr, "一条诊断阈值都没登记 ⇒ GEN26 恒零对象"
    for sig, rule in thr.items():
        assert str(rule.get("source") or "").strip(), f"{sig} 的阈值没有出处"
        assert rule.get("condition"), f"{sig} 没说这个阈值下的是什么诊断"
        assert isinstance(rule.get("ge"), (int, float)), sig


# ─────────────────────────────────────────────── the split changes the baseline

def test_obesity_without_diabetes_gets_a_non_diabetic_baseline():
    """The fix itself: an obesity-only patient does not start in the diabetic HbA1c range."""
    thr = (_clinical_baselines()["diagnostic_thresholds"]["HbA1c"])["ge"]
    base = clinical_plan("obesity", DEVICES, [])["HbA1c"]["base"]
    assert base < thr, f"obesity 无糖尿病的 HbA1c 基线 {base} 仍 ≥ 诊断阈值 {thr}"


def test_declared_diabetes_keeps_the_calibrated_baseline():
    """The other direction: diabetic patients keep their calibrated baseline.

    Baselines are drawn per case from the cohort distribution, so the check is that the
    distribution's median stays anchored on the registered value.

    The second check pins the "diagnostic basis first" choice: a patient with primary
    obesity and comorbid T2D matches both cohorts, and `HbA1c` is a diagnostic basis of T2D,
    so the T2D cohort applies. Taking the disease before the comorbidities would return the
    obesity cohort's 5.6, which is wrong.
    """
    import statistics

    from haenv import indicators as I
    # The registered T2D median is 8.0, the measured per-patient p50. A median of 7.4 taken
    # from a different calibration than `p_abnormal` (0.926) would solve to sigma = 0.145,
    # below the measured between-patient sigma of 0.215, which is impossible.
    assert I.moments_of("HbA1c", "T2D")[0] == 8.0, "T2D 档的中位被改掉了"

    def _med(disease, comorb):
        return statistics.median(
            clinical_plan(disease, DEVICES, comorb, f"PROBE-{i}")["HbA1c"]["base"]
            for i in range(400))

    # Anchored on the registered median, 8.0 (see above).
    assert abs(_med("T2D", []) - 8.0) < 0.25, _med("T2D", [])
    # An obesity patient with comorbid T2D lands in the T2D cohort, not obesity's 5.6
    assert abs(_med("obesity", ["T2D"]) - 8.0) < 0.25, _med("obesity", ["T2D"])
    assert abs(_med("obesity", []) - 5.6) < 0.25, "非糖尿病档被拉走了"


@pytest.mark.parametrize("comorb", [["T2D"], ["type_2_diabetes"], ["糖尿病"], ["diabetes"]])
def test_diabetes_recognised_by_every_registered_spelling(comorb):
    """The spellings live in one place in the registry, so a missing spelling is one fix."""
    assert clinical_cohort("HbA1c", "obesity", comorb) == "diabetic"


def test_unrelated_comorbidity_does_not_flip_the_cohort():
    """The comorbidity values in use are the OSA / thyroid family; none reads as diabetes."""
    for c in (["OSA"], ["hypothyroidism"], ["hyperthyroidism_treated"], []):
        assert clinical_cohort("HbA1c", "obesity", c) == "non_diabetic", c


def test_missing_cohort_raises_instead_of_falling_back():
    """A cohort that is not registered raises. A silent fallback would hide registry errors.

    Baselines come from `registry/indicators.yaml`. The fixture uses a disease name that by
    definition is never registered: the property under test is "raise, never fall back",
    not the absence of one particular cell, and a real cell can be filled in at any time --
    which would silently turn this negative control green.
    """
    from haenv import indicators as I
    with pytest.raises(I.IndicatorUnregistered, match="cohort"):
        I.baseline("HbA1c", "__never_registered_disease__")


# ─────────────────────────────────────────────── GEN26 emission gate

def _hba1c(v, n=3):
    return [{"ts": d * 90, "value": v} for d in range(n)]


def test_gate_catches_the_original_defect():
    """Negative control: the original defect -- an obesity patient starting at 7.4."""
    hits = gates.check_clinical_baseline_cohort(
        _raw({"HbA1c": _hba1c(7.4)}, ["obesity"]), _premise("obesity"))
    assert _gate_kinds(hits) == ["clinical_baseline_without_diagnosis"], hits


def test_gate_passes_when_the_diagnosis_is_declared():
    """Positive control: 7.4 for a declared diabetic patient is a correct case, not a defect."""
    hits = gates.check_clinical_baseline_cohort(
        _raw({"HbA1c": _hba1c(7.4)}, ["T2D"]), _premise("T2D"))
    assert not [h for h in hits if h["severity"] == "gate"], hits


def test_gate_passes_on_the_fixed_baseline():
    hits = gates.check_clinical_baseline_cohort(
        _raw({"HbA1c": _hba1c(5.0)}, ["obesity"]), _premise("obesity"))
    assert not [h for h in hits if h["severity"] == "gate"], hits


def test_gate_judges_the_start_not_the_whole_course():
    """Progressing to diabetes during the course is a valid story; only starting there on
    day 0 contradicts the patient profile."""
    prog = [{"ts": 0, "value": 5.4}, {"ts": 90, "value": 6.2}, {"ts": 180, "value": 7.1}]
    hits = gates.check_clinical_baseline_cohort(
        _raw({"HbA1c": prog}, ["obesity"]), _premise("obesity"))
    assert not [h for h in hits if h["severity"] == "gate"], hits


def test_gate_reports_zero_object_rather_than_passing():
    """A stream with no registered threshold is reported as zero objects, not as a pass."""
    hits = gates.check_clinical_baseline_cohort(
        _raw({"weight": [{"ts": 0, "value": 90.0}]}, ["obesity"]), _premise("obesity"))
    assert sorted(h["kind"] for h in hits) == ["clinical_baseline_scan_empty"], hits
    assert all(h["severity"] == "warn" for h in hits)


def test_gate_reads_comorbidities_too_not_only_known_conditions():
    """`known_conditions` lists only the primary disease; comorbidities are in the premise,
    and both are read."""
    hits = gates.check_clinical_baseline_cohort(
        _raw({"HbA1c": _hba1c(7.4)}, ["obesity"]), _premise("obesity", ["T2D"]))
    assert not [h for h in hits if h["severity"] == "gate"], hits
