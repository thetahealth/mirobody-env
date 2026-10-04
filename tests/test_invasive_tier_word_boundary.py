"""Two-sided controls for `haenv_kernel/verifier.py::_test_tier` and the
`invasive_before_firstline` hard gate.

Background
----------
A tier classifier that is a pure substring match::

    if any(w in _t.lower() for w in ("ct", "mri", "mrcp", "pet", ...)):

fires on a large family of ordinary first-line labs, because `"ct"` is a
two-letter bare substring: `A(CT)H`, `Inta(ct) PTH`, `Plasma Renin A(ct)ivity`,
`fra(ct)ionated metanephrines`, `C-rea(ct)ive protein`, `Rheumatoid fa(ct)or`,
`24-hour urine colle(ct)ion`, `Thyroid fun(ct)ion tests`, `ele(ct)rolytes`,
`Ele(ct)rocardiogram`, ... The gate is non-compensatory, so a misfire fails
the entire case, and the most expensive shape is a *correct* workup being
failed: "order the first-line biochemistry, then image" trips the gate when
the biochemistry line happens to contain the letters `ct`.

The two sides below keep the classifier honest: the negative controls are real
strings that a substring rule misfires on, the positive controls are real invasive/imaging
orders (including the ones the repo's own `gatetrip_invasive` positive-control stub
emits) that must still fail.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from haenv_kernel.verifier import _test_tier  # noqa: E402


# --------------------------------------------------------------------------------------
# Negative controls -- ordinary first-line tests that must NOT be priced as imaging
# --------------------------------------------------------------------------------------
# Every string here was observed on disk being classified `imaging` by the substring rule.
NOT_IMAGING = [
    # the eight the external review named
    "Plasma Renin Activity (PRA)",
    "lactate",
    "24-hour urine collection",
    "Thyroid function tests",
    "fractionated metanephrines",
    "electrolytes",
    "fructosamine",
    "ACTH",
    # the rest of the family, all real strings from `responses.jsonl`
    "血浆ACTH",
    "ACTH兴奋试验",
    "Plasma ACTH level",
    "Intact PTH (iPTH)",
    "Serum intact PTH",
    "Intact parathyroid hormone (iPTH)",
    "C-reactive protein (CRP)",
    "Rheumatoid factor (RF)",
    "Cosyntropin (ACTH) stimulation test",
    "24-hour urine fractionated metanephrines and catecholamines",
    "Follow-up Liver Function Test (AST/ALT)",
    "Repeat fasting glucose and electrolyte panel",
    "KOH skin scraping for fungal infection",
    "Repeat Renal Panel (Creatinine, BUN, Electrolytes)",
    "Prolactin",
    "Neurological examination for tremor characteristics",
    # `pet` hides inside clinical words too
    "anti-tTG IgA for dermatitis herpetiformis",   # `her(pet)iformis`
    "appetite and satiety questionnaire",          # `ap(pet)ite`
    "petechiae count on exam",                     # `(pet)echiae`
    "competitive ELISA for insulin antibodies",    # `com(pet)itive`
]

# --------------------------------------------------------------------------------------
# Positive controls -- genuine imaging / invasive orders that MUST still be priced imaging
# --------------------------------------------------------------------------------------
IS_IMAGING = [
    # the three the task named explicitly
    "Adrenal CT scan",
    "MRI pituitary",
    "肾活检",
    # abbreviation family: boundary matching must not lose what substring matching caught
    "CT",
    "CT scan",
    "CTA of the chest",
    "HRCT chest",
    "LDCT lung cancer screening",
    "口腔全景片或CBCT评估阻生齿",
    "18F-DOPA PET/CT",
    "68Ga-DOTATATE PET-CT",
    "PET",
    "SPECT myocardial perfusion",
    "fMRI",
    "MRCP",
    "Pituitary MRI with contrast",
    # CJK, which never had a boundary problem -- must stay exactly as it was
    "腹部CT增强",
    "肾上腺CT或MRI",
    "垂体增强 MRI",
    "肾脏活检(用于狼疮肾炎病理分型)",
    "皮肤活检(必要时)",
    "上消化道内镜",
    "冠状动脉造影",
    # latin stems
    "Renal biopsy",
    "Upper endoscopy",
    "Coronary angiography",
]

VITALS = [
    "Electrocardiogram (ECG)",
    "12-lead ECG",
    "Wrist Ultrasound or Nerve Conduction Studies (if pain persists)",
    "Liver ultrasound if NAFLD suspected",
    "Hand/wrist X-ray (if inflammatory arthritis suspected)",
    "心电图",
    "甲状腺超声",
]


@pytest.mark.parametrize("text", NOT_IMAGING)
def test_negative_control_first_line_tests_are_not_imaging(text):
    """A first-line lab must never be priced as imaging just because it spells `ct`."""
    assert _test_tier(text) != "imaging", (
        f"{text!r} was priced as imaging -- this is the substring misfire coming back. "
        "The gate that reads this tier is non-compensatory: one hit fails the whole case."
    )


@pytest.mark.parametrize("text", IS_IMAGING)
def test_positive_control_real_imaging_still_priced_imaging(text):
    """A real imaging / invasive order must still be priced imaging."""
    assert _test_tier(text) == "imaging", (
        f"{text!r} stopped being priced as imaging -- the fix over-shot and the gate can no "
        "longer see the thing it exists to catch."
    )


@pytest.mark.parametrize("text", VITALS)
def test_bedside_tests_are_vitals(text):
    assert _test_tier(text) == "vitals", f"{text!r} was not priced as a bedside/vitals test"


def test_cjk_is_not_broken_by_word_boundaries():
    """The trap a naive `\\b` fix falls into.

    `\\b` is defined against `\\w`, and under Unicode `\\w` includes CJK -- so `\\bCT\\b`
    finds no boundary between `部` and `C`, and `腹部CT增强` silently stops being imaging.
    The Chinese side never had a substring problem; a fix that breaks it is a net loss.
    """
    import re
    naive = re.compile(r"\bct\b")
    assert not naive.search("腹部ct增强"), (
        "premise of this test is stale: `\\b` now matches at a CJK/ASCII transition"
    )
    assert _test_tier("腹部CT增强") == "imaging"
    assert _test_tier("肾上腺CT平扫+增强") == "imaging"


# --------------------------------------------------------------------------------------
# Gate-level controls: the tier feeds `invasive_before_firstline`
# --------------------------------------------------------------------------------------
def _gate_fires(tests: list[str]) -> bool:
    """Mirror of the gate's own condition, so the controls below read at gate level."""
    if len(tests) < 2:
        return False
    kinds = [_test_tier(t) for t in tests]
    return "imaging" in kinds and not any(k != "imaging" for k in kinds)


def test_gate_positive_control_imaging_only_proposal_still_fails():
    """`gatetrip_invasive` (haenv/baselines.py) emits exactly this -- it must keep hitting,
    or the repo's only positive control for this gate goes silent."""
    assert _gate_fires(["腹部CT增强", "MRCP"])
    assert _gate_fires(["Adrenal CT scan", "MRI pituitary"])
    assert _gate_fires(["肾脏活检(用于狼疮肾炎病理分型)", "皮肤活检(必要时)"])
    assert _gate_fires(["下肢肌肉骨骼超声或MRI(若疼痛持续或加重)", "口腔全景片或CBCT评估阻生齿"])


def test_gate_negative_control_correct_workup_no_longer_fails():
    """Biochemistry first, imaging after -- the textbook-correct pheochromocytoma and
    primary-aldosteronism workups. These were real `FAIL(gate)` rows on disk."""
    assert not _gate_fires([
        "Plasma free metanephrines or 24-hour urine fractionated metanephrines and catecholamines",
        "Adrenal CT/MRI",
    ])
    assert not _gate_fires([
        "Renin and Aldosterone levels (Plasma Renin Activity and Plasma Aldosterone Concentration)",
        "Urinary Sodium and Potassium excretion (24-hour urine collection)",
        "Adrenal CT scan (if RAAS ratio is abnormal)",
    ])


def test_gate_negative_control_all_lab_proposal_never_fires():
    """Proposals with no imaging in them at all, which the substring rule failed anyway."""
    assert not _gate_fires([
        "Morning serum cortisol and plasma ACTH",
        "ACTH stimulation test (Cosyntropin test)",
        "Plasma renin activity and aldosterone",
    ])
    assert not _gate_fires([
        "Follow-up Liver Function Test (AST/ALT)",
        "Wrist Ultrasound or Nerve Conduction Studies (if pain persists)",
    ])
    assert not _gate_fires([
        "Repeat Thyroid Function Tests (Free T4, TSH)",
        "Repeat Renal Panel (Creatinine, BUN, Electrolytes)",
    ])


def test_single_test_proposal_is_never_failed():
    """Unchanged behaviour: ordering only an imaging test can be the correct first move."""
    assert not _gate_fires(["Adrenal CT scan"])
    assert not _gate_fires(["腹部CT增强"])
