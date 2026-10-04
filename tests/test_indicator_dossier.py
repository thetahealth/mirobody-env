"""Invariants of the indicator dossiers.

Design: `docs/design/indicator-dossier.md`

## The structure alone does not fix anything -- the reconciliation does

A dossier gathers attributes that would otherwise live in 3-6 places into one, but what catches
defects is that the attributes have to agree with each other:

| | Invariant | Defect it catches |
|---|---|---|
| INV-1 | for a disease in `diagnostic_for`, `cohorts[disease].median` is on the diagnostic side | TG 1.5 below the 1.7 threshold |
| INV-3 | the `physiological/payload` upper bound is >= `reference_range.high` | "a bound is what is possible, not what is normal" |
| INV-4 | every value has `source:`; `review: pending` reaches the report | a fake citation is worse than `pending` |
| INV-6 | every clinical indicator in a case has a dossier (fail-closed) | guessing by name, like the `ct` substring |

## An exemption is an assertion that expires

The dossiers carry values over unchanged, so INV-1 fails on `triglycerides/dyslipidemia`.
The check is not loosened: that cohort is exempted with a written reason (`known_defect`),
and a reverse control requires the exemption to be removed the day the cohort complies.
Otherwise "registered, reasoned, never fires" becomes the next defect.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import indicators as I  # noqa: E402

D = I.dossiers()


def _violates_inv1(name, dossier) -> list[str]:
    """INV-1: when a marker is a diagnostic basis of a disease, that disease's cohort median
    is on the diagnostic side."""
    out = []
    for disease, rule in (dossier.get("diagnostic_for") or {}).items():
        coh = (dossier.get("cohorts") or {}).get(disease)
        if coh is None:
            continue                       # no cohort of this disease for the marker: no object
        ge = rule.get("ge")
        if ge is None:
            continue
        if float(coh["median"]) < float(ge):
            out.append(f"{name}/{disease}: 中位 {coh['median']} < 诊断阈值 {ge}")
    return out


# ─────────────────────────────────────────────── INV-1 diagnostic consistency

def test_inv1_diagnostic_cohorts_are_on_the_diagnostic_side():
    """A patient "diagnosed with X" is past X's diagnostic line in the case.

    This is consistency by definition: dyslipidemia is diagnosed from lipid values, and a
    "dyslipidemia patient" with normal TG contradicts itself.
    """
    bad, exempt = [], []
    for n, d in D.items():
        for v in _violates_inv1(n, d):
            coh_name = v.split("/")[1].split(":")[0]
            if (d["cohorts"][coh_name].get("known_defect")):
                exempt.append(v)
            else:
                bad.append(v)
    assert not bad, ("这些队列在自己的诊断阈值下面(且没登记 known_defect):\n  "
                     + "\n  ".join(bad))
    # There are exemptions -- state them rather than let them pass silently
    assert exempt, "一条豁免都没有 ⇒ 要么真的全合规(那就把本断言改掉),要么 INV-1 失去对象"


def test_inv1_exemptions_are_still_violating():
    """Reverse control: a cohort registered with `known_defect` really still violates.

    An exemption expires. A fixed cohort that keeps its exemption becomes "registered,
    reasoned, never fires".
    """
    stale = []
    for n, d in D.items():
        viol = {v.split("/")[1].split(":")[0] for v in _violates_inv1(n, d)}
        for coh, c in (d.get("cohorts") or {}).items():
            if not c.get("known_defect"):
                continue
            # `known_defect` can also record a defect other than INV-1 (the ALT distribution,
            # for one); only "claims to violate INV-1 but no longer does" is checked here.
            if coh in (d.get("diagnostic_for") or {}) and coh not in viol:
                stale.append(f"{n}/{coh}")
    assert not stale, f"这些 known_defect 已经不违反 INV-1 了,豁免该摘掉:{stale}"


def test_inv1_would_catch_a_regression():
    """Negative control: a compliant cohort pushed below the threshold is caught by INV-1."""
    probe = {"diagnostic_for": {"T2D": {"ge": 6.5}},
             "cohorts": {"T2D": {"median": 5.0, "p_abnormal": None, "source": "x"}}}
    assert _violates_inv1("PROBE", probe), "INV-1 抓不到人造违规 ⇒ 它是恒真的"
    probe["cohorts"]["T2D"]["median"] = 7.4
    assert not _violates_inv1("PROBE", probe), "INV-1 把合规的也判了 ⇒ 会被人关掉"


# ─────────────────────────────────────────────── INV-3 bound order

def test_inv3_payload_range_covers_the_reference_high_side():
    """The case value range's upper bound is >= the reference upper bound; otherwise an
    abnormal value cannot be expressed in a case.

    Only the upper bound is checked. Plain containment (`payload.low <= ref.low`) would fire
    on `ALT`: its `ref.low = 0` is a notation for "anything below the upper bound is
    normal", not "0 is reachable", and a physiological lower bound of 5.0 above it is right.
    The invariant is stated in the form in which it is true, not the symmetric-looking one.
    """
    bad = []
    for n, d in D.items():
        ref, pay = d.get("reference_range") or {}, d.get("payload_range") or {}
        if ref.get("high") is None or pay.get("high") is None:
            continue
        if float(pay["high"]) < float(ref["high"]):
            bad.append(f"{n}: 题面上界 {pay['high']} < 参考上界 {ref['high']}")
    assert not bad, "\n  ".join(["异常值在题面上不可表达:"] + bad)


def test_inv3_has_objects():
    """The check above has objects: reference ranges cover at least 5 labs."""
    n = sum(1 for d in D.values()
            if (d.get("reference_range") or {}).get("high") is not None
            and (d.get("payload_range") or {}).get("high") is not None)
    assert n >= 5, f"INV-3 只有 {n} 个对象 —— 扫描面塌了"


# ─────────────────────────────────────────────── INV-4 provenance

def test_inv4_every_cohort_has_a_source():
    """Enforced at load time as well: a missing source is a decision nobody made."""
    for n, d in D.items():
        for coh, c in (d["cohorts"]).items():
            assert str(c.get("source") or "").strip(), f"{n}/{coh} 没有 source"


def test_inv4_pending_is_visible():
    """`review: pending` is readable -- it is the only disclaimer those values have."""
    p = I.pending_cohorts()
    assert p, "一条 pending 都没有 ⇒ 要么全复核过了(那该改断言),要么这条通路断了"
    assert "HbA1c/obesity" in p


def test_known_defects_are_enumerable_not_remembered():
    """Values carried over unchanged that are known to need a change are recorded in the
    dossier, not remembered."""
    kd = I.known_defects()
    assert "triglycerides/dyslipidemia" in kd, "TG 越阈那条没登记进 known_defect"
    assert "ALT/MASLD" in kd, "MASLD 的 ALT 分布缺陷没登记"
    for k, v in kd.items():
        assert len(v) > 40, f"{k} 的 known_defect 说明太短,写不清缺陷是什么"


# ─────────────────────────────────────────────── INV-6 coverage, fail-closed

def test_inv6_every_clinical_spec_signal_has_a_dossier():
    """Every clinical indicator a case carries has a dossier; a missing one is a place where
    the value is guessed by name."""
    from haenv.build import CLINICAL_SPEC
    missing = sorted(set(CLINICAL_SPEC) - set(D))
    assert not missing, f"这些信号没有档案:{missing}"


def test_unregistered_indicator_raises_instead_of_defaulting():
    with pytest.raises(I.IndicatorUnregistered):
        I.of("从来没建过档的指标")


def test_unknown_cohort_raises_instead_of_falling_back():
    """An unknown cohort raises. A fallback would make "this disease has no dossier" look the
    same as "the baseline happens to equal the default cohort".

    The fixture uses a disease name that by definition is never registered: the property
    under test is "raise, never fall back", not the absence of one particular cell, and a
    real cell can be filled in at any time -- which would silently turn this negative
    control green.
    """
    with pytest.raises(I.IndicatorUnregistered, match="cohort"):
        I.baseline("HbA1c", "__never_registered_disease__")


# ─────────────────────────────────────────────── the dossier is the only source of the values

def test_clinical_spec_no_longer_carries_base_or_ndigits():
    """A structural guarantee: `CLINICAL_SPEC` entries are pairs `(per_kg, step)`.

    ## Why this is stronger than "the two copies must agree"

    With a `base` in the tuple as well as in the dossier there are two copies and one
    reconciliation, and the reconciliation only finds a divergence when someone runs it.
    Without the copy, a divergence cannot happen.

    A dead slot left in place makes "still used" look the same as "not deleted yet".
    """
    from haenv.build import CLINICAL_SPEC
    assert CLINICAL_SPEC, "空表 ⇒ 本条失去对象"
    for sig, tup in CLINICAL_SPEC.items():
        assert len(tup) == 2, f"{sig}: 元组还是 {len(tup)} 元 —— base/ndigits 没删干净"
        per_kg, step = tup
        assert isinstance(step, int) and step > 0, f"{sig}: step={step!r}"
        assert per_kg != 0, f"{sig}: per_kg 为 0 ⇒ 该化验与体重脱钩,GEN15 判不了方向"


def test_ndigits_has_exactly_one_declaration_site():
    """INV-5: precision (`ndigits`) is declared in exactly one place, the dossier.

    With `ndigits` spread over `events.METRICS`, `build.CLINICAL_SPEC` and
    `events._NON_METRIC_NDIGITS`, precision declared in one place can be overridden in
    another and put spurious decimal places on the numbers in cases.
    """
    from haenv import events as E
    from haenv.build import CLINICAL_SPEC
    assert E._NON_METRIC_NDIGITS == {}, (
        f"第三处声明又回来了:{E._NON_METRIC_NDIGITS}")
    assert all(len(t) == 2 for t in CLINICAL_SPEC.values()), "第二处声明(元组第4槽)还在"
    # Reads go through the dossier: a stream without a dossier returns None (no guess), and
    # one with a dossier matches it digit for digit.
    # `ndigits` may be `null` for `kind: panel` items: they go through the text lab channel,
    # where `findings_render._round` sets the precision by value. `null` maps to `None` as
    # well (leave it alone), the same result as "no dossier" for a different reason: this
    # stream's precision is not the dossier's to set, rather than nobody declared it.
    n_null = 0
    for name, d in D.items():
        nd = d["ndigits"]
        if nd is None:
            n_null += 1
            assert d.get("kind") == "panel", (
                f"{name}:ndigits 为 null 却不是 panel —— 只有文本通道的项可以不声明位数")
        assert E._declared_ndigits(name) == (None if nd is None else int(nd)), name
    # Since 2026-09-30 (realism P1-4) every panel item declares its printed precision,
    # read by `findings_render._round`; `null` stays legal (magnitude rounding) but unused.
    n_panel = sum(1 for d in D.values() if d.get("kind") == "panel")
    assert n_panel >= 5, f"kind: panel 的只有 {n_panel} 条 —— 这一支的扫描面塌了"
    from haenv import findings_render as _FR
    for name, d in D.items():
        if d.get("kind") == "panel" and d["ndigits"] is not None:
            assert _FR._ndigits(name) == int(d["ndigits"]), name
    assert E._declared_ndigits("从来没建过档的流") is None


def test_metrics_ndigits_agrees_with_the_dossier():
    """The `ndigits` field of `METRICS` (a dataclass slot) agrees with the dossier.

    It is not a second declaration: rendering reads `_declared_ndigits`. This catches a
    stream left out of the dossier and an edit made on one side only.
    """
    from haenv import events as E
    bad = [m.name for m in E.METRICS
           if m.name in D and int(m.ndigits) != int(D[m.name]["ndigits"])]
    assert not bad, f"这些流的 METRICS.ndigits 与档案分叉:{bad}"
    uncovered = [m.name for m in E.METRICS if m.name not in D]
    assert not uncovered, f"这些 METRICS 流没建档 ⇒ 精度又只剩一处私有声明:{uncovered}"


def test_dossier_is_the_only_source_of_the_baseline():
    """The code does not read a baseline from `CLINICAL_SPEC`; the dossier is the source."""
    import ast
    sys.path.insert(0, str(ROOT / "tests"))
    from _module_family import family_source
    src = family_source("haenv/build.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "CLINICAL_SPEC"):
            # Unpacking the tuple is fine; taking [0] out as the baseline is not.
            if isinstance(node.slice, ast.Constant) and node.slice.value == 0:
                pytest.fail(f"build.py:{node.lineno} 直接取 CLINICAL_SPEC[...][0] 当基线")
    # Baselines come from `sample_case` (the whole set conditioned on the umbrella diagnosis),
    # not from a per-signal `baseline`
    assert "_indicators.sample_case(" in src, "build.py 没有走档案取基线"


def test_calibration_provenance_is_a_structured_field_not_prose():
    """The calibration marker is a structured field, not a substring of the `source` prose.

    Selecting the scan surface by a substring of the source text breaks as soon as the
    sources are reworded. A locator in prose is the fastest kind to rot.
    """
    marked = [f"{n}/{coh}" for n, d in D.items()
              for coh, c in (d.get("cohorts") or {}).items()
              if c.get("calibration") == "real_emr_671_diabetes"]
    assert len(marked) >= 3, f"带校准标记的队列只有 {len(marked)} 个:{marked}"
    # The other side: the marker agrees with the source; a marker with no source is a fake one
    for key in marked:
        n, coh = key.split("/")
        src = str(D[n]["cohorts"][coh].get("source") or "")
        assert "重算" in src or "calibrate_from_real_emr" in src, f"{key} 标了校准但出处对不上"


def test_recomputed_rates_are_recorded_with_the_gap():
    """A recomputed rate records how far it is from the lab report's own flags."""
    # `fasting_glucose` is +0.193 (urine glucose rows excluded).
    gaps = {"fasting_glucose": "+0.193", "AST": "−0.107"}
    for sig, gap in gaps.items():
        coh = next(iter(D[sig]["cohorts"]))
        src = str(D[sig]["cohorts"][coh].get("source") or "")
        assert gap in src, f"{sig} 的出处里没写出与化验单自带标记的差 {gap}"


# ─────────────────────────────────────────────── INV-8 unit consistency
#
# Catches one cell of a quantity written in a different unit. The same family in Synthea:
# `VitalSign.BLOOD_GLUCOSE` holding HbA1c, `VitalSign.EGFR` holding creatinine clearance.
#
# The check reads only threshold-class declarations (reference bounds, diagnostic
# thresholds, cohort medians), not `payload_range`, which is wide by design (ALT 5..300 is
# 60x) and would make the check always pass.

def test_inv8_whole_table_passes():
    """Positive control: every current dossier passes. A false alarm costs more than a miss."""
    bad = I.check_unit_consistency()
    assert not bad, (
        "阈值类声明跨了一个量级以上,多半是某一格写成了另一个单位:\n  "
        + "\n  ".join(f"{n}: {r:.1f}x  {v}" for n, (r, v) in bad.items()))


def test_inv8_band_sits_between_the_measured_max_and_the_smallest_slip():
    """The band is measured, not chosen: above the widest current dossier, below the smallest
    unit slip.

    This turns red when someone widens the band or adds a wider dossier -- exactly when it
    should speak.
    """
    widest = max((r for r in (I.unit_slip_of(n) for n in I.dossiers()) if r), 
                 key=lambda t: t[0])[0]
    assert widest < I._UNIT_SLIP_BAND, (
        f"现有最宽 {widest:.2f}x 已经够到带宽 {I._UNIT_SLIP_BAND} —— 判据要误伤了")
    # The smallest mg/dL -> mmol/L factor: glucose 18.0 (cholesterol 38.67 and TG 88.5 are larger)
    assert I._UNIT_SLIP_BAND < 18.0, (
        f"带宽 {I._UNIT_SLIP_BAND} ≥ 18.0 ⇒ 连葡萄糖的 mg/dL 滑移都抓不住,判据恒真")


def test_inv8_catches_a_mgdl_slip():
    """Negative control: a diagnostic threshold written in mg/dL (x18) is caught.

    The probe is synthetic and bound to no real dossier: a negative control bound to a real
    entry loses its object when that entry is fixed, and still "passes".
    """
    probe = {"unit": "mmol/L",
             "reference_range": {"low": None, "high": 5.6},
             "cohorts": {"T2D": {"median": 8.2}},
             "diagnostic_for": {"T2D": {"ge": 126.0}}}     # 7.0 mmol/L written as 126 mg/dL
    r = I.violates_inv8(probe)
    assert r is not None and r > 18.0, f"INV-8 抓不到 mg/dL 滑移(比值 {r})⇒ 它是恒真的"


def test_inv8_does_not_fire_on_a_clean_probe():
    """Reverse control: the same probe with the right unit is not caught.

    A check tested only for "catches it" and never for "leaves the clean case alone" stays
    green with its constant set to 1.0.
    """
    probe = {"unit": "mmol/L",
             "reference_range": {"low": None, "high": 5.6},
             "cohorts": {"T2D": {"median": 8.2}},
             "diagnostic_for": {"T2D": {"ge": 7.0}}}
    assert I.violates_inv8(probe) is None


def test_inv8_ignores_payload_range():
    """`payload_range` stays out of the check: with it in, the band would have to exceed 60 and
    would miss an 18x slip."""
    probe = {"unit": "U/L",
             "reference_range": {"low": None, "high": 40.0},
             "payload_range": {"low": 5.0, "high": 300.0},   # 60x, valid
             "cohorts": {"MASLD": {"median": 30.0}}}
    assert I.violates_inv8(probe) is None, "payload_range 漏进了阈值类"
    assert not any("payload" in k for k in I.threshold_class_of(probe))


def test_inv8_has_objects():
    """The scan surface holds: at least 8 dossiers have two or more positive threshold-class
    values, so the check is looking at something."""
    n = sum(1 for x in I.dossiers() if I.unit_slip_of(x) is not None)
    assert n >= 8, f"INV-8 只有 {n} 个对象 —— 扫描面塌了"


def test_inv8_is_necessary_not_sufficient_and_says_so():
    """The check's boundary is written in its own docstring.

    INV-8 catches order-of-magnitude slips, not a same-magnitude mix-up: eGFR and creatinine
    clearance are both ml/min, and the Synthea case passes. Without saying so, someone takes
    it as "units verified". (INV-7 has the same kind of boundary: with blood pressure, the
    systolic value is refused while the diastolic one passes.)
    """
    # The check is bound to "is the boundary written", not to the language it is written in:
    # both wordings count, and the `eGFR` example, which is language-independent, is required.
    doc = (I.unit_slip_of.__doc__ or "").lower()
    _says_boundary = ("必要不充分" in doc
                      or ("necessary" in doc and "sufficient" in doc))
    assert _says_boundary and "egfr" in doc, "INV-8 没有写清自己抓不到什么"


def test_inv8_does_not_judge_a_wide_reference_interval():
    """Negative control: a wide reference interval on its own is not judged, rather than failed.

    Comparing all threshold-class values with max/min fires on `TSH`, whose reference
    interval 0.27-4.2 spans 15.6x. That is a property of the analyte (TSH is roughly
    log-distributed), not a unit slip. So the reference interval is the anchor that other
    values are compared with; `TSH` has no other value, which makes it zero objects.
    """
    wide = {"unit": "mIU/L", "reference_range": {"low": 0.27, "high": 4.2},
            "cohorts": {}, "diagnostic_for": {}}
    assert I.violates_inv8(wide) is None, "参考区间宽被当成了单位滑移 —— 判据会误伤"
    # The registered TSH dossier is not judged either
    assert I.violates_inv8(I.of("TSH")) is None


def test_inv8_catches_a_slip_on_the_low_side_too():
    """Negative control: a low-side slip is caught too; a high-side-only check is blind on
    `CGM_TIR` and its kind."""
    probe = {"unit": "%", "reference_range": {"low": 70.0, "high": None},
             "cohorts": {"T2D": {"median": 3.0}},          # 70% written as a fraction, and a digit lost
             "diagnostic_for": {}}
    assert (I.violates_inv8(probe) or 0) > 10.0, "低侧滑移抓不到"


def test_inv8_has_exactly_one_computation_core():
    """Both entry points (`unit_slip_of` / `violates_inv8`) use one computation core, so they
    cannot drift apart."""
    import inspect
    for fn in (I.unit_slip_of, I.violates_inv8):
        assert "_worst_deviation" in inspect.getsource(fn), f"{fn.__name__} 自己算了一遍"


def test_panel_items_are_registered_with_empty_cohorts():
    """All 16 routine-panel items have a dossier; the `panel` items have empty `cohorts`."""
    panel = ["fasting_glucose", "HbA1c", "LDL", "triglycerides", "ALT", "AST",
             "TC", "HDL", "Alb", "Ca", "Cr", "Hb", "K", "Na", "Cl", "TSH"]
    missing = [p for p in panel if p not in D]
    assert not missing, f"§C5 面板项没建档:{missing}"
    kinds = {p: D[p].get("kind", "clinical") for p in panel}
    for p, k in kinds.items():
        if k == "panel":
            assert D[p]["cohorts"] == {}, f"{p} 是 panel 却有队列 —— 时序通道不产它"
            # 2026-09-30 (realism P1-4): panel items declare the printed precision.
            assert isinstance(D[p]["ndigits"], int), f"{p} 是 panel 却没声明 ndigits"


def test_panel_reference_ranges_are_not_self_authored():
    """No panel item's reference range is authored here: each traces to the upstream
    indicator table.

    (A self-authored range would carry `review: pending`, which reaches the report and
    needs a clinician's review.)
    """
    for n, d in D.items():
        if d.get("kind") != "panel":
            continue
        src = str((d.get("reference_range") or {}).get("source") or "")
        assert "upstream:" in src, f"{n} 的参考区间不是上游来的:{src!r}"


def test_panel_orphans_are_named_not_hidden():
    """The four panel items no relation constrains are named, not covered by a made-up relation."""
    orph = {n for n, d in D.items()
            if d.get("kind") == "panel" and "orphan" in str(d.get("relation_role") or "")}
    assert orph == {"Cr", "Hb", "K", "TSH"}, f"孤儿名单与 §C5 对不上:{sorted(orph)}"
