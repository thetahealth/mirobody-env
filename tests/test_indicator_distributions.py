"""Indicator dossiers: constant -> distribution.

Design: `docs/design/indicator-dossier.md` (Section 2, INV-2)

## Why a distribution

A single median per cohort is still a constant. Ma X et al. (PMC6961232,
11 studies, 4084 cases) report that **25%** of NAFLD patients have normal
ALT, so a constant of 21.0 (normal) puts **every** synthetic patient into
that minority. A constant cannot express a distribution.

## Load-bearing tests

- `test_sigma_is_solved_not_invented` -- the spread is pinned by solving two
  simultaneous **published** quantities;
- `test_umbrella_diagnosis_always_shows_at_least_one_abnormal` -- an
  umbrella diagnosis is a **definitional** requirement, it must not depend
  on chance;
- `test_pair_constraint_kills_the_absurd_de_ritis` -- AST and ALT are
  sampled jointly so the De Ritis ratio stays inside its registered envelope.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import math
import pathlib
import statistics
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import indicators as I  # noqa: E402

N = 2000
IDS = [f"PROBE-{i:05d}" for i in range(N)]


# ─────────────────────────────────────────────── the spread is solved, not made up

def test_sigma_is_solved_not_invented():
    """Sigma is **pinned by solving simultaneously** from either
    (median, abnormal rate) or (mean, sd) -- there is no free parameter.

    A constant baseline has one degree of freedom; a distribution needs two,
    and the second one is usually not available -- which makes it tempting
    to just invent an sd. Both parameterizations here consume only
    **published** quantities.
    """
    med, sd = I.moments_of("HbA1c", "T2D")
    d = I.of("HbA1c")
    hi = d["reference_range"]["high"]
    p = d["cohorts"]["T2D"]["p_abnormal"]
    # Back-substitution: the abnormal rate computed from the solved sigma must
    # return to the registered p
    got = 1.0 - statistics.NormalDist().cdf(math.log(hi / med) / sd)
    assert abs(got - p) < 1e-6, f"σ 解错了:回算异常率 {got:.4f} vs 登记 {p}"


def test_mean_sd_parameterisation_round_trips():
    """The `(mean, sd)` path -- this is the one `ALT/MASLD` uses."""
    med, sig = I.moments_of("ALT", "MASLD")
    c = I.of("ALT")["cohorts"]["MASLD"]
    mean_back = med * math.exp(sig * sig / 2.0)
    sd_back = mean_back * math.sqrt(math.exp(sig * sig) - 1.0)
    assert abs(mean_back - c["mean"]) < 1e-6
    assert abs(sd_back - c["sd"]) < 1e-6


def test_underdetermined_raises_instead_of_inventing_a_spread():
    """Raise when the three numbers are incomplete. Filling in a
    default spread would be inventing the one free parameter this system
    can actually fabricate."""
    with pytest.raises(I.DistributionUnderdetermined):
        I.sigma_of("CGM_TIR", "T2D")        # neither ref.high nor p_abnormal is registered


def test_distribution_status_tells_constant_from_distribution():
    """"the distribution is built" and "the three numbers are
    incomplete so it's still a constant" look identical on the item's
    surface."""
    st = I.distribution_status()
    assert st["ALT/MASLD"].startswith("分布"), st["ALT/MASLD"]
    assert st["CGM_TIR/T2D"].startswith("仅中位数"), st["CGM_TIR/T2D"]


# ─────────────────────────────────────────────── INV-2: the emitted values really produce the declared rate

def test_inv2_emitted_rate_matches_the_declared_one():
    """**INV-2**: once `p_abnormal` is given, the emitted values
    must actually produce that rate (for a single indicator with no pair
    constraint).

    `LDL/dyslipidemia` is not governed by a pair constraint => its marginal
    is not clipped, so it can be checked directly.
    """
    d = I.of("LDL")
    p = d["cohorts"]["dyslipidemia"]["p_abnormal"]
    hi = d["reference_range"]["high"]
    vals = [I.sample_baseline("LDL", "dyslipidemia", (), cid) for cid in IDS]
    got = sum(1 for v in vals if v > hi) / len(vals)
    assert abs(got - p) < 0.04, f"实测异常率 {got:.3f} vs 登记 {p}"


def test_the_masld_constant_defect_is_gone():
    """A constant ALT of 21.0 would freeze the abnormal rate at **0.0%**
    (every patient in the normal minority); the sampled distribution does not."""
    hi = I.of("ALT")["reference_range"]["high"]
    assert 21.0 < hi, "负对照前提变了:旧常数本该是正常值"
    vals = [I.sample_baseline("ALT", "MASLD", (), cid) for cid in IDS]
    got = sum(1 for v in vals if v > hi) / len(vals)
    assert got > 0.15, f"ALT 异常率仍只有 {got:.1%} —— 分布没生效"


# ─────────────────────────────────────────────── umbrella diagnosis

def test_umbrella_diagnosis_always_shows_at_least_one_abnormal():
    """**load bearing**: a patient diagnosed with dyslipidemia must
    show at least one abnormal lipid value in the prompt.

    That is a **definitional** fact (ACC/AHA: dyslipidemia is diagnosed by
    these very values), not a matter of probability.
    """
    sig = ["LDL", "triglycerides"]
    bad = []
    for cid in IDS[:500]:
        v = I.sample_case("dyslipidemia", sig, (), cid)
        if not any(I.abnormal_side(n, v[n]) for n in sig):
            bad.append((cid, v))
    assert not bad, f"{len(bad)} 例确诊血脂异常却一条血脂都不异常,例:{bad[:2]}"


def test_independent_sampling_would_have_failed():
    """**negative control**: when each signal is sampled
    independently, roughly half the patients end up with no abnormal value
    at all.

    Without this test, the one above might just be holding by coincidence.
    """
    sig = ["LDL", "triglycerides"]
    ok = sum(1 for cid in IDS[:500]
             if any(I.abnormal_side(n, I.sample_baseline(n, "dyslipidemia", (), cid))
                    for n in sig))
    frac = ok / 500
    assert frac < 0.75, (
        f"独立抽也有 {frac:.1%} 满足 ⇒ 条件化没有对象,上一条是恒真的")


def test_no_umbrella_means_no_extra_condition():
    """When the disease is not in `diagnoses`, `sample_case` degenerates to
    independent per-signal sampling, identical value for value."""
    a = I.sample_case("MASLD", ["FIB4"], (), "X-1")
    b = {"FIB4": I.sample_baseline("FIB4", "MASLD", (), "X-1")}
    assert a == b


# ─────────────────────────────────────────────── pair constraint

def _de_ritis(cid):
    v = I.sample_case("MASLD", ["ALT", "AST", "FIB4"], (), cid)
    return v["AST"] / v["ALT"]


def test_pair_constraint_kills_the_absurd_de_ritis():
    """With AST and ALT sampled independently, about 32% of cases land the
    De Ritis ratio outside 0.3-3 (p5 0.11 / p95 4.92). The pair constraint
    keeps it inside the registered envelope (`known_defect` note on the AST
    dossier entry: change neither side without the other).
    """
    r = sorted(_de_ritis(cid) for cid in IDS[:500])
    absurd = sum(1 for x in r if x < 0.3 or x > 3.0) / len(r)
    assert absurd < 0.01, f"仍有 {absurd:.1%} 的 De Ritis 离谱(修前 32.1%)"
    c = I.pair_constraints()["de_ritis"]
    assert all(c["low"] <= x <= c["high"] for x in r), "有值越出登记的包络"


def test_de_ritis_centre_matches_nafld_reading():
    """Early NAFLD **typically <1** (ALT dominant) -- the median must fall
    on that side."""
    r = [_de_ritis(cid) for cid in IDS[:500]]
    assert statistics.median(r) < 1.0, f"中位 {statistics.median(r):.2f},与 NAFLD 读法不符"


def test_the_pair_constraint_shifts_the_marginal_and_that_is_reported():
    """Counter-metric: the pair constraint clips the joint distribution, so
    **the marginal moves**.

    The ALT abnormal rate drops from a declared 35.9% to roughly 27.5%. The
    test bounds how far it moves (less than 0.15) so a constraint that
    silently changes a second quantity is caught.
    """
    hi = I.of("ALT")["reference_range"]["high"]
    med, sig = I.moments_of("ALT", "MASLD")
    declared = 1.0 - statistics.NormalDist().cdf(math.log(hi / med) / sig)
    got = sum(1 for cid in IDS[:800]
              if I.sample_case("MASLD", ["ALT", "AST", "FIB4"], (), cid)["ALT"] > hi) / 800
    assert got < declared, "成对约束没有截断边际 ⇒ 它没被行使"
    assert declared - got < 0.15, (
        f"边际被拉走 {declared - got:.3f}(登记 {declared:.3f} → 实测 {got:.3f})"
        " —— 两条的边际参数在互相打架,该改的是边际不是约束")


# ─────────────────────────────────────────────── determinism

def test_sampling_is_a_pure_function_of_the_case_id():
    """The same case_id always yields the same set of values -- otherwise
    `--rebuild` and a recompute would produce different prompts."""
    for cid in IDS[:20]:
        assert (I.sample_case("MASLD", ["ALT", "AST"], (), cid)
                == I.sample_case("MASLD", ["ALT", "AST"], (), cid))


def test_different_cases_get_different_values():
    """Sampling must not collapse to a constant."""
    v = [I.sample_baseline("ALT", "MASLD", (), cid) for cid in IDS[:200]]
    assert len(set(v)) > 190, f"200 例只有 {len(set(v))} 个不同值"


# ═══════════════════════════════ direction awareness and INV-7

def test_cgm_tir_direction_is_inverted():
    """`CGM_TIR` is "the fraction of time glucose is in the target
    range" -- **higher is better**.

    International consensus (Diabetes Care 2019;42:1593): target range
    3.9-10.0 mmol/L, **TIR >= 70%**. => abnormal means **below** 70, the
    opposite of a lab-test indicator. Judging it by a lab test's default
    direction would call patients at goal abnormal and patients off goal
    normal -- flipping the conclusion entirely.
    """
    assert I.of("CGM_TIR")["direction"] == "lower_abnormal"
    assert I.abnormal_side("CGM_TIR", 54.0) is True      # off goal
    assert I.abnormal_side("CGM_TIR", 85.0) is False     # at goal


def test_default_direction_is_higher_abnormal():
    """The flip-side control: do not invert direction across the
    board -- lab-test indicators must still be judged as "above the upper
    bound is abnormal.\""""
    for n in ("ALT", "LDL", "triglycerides", "HbA1c"):
        assert I.of(n).get("direction", "higher_abnormal") == "higher_abnormal", n
    assert I.abnormal_side("ALT", 60.0) is True
    assert I.abnormal_side("ALT", 20.0) is False


def test_inv7_refuses_a_distribution_that_escapes_the_payload_range():
    """**INV-7**: the solved distribution's p5/p95 must fall inside
    `payload_range`.

    ## What this catches

    `p_abnormal`'s original source takes **the source data's own abnormal
    flag from the lab report** (`a not in ("N","n","正常","0")`) -- which
    uses **the issuing lab's own reference range**, not this repo's
    `findings.yaml` bound. When the two disagree, sigma solves to an absurd
    value.

    ## The negative control uses a **synthetic probe**, not live data

    A negative control tied to a real registry entry would lose its target
    as soon as that entry is consistent. So the control uses a constructed
    probe whose abnormal rate does not fit its payload range.
    """
    probe = {"unit": "x", "ndigits": 2,
             "reference_range": {"low": None, "high": 5.6},
             "payload_range": {"low": 3.0, "high": 25.0},
             "diagnostic_for": {}, "kind": "clinical",
             "cohorts": {"C": {"median": 8.2, "p_abnormal": 0.642, "source": "合成探针"}}}
    saved = I.dossiers().get("__INV7_PROBE__")
    I.dossiers()["__INV7_PROBE__"] = probe
    try:
        with pytest.raises(I.DistributionUnderdetermined, match="does not fit the payload range"):
            I.moments_of("__INV7_PROBE__", "C")
        # Change the bound to one that is self-consistent with the abnormal rate
        # => this must now pass (otherwise the gate is an unconditional reject)
        probe["cohorts"]["C"]["p_abnormal"] = 0.823
        med, sig = I.moments_of("__INV7_PROBE__", "C")
        assert 0.3 < sig < 0.6, sig
    finally:
        if saved is None:
            I.dossiers().pop("__INV7_PROBE__", None)
        else:
            I.dossiers()["__INV7_PROBE__"] = saved


def test_inv7_caught_a_real_instance_and_it_got_fixed():
    """`fasting_glucose/T2D` registers the abnormal rate recomputed against
    this repo's bound (ADA 5.6), 0.833; the lab report's own flag gives 0.640.
    The registered value is aligned to the repo bound, and sigma solves to a
    sensible value.

    The extraction path matters: a value extractor whose regex only anchors
    the start of the string reads `"4+"` as 4.0 and mixes semi-quantitative
    urinalysis glucose readings into the blood-glucose values (mixed, the rate
    is 0.736; clean, 0.833).
    """
    c = I.of("fasting_glucose")["cohorts"]["T2D"]
    assert c["p_abnormal"] == 0.833, "重算值被改回去了"
    assert "Recalculated" in str(c["source"]), "the source no longer records that the rate was recalculated"
    med, sig = I.moments_of("fasting_glucose", "T2D")   # passes as long as it does not raise
    assert sig < 0.6, f"σ={sig:.3f} 又回到荒唐区间"


def test_inv7_refuses_rather_than_clipping():
    """When the parameters do not fit, **refuse to build the distribution**
    rather than clip the tail back in: clipping would make "self-consistent"
    and "conflicting but clipped" look identical. Targets the cohorts that
    lack `p_abnormal`.
    """
    for n, coh, med in (("FIB4", "MASLD", 1.35), ("systolic_bp", "hypertension", 142.0)):
        vals = {I.sample_baseline(n, coh, (), f"P{i}") for i in range(40)}
        assert vals == {med}, f"{n}/{coh} 应退回中位数常数,实得 {sorted(vals)[:4]}"


def test_inv7_does_not_fire_on_the_healthy_ones():
    """Flip-side control: 5 cohorts whose parameters are self-consistent must
    not be falsely rejected."""
    for key in ("HbA1c/T2D", "LDL/dyslipidemia", "triglycerides/dyslipidemia",
                "ALT/MASLD", "AST/MASLD"):
        n, coh = key.split("/")
        med, sig = I.moments_of(n, coh)          # passes as long as it does not raise
        assert med > 0 and sig > 0, key


def test_newly_bounded_indicators_did_not_change_the_payload():
    """The bounds are **declarations**, not changed values -- the four
    bounded indicators lack `p_abnormal`, so they fall back to the median."""
    for n, coh, med in (("FIB4", "MASLD", 1.35), ("systolic_bp", "hypertension", 142.0),
                        ("diastolic_bp", "hypertension", 88.0), ("CGM_TIR", "T2D", 54.0)):
        vals = {I.sample_baseline(n, coh, (), f"P{i}") for i in range(30)}
        assert vals == {med}, f"{n}/{coh} 变成了分布:{sorted(vals)[:4]}"


def test_every_clinical_indicator_now_has_a_judgeable_bound():
    """`abnormal_side` can judge every clinical indicator."""
    unjudgeable = [n for n, d in I.dossiers().items()
                   if d.get("kind", "clinical") == "clinical"
                   and I.abnormal_side(n, 1.0) is None]
    assert not unjudgeable, f"这些临床指标仍判不了异常侧:{unjudgeable}"


# ─────────────────────────────────────────────── a single factor for shared severity
#
# Independent per-indicator sampling would not make a "severe" patient severe
# across every indicator at once. Synthea ties three lipid indicators together
# with an integer-tiered `diabetes_severity` (`LifecycleModule.java:733-741`);
# this is a continuous version, with **the loading solved from the measured
# between-patient correlation**.

_SEV_IDS = [f"SEV-{i:05d}" for i in range(4000)]


def _pearson(xs, ys):
    import math
    import statistics as st
    mx, my = st.mean(xs), st.mean(ys)
    return (sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)))


def test_severity_reproduces_the_measured_between_patient_correlation():
    """Load bearing: the raw-scale r produced by emission must hit the
    measured anchor.

    The anchor +0.7133 is measured on a de-identified diabetes cohort
    (per-patient mean, then between-patient correlation, n=670; see
    docs/ETHICS.md §2 for how the real records enter). At n=4000, Pearson's SE is approximately
    (1-r^2)/sqrt(n) ~= 0.008.
    """
    spec = I.severity()["T2D"]
    a, b = spec["pair"]
    xs = [I.sample_baseline(a, "T2D", (), c) for c in _SEV_IDS]
    ys = [I.sample_baseline(b, "T2D", (), c) for c in _SEV_IDS]
    got = _pearson(xs, ys)
    assert abs(got - spec["between_patient_r"]) < 0.03, (
        f"合成 r={got:+.4f} 打不到实测锚 {spec['between_patient_r']:+.4f}")


def test_severity_does_not_move_the_marginals():
    """Counter-metric: adding co-movement leaves the marginals unchanged
    (lambda^2+(1-lambda^2)=1). If the marginal moved, `p_abnormal` would no
    longer hold, and INV-2 depends on it.
    """
    for name in I.severity()["T2D"]["pair"]:
        d = I.of(name)
        p = d["cohorts"]["T2D"]["p_abnormal"]
        hi = d["reference_range"]["high"]
        vals = [I.sample_baseline(name, "T2D", (), c) for c in _SEV_IDS]
        got = sum(1 for v in vals if v > hi) / len(vals)
        assert abs(got - p) < 0.02, f"{name}:边际被共动改了,实测 {got:.3f} vs 登记 {p}"


def test_without_the_loading_there_would_be_no_comovement():
    """Negative control: with the loading removed (lambda=None) the r between
    the two indicators collapses to near 0, so the co-movement comes from the
    loading and not from a shared case_id seed.
    """
    import unittest.mock as mock
    a, b = I.severity()["T2D"]["pair"]
    with mock.patch.object(I, "_loading_of", lambda c: None):
        xs = [I.sample_baseline(a, "T2D", (), c) for c in _SEV_IDS]
        ys = [I.sample_baseline(b, "T2D", (), c) for c in _SEV_IDS]
    got = _pearson(xs, ys)
    assert abs(got) < 0.05, f"去掉载荷后 r 仍有 {got:+.4f} —— 共动不是这个机制给的"


def test_loading_is_derived_not_stored():
    """lambda is a **derived quantity**; storing it in the registry would let
    it go stale when sigma changes."""
    for coh, spec in I.severity().items():
        assert "loading" not in spec, f"{coh} 把 λ 存下来了(σ 一变就过期)"
    # And it must **not equal** the naive sqrt(r) -- if it did, that would mean
    # the log-normal decay was never solved for
    import math
    lam = I._loading_of("T2D")
    naive = math.sqrt(I.severity()["T2D"]["between_patient_r"])
    assert abs(lam - naive) > 0.005, (
        f"λ={lam:.4f} 与朴素 √r={naive:.4f} 一样 —— 对数正态衰减没被反解")


def test_sigma_between_is_registered_but_deliberately_unused():
    """`sigma_between` is a **measured fact** that sampling deliberately
    does not use; the reason lives in the code.

    `HbA1c`/`fasting_glucose` are not in `physio_streams.yaml:streams`, so
    clinical streams **do not go through the observation-noise layer** and
    all the spread is between-patient spread. Sampling with sigma_between
    would push the emitted abnormal rate away from `p_abnormal`.
    """
    import inspect
    reg = [n for n, d in I.dossiers().items()
           for c in (d.get("cohorts") or {}).values() if c.get("sigma_between")]
    assert len(reg) >= 2, f"登记 sigma_between 的只有 {len(reg)} 条"
    src = inspect.getsource(I.sample_baseline)
    assert "sigma_between" not in src.split("lam = _loading_of")[1], "抽样真用上了它?"
    assert "physio_streams" in src, "「为什么不用它」的理由不在代码里,那就会被当成疏漏改掉"


def test_cohorts_without_severity_are_bit_identical():
    """For a cohort not registered under severity, the sampled result equals
    the plain per-indicator baseline **bit for bit**."""
    before = [round(I.sample_baseline("LDL", "dyslipidemia", (), c), 10)
              for c in _SEV_IDS[:200]]
    assert I._loading_of("dyslipidemia") is None, "dyslipidemia 不该有载荷(人群不匹配)"
    # Recomputing with the same seeds must be fully consistent (pure function);
    # and must not be affected by the T2D severity registration
    after = [round(I.sample_baseline("LDL", "dyslipidemia", (), c), 10)
             for c in _SEV_IDS[:200]]
    assert before == after
