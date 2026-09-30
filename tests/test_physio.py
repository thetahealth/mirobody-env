"""Assertions and **negative controls** for `haenv.physio`.

## This package's testing discipline

Every assertion is paired with a negative control: an assertion that can
only ever pass proves nothing. So every `test_*_ok` below
has a `test_*_negative` next to it — **the deliberately-broken version must
be caught by that same function**.

Three negative controls are worth special attention — they guard the places
where this package is stricter than the external implementation it's based on:
  · `test_kernel_naive_form_jumps`      — the original formula jumps at short
    events (our fix is meaningful)
  · `test_audit_catches_pure_sine`      — a pure sine wave must be caught by
    the ACF check (otherwise the audit does nothing)
  · `test_coupling_cycle_detected`      — a cycle must be judged a failure
    (otherwise that `raise` is dead code)

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from haenv.physio import apply as physio_apply
from haenv.physio import audit, bounds, coupling, kernel, noise, superpose

REPO = Path(__file__).resolve().parents[1]
KERNEL_TABLE = REPO / "registry" / "physio_kernels.yaml"
COUPLING_TABLE = REPO / "registry" / "comorbid_coupling.yaml"


# ══════════════════════════════════════════════════════════════════
# kernel
# ══════════════════════════════════════════════════════════════════

def _kp(**kw) -> kernel.KernelParams:
    # `generator_source` is required: the registry must not write
    # parameters against an event name that doesn't exist.
    # Defaults to `NO_GENERATOR` here — synthetic parameters don't come from
    # any pool to begin with, so filling in a real topic would be a lie.
    base = dict(beta=1.0, tau_rise=10.0, tau_fade=10.0, duration=30.0,
                source="test", review="done",
                generator_source=kernel.NO_GENERATOR)
    base.update(kw)
    return kernel.KernelParams(**base)


def test_kernel_four_segments_ok() -> None:
    p = _kp()
    assert kernel.kernel_value(p, -1) == 0.0          # before the event
    assert kernel.kernel_value(p, 0) == 0.0
    mid = kernel.kernel_value(p, p.duration)          # end of the active phase
    assert 0.0 < mid <= 1.0
    assert kernel.kernel_value(p, p.duration + p.tau_fade + 0.1) == 0.0   # outside the decay window


def test_kernel_monotone_rise_then_decay() -> None:
    p = _kp()
    rise = [kernel.kernel_value(p, d) for d in range(1, int(p.duration) + 1)]
    assert rise == sorted(rise), "进行段必须单调不减"
    fade = [kernel.kernel_value(p, p.duration + d) for d in range(0, int(p.tau_fade) + 1)]
    assert fade == sorted(fade, reverse=True), "消退段必须单调不增"


def test_kernel_continuous_at_t_end_ok() -> None:
    """A short event (duration < tau_rise) must not jump at t_end — this is
    our fix to the original formula."""
    p = _kp(duration=1.0, tau_rise=10.0, tau_fade=5.0)
    left = kernel.kernel_value(p, p.duration)
    right = kernel.kernel_value(p, p.duration + 1e-9)
    assert abs(left - right) < 1e-6, f"t_end 处跳变:{left} -> {right}"


def test_kernel_naive_form_jumps() -> None:
    """Negative control: the original formula (decay phase unconditionally
    starts from 1.0) really does **jump** under the same parameters.

    This test proves the fix above isn't fixing a non-problem — if the
    original formula were also continuous, we shouldn't have changed it.
    """
    p = _kp(duration=1.0, tau_rise=10.0, tau_fade=5.0)
    ours_at_end = kernel.kernel_value(p, p.duration)
    naive_at_end_plus = 1.0 * math.exp(-3.0 / p.tau_fade * 0.0)   # original formula: exp(-alpha*0) = 1.0
    assert abs(naive_at_end_plus - ours_at_end) > 0.5, (
        "原式应当在 t_end 处从 ~0.08 跳到 1.0;若这条失败,说明修正的前提不成立"
    )


@pytest.mark.parametrize("bad", [
    dict(tau_rise=0.0), dict(tau_rise=-1.0), dict(tau_fade=0.0),
    dict(duration=-1.0), dict(beta=float("nan")), dict(source="  "),
    dict(review=""), dict(review="maybe"),
])
def test_kernel_bad_params_raise(bad) -> None:
    """Negative control: invalid parameters must raise — **no default-value
    fallback is allowed**.

    A fallback's failure mode is "fabricate a plausible value when a real
    result can't be obtained, and leave only a warning behind" — data keeps
    getting produced and downstream has no way to notice. That's far worse
    than crashing outright.
    """
    with pytest.raises(kernel.KernelParamError):
        _kp(**bad)


def test_kernel_table_loads_and_is_sourced() -> None:
    table = kernel.load_kernel_table(KERNEL_TABLE)
    assert table, "登记表不许为空"
    for (event, ind), p in table.items():
        assert p.source.strip(), f"{event}/{ind} 缺 source"
        assert p.review in ("pending", "done"), f"{event}/{ind} 的 review 非法"


def test_kernel_table_exercises_continuity_fix() -> None:
    """A mechanism existing is not the same as it being exercised.

    Without a short-duration entry, the lowest end-value across the whole
    table is 0.999877: the continuity fix would be **exercised by zero
    registry entries**, covered only by one synthetic test parameter. This
    test holds because of `acute_uri/weight` (duration 5 < tau_rise 21).
    Remove that entry and this test fails.
    """
    table = kernel.load_kernel_table(KERNEL_TABLE)
    ends = {}
    for (event, ind), p in table.items():
        k = 6.0 / p.tau_rise
        ends[(event, ind)] = 1.0 / (1.0 + math.exp(-k * (p.duration - p.tau_rise / 2.0)))
    lowest = min(ends.values())
    assert lowest < 0.9, (
        f"全表最小末值 {lowest:.6f} —— 没有任何条目的 duration 短到能行使连续性修正。"
        f"那个修正就成了只有测试覆盖、生产从不触发的机制。各条末值:{ends}"
    )


def test_saturation_covers_every_kernel_indicator() -> None:
    """A `saturation` section with **no code reading it** would be
    declared but never triggered.

    Both sections load together and cross-validate: every kernel
    parameter must have a saturation ceiling, and, symmetrically, a leftover
    ceiling with no kernel parameter is also a failure.
    """
    reg = kernel.load_physio_registry(KERNEL_TABLE)
    used = {ind for _, ind in reg.kernels}
    assert used, "kernels 为空"
    assert used == set(reg.saturation), "核参数指标与饱和上限必须一一对应"
    for ind in used:
        assert reg.m_for(ind) > 0


def test_saturation_gap_raises(tmp_path: Path) -> None:
    """Negative control: a missing saturation ceiling must fail, not
    silently produce an unsaturated result."""
    bad = tmp_path / "gap.yaml"
    bad.write_text(
        "saturation:\n  a: {m: 1.0, source: s}\n"
        "kernels:\n  e:\n    a: {beta: 1.0, tau_rise: 1.0, tau_fade: 1.0, duration: 1.0,"
        " source: s, review: pending, generator_source: none}\n"
        "    b: {beta: 1.0, tau_rise: 1.0, tau_fade: 1.0, duration: 1.0,"
        " source: s, review: pending, generator_source: none}\n",
        encoding="utf-8")
    with pytest.raises(kernel.KernelParamError, match="no saturation cap"):
        kernel.load_physio_registry(bad)


def test_unused_saturation_raises(tmp_path: Path) -> None:
    """Negative control: an unused saturation ceiling also fails — it would
    make coverage look bigger than it actually is."""
    bad = tmp_path / "unused.yaml"
    bad.write_text(
        "saturation:\n  a: {m: 1.0, source: s}\n  ghost: {m: 2.0, source: s}\n"
        "kernels:\n  e:\n    a: {beta: 1.0, tau_rise: 1.0, tau_fade: 1.0, duration: 1.0,"
        " source: s, review: pending, generator_source: none}\n",
        encoding="utf-8")
    with pytest.raises(kernel.KernelParamError, match="not used by any kernel parameter"):
        kernel.load_physio_registry(bad)


def test_pending_review_is_reportable() -> None:
    """`review: pending` must be able to reach the report — otherwise
    unreviewed parameters would silently take effect as if "already live"."""
    reg = kernel.load_physio_registry(KERNEL_TABLE)
    assert reg.pending(), "当前全表都应是 pending(医生复核 H4 尚未完成)"
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    assert coupling.pending_rules(rules), "耦合规则同样应是 pending"


def test_kernel_table_missing_field_raises(tmp_path: Path) -> None:
    """Negative control: a table missing a field must be rejected as a
    whole, not just skip that one entry."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("kernels:\n  e1:\n    i1:\n      beta: 1.0\n", encoding="utf-8")
    with pytest.raises(kernel.KernelParamError):
        kernel.load_kernel_table(bad)


# ══════════════════════════════════════════════════════════════════
# superpose
# ══════════════════════════════════════════════════════════════════

def test_superpose_saturates_below_m() -> None:
    """Extreme overlap is clamped within m. **The boundary is `<=`** — see
    saturate's docstring: under double precision, tanh(20) == 1.0, so it can
    land exactly on m."""
    m = 5.0
    assert abs(superpose.saturate([100.0, 100.0, 100.0], m)) <= m
    assert abs(superpose.saturate([-100.0], m)) <= m
    # Moderate overlap must be **strictly** less than m — otherwise the
    # assertion above would be fooled by a broken implementation that always
    # returns m
    assert abs(superpose.saturate([3.0, 3.0], m)) < m


def test_superpose_near_linear_for_small_overlap() -> None:
    """Small overlap keeps approximate additivity — this is the reason to
    pick tanh over a hard clamp."""
    m = 100.0
    got = superpose.saturate([1.0, 2.0], m)
    assert abs(got - 3.0) < 0.01


@pytest.mark.parametrize("bad_m", [0.0, -1.0, float("inf"), float("nan")])
def test_superpose_bad_m_raises(bad_m) -> None:
    """Negative control: there is no "not registered, so no saturation"
    escape hatch."""
    with pytest.raises(superpose.SaturationError):
        superpose.saturate([1.0], bad_m)


def test_saturation_bite_reports_how_much_was_eaten() -> None:
    lin = superpose.linear_sum([10.0, 10.0])
    sat = superpose.saturate([10.0, 10.0], 5.0)
    assert audit.saturation_bite(lin, sat) > 0.5


# ══════════════════════════════════════════════════════════════════
# noise — the co-movement layer
# ══════════════════════════════════════════════════════════════════

def _specs() -> list[noise.NoiseSpec]:
    return [
        noise.NoiseSpec("systolic_bp", {"cardio": 1.0}, 0.30),
        noise.NoiseSpec("resting_hr",  {"cardio": 1.0}, 0.30),
        noise.NoiseSpec("FBG",         {"metabolic": 1.0}, 0.30),
    ]


def test_noise_reproducible_bytewise() -> None:
    """Two runs with the same (case_id, indicator, t) are byte-for-byte
    identical."""
    a = noise.correlated_noise("C1", 7, _specs())
    b = noise.correlated_noise("C1", 7, _specs())
    assert a == b


def test_noise_adding_a_stream_does_not_move_others() -> None:
    """The most important of `rng.py`'s three properties: adding one stream
    must not move the numbers of any other stream."""
    base = noise.correlated_noise("C1", 7, _specs())
    more = noise.correlated_noise("C1", 7, _specs() + [
        noise.NoiseSpec("steps", {"activity": 1.0}, 0.5)
    ])
    for k in base:
        assert base[k] == more[k], f"{k} 被新增的流挪动了"


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy)


def test_noise_same_group_co_moves() -> None:
    """Positive case: indicators in the same group have high
    correlation."""
    days = range(300)
    series = {k: [] for k in ("systolic_bp", "resting_hr", "FBG")}
    for t in days:
        e = noise.correlated_noise("C1", t, _specs())
        for k, v in e.items():
            series[k].append(v)
    assert _corr(series["systolic_bp"], series["resting_hr"]) > 0.7


def test_noise_cross_group_not_correlated() -> None:
    """Negative case: cross-group correlation is not significant.
    Testing only the same-group case would miss a broken implementation
    where "every indicator co-moves"."""
    days = range(300)
    series = {k: [] for k in ("systolic_bp", "resting_hr", "FBG")}
    for t in days:
        e = noise.correlated_noise("C1", t, _specs())
        for k, v in e.items():
            series[k].append(v)
    assert abs(_corr(series["systolic_bp"], series["FBG"])) < 0.3


def test_noise_zero_idio_sd_raises() -> None:
    """Negative control: no loading and no idiosyncratic noise means a
    deterministic series — exactly what this module exists to replace."""
    with pytest.raises(noise.NoiseSpecError):
        noise.NoiseSpec("x", {}, 0.0)


def test_normal_dev_roughly_standard() -> None:
    xs = [noise.normal_dev("C1", "probe", t) for t in range(4000)]
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    assert abs(mean) < 0.08, f"均值偏离:{mean}"
    assert abs(sd - 1.0) < 0.08, f"标准差偏离:{sd}"


# ══════════════════════════════════════════════════════════════════
# bounds
# ══════════════════════════════════════════════════════════════════

def _bs(**kw) -> bounds.BoundSpec:
    base = dict(indicator="weight", lo=30.0, hi=300.0, max_step=1.0, source="test")
    base.update(kw)
    return bounds.BoundSpec(**base)


def test_project_respects_range_and_slope() -> None:
    spec = _bs()
    assert bounds.project(1000.0, None, spec) == spec.hi
    assert bounds.project(80.0, 70.0, spec) == 71.0      # slope-limited to 1.0


def test_violations_counted_before_projection() -> None:
    """This module's core design: counts are recorded against the proposal
    **before** projection."""
    spec = _bs()
    rep = bounds.ProjectionReport()
    bounds.project(1000.0, 70.0, spec, rep)
    assert rep.n_range_violation == 1
    assert rep.n_slope_violation == 1
    assert rep.n_clipped == 1
    assert rep.clip_rate == 1.0


def test_no_violation_when_proposal_is_clean() -> None:
    """Negative control: a clean proposal must not be recorded as a
    violation (otherwise the previous test would be vacuously true)."""
    spec = _bs()
    rep = bounds.ProjectionReport()
    bounds.project(70.5, 70.0, spec, rep)
    assert (rep.n_range_violation, rep.n_slope_violation, rep.n_clipped) == (0, 0, 0)


@pytest.mark.parametrize("tf,val", [("identity", 70.0), ("log", 70.0), ("logit", 70.0)])
def test_transform_roundtrip(tf, val) -> None:
    # A non-identity transform requires a reference
    # point `at` (see `BoundSpec.__post_init__`).
    # Any value inside (lo, hi) works here — this test is about the
    # transform's **math**, not about which reference point is used.
    spec = _bs(transform=tf, **({} if tf == "identity" else {"at": 100.0}))
    back = bounds.from_transform(bounds.to_transform(val, spec), spec)
    assert abs(back - val) < 1e-6


@pytest.mark.parametrize("bad", [
    dict(lo=300.0, hi=30.0), dict(max_step=0.0), dict(max_step=-1.0),
    dict(transform="sqrt"), dict(source=""), dict(transform="log", lo=-1.0),
])
def test_bad_bound_spec_raises(bad) -> None:
    with pytest.raises(bounds.BoundError):
        _bs(**bad)


# ══════════════════════════════════════════════════════════════════
# coupling — the discrete layer
# ══════════════════════════════════════════════════════════════════

def test_coupling_table_loads() -> None:
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    assert rules
    assert len({r.rule_id for r in rules}) == len(rules), "id 不许重复"


def test_coupling_fires_and_is_observable() -> None:
    """A change must be observable in the output, not just in memory."""
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    bb = coupling.Blackboard({
        "has_diabetes": True, "ckd_stage": 4, "has_hypertension": False,
        "has_hypothyroidism": False, "has_osa": False, "bmi": 25.0,
    })
    trace = coupling.apply_couplings(bb, rules)
    assert trace.fired == ["ckd4_metformin_avoid"]
    rec = trace.as_record()
    assert rec["coupled_attrs"]["dm_care_branch"] == "metformin_avoid"
    assert rec["coupled_attrs"]["renal_dosing_flag"] is True
    assert rec["fired_rules"], "trace 必须非空,否则 reachability 无从查"


def test_coupling_stages_are_mutually_exclusive() -> None:
    """Negative control — staging rules must be mutually exclusive.

    With `>= 3` and `>= 4`, a stage-4 patient matches both and writes
    conflicting values into the same attribute, causing a cycle. This test
    pins down that the ranges are mutually exclusive intervals, with cycle
    detection unchanged.
    """
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    for stage, want in ((3, "metformin_reassess"), (4, "metformin_avoid"), (5, "metformin_avoid")):
        bb = coupling.Blackboard({
            "has_diabetes": True, "ckd_stage": stage, "has_hypertension": False,
            "has_hypothyroidism": False, "has_osa": False, "bmi": 25.0,
        })
        trace = coupling.apply_couplings(bb, rules)   # must not raise a cycle error
        assert bb.get("dm_care_branch") == want, f"stage={stage}"
        assert len([r for r in trace.fired if r.startswith("ckd")]) == 1, (
            f"stage={stage} 命中了多于一条分期规则:{trace.fired}"
        )


def test_coupling_does_not_fire_without_comorbidity() -> None:
    """Negative control: a single-disease patient should not trigger a
    comorbidity rule (otherwise the previous test would be vacuously true)."""
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    bb = coupling.Blackboard({
        "has_diabetes": True, "ckd_stage": 1, "has_hypertension": False,
        "has_hypothyroidism": False, "has_osa": False, "bmi": 22.0,
    })
    trace = coupling.apply_couplings(bb, rules)
    assert trace.fired == []
    assert trace.writes == {}


def test_coupling_missing_attribute_raises() -> None:
    """Negative control: a misspelled attribute and an unsatisfied condition
    are indistinguishable from the readings alone, so this must fail rather
    than silently skip."""
    rules = coupling.load_coupling_rules(COUPLING_TABLE)
    bb = coupling.Blackboard({"has_diabetes": True})     # ckd_stage missing
    with pytest.raises(coupling.CouplingError):
        coupling.apply_couplings(bb, rules)


def test_coupling_chain_converges_regardless_of_order() -> None:
    """A rule chain must converge even when listed in reverse order — the
    default round-count ceiling must be high enough."""
    mk = coupling.CouplingRule
    chain = [
        mk("r3", {"b": ["==", 1]}, {"c": 1}, "test"),
        mk("r2", {"a": ["==", 1]}, {"b": 1}, "test"),
        mk("r1", {"seed": ["==", 1]}, {"a": 1}, "test"),
    ]
    bb = coupling.Blackboard({"seed": 1, "a": 0, "b": 0, "c": 0})
    trace = coupling.apply_couplings(bb, chain)
    assert bb.get("c") == 1
    assert set(trace.fired) == {"r1", "r2", "r3"}


def test_coupling_cycle_detected() -> None:
    """Negative control: a genuine cycle must be judged a failure —
    otherwise that `raise` is dead code."""
    mk = coupling.CouplingRule
    cyc = [
        mk("up",   {"flag": ["==", 0]}, {"flag": 1}, "test"),
        mk("down", {"flag": ["==", 1]}, {"flag": 0}, "test"),
    ]
    bb = coupling.Blackboard({"flag": 0})
    with pytest.raises(coupling.CouplingError, match="did not converge"):
        coupling.apply_couplings(bb, cyc)


@pytest.mark.parametrize("bad", [
    dict(when={}), dict(then={}), dict(source=""), dict(rule_id=""),
    dict(when={"a": ["~~", 1]}), dict(when={"a": [1]}),
])
def test_bad_coupling_rule_raises(bad) -> None:
    base = dict(rule_id="r", when={"a": ["==", 1]}, then={"b": 1}, source="s")
    base.update(bad)
    with pytest.raises(coupling.CouplingError):
        coupling.CouplingRule(**base)


# ══════════════════════════════════════════════════════════════════
# audit
# ══════════════════════════════════════════════════════════════════

def _noisy_series(n: int = 200) -> list[float]:
    """AR(1) plus correlated noise — the shape we expect generated streams
    to have."""
    specs = [noise.NoiseSpec("systolic_bp", {"cardio": 1.0}, 0.3)]
    out, prev = [], 120.0
    for t in range(n):
        e = noise.correlated_noise("AUDIT", t, specs)["systolic_bp"]
        prev = 120.0 + 0.5 * (prev - 120.0) + e
        out.append(prev)
    return out


def test_audit_passes_on_noisy_series() -> None:
    rep = audit.describe("systolic_bp", _noisy_series())
    assert rep.check((-0.2, 0.9), 0.35, (0.0, 0.2)) == []


def test_audit_catches_pure_sine() -> None:
    """Negative control — the single most important test in this
    package.

    The helper stream in `events.py:1039-1040` is currently two sine waves.
    If the audit can't catch it, this whole layer might as well not exist.
    """
    sine = [10.0 * math.sin(2 * math.pi * t / 7.0) + 3.0 * math.sin(t / 2.0)
            for t in range(200)]
    rep = audit.describe("aux_sine", sine)
    bad = rep.check((-0.2, 0.9), 0.35, (0.0, 0.2))
    assert bad, "纯正弦必须被 ACF(7) 抓出来"
    assert any("ACF(7)" in b for b in bad)


def test_acf_refuses_constant_series() -> None:
    """Negative control: a constant series has no meaningful autocorrelation
    — it must raise rather than return 0.0."""
    with pytest.raises(audit.AuditError):
        audit.acf([5.0] * 50, 1)


def test_acf_is_calendar_aligned_under_missingness() -> None:
    """The lag is taken on the calendar, not on compacted indices.

    Dropping `None`s and then taking the lag against the compacted indices
    breaks once there is enough missingness: "lag=7" no longer means "seven
    days ago". On the same period-7 sine wave, missing every 3rd day:

        no missingness   ACF(7) = +0.967
        with missingness ACF(7) = -0.856   <- misaligned after compaction,
                                              even the sign flips

    It uses pairwise-complete lags (only pairs where both t and t-lag
    exist), so lag always equals the calendar-day lag. This test pins down
    that ACF(7) with missingness must still be **positive and close to the
    no-missingness value**.
    """
    full = [10.0 * math.sin(2 * math.pi * t / 7.0) for t in range(210)]
    gappy = [None if t % 3 == 0 else v for t, v in enumerate(full)]
    a_full = audit.acf(full, 7)
    a_gappy = audit.acf(gappy, 7)
    assert a_full > 0.9, f"无缺失基准异常:{a_full}"
    assert a_gappy > 0.9, (
        f"有缺失时 ACF(7)={a_gappy:+.3f} —— 若为负或接近 0,说明又压紧了时间轴"
    )
    assert abs(a_full - a_gappy) < 0.15, f"两者差距过大:{a_full:+.3f} vs {a_gappy:+.3f}"


def test_acf_refuses_when_pairs_too_few() -> None:
    """Negative control: when missingness is concentrated enough that there
    aren't enough pairs, it must raise, not return a number that looks
    plausible."""
    s: list[float | None] = [None] * 60
    for i in range(0, 60, 20):
        s[i] = float(i)
    with pytest.raises(audit.AuditError, match="valid pairs"):
        audit.acf(s, 7)


def test_acf_refuses_short_series() -> None:
    with pytest.raises(audit.AuditError):
        audit.acf([1.0, 2.0, 3.0], 7)


def test_assert_clean_rejects_any_clipping() -> None:
    """`clip_rate > 0` is a failure — the one place we're stricter
    than ESL-Bench."""
    rep = bounds.ProjectionReport(n_total=100, n_clipped=1)
    s = audit.AuditSummary(projection=rep)
    with pytest.raises(audit.AuditError, match="hard projection applied"):
        audit.assert_clean(s)


def test_assert_clean_passes_when_untouched() -> None:
    """Negative control: it should pass when the hard clamp never triggered
    (otherwise the previous test would be vacuously true)."""
    s = audit.AuditSummary(projection=bounds.ProjectionReport(n_total=100))
    audit.assert_clean(s)


# ══════════════════════════════════════════════════════════════════
# apply — the end-to-end assembly point
# ══════════════════════════════════════════════════════════════════

STREAM_TABLE = REPO / "registry" / "physio_streams.yaml"


def test_stream_registry_loads() -> None:
    reg = physio_apply.load_stream_registry(STREAM_TABLE)
    assert reg.specs and reg.excluded
    assert reg.pending(), "当前全表应是 pending(H4 医生复核未完成)"


def test_noise_is_bounded_by_construction() -> None:
    """Unbounded noise plus a hard slope ceiling means the tail **must** get
    clipped, which would make `clip_rate > 0 => failure` unsatisfiable. The
    noise is therefore truncated and its budget validated at load time.
    """
    spec = noise.NoiseSpec("x", {"g": 2.0}, 1.0, trunc=3.0)
    assert spec.bound == pytest.approx(9.0)
    worst = max(abs(noise.correlated_noise("C", t, [spec])["x"]) for t in range(3000))
    assert worst <= spec.bound + 1e-9, f"噪声越过了自称的上确界:{worst} > {spec.bound}"


def test_registry_rejects_max_step_too_tight(tmp_path: Path) -> None:
    """Negative control: it must fail when the slope ceiling can't
    accommodate both ends of the noise."""
    bad = tmp_path / "tight.yaml"
    bad.write_text(
        "streams:\n  x:\n    group: g\n    loading: 5.0\n    idio_sd: 5.0\n"
        "    lo: 0.0\n    hi: 100.0\n    max_step: 1.0\n    transform: identity\n"
        "    source: s\n    review: pending\n", encoding="utf-8")
    with pytest.raises(physio_apply.StreamRegistryError, match="cannot absorb both noise tails"):
        physio_apply.load_stream_registry(bad)


def test_registry_rejects_noise_too_big_for_max_step(tmp_path: Path) -> None:
    """Negative control for transformed streams.

    Noise is **additive in the transformed domain**, and the conversion
    lives in `load_stream_registry`: the registry always writes
    values in the **original unit**, and flipping a stream to a transform
    only requires changing `transform` and adding `at`. So there is no
    "change one string without touching those two numbers" error to make —
    nobody has to remember to recompute anything.

    What this test now pins down instead: the noise really is too big for
    that stream. The units are already aligned (`bounds.induced_step` folds
    the transformed-domain budget back into the original unit), so this
    check is now meaningful.
    """
    bad = tmp_path / "tf.yaml"
    bad.write_text(
        "streams:\n  x:\n    group: g\n    loading: 0.1\n    idio_sd: 5.0\n"
        "    lo: 1.0\n    hi: 100.0\n    max_step: 1.0\n    transform: log\n"
        "    at: 50.0\n    source: s\n    review: pending\n", encoding="utf-8")
    with pytest.raises(physio_apply.StreamRegistryError, match="max_step"):
        physio_apply.load_stream_registry(bad)
    # Positive control: same shape, noise scaled down => must load fine
    # (proves the previous check isn't just "fails whenever it sees log")
    ok = tmp_path / "tf_ok.yaml"
    ok.write_text(
        "streams:\n  x:\n    group: g\n    loading: 0.05\n    idio_sd: 0.1\n"
        "    lo: 1.0\n    hi: 100.0\n    max_step: 1.0\n    transform: log\n"
        "    at: 50.0\n    source: s\n    review: pending\n", encoding="utf-8")
    assert "x" in physio_apply.load_stream_registry(ok).specs


def test_registry_rejects_unknown_stream() -> None:
    """Negative control: an unregistered stream must fail by default —
    silently allowing it would mean adding a new stream bypasses this whole
    protection layer."""
    reg = physio_apply.load_stream_registry(STREAM_TABLE)
    ld = {"未登记的流": [{"ts": i, "value": float(i)} for i in range(30)]}
    with pytest.raises(physio_apply.StreamRegistryError, match="no spec and not in exclude"):
        physio_apply.apply_physio("C1", ld, reg)


def test_apply_preserves_sampling_and_excluded() -> None:
    """Only `value` changes, `ts` never does; streams in `exclude` pass
    through unchanged (they are determined by hidden control variables, and
    adding noise to them would change the gold label)."""
    reg = physio_apply.load_stream_registry(STREAM_TABLE)
    # The fixture stream must be one this layer adds noise to; the calibrated
    # wearable streams (resting_hr among them) get their noise upstream
    # (`noise_external`).
    ld = {
        "activity_index": [{"ts": i, "value": 50.0} for i in range(0, 60, 2)],
        "medication_adherence": [{"ts": i, "value": 0.9} for i in range(0, 60, 10)],
    }
    out, _ = physio_apply.apply_physio("C1", ld, reg)
    assert [p["ts"] for p in out["activity_index"]] == [p["ts"] for p in ld["activity_index"]]
    assert out["medication_adherence"] == ld["medication_adherence"]
    assert out["activity_index"] != ld["activity_index"], "已登记的流必须真的被加噪"


def test_external_noise_stream_is_not_noised_twice() -> None:
    """A stream whose noise is declared external gets none added here.

    Its spread and co-movement are rendered upstream from measured parameters;
    adding this layer's authored noise on top inflated the amplitude 21-29 percent.
    """
    reg = physio_apply.load_stream_registry(STREAM_TABLE)
    assert reg.specs["resting_hr"].noise.external, "fixture assumption: resting_hr is external"
    ld = {"resting_hr": [{"ts": i, "value": 65.0} for i in range(0, 60, 2)]}
    out, _ = physio_apply.apply_physio("C1", ld, reg)
    assert [round(p["value"], 6) for p in out["resting_hr"]] == [65.0] * len(ld["resting_hr"])


def test_external_noise_must_add_none() -> None:
    """Declaring noise external while still adding some counts the variance twice."""
    with pytest.raises(noise.NoiseSpecError):
        noise.NoiseSpec(indicator="x", loadings={"g": 0.5}, idio_sd=0.0, external="haenv.wearable")


def test_apply_is_reproducible() -> None:
    reg = physio_apply.load_stream_registry(STREAM_TABLE)
    ld = {"resting_hr": [{"ts": i, "value": 65.0} for i in range(40)]}
    a, _ = physio_apply.apply_physio("C1", ld, reg)
    b, _ = physio_apply.apply_physio("C1", ld, reg)
    assert a == b


def test_audit_summary_record_is_plain_data() -> None:
    """A reading must be persistable, otherwise nobody can recompute it."""
    s = audit.AuditSummary(
        projection=bounds.ProjectionReport(n_total=10),
        distributions=[audit.describe("systolic_bp", _noisy_series(100))],
    )
    rec = s.as_record()
    assert rec["projection"]["n_total"] == 10
    assert rec["distributions"][0]["indicator"] == "systolic_bp"
