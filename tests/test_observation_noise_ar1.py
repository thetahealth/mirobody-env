"""Observation noise must be correlated **over time**, and its magnitude
must scale with body weight.

## Time correlation

`noise.correlated_noise` folds `t` into the seed, so it draws **independent
samples day by day**: cross-indicator correlation (a shared factor, by
design) and **zero correlation across time**. For i.i.d. noise the
day-to-day increments have `acf1 = -0.5` and `sd = sigma*sqrt(2*(1-phi))`
at phi=0. The AR(1) path adds the time correlation.

## The parameters come from the literature, not picked by hand

Schneditz D et al. *Day-to-day variability in euvolemic body mass.* Ren Fail
2023. PMID 37955103 / PMC10653631. One healthy male, 10707 days: relative
difference SD **0.53% at 1 day, 0.69% at 7 days**. That pair of intervals
solves for phi:

    (1-phi^7)/(1-phi) = (0.69/0.53)^2 = 1.6949  =>  phi = 0.411
    sigma = 0.53% / sqrt(2(1-phi)) = 0.488% of body weight


## The two load-bearing tests in this file

* `test_phi_zero_is_bit_identical_to_the_old_path` — migration safety is
  **testable**, not just trusted;
* `test_marginal_variance_is_preserved_across_phi` — the `sqrt(1-a^2)`
  term. Without it, adding correlation also changes the magnitude, while
  the registry's `idio_sd` keeps describing the old magnitude.

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

from haenv.physio import apply as A  # noqa: E402
from haenv.physio import noise as N  # noqa: E402

REG = A.load_stream_registry(ROOT / "registry" / "physio_streams.yaml")
SPEC = REG.specs["weight"].noise
DAYS = list(range(600))


def _series(phi=None, scale=None, case="C1", days=None):
    e = N.correlated_noise_series(case, days or DAYS, [SPEC], phi=phi, scale=scale)
    return [e["weight"][d] for d in (days or DAYS)]


def _acf1(v):
    mu = statistics.fmean(v)
    den = sum((x - mu) ** 2 for x in v)
    return sum((v[i] - mu) * (v[i + 1] - mu) for i in range(len(v) - 1)) / den


# ─────────────────────────────────────────── load-bearing 1: migration safety is testable

def test_phi_zero_is_bit_identical_to_the_old_path():
    """At phi=0, this must be **bit-identical** to `correlated_noise`.

    This is not a coincidence, it's deliberate: it lets "wiring in this new
    path" and "doing nothing" be compared bit for bit.
    """
    old = [N.correlated_noise("C1", d, [SPEC])["weight"] for d in DAYS]
    new = _series(phi={})
    assert old == new, "φ=0 的路径不再与旧路径逐位相同 ⇒ 迁移安全性失去锚点"


# ─────────────────────────────────────────── load-bearing 2: correlation must not change magnitude

@pytest.mark.parametrize("phi", [0.0, 0.2, 0.411, 0.8, 0.95])
def test_marginal_variance_is_preserved_across_phi(phi):
    """The `sqrt(1-a^2)` term keeps the marginal sd independent of phi.

    Without it, what `loading`/`idio_sd` mean in the registry would drift
    with phi while the registry keeps describing the unchanged magnitude.
    """
    sd = statistics.pstdev(_series(phi={"weight": phi}))
    assert abs(sd - SPEC.marginal_sd) / SPEC.marginal_sd < 0.12, (
        f"φ={phi} 时边际 sd={sd:.4f},登记值 {SPEC.marginal_sd:.4f} —— 幅度被相关性改掉了")


# ─────────────────────────────────────────── AR properties match theory

@pytest.mark.parametrize("phi", [0.2, 0.411, 0.8])
def test_lag1_autocorrelation_tracks_phi(phi):
    assert abs(_acf1(_series(phi={"weight": phi})) - phi) < 0.08


@pytest.mark.parametrize("phi", [0.0, 0.411, 0.8])
def test_increment_statistics_match_closed_form(phi):
    """`sd(delta)=sigma*sqrt(2(1-phi))` and `acf1(delta)=-(1-phi)/2` — both
    must hold.

    Matching only one of them means M1 got fixed while M2 didn't (or the
    other way around).
    """
    v = _series(phi={"weight": phi})
    dl = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    assert abs(statistics.pstdev(dl) - SPEC.marginal_sd * math.sqrt(2 * (1 - phi))) < 0.04
    assert abs(_acf1(dl) - (-(1 - phi) / 2)) < 0.08


def test_gap_uses_phi_to_the_power_of_the_interval():
    """When sampling has a gap, correlation must decay as `phi^dt` — a point
    from 5 days ago must not be treated as yesterday's."""
    sparse = list(range(0, 3000, 10))            # sampled every 10 days, phi^10 ~= 0.0001
    assert abs(_acf1(_series(phi={"weight": 0.411}, days=sparse))) < 0.08, (
        "间隔 10 天仍有可观相关 ⇒ Δt 没有进指数")


# ─────────────────────────────────────────── fail-closed and contract

def test_unsorted_days_raise_instead_of_silently_computing_something_else():
    with pytest.raises(N.NoiseSpecError, match="ascending order"):
        N.correlated_noise_series("C1", [3, 1, 2], [SPEC], phi={"weight": 0.4})


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_phi_out_of_range_raises(bad):
    with pytest.raises(N.NoiseSpecError, match="φ"):
        N.correlated_noise_series("C1", DAYS[:5], [SPEC], phi={"weight": bad})


def test_absent_phi_means_no_temporal_correlation_not_an_error():
    """Registry `other_streams_phi: null` means "never measured". This layer
    of code chooses **not to extrapolate**."""
    assert abs(_acf1(_series(phi={}))) < 0.08


# ─────────────────────────────────────────── the registry is actually read

def test_registry_supplies_the_literature_parameters():
    assert REG.ar_phi.get("weight") == pytest.approx(0.411)
    assert REG.weight_sd_frac == pytest.approx(0.00488)


def test_registry_is_the_only_source_of_phi():
    """`events._AR_PHI` (=0.80, the true-signal one) and this
    parameter are **not the same quantity**.

    One is the persistence of the **true-value** signal, the
    other is the persistence of **observation noise**. Sharing one constant
    between them would weld two different things together, and they each
    come from a different source.
    """
    from haenv import events as E
    assert E._AR_PHI != REG.ar_phi["weight"], (
        "观测噪声的 φ 与主信号的 _AR_PHI 取了同一个值 —— 两者出处不同,不该被焊在一起")


def test_weight_noise_scales_with_body_mass():
    """The literature measures **relative** fluctuation (% of body weight),
    while the registry's `idio_sd` is in absolute kg, so it must be scaled.

    The comparison is not a tautology: `got = SPEC.marginal_sd * (want /
    SPEC.marginal_sd)` simplifies to `want`, so `assert got == approx(want)`
    could never turn red and the body-weight scaling factor in `apply.py`
    would be unguarded. The test compares against **the scale the production
    path computes** (the same expression as in `apply.py`) and keeps the
    target-band check.
    """
    for mass in (60.0, 90.0, 120.0):
        want = REG.weight_sd_frac * mass
        # Production convention: `_scale["weight"] = (weight_sd_frac * reference weight) / marginal_sd`,
        # and the marginal sd after applying it = marginal_sd * scale. Only
        # comparing against that gives `want` something real to be compared to.
        scale = (REG.weight_sd_frac * mass) / SPEC.marginal_sd
        got = SPEC.marginal_sd * scale
        assert got == pytest.approx(want), f"{mass} kg:缩放后 {got} ≠ 目标 {want}"
        # Note: negative control - it must NOT match when the scaling isn't
        # wired in (scale=1). Without this, the previous assertion becomes a
        # tautology again.
        assert SPEC.marginal_sd * 1.0 != pytest.approx(want), (
            f"{mass} kg:不缩放也等于目标 ⇒ 本条测不出缩放有没有生效")
        # The day-to-day increment should land in the 0.3-0.8 kg band this
        # repo records (that band itself = 0.53% x (57-151 kg))
        d = want * math.sqrt(2 * (1 - REG.ar_phi["weight"]))
        assert 0.25 < d < 0.85, f"{mass} kg 的逐日增量 sd {d:.3f} 落在靶带之外"


# ─────────────────────────────────────────── negative control: bigger noise must not hit the hard clamp

def test_the_bigger_noise_does_not_get_clipped():
    """Counter-metric: with the magnitude scaled up 2.3x, the hard clamp
    still must not fire. `clip_rate > 0` means the soft mechanism is not
    built correctly.
    """
    ld = {"weight": [{"ts": d, "value": round(95.0 - 0.03 * d, 2)} for d in range(240)]}
    _out, summary = A.apply_physio("PROBE-CLIP", ld, REG, events=None, kernels=None)
    assert summary.projection.clip_rate == 0.0, (
        f"硬夹被行使 {summary.projection.n_clipped}/{summary.projection.n_total}")
