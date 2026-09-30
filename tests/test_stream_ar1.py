"""The AR(1) waveform of `events.render_stream` -- the deliverable itself, not a proxy for it.

## What this guards against

An auxiliary stream built as `base + amp*sin(2*pi*t/period) + 0.3*amp*sin(t/2)` -- two
sines, no randomness -- puts a strong periodic fingerprint at lag 7 on every series.
Post-processing cannot remove it: additive noise only dilutes the autocorrelation
(`r_new ~= r0 * var_s / (var_s + var_n)`) and leaves the periodic structure, and pushing
|ACF(7)| below 0.35 takes noise that drowns the physiological signal.

So the source is AR(1), whose autocorrelation decays as `phi^k` and carries no periodic
component.

## Why each assertion has a counter-metric

The tests measure the deliverable's definition (|ACF(7)| is the acceptance quantity), not a
shape, a line number or a piece of text. Pushing ACF(7) down is trivial on its own -- a
constant or pure white noise does it -- so the same file pins that day-to-day persistence
survives, that series differ across cases, and that values stay in the physiological
range. Dropping any one of these leaves the acceptance quantity satisfiable by a degenerate
stream.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haenv import events as E                              # noqa: E402

#: Acceptance bound for |ACF(7)| of an auxiliary stream.
ACF7_LIMIT = 0.35


def _acf(vals: list[float], lag: int) -> float:
    """Lag-k autocorrelation. Too few pairs raises: returning 0.0 would read as a real value."""
    n = len(vals)
    if n <= lag + 2:
        raise AssertionError(f"序列长 {n},算不了 lag={lag} 的自相关 —— 夹具太短")
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals)
    if var <= 0:
        raise AssertionError("序列方差为 0(常数流)—— 自相关无定义,而不是 0")
    cov = sum((vals[i] - mu) * (vals[i + lag] - mu) for i in range(n - lag))
    return cov / var


def _plan(name: str = "hrv", step: int = 1) -> E.StreamPlan:
    m = next(x for x in E.METRICS if x.name == name)
    lo, hi = m.hard_range
    base = min(max(m.base, lo + m.amp + 1e-9), hi - m.amp - 1e-9)
    return E.StreamPlan(spec=m, base_eff=base, step_days=step, source="test", phase=3)


def _vals(p: E.StreamPlan, end_day: int, seed: str) -> list[float]:
    return [float(x["value"]) for x in E.render_stream(p, end_day, seed=seed)]


def test_lag7_periodicity_is_gone() -> None:
    """The deliverable: an auxiliary stream's |ACF(7)| is within the acceptance bound."""
    for name in ("hrv", "spo2", "steps", "sleep_hours", "resting_hr"):
        v = _vals(_plan(name), 180, "CASE-A")
        a7 = abs(_acf(v, 7))
        assert a7 <= ACF7_LIMIT, f"{name} 的 |ACF(7)| = {a7:.3f} > {ACF7_LIMIT} —— 周期性没去掉"


def test_counter_metric_day_to_day_persistence_survives() -> None:
    """Counter-metric: pushing ACF(7) down is trivial -- white noise does it.

    So lag-1 persistence must survive too; otherwise the fix replaces the physiological
    signal with noise. AR(1) ties the two together through `ACF(k) = phi^k`:
    phi = 0.80 gives ACF(1) ~= 0.80 and ACF(7) ~= 0.21.

    The 0.55 bound applies to uncalibrated streams only. It encodes strong day-to-day
    persistence, while the reference lag-1 values of calibrated streams are resting_hr 0.336,
    stress 0.212, steps 0.023, hrv 0.013 and sleep -0.051 -- a day-to-day hrv series is
    close to white noise, and holding calibrated streams to 0.55 would demand synthetic
    streams smoother than real people. Calibrated streams are checked against their reference
    values elsewhere; uncalibrated ones render with the global phi = 0.80, so turning into
    white noise is a failure for them.
    """
    from haenv import wearable as _w
    checked = [n for n in ("hrv", "spo2", "steps", "sleep_hours", "resting_hr",
                           "skin_temp", "body_temp")
               if not _w.is_calibrated(n)]
    assert checked, "所有流都被标定了 ⇒ 这条没有对象,应当退役而不是空转"
    for name in checked:
        v = _vals(_plan(name), 180, "CASE-A")
        a1 = _acf(v, 1)
        assert a1 >= 0.55, f"{name} 的 ACF(1) = {a1:.3f} —— 日间持续性没了,这是白噪声不是生理流"


def test_counter_metric_series_differ_across_cases() -> None:
    """Counter-metric: `phase=(i*3)%7` depends only on the stream index, so every case in a
    batch shares the same phases.

    If the seed did not carry the case identity, every patient would get a byte-identical
    stream. This pins that the seed takes effect.
    """
    p = _plan("hrv")
    a, b = _vals(p, 120, "CASE-A"), _vals(p, 120, "CASE-B")
    assert a != b, "两个不同 case_id 渲染出逐字节相同的流 —— 种子没生效"
    # The other side: the same case_id renders byte-identical (the idempotency gate relies on it)
    assert a == _vals(p, 120, "CASE-A"), "同一个种子两次渲染不一致 —— 噪声源带了全局状态"


def test_counter_metric_values_stay_in_physio_range() -> None:
    """Counter-metric: AR(1) has heavier tails than a sine, and an out-of-range value is
    clipped back by the final `min/max`, which leaves no trace in the output -- so the
    clip rate is measured directly and must be 0."""
    for name in ("hrv", "spo2", "steps", "sleep_hours", "resting_hr", "skin_temp"):
        p = _plan(name)
        st: dict = {}
        E.render_stream(p, 365, seed="CASE-A", stats=st)
        lo, hi = p.spec.hard_range
        # The bound only catches a fixture that never filled in; it does not pin a wear
        # rate (non-wear days drop points from the 366 rendered days).
        assert st["n"] > 200, f"{name} 只渲染了 {st['n']} 点 —— 夹具没铺满"
        assert st["n_clipped"] == 0, (
            f"{name} 有 {st['n_clipped']}/{st['n']} 点被夹回 {lo}~{hi} —— "
            f"`_AR_SD_FRAC` 要收窄")


def test_negative_control_the_acf_probe_can_actually_fail() -> None:
    """Negative control: the assertions above can fail.

    The two-sine waveform, fed to the same `_acf`, must exceed the bound; otherwise the probe
    cannot detect periodicity and "the new waveform passes" carries no information. A gate
    that cannot turn red looks the same as no gate.
    """
    import math
    m = next(x for x in E.METRICS if x.name == "hrv")
    old = [m.base
           + m.amp * math.sin(2 * math.pi * (d + 3) / m.period)
           + 0.3 * m.amp * math.sin((d + 3) / 2.0)
           for d in range(0, 181)]
    a7 = abs(_acf(old, 7))
    assert a7 > ACF7_LIMIT, (
        f"旧的双正弦波形量出 |ACF(7)| = {a7:.3f} ≤ {ACF7_LIMIT} —— "
        f"这把尺子测不出周期性,新波形「达标」因此没有信息量")
