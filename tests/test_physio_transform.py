"""Unit semantics for transform-domain noise: the pinned statement must be falsifiable.

## The one statement this file pins down

> `loading` / `idio_sd` are in transform-domain units; noise is additive in
> the transform domain:
> `y = from_transform(to_transform(base) + eps)`.

Without this statement, `apply.load_stream_registry` has to reject any
transform other than identity, and that fail-closed behavior is right: the
ambiguity would silently change every value -- the same `idio_sd: 0.20`
means +/-0.2 percentage points in raw units, and something completely
different in the logit domain.

`apply.py`'s invariant `max_step < 2 * noise.bound` compares the raw-unit
`max_step` against the transform-domain `noise.bound`. Under identity the two
share the same dimension; once any stream switches to log/logit, the
comparison mixes two units and raises no error. The load-bearing piece is
therefore `bounds.induced_step` (which converts back to raw units).

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from haenv.physio import apply as physio_apply       # noqa: E402
from haenv.physio import bounds as B                 # noqa: E402
from haenv.physio import noise as N                  # noqa: E402

REG = pathlib.Path(__file__).resolve().parents[1] / "registry" / "physio_streams.yaml"


def _ident(**kw) -> B.BoundSpec:
    d = dict(indicator="x", lo=0.0, hi=100.0, max_step=10.0, source="test")
    d.update(kw)
    return B.BoundSpec(**d)


# ---------------------------------------------------------------- 1. positive control: identity is bit-identical

def test_identity_path_is_bit_identical() -> None:
    """Note: the most important test here -- under identity, the transform path and the plain path must be bit-identical.

    The transform form is `from_transform(to_transform(base) + eps)`, the
    plain form is `base + eps`. Under identity, `to`/`from` are the identity
    function, so the two must be exactly equal, not "approximately equal."
    If this test fails, transform-domain noise has changed the readings of
    every existing stream along the way.
    """
    spec = _ident()
    for base in (0.0, 1.0, 37.5, 99.999, 1e-9, 12345.678):
        for eps in (0.0, 1e-12, -0.3, 7.25, -1e3):
            got = B.from_transform(B.to_transform(base, spec) + eps, spec)
            assert got == base + eps, f"identity 下 {base}+{eps} 走变换域后变成了 {got}"


def test_induced_step_is_identity_under_identity() -> None:
    """The other half of the same test: the conversion function must be the identity under identity => the load-time invariant's reading is unchanged, bit for bit."""
    spec = _ident()
    for b in (0.0, 0.001, 1.0, 42.0):
        assert B.induced_step(b, spec) == b


# ---------------------------------------------------------------- 2. what the transform actually buys

def test_logit_cannot_reach_the_ceiling() -> None:
    """The entire reason logit exists: an update lands inside `(lo, hi)` by construction, not by a hard clamp as a backstop.

    `spo2` sits right against its 100% physical ceiling -- under identity,
    noise would push a proposal past `hi` and get clamped back, and "got
    clamped" is invisible in the output (only `clip_rate` would show it).
    """
    spec = _ident(indicator="spo2", lo=70.0, hi=100.0, transform="logit", at=97.4)
    x = B.to_transform(97.4, spec)
    for k in (1, 5, 10, 50):
        y = B.from_transform(x + k * 0.4, spec)
        assert y < spec.hi, f"+{k}×预算后 y={y} 够到了上界 {spec.hi}"
        assert y > spec.lo


def test_log_cannot_go_nonpositive() -> None:
    """The dual case for log: no matter how much is subtracted, the value stays positive."""
    spec = _ident(indicator="w", lo=30.0, hi=300.0, transform="log", at=89.35)
    x = B.to_transform(89.35, spec)
    for k in (1, 5, 20, 100):
        assert B.from_transform(x - k * 0.05, spec) > 0.0


# ---------------------------------------------------------------- 3. conversion is computed, not guessed

def test_raw_sd_round_trips_to_the_intended_raw_step() -> None:
    """Converting a raw-unit sd into the transform domain, then back into raw units, should land within roughly the same step size.

    It's a first-order conversion, so some curvature error is expected;
    requiring it within 20% is enough to pin down "the dimension is
    right." Without this test, a wrong factor in `raw_sd_to_transform_sd`
    would go unnoticed.
    """
    for spec, at, raw in (
        (_ident(indicator="spo2", lo=70.0, hi=100.0, transform="logit", at=97.4), 97.4, 0.20),
        (_ident(indicator="w", lo=30.0, hi=300.0, transform="log", at=89.35), 89.35, 0.12),
    ):
        tsd = B.raw_sd_to_transform_sd(raw, at, spec)
        back = B.induced_step(tsd, spec)
        assert 0.8 * raw <= back <= 1.2 * raw, (
            f"{spec.indicator}:原单位 {raw} → 变换域 {tsd:.6f} → 折回 {back:.6f},量纲对不上")


# ---------------------------------------------------------------- 4. negative control: a wrong statement must be caught

def _raw_entries() -> dict:
    import yaml
    return yaml.safe_load(REG.read_text(encoding="utf-8"))["streams"]


def _expected(v: float, bound: B.BoundSpec) -> float:
    """What the loader must put in effect for a raw-unit registry value `v`."""
    if bound.transform == "identity" or v == 0.0:
        return v
    s = B.raw_sd_to_transform_sd(abs(v), float(bound.at), bound)
    return s if v >= 0 else -s


def test_conversion_happens_in_the_loader_not_by_hand(tmp_path) -> None:
    """Note: the conversion is done by `load_stream_registry`; the registry always writes raw units.

    Leaving the conversion to whoever writes the registry entry (storing
    transform-domain numbers in the table) has two problems: (1) the numbers
    in the table get two possible readings (raw units under identity,
    transform-domain under log/logit) -- the ambiguity this file exists to
    remove, grown back in a different place; (2) the conversion function has
    no production call site -- a mechanism that exists but is never
    exercised.

    What this pins: for every entry on disk, the value in effect is the loader's
    conversion of the raw-unit number written in the table (read from the table,
    not restated here). The only non-identity stream, `spo2`, is
    `noise_external: haenv.wearable` with `loading`/`idio_sd` 0.0 -- its spread is
    the measured calibration in `wearable.CALIBRATION`, and a second layer of
    variance on top would count it twice (the rejection of that is pinned in
    tests/test_physio.py). A zero converts to zero, so the shipped table alone
    does not show that a conversion happens. The discriminating side therefore loads
    a copy of the table in which `spo2` carries non-zero raw-unit noise
    (-0.12 / 0.20): the effective values must be those numbers times the
    conversion factor, not the numbers themselves.
    """
    import yaml

    reg = physio_apply.load_stream_registry(REG)
    raw = _raw_entries()
    for name, sp in reg.specs.items():
        r = raw[name]
        assert sp.noise.idio_sd == pytest.approx(_expected(float(r["idio_sd"]), sp.bound),
                                                 abs=1e-12), f"{name}: idio_sd"
        (w,) = sp.noise.loadings.values()
        assert w == pytest.approx(_expected(float(r["loading"]), sp.bound), abs=1e-12), \
            f"{name}: loading"
        assert sp.noise.external == str(r.get("noise_external") or ""), f"{name}: external"
    assert reg.specs["spo2"].noise.external == "haenv.wearable"

    doc = yaml.safe_load(REG.read_text(encoding="utf-8"))
    ent = doc["streams"]["spo2"]
    ent.pop("noise_external", None)
    ent["loading"], ent["idio_sd"] = -0.12, 0.20
    alt = tmp_path / "physio_streams.yaml"
    alt.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    sp = physio_apply.load_stream_registry(alt).specs["spo2"]
    f = B.raw_sd_to_transform_sd(1.0, 97.4, sp.bound)        # the unit-conversion factor
    assert abs(sp.noise.idio_sd - 0.20 * f) < 1e-9, "idio_sd 没有被加载器换算"
    assert abs(list(sp.noise.loadings.values())[0] - (-0.12 * f)) < 1e-9, "loading 没换算(含符号)"
    assert sp.noise.idio_sd != 0.20, "生效值与表里写的相同 ⇒ 换算根本没发生"


def test_negative_control_noise_too_big_for_max_step_is_rejected() -> None:
    """Note: negative control -- once the units are aligned, "this stream's noise is genuinely too large" must still be catchable.

    Moving the conversion into the loader eliminates human error, not the
    invariant itself -- if genuinely excessive noise were allowed through,
    the conversion would have removed a gate without putting it back.
    """
    spec = _ident(indicator="x", lo=1.0, hi=100.0, max_step=1.0, transform="log", at=50.0)
    big = N.NoiseSpec(indicator="x", loadings={"g": 0.002}, idio_sd=0.1)    # 5.0 in raw units
    small = N.NoiseSpec(indicator="x", loadings={"g": 0.001}, idio_sd=0.002)
    assert 2 * B.induced_step(big.bound, spec) > spec.max_step, "过大的噪声没被抓"
    assert 2 * B.induced_step(small.bound, spec) <= spec.max_step, "合理的噪声反而过不去"


def test_negative_control_transform_without_operating_point_is_rejected() -> None:
    """A non-identity transform must specify an operating point. Without one, the only option is computing against the worst case across the full range, which cancels out the entire reason for using a transform."""
    with pytest.raises(B.BoundError, match="at"):
        _ident(transform="logit")
    with pytest.raises(B.BoundError, match="at"):
        _ident(transform="log", lo=1.0)


def test_negative_control_identity_with_operating_point_is_rejected() -> None:
    """Converse: identity must not be given an `at` -- writing an operating point that's never used would make it look like it's doing something."""
    with pytest.raises(B.BoundError, match="at"):
        _ident(at=50.0)


def test_negative_control_operating_point_must_be_inside_range() -> None:
    for bad_at in (70.0, 100.0, 101.0, 69.0):
        with pytest.raises(B.BoundError):
            _ident(indicator="spo2", lo=70.0, hi=100.0, transform="logit", at=bad_at)


# ---------------------------------------------------------------- 5. the registry on disk actually passes

def test_shipped_registry_loads_and_spo2_is_on_logit() -> None:
    """The registry actually loads, and `spo2` has actually switched over -- otherwise everything above is a fixture-only victory."""
    reg = physio_apply.load_stream_registry(REG)
    sp = reg.specs["spo2"]
    assert sp.bound.transform == "logit" and sp.bound.at is not None
    assert 2 * B.induced_step(sp.noise.bound, sp.bound) <= sp.bound.max_step
    # every other stream is on identity, and none of them may carry `at` (an `at` without a transform is a half-configured stream)
    for n, s in reg.specs.items():
        if n == "spo2":
            continue
        assert s.bound.transform == "identity" and s.bound.at is None, f"{n} 的状态不对"
