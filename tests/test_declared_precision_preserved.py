"""The decimal precision in the prompt must be the **declared** one, not a digit
count hard-coded in the physiology layer.

## Declared precision

Rounding every stream to a fixed 4 digits in `physio/apply.py` would put
values such as `weight = 88.4011 kg`, `steps = 7431.0026 steps` into the
prompt once observation noise is added. Each stream keeps its declared
precision instead.

## Why this is tested as a "shortcut", not as "cosmetics"

The second-order harm matters more than the first: a stream has 4 decimal
digits when physio is on and only 2 when it's off, so "the reading with more
decimal digits" maps one-to-one to "was touched by the physiology layer". The
benchmark does not let "the smoothest reading is the answer" work as a shortcut,
and it must not grow an isomorphic shortcut on decimal digits instead.

So the load-bearing test here is `test_digit_profile_cannot_tell_physio_on_from_off`
— it measures **distinguishability**, not appearance.

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

from haenv import events as E  # noqa: E402


def _digits(v) -> int:
    s = repr(float(v))
    return len(s.split(".")[-1].rstrip("0")) if "." in s and not s.endswith(".0") else 0


def _stream(vals):
    return [{"ts": i, "value": float(v)} for i, v in enumerate(vals)]


# ─────────────────────────────────────────────── There is exactly one source of declared precision

def test_declared_digits_come_from_metrics_not_a_second_table():
    """`METRICS` is the sole source; the `weight` entry is patched in separately
    and must agree with `build`'s `round(v,2)`."""
    for m in E.METRICS:
        assert E._declared_ndigits(m.name) == int(m.ndigits), m.name
    assert E._declared_ndigits("weight") == 2, "与 build.py::_weight_series 的 round(v,2) 对不上"


def test_unknown_stream_is_left_alone_not_guessed():
    """"precision unknown" and "precision is 4 digits" are not the same
    thing — leave it untouched if nothing was declared."""
    assert E._declared_ndigits("完全没登记过的流") is None
    ld = {"完全没登记过的流": _stream([1.23456789])}
    E._round_to_declared_digits(ld)
    assert ld["完全没登记过的流"][0]["value"] == 1.23456789


# ─────────────────────────────────────────────── Negative control: the raw defect must be caught

def test_catches_the_four_decimal_defect():
    """Negative control — the physiology layer's raw output (4 digits) must
    be brought back to the declared precision."""
    ld = {"weight": _stream([88.4011, 90.6013, 79.2507])}
    E._round_to_declared_digits(ld)
    got = [p["value"] for p in ld["weight"]]
    assert got == [88.4, 90.6, 79.25], got
    assert all(_digits(v) <= 2 for v in got)


@pytest.mark.parametrize("name,nd", [("steps", 0), ("sleep_hours", 1),
                                     ("skin_temp", 2), ("spo2", 1),
                                     ("body_temp", 1), ("resting_hr", 0)])
def test_each_stream_gets_its_own_declared_digits(name, nd):
    """Each stream gets its own precision — flattening everything to one digit
    count is also wrong (steps should have no decimals, skin_temp should have
    two)."""
    assert E._declared_ndigits(name) == nd
    ld = {name: _stream([12.3456789, 98.7654321])}
    E._round_to_declared_digits(ld)
    assert all(_digits(p["value"]) <= nd for p in ld[name]), ld[name]


def test_integer_streams_become_integral():
    """A stream with `ndigits=0` must not keep a decimal tail — "7431.0026 steps"
    is a number no wearable can actually report."""
    ld = {"steps": _stream([7431.0026, 8102.9974])}
    E._round_to_declared_digits(ld)
    assert [p["value"] for p in ld["steps"]] == [7431.0, 8103.0]


# ─────────────────────────────────────────────── Load-bearing: decimal digits must not leak the physio switch

def test_digit_profile_cannot_tell_physio_on_from_off():
    """This is this file's load-bearing wall.

    With physio off, the prompt comes from `round(v, ndigits)`; with physio on,
    after passing through this function it must have the **same shape**. If the
    two decimal-digit profiles are distinguishable, "the reading with more
    decimal digits" becomes a shortcut that can be judged without reading any
    clinical content — the same family as the "smoothest reading is the answer"
    shortcut.
    """
    off = {"weight": _stream([88.40, 90.60, 79.25]),
           "steps": _stream([7431.0, 8103.0]),
           "sleep_hours": _stream([7.2, 6.8])}
    on = {"weight": _stream([88.4011, 90.6013, 79.2507]),
          "steps": _stream([7431.0026, 8102.9974]),
          "sleep_hours": _stream([7.2331, 6.7688])}
    E._round_to_declared_digits(on)

    def profile(ld):
        return {k: sorted({_digits(p["value"]) for p in v}) for k, v in ld.items()}

    assert profile(on) == profile(off), (
        f"physio 开/关的小数位剖面可分 ⇒ 捷径还在\n开={profile(on)}\n关={profile(off)}")


def test_idempotent():
    """Running it twice must be byte-for-byte identical — otherwise `--regen` and
    recompute would produce different prompts."""
    ld = {"weight": _stream([88.4011]), "steps": _stream([7431.0026])}
    E._round_to_declared_digits(ld)
    once = [(k, [p["value"] for p in v]) for k, v in sorted(ld.items())]
    E._round_to_declared_digits(ld)
    twice = [(k, [p["value"] for p in v]) for k, v in sorted(ld.items())]
    assert once == twice
