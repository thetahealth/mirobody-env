"""A raw field's default value is allowed to exist in exactly one place.

## Invariant

Inline default values for the same raw field in two modules (for example
`events.facts_of` and `build.premise_spec`) can silently disagree:

| Field | events.py | build.py |
|---|---|---|
| `drug` | `metformin` | `tirzepatide` |
| `start_weight` | `90.0` | `98.0` |
| `nadir_weight` | `start - 8` | `start - 12` |

Whoever leaves out a `drug`, two modules would disagree about what medication
that person is on, and neither would raise an error (the same shape as a
`gold_drivers` field silently becoming `[None]`). A default therefore lives in
one place and every consumer reads it from there.

## The load-bearing test

`test_no_module_inlines_a_raw_default` -- it guards against the next
copy. Merging the two into one without blocking a third would let this
defect come right back.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv.job import CaseSpec, raw_field  # noqa: E402

MANAGED = set(CaseSpec.RAW_DEFAULTS) | set(CaseSpec.RAW_DERIVED_DEFAULTS)
#: `job.py` itself is the single source of truth; `extract.py` only mentions these names in a docstring.
_OWNER = "haenv/job.py"


def test_the_two_old_values_cannot_both_survive():
    """Negative control: the two candidate value sets (`events.py` and `build.py` inline defaults) differ from each other -- if they were the same, the disagreement this file guards against could not exist."""
    assert raw_field({}, "drug") in ("tirzepatide", "metformin")
    assert raw_field({}, "start_weight") in (90.0, 98.0)
    # The set comparison below holds while the two candidates differ; the
    # single-source resolution itself is asserted by the next tests.
    assert {"tirzepatide", "metformin"} != {raw_field({}, "drug")}, "只剩一个值 = 收口成功"


def test_single_source_resolves_every_managed_key():
    for k in MANAGED:
        v = raw_field({}, k)
        assert v is not None, k


def test_explicit_value_wins_over_default():
    assert raw_field({"drug": "semaglutide"}, "drug") == "semaglutide"
    assert raw_field({"start_weight": 77.0}, "start_weight") == 77.0


def test_derived_default_follows_the_field_it_derives_from():
    """`nadir_weight` derives from `start_weight` -- change the starting point, and the derived value must follow it."""
    assert raw_field({"start_weight": 90.0}, "nadir_weight") == 78.0
    assert raw_field({}, "nadir_weight") == raw_field({}, "start_weight") - 12.0


def test_unregistered_key_raises_instead_of_returning_none():
    """"No default found" does not collapse into `None` -- if it did, downstream code would get `None` and collapse that into 0."""
    with pytest.raises(KeyError, match="has no registered default"):
        raw_field({}, "这个字段没登记过")


def test_none_value_falls_through_to_default():
    """`dict.get(k, d)` does not kick in when the key is present but the value is `None` -- the root cause of a `gold_drivers` field becoming `[None]`."""
    assert raw_field({"drug": None}, "drug") == raw_field({}, "drug")


# ─────────────────────────────────────────────── load-bearing: block the next copy

#: Sentinel values for "this key is absent" -- they fabricate no fact,
#: downstream code treats them as absence. In contrast, defaults like
#: `"obesity"` / `"F"` / `98.0` that look like real values are exactly
#: what this test blocks: they make "the source text never said this" and
#: "the source text said obesity" look identical downstream.
_ABSENT_SENTINELS = ("", 0, 0.0, None, False)


def _inline_raw_defaults(path: pathlib.Path) -> list[str]:
    """Scans for the shape `<dict>.get("<managed key>", <fabricated default>)`.

    The scan is by AST, not by text -- a comment reading
    `raw.get("drug", "metformin")` does not count as a call (this repo's
    own `extract.py` docstring has exactly one).

    The sentinel exemption is semantic, not shape-based: a default that falls inside `_ABSENT_SENTINELS` expresses
    "this key is absent," fabricating no fact; anything else counts as a
    fabricated default, including a module-level constant
    (`.get("sampling_days", WEIGHT_STEP)` is a second copy of the same
    number that's already in the registry as `1`).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2):
            continue
        key, dflt = node.args
        if not (isinstance(key, ast.Constant) and key.value in MANAGED):
            continue
        if isinstance(dflt, ast.Constant) and any(
                dflt.value is s or dflt.value == s for s in _ABSENT_SENTINELS):
            continue                       # sentinel: expresses "absent," fabricates nothing
        out.append(f"{path.name}:{node.lineno} .get({key.value!r}, {ast.unparse(dflt)})")
    return out


def test_no_module_inlines_a_raw_default():
    """This file's load-bearing test -- nobody but `job.py` is allowed to write `raw.get(<managed>, <default>)`.

    Merging the two copies into one without blocking a third would let
    this defect come right back -- the family covers at least
    `gold_drivers` and `drug`.
    """
    bad: list[str] = []
    for p in sorted((ROOT / "haenv").glob("*.py")):
        if p.relative_to(ROOT).as_posix() == _OWNER:
            continue
        bad += _inline_raw_defaults(p)
    assert not bad, (
        "这些地方又内联了一份 raw 缺省值(唯一来源是 job.CaseSpec.RAW_DEFAULTS):\n  "
        + "\n  ".join(bad))


def test_the_scanner_actually_catches_things():
    """Negative control: the scanner must genuinely be able to report, otherwise the previous test is vacuously true.

    One case for each failure mode: a comment posing as a call (must not
    count), and a real call (must count).
    """
    import tempfile
    src = (
        "# raw.get('drug', 'metformin') 这一行只是注释\n"
        "def f(raw):\n"
        "    a = raw.get('drug', 'metformin')\n"     # fabricated -> must catch
        "    b = raw.get('drug')\n"                  # no default -> doesn't count
        "    c = raw.get('没登记过的键', 1)\n"        # not in the managed set -> doesn't count
        "    d = raw.get('sex', '')\n"               # sentinel -> doesn't count
        "    e = raw.get('sampling_days', 0)\n"      # sentinel -> doesn't count
        "    g = raw.get('sampling_days', SOME_CONST)\n"   # a module constant is fabricated too -> must catch
        "    return a, b, c, d, e, g\n"
    )
    with tempfile.TemporaryDirectory(dir=ROOT) as td:
        p = pathlib.Path(td) / "probe.py"
        p.write_text(src, encoding="utf-8")
        hits = _inline_raw_defaults(p)
    assert len(hits) == 2, f"应当抓到第 3、8 行两处,实得 {hits}"
    assert [h.split(":")[1].split(" ")[0] for h in hits] == ["3", "8"], hits
