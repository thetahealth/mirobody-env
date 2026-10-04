"""The polarity key value for the noop probe must use the same convention on both sides.

## Convention

The probe, in `evaluate._noop_probe_for`, produces `gap` / `covered`, and the
table in `report.py` that flags "always claims data is insufficient" filters
on the same two values. If the two sides disagreed, the table's columns
would be empty and the detector could never fire.

## Which side carries the convention

The report reads the value the probe writes. Changing the judge side instead
would change `noop_polarity`'s on-disk value, which is a definition change
(requiring a recompute). The legacy values `absent`/`present` are still
accepted on the reading side, so rows that carry them are not judged as
never measured.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))


def _probe_polarities() -> set[str]:
    """The polarity values the production probe actually emits -- read from the source, not hand-copied a second time."""
    import ast
    import inspect
    from haenv import evaluate as E
    # The file that defines the probe builder, wherever it lives.
    src = pathlib.Path(inspect.getsourcefile(E.build_noop_probe)).read_text(encoding="utf-8")
    out: set[str] = set()
    for n in ast.walk(ast.parse(src)):
        # `"polarity": ("gap" if want_gap else "covered")`
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value == "polarity" and isinstance(v, ast.IfExp):
                    for side in (v.body, v.orelse):
                        if isinstance(side, ast.Constant) and isinstance(side.value, str):
                            out.add(side.value)
    return out


def test_report_covers_every_polarity_the_probe_emits():
    """The set of values the report side filters on must cover every value the probe actually emits.

    This isn't a string-comparison wording check: it takes the probe's
    own set of emitted values from the source, feeds each one through the
    report side's two filters, and requires that every value lands on
    some side. If one doesn't, that tier's rows are permanently empty in
    the table.
    """
    from haenv.report import rank_models      # noqa: F401  ensures the module actually imports
    import haenv.report as R
    src = pathlib.Path(R.__file__).read_text(encoding="utf-8")
    # get the two sets the report side declares (they're literal tuples in the module)
    import ast
    gap = cov = None
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name) and t.id in ("_GAP", "_COV") and isinstance(n.value, ast.Tuple):
                vals = {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
                if t.id == "_GAP":
                    gap = vals
                else:
                    cov = vals
    assert gap and cov, "报告侧没有声明极性取值集 ⇒ 这条断言没有对象"
    emitted = _probe_polarities()
    assert emitted, "从探针所在模块取不到极性取值 ⇒ 前提不成立"
    missing = sorted(emitted - (gap | cov))
    assert not missing, (
        f"探针会产出 {sorted(emitted)},而报告侧只认 {sorted(gap | cov)} ⇒ "
        f"{missing} 这些行在表上**恒空**")


def test_the_two_sides_do_not_overlap():
    """Converse: the two sides' value sets must not overlap -- the same value counting as both "no data" and "has data" would be meaningless."""
    import ast
    import haenv.report as R
    src = pathlib.Path(R.__file__).read_text(encoding="utf-8")
    sets = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name) and t.id in ("_GAP", "_COV") and isinstance(n.value, ast.Tuple):
                sets[t.id] = {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
    assert set(sets) == {"_GAP", "_COV"}, sets
    assert not (sets["_GAP"] & sets["_COV"]), f"两侧取值相交:{sets['_GAP'] & sets['_COV']}"


def test_probe_and_judge_agree_on_the_key():
    """The probe writes `polarity`, and the judge renames it to `noop_polarity` -- the meaning must not change in transit."""
    from haenv.judges import judge_noop_probe
    probe = {"target": "steps", "polarity": "gap", "truth_present": False,
             "window": [10, 20], "n_pts_in_window": 0, "T": 30}

    class _Out:
        data_quality = {}
        _raw = {"signal_quality": {"steps": "该时段没有读数"}}
        forecast = {}
        drivers = []
        action = {}
        cited_evidence = []

    r = judge_noop_probe(_Out(), probe)
    assert r["noop_polarity"] == "gap", f"极性被改写了:{r.get('noop_polarity')}"
