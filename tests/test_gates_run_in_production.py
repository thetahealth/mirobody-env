"""Every gate in `haenv/gates.py` runs in production, with real inputs.

A gate can be unit-tested with a `raw` argument while production calls it with
`raw=None`, so it judges nothing and its tests stay green. Testing the function
does not test its call site.

This test watches the call sites. It wraps every `check_*` defined in
`haenv/gates.py`, runs the production entry (`haenv build`, in process, on cases of
the production pack) and requires each gate to be called and each of its arguments
to be non-empty in at least one call. Per-case gates run inside `build_case`,
batch gates in the CLI after it; both are covered by the same run.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import haenv                                                        # noqa: E402,F401  mounts the kernel
from haenv import cli                                              # noqa: E402
from haenv import gates                                             # noqa: E402
from haenv import job as J                                          # noqa: E402

JOB = ROOT / "inputs" / "ddx-timeline.job.yaml"
N_CASES = 8

#: (gate, argument) pairs that are empty by design, with the reason. Anything
#: else that is empty in every call is a gate that judges nothing.
EMPTY_BY_DESIGN: dict[tuple[str, str], str] = {
    ("check_demographic_plausibility", "raw"):
        "declaration gate: runs before the world exists and reads cs.raw",
    ("check_gold_line_in_known_conditions", "raw"):
        "declaration gate: runs before the world exists and reads cs.raw['disease']",
}


def _defined_gates() -> list[str]:
    src = (ROOT / "haenv" / "gates.py").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^def (check_\w+)\(", src, flags=re.M)))


def _empty(v) -> bool:
    return v is None or (hasattr(v, "__len__") and not isinstance(v, (int, float)) and len(v) == 0)


def _hollow(calls: dict[str, list[dict]]) -> list[str]:
    out = []
    for name, rows in calls.items():
        for arg in sorted({a for r in rows for a in r}):
            if (name, arg) in EMPTY_BY_DESIGN:
                continue
            if all(_empty(r.get(arg)) for r in rows):
                out.append(f"{name}({arg}=empty in all {len(rows)} calls)")
    return out


def _run(monkeypatch, tmp_path) -> dict[str, list[dict]]:
    import inspect
    calls: dict[str, list[dict]] = {}
    for name in _defined_gates():
        orig = getattr(gates, name)
        sig = inspect.signature(orig)

        def spy(*args, __orig=orig, __name=name, __sig=sig, **kw):
            bound = __sig.bind(*args, **kw)
            calls.setdefault(__name, []).append(dict(bound.arguments))
            return __orig(*args, **kw)

        monkeypatch.setattr(gates, name, spy)
    monkeypatch.setenv("HAENV_OUTPUT_ROOT", str(tmp_path))
    ids = ",".join(cs.case_id for cs in J.load_job(JOB).cases[:N_CASES])
    rc = cli.main(["build", str(JOB), "--gen", "deterministic", "--fresh", "--cases", ids])
    assert rc in (0, 4), f"haenv build exited {rc}"
    return calls


def test_every_gate_is_called_with_real_inputs(monkeypatch, tmp_path):
    calls = _run(monkeypatch, tmp_path)
    names = _defined_gates()
    assert len(names) >= 25, f"scan surface looks wrong: {names}"

    never = [n for n in names if n not in calls]
    assert not never, f"gates defined in gates.py but never called by `haenv build`: {never}"

    hollow = _hollow(calls)
    assert not hollow, "gates that received no input to judge: " + "; ".join(hollow)


def test_hollow_detection_controls():
    """Negative control: an argument empty in every call is reported. Positive
    controls: one non-empty call, or a listed design exception, is not."""
    assert _hollow({"check_x": [{"raw": None, "cs": 1}, {"raw": {}, "cs": 2}]}) == [
        "check_x(raw=empty in all 2 calls)"]
    assert _hollow({"check_x": [{"raw": None}, {"raw": {"a": 1}}]}) == []
    assert _hollow({"check_demographic_plausibility": [{"raw": None, "cs": 1}]}) == []


def test_design_exceptions_are_still_empty(monkeypatch, tmp_path):
    """An exception that no longer applies is removed, not kept as a blanket pass."""
    calls = _run(monkeypatch, tmp_path)
    stale = [f"{g}.{a}" for (g, a) in EMPTY_BY_DESIGN
             if g in calls and not all(_empty(r.get(a)) for r in calls[g])]
    assert not stale, f"EMPTY_BY_DESIGN entries that now receive input: {stale}"
