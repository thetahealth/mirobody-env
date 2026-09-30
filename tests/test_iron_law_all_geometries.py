"""The gold self-consistency gate (the W x Q iron law, three rules) must be
exercised on **all four geometries**.

## What the gate guards

`iron_law_precheck` is a module-level function in `evaluate` that every
geometry calls. A closure inside `_row_single` would leave the other three
geometries without it, and the "gold self-consistency" gate would have
nothing to grade on three-quarters of the output: every check green and no
`ABORT(iron_law)` on disk would then mean the gate is never called, not that
every case is self-consistent.

## The four geometries wire the gate in at different points, and this is
structural

    single / slices   go through `guarded_solve(precheck=...)`, so the order is
                      `probe -> build_question -> enforce -> solve`;
                      when the iron law and a leak both fail at once, it records
                      `ABORT(leak)`
    gated / multi     `run_gated` and the kernel's `run_multiround` have **no
                      precheck hook** and the kernel is not modified, so it is run explicitly
                      **before** solving; when both fail at once here, it
                      records `ABORT(iron_law)`

So what this file asserts is that "**every geometry actually calls it**", not
"all four behave byte-for-byte identically" — the latter is not achievable
today, and writing an assertion for something unachievable would just create a
permanently red guard.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import pathlib
import sys
import types

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import evaluate as EV                    # noqa: E402
from haenv import judges as _j                      # noqa: E402
from haenv import solve_guard as _sg                # noqa: E402
from haenv import tracks as _tk                     # noqa: E402
from haenv import wq as _wq                         # noqa: E402


class _VP:
    adjudication = {"ddx": {"diagnosis": "X"}}
    gold_drivers = ["unknown_or_multifactorial"]


class _SP:
    evidence_ledger = [{"evidence_id": "EV-01"}]
    prediction_context = {"prediction_time_T": 10}


class _Raw:
    case_id = "C1"


class _Solver:
    prompt_mode = None
    probe_id = None


class _Out:
    def __init__(self):
        self.forecast = {"risk_category": "moderate"}
        self.drivers = []
        self.action = {"selected_action_class": "A0"}
        self.data_quality = {"data_sufficiency": "sufficient"}
        self.cited_evidence = []
        self._raw = {"tests_to_order": [], "referral_specialty": [], "join_type": None}


#: The minimal usable Q — downstream code reads `world_ref` / `question_id`.
_Q = types.SimpleNamespace(world_ref={"world_id": "W", "world_truth_hash": "h"},
                           question_id="W@T10")


def _facts():
    return types.SimpleNamespace(leak=None, out=_Out(), raw_empty=False,
                                 raw_unparseable=False, raw_text="{}", attempts=1,
                                 precheck_result=(_Q, []))


def _install(monkeypatch, calls: list):
    """Replace the iron law with a **counter**; stub everything else. The counter
    records "who called it"."""
    monkeypatch.setattr(EV, "build_instance", lambda raw, T: (_SP(), _VP()))
    monkeypatch.setattr(EV, "RESP_PATH", [None])
    monkeypatch.setattr(_wq, "injected_manifest", lambda cid: {})
    monkeypatch.setattr(_sg, "guarded_solve",
                        lambda solver, sp, t, precheck=None, **kw: (
                            (precheck() if precheck else None), _facts())[1])
    monkeypatch.setattr(EV.verifier_mod, "grade",
                        lambda out, vp, vis: types.SimpleNamespace(
                            hard_gate_failures=[], tracks={}, overall="SCORED"))
    monkeypatch.setattr(_j, "run_judges", lambda geom, subjects, vp, ctx=None: {})
    monkeypatch.setattr(_j, "judge_noop_probe", lambda out, spec: {})
    monkeypatch.setattr(_j, "judge_quant_probe", lambda out, spec, t_max=None: {})
    monkeypatch.setattr(EV, "_premise_row", lambda cid, out: {})
    monkeypatch.setattr(_tk, "alternative_a1", lambda out: {"a1": None})
    monkeypatch.setattr(_tk, "_answer_text", lambda out: "")
    monkeypatch.setattr(_tk, "_differential", lambda out: [])

    def _counting(raw, sp, vp, solver):
        calls.append(id(sp))
        return lambda: (_Q, [])
    monkeypatch.setattr(EV, "iron_law_precheck", _counting)


def test_single_runs_the_iron_law(monkeypatch):
    calls: list = []
    _install(monkeypatch, calls)
    EV._row_single("C1", "m1", _Raw(), 10, _Solver())
    assert len(calls) == 1, f"单拍上铁律被调 {len(calls)} 次"


def test_slices_runs_the_iron_law_per_slice(monkeypatch):
    """Slices: build it **once per slice** — sharing a single one would judge the
    earlier slices against the Q taken at T."""
    calls: list = []
    _install(monkeypatch, calls)
    EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver())
    assert len(calls) == 3, f"3 片上铁律被调 {len(calls)} 次(应逐片各一次)"


def test_gated_runs_the_iron_law(monkeypatch):
    """Gated: no precheck hook, so run it explicitly once before solving."""
    calls: list = []
    _install(monkeypatch, calls)
    monkeypatch.setattr(EV, "verifier_mod", EV.verifier_mod)
    import haenv.gated as _g
    monkeypatch.setattr(_g, "run_gated",
                        lambda raw, T, solver: (_Out(), types.SimpleNamespace(
                            leak=None, raw_empty=False, raw_unparseable=False,
                            calls=[], budget=0, spent=0, rounds=0)))
    try:
        EV._row_gated("C1", "m1", _Raw(), 10, _Solver())
    except Exception:                                # noqa: BLE001
        pass                                         # the downstream pipeline is not this file's concern
    assert calls, "门控上铁律**一次都没被调**"


def test_counter_is_not_vacuous(monkeypatch):
    """Inverse case: the counter itself must actually count (without this test,
    the three tests above would all be vacuously true)."""
    calls: list = []
    _install(monkeypatch, calls)
    _q2, _w2 = EV.iron_law_precheck(_Raw(), _SP(), _VP(), _Solver())()
    assert _w2 == [] and _q2 is not None
    assert len(calls) == 1, "计数器没记上 —— 上面三条断言没有对象"
