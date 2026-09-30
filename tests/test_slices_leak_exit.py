"""The slices geometry's leak exit.

Single-shot and multi-round cells end in `ABORT(leak)` when the solver output leaks.
The slices geometry has the same exit: a slice that leaks keeps only `{"t", "leak"}`
and the loop continues; when every slice leaks, `_out_last` stays `None`, `_gates`
gets `[]`, and the cell must still end in `ABORT(leak)` with no score, not `SCORED`.

## Three tests, two directions

* every slice leaks => `ABORT(leak)`, and the leak reason is named in the output;
* partial leak => no ABORT (the leaked slice is voided, the rest keep running);
* the last slice leaks => `_out_last` is `None`, and the 8 judges on the `last`
  subject go through `mount_subject_missing`, never substituted with an earlier slice.

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
from haenv import wq as _wq                        # noqa: E402


class _VP:
    adjudication = {"ddx": {"diagnosis": "X"}}
    gold_drivers = ["unknown_or_multifactorial"]


class _SP:
    evidence_ledger = [{"evidence_id": "EV-01"}]
    # All four geometries go through `wq.build_question`, so the stub
    # supplies the two surfaces it reads.
    prediction_context = {"prediction_time_T": 10}


class _Raw:
    case_id = "C1"


class _Solver:
    prompt_mode = None
    probe_id = None


class _Out:
    """A minimal usable solver output -- fills in only the surfaces the
    judges actually read."""

    def __init__(self):
        self.forecast = {"risk_category": "moderate"}
        self.drivers = []
        self.action = {"selected_action_class": "A0"}
        self.data_quality = {"data_sufficiency": "sufficient"}
        self.cited_evidence = []
        self._raw = {"tests_to_order": [], "referral_specialty": [], "join_type": None}


def _facts(*, leak=None, out=None):
    return types.SimpleNamespace(
        leak=leak, out=out or _Out(), raw_empty=False, raw_unparseable=False,
        raw_text="{}", attempts=1)


def _install(monkeypatch, per_slice):
    """`per_slice` is a `t -> facts` function. Everything else is stubbed
    out, leaving only the path this file needs to test."""
    monkeypatch.setattr(EV, "build_instance", lambda raw, T: (_SP(), _VP()))
    monkeypatch.setattr(_wq, "injected_manifest", lambda cid: {})
    monkeypatch.setattr(EV, "RESP_PATH", [None])
    # Note: `guarded_solve` is imported **inside `_row_slices`'s function
    #    body** => it must be stubbed on the `solve_guard` module; stubbing
    #    it on `evaluate` has no effect.
    monkeypatch.setattr(_sg, "guarded_solve",
                        lambda solver, sp, t, **kw: per_slice(int(t)))
    monkeypatch.setattr(EV.verifier_mod, "grade",
                        lambda out, vp, vis: types.SimpleNamespace(hard_gate_failures=[]))
    monkeypatch.setattr(_j, "run_judges", lambda geom, subjects, vp, ctx=None: {
        "_saw_last": (subjects.get("last") is not None)
        if isinstance(subjects, dict) else None})
    monkeypatch.setattr(_j, "judge_noop_probe", lambda out, spec: {})
    monkeypatch.setattr(_j, "judge_quant_probe", lambda out, spec, t_max=None: {})
    monkeypatch.setattr(EV, "_premise_row", lambda cid, out: {})
    # same as above: `alternative_a1` / `_answer_text` / `_differential` are also imported from `tracks` inside the function body
    monkeypatch.setattr(_tk, "alternative_a1", lambda out: {"a1": None})
    monkeypatch.setattr(_tk, "_answer_text", lambda out: "")
    monkeypatch.setattr(_tk, "_differential", lambda out: [])


def test_all_slices_leak_aborts(monkeypatch):
    """Every slice leaking => `ABORT(leak)`, with the reason named
    explicitly."""
    _install(monkeypatch, lambda t: _facts(leak=[f"leak_at_{t}"]))
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver())
    assert row["overall"] == "ABORT(leak)", row.get("overall")
    assert row.get("leak_scope") == "all_slices"
    assert set(row.get("leak") or []) == {"leak_at_10", "leak_at_20", "leak_at_30"}, row.get("leak")
    assert row.get("answered_slices") == 0


def test_partial_leak_does_not_abort(monkeypatch):
    """Negative control: a **partial** leak must not ABORT.

    An assertion that any leak fails the whole cell would also condemn the
    intended behavior of voiding that slice and running the rest.
    """
    _install(monkeypatch, lambda t: _facts(leak=["x"]) if t == 10 else _facts())
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver())
    assert row["overall"] != "ABORT(leak)", row.get("overall")
    assert row.get("leak_scope") is None


def test_last_slice_leak_drops_out_last(monkeypatch):
    """The last slice leaking => the `last` subject is absent, **never
    substituted with an earlier slice**."""
    _install(monkeypatch, lambda t: _facts() if t in (10, 20) else _facts(leak=["x"]))
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver())
    assert row["overall"] != "ABORT(leak)", "只有末片泄漏,不该整格 ABORT"
    assert row.get("_saw_last") is False, "末片泄漏了,`last` 仍被喂进判据 ⇒ 拿早片顶替"


def test_healthy_path_still_passes_last(monkeypatch):
    """Reverse: with no leak, `last` **must** still be fed in as usual
    (otherwise the previous test would be vacuously true)."""
    _install(monkeypatch, lambda t: _facts())
    row = EV._row_slices("C1", "m1", _Raw(), [10, 20, 30], _Solver())
    assert row.get("_saw_last") is True, "正常路径下 `last` 没被喂进判据"
