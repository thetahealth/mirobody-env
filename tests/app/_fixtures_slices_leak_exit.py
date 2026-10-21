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
from _paths import ROOT as _TEST_ROOT

import pathlib

import sys

import types

import yaml

ROOT = _TEST_ROOT

sys.path.insert(0, str(ROOT))

_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

import haenv.app.evaluate as EV                    # noqa: E402

from haenv import judges as _j                      # noqa: E402

import haenv.interaction.solve_guard as _sg                # noqa: E402

import haenv.scoring.tracks as _tk                     # noqa: E402

import haenv.world.wq as _wq                        # noqa: E402

from _patch_bound import patch_bound  # noqa: E402

from haenv.runtime.state import RunContext as _RunContext  # noqa: E402

_CTX = _RunContext()   # the run these tests drive; monkeypatch restores its fields

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
    patch_bound(monkeypatch, "build_instance", lambda raw, T: (_SP(), _VP()))
    monkeypatch.setattr(_wq, "injected_manifest", lambda cid: {})
    monkeypatch.setattr(_CTX, "resp_path", None)
    # Note: `guarded_solve` is imported **inside `_row_slices`'s function
    #    body** => it must be stubbed on the `solve_guard` module; stubbing
    #    it on `evaluate` has no effect.
    monkeypatch.setattr(_sg, "guarded_solve",
                        lambda solver, sp, t, **kw: per_slice(int(t)))
    monkeypatch.setattr(EV.verifier_mod, "grade",
                        lambda out, vp, vis, **kw: types.SimpleNamespace(hard_gate_failures=[]))
    monkeypatch.setattr(_j, "run_judges", lambda geom, subjects, vp, ctx=None: {
        "_saw_last": (subjects.get("last") is not None)
        if isinstance(subjects, dict) else None})
    monkeypatch.setattr(_j, "judge_noop_probe", lambda out, spec, **kw: {})
    monkeypatch.setattr(_j, "judge_quant_probe", lambda out, spec, t_max=None: {})
    patch_bound(monkeypatch, "_premise_row", lambda cid, out, **kw: {})
    # same as above: `alternative_a1` / `_answer_text` / `_differential` are also imported from `tracks` inside the function body
    monkeypatch.setattr(_tk, "alternative_a1", lambda out: {"a1": None})
    monkeypatch.setattr(_tk, "_answer_text", lambda out: "")
    monkeypatch.setattr(_tk, "_differential", lambda out: [])


__all__ = [name for name in globals() if not name.startswith("__")]
