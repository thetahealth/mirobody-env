"""A judge raising an error must not collapse into `SCORED`; zero evidence
citations must not be recorded as 100% grounded.

## `<judge>_error` needs a consumer

When a judge crashes, the row gets a `<judge>_error` field written, and left
at that, the row still records `SCORED`. `run_judges`'s own comment says:
"to whoever reads it, 'this dim has no value' looks identical to 'this dim
doesn't apply'." But those are two different things -- **the former is a
measurement failure, the latter is a measurement result** -- and treating
them as the same shape in the denominator means counting a failure as a
valid measurement.

=> Collected into a named `judge_errors` field, rejected by
`assert_publishable`. Same rule as "no fingerprint": **"a dim that failed to
be measured" is not a publishable state.**

## A zero denominator must not get a perfect score

With `grounded_rate = ... if total_ev_refs > 0 else 1.0`, **citing zero
evidence** would be recorded as **100% grounded**. The direction is backwards:
it rewards an answer that cites no evidence at all, the exact opposite of
what this dim is meant to measure.

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

from verifier_core.vintage import NotPublishable, assert_publishable   # noqa: E402

_FP = "a48408afeea9545f"


def _row(**kw):
    r = {"case": "C1", "solver": "m1", "judging_sha16": _FP, "overall": "SCORED"}
    r.update(kw)
    return r


# ──────────────────────────────────────────── judge errors

def test_judge_error_blocks_publication():
    """A row with `judge_errors` => the publish gate must reject
    it, and **name** which judge."""
    with pytest.raises(NotPublishable) as e:
        assert_publishable([_row(), _row(case="C2", judge_errors=["dx_rival"])], _FP)
    assert "judge threw an error" in str(e.value), str(e.value)
    assert "dx_rival" in str(e.value), "没点名是哪条判据崩的"


def test_clean_rows_still_pass():
    """Reverse: a batch with no judge errors **must still pass as usual** --
    otherwise this gate would always be red."""
    assert_publishable([_row(), _row(case="C2")], _FP)     # not raising counts as passing


def test_empty_judge_errors_is_not_a_failure():
    """Reverse: `judge_errors` present but as an **empty list** => doesn't
    count as an error (empty != present)."""
    assert_publishable([_row(judge_errors=[])], _FP)


def test_run_judges_names_the_failing_judge():
    """One judge crashing inside `run_judges` => both `<name>_error` gets
    written, and it goes into `judge_errors`."""
    from haenv import judges as J

    class _Boom:
        name = "boom_judge"
        kinds = ()

        @staticmethod
        def fn(subject, vp, ctx):
            raise ValueError("故意炸")

    # **Installs a judge guaranteed to crash into the dispatch table**, runs
    # it once, and checks whether it got named.
    # Calls the real `run_judges`; does not scan source strings.
    import haenv.mount_table as MT
    j = _Boom()
    orig_app = J.applicable
    orig_sub = MT.subject_of
    try:
        J.applicable = lambda geom, kind, vp: (j,)
        MT.subject_of = lambda name, geom: "out" if name == "boom_judge" else orig_sub(name, geom)
        out = J.run_judges("single", {"out": object()}, _VPStub())
    finally:
        J.applicable = orig_app
        MT.subject_of = orig_sub
    assert out.get("judge_errors") == ["boom_judge"], (
        f"判据崩了却没被点名进 `judge_errors`:{out}")
    assert "boom_judge_error" in out, "逐判据的 `_error` 消息也要留着(查错要用)"


class _VPStub:
    adjudication = {}
    gold_drivers = []
    outcome_label = "event_occurred"
    case_id = "C1"


# ──────────────────────────────────────────── zero denominator

def test_zero_refs_is_unmeasured_not_perfect():
    """Zero evidence citations => `grounded_rate` is recorded as `None` plus
    a named flag, **not 1.0**.

    This test **calls the real function** instead of scanning source
    strings: a string scan tests what the code looks like, not what it does.
    """
    from haenv.process import judge_commitment_consistency as f
    # has a commitment, but **cites zero evidence**
    r = f({"commitments": [{"claim": "体重在降", "falsified_by": "体重回升"}]}, [])
    assert r["trace_commitments_n_refs"] == 0, r
    assert r["trace_commitments_grounded_rate"] is None, (
        f"零引用被记成 {r['trace_commitments_grounded_rate']} —— "
        f"那是「没量到」冒充「量到了没问题」")
    assert r["trace_commitments_grounded_unmeasured"] is True, r


def test_grounded_rate_still_measured_when_refs_exist():
    """Reverse: with citations present, the rate is still computed normally
    (otherwise the zero-refs test could be satisfied by "always return
    `None`")."""
    from haenv.process import judge_commitment_consistency as f
    r = f({"commitments": [{"claim": "见 EV-01 与 EV-99", "falsified_by": "反证"}]},
          ["EV-01"])
    assert r["trace_commitments_n_refs"] >= 1, r
    assert r["trace_commitments_grounded_rate"] is not None, r
    assert r["trace_commitments_grounded_unmeasured"] is False, r
