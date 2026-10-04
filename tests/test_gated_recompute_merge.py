"""Gated rows: the answer the recompute grades is the answer the live path graded.

On the gated geometry tests are bought from the menu, not written into the answer. The live path
(`evaluate._row_gated`) merges the bought tests into `tests_to_order` before the hard gates and
the judges read the answer; `tools/recompute_judges.py` makes the same merge from the row's
`tool_targets` and a menu rebuilt from the case (`evaluate.gated_menu`), through the same
function (`evaluate.merge_bought_tests`). Without the merge a recompute reads "bought tests" as
"ordered nothing" and every test-ordering judge on those rows comes out low.

| Check | Catches | Turns red when |
|---|---|---|
| `test_live_and_recompute_grade_the_same_answer` | the two paths merging differently | either side stops calling the shared merge, or changes what it passes |
| `test_rebuilt_menu_that_merges_a_different_count_is_refused` | a menu drifted since the batch | the `tests_from_queries` cross-check is dropped |
| `test_unrecorded_purchases_are_skipped_not_guessed` | "not recorded" read as "bought nothing" | the `tool_n_calls` / trace distinction is dropped |
| `test_source_answer_is_not_mutated` | a superseded row merged twice from the shared response | the merge edits the response in place |

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import types

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str((ROOT / _cfg["kernel_path"]).resolve()))

from haenv import evaluate as EV                    # noqa: E402
import recompute_judges as RJ                       # noqa: E402
from _patch_bound import patch_bound  # noqa: E402
from haenv.run_state import RunContext as _RunContext  # noqa: E402
_CTX = _RunContext()   # the run these tests drive; monkeypatch restores its fields

MENU = [{"target": "TSH", "is_test": True},
        {"target": "ACTH", "is_test": True},
        {"target": "resting_hr", "is_test": False, "real": True},
        {"target": "cortisol_am", "is_test": False, "real": False}]
#: Signals are not merged; a truncated test is still in `targets`; repeats stay repeats.
TARGETS = ["resting_hr", "TSH", "cortisol_am", "ACTH", "TSH"]
ANSWER = {"tests_to_order": ["HbA1c"], "differential": [{"diagnosis": "X"}]}


class _Out:
    def __init__(self):
        self._raw = copy.deepcopy(ANSWER)


class _Trace:
    leak = None
    raw_empty = False
    raw_unparseable = False
    committed = True
    rounds_log = []
    rounds_used = 2
    spent = 1.5
    budget = 25
    calls = [{"target": t} for t in TARGETS]
    targets = list(TARGETS)
    menu = MENU


class _VP:
    adjudication = {"ddx": {"diagnosis": "X"}}
    gold_drivers = ["unknown_or_multifactorial"]


class _SP:
    evidence_ledger = [{"evidence_id": "EV-01"}]
    prediction_context = {"prediction_time_T": 10}


class _Solver:
    prompt_mode = None
    probe_id = None


def _live_row(monkeypatch):
    """Run `_row_gated` on stubs; return the row and the answer the kernel graded."""
    from haenv import gated as _g, judges as _j, tracks as _tk
    seen: list = []

    def _grade(out, vp, ev, **kw):
        seen.append(copy.deepcopy(out._raw))
        return types.SimpleNamespace(tracks={"A": 1.0}, hard_gate_failures=[], overall="SCORED")

    patch_bound(monkeypatch, "build_instance", lambda raw, T: (_SP(), _VP()))
    monkeypatch.setattr(_g, "run_gated", lambda raw, T, solver, trace=None: (_Out(), _Trace()))
    monkeypatch.setattr(EV.verifier_mod, "grade", _grade)
    monkeypatch.setattr(_tk, "tool_track", lambda tr, vp, key_signals=None: {})
    monkeypatch.setattr(_tk, "key_signals_for", lambda vp: [])
    monkeypatch.setattr(_j, "run_judges", lambda shape, out, vp: {})
    monkeypatch.setattr(_j, "judge_noop_probe", lambda out, spec, **kw: {})
    monkeypatch.setattr(_j, "judge_quant_probe", lambda out, spec, t_max=None: {})
    monkeypatch.setattr(_j, "judge_abstention_calibration", lambda out, vp: {})
    monkeypatch.setattr(_CTX, "resp_path", None)
    patch_bound(monkeypatch, "iron_law_precheck", lambda raw, sp, vp, solver: (lambda: (None, [])))
    monkeypatch.setattr(EV.process, "run_process_judges", lambda raw, ev: {})
    row = EV._row_gated("JD-01", "stub", object(), 84, _Solver(), ctx=_CTX)
    assert len(seen) == 1, "live path did not grade exactly once"
    return row, seen[0]


def _recompute(monkeypatch, row, menu=MENU, trace_p=pathlib.Path("/nonexistent/trace.jsonl")):
    patch_bound(monkeypatch, "gated_menu", lambda raw, T: list(menu))
    RJ._MENUS.clear()
    RJ._TRACE_TARGETS.clear()
    return RJ.merge_gated_purchases(row, copy.deepcopy(ANSWER), object(), trace_p)


def test_live_and_recompute_grade_the_same_answer(monkeypatch):
    row, graded_live = _live_row(monkeypatch)
    assert graded_live["tests_to_order"] == ["HbA1c", "TSH", "ACTH", "TSH"]
    assert row["tests_from_queries"] == 3 and row["tool_targets"] == TARGETS
    answer, gap = _recompute(monkeypatch, row)
    assert gap is None, gap
    assert answer == graded_live


def test_rebuilt_menu_that_merges_a_different_count_is_refused(monkeypatch):
    row, _ = _live_row(monkeypatch)
    answer, gap = _recompute(monkeypatch, row, menu=[m for m in MENU if m["target"] != "ACTH"])
    assert gap and "现场并入 3 项" in gap, gap
    assert answer == ANSWER, "a refused merge must hand back the unmerged answer"


def test_unrecorded_purchases_are_skipped_not_guessed(monkeypatch, tmp_path):
    base = {"case": "JD-01", "solver": "m", "T": 84, "tool_rounds_used": 1, "tool_targets": None}
    # no `tool_n_calls`: not recorded
    assert _recompute(monkeypatch, dict(base))[1]
    # calls recorded, targets empty, no trace on disk: not recorded
    assert _recompute(monkeypatch, dict(base, tool_n_calls=2))[1]
    # the trace shows the calls had targets the row lost: not recorded
    tp = tmp_path / "trace.jsonl"
    tp.write_text("".join(json.dumps({"case": "JD-01", "solver": "m", "type": "tool/call",
                                      "data": {"target": t}}) + "\n" for t in ("TSH", None)),
                  encoding="utf-8")
    assert _recompute(monkeypatch, dict(base, tool_n_calls=2), trace_p=tp)[1]
    # the trace shows every call had no target: nothing bought
    tp.write_text("".join(json.dumps({"case": "JD-01", "solver": "m", "type": "tool/call",
                                      "data": {"target": None}}) + "\n" for _ in range(2)),
                  encoding="utf-8")
    answer, gap = _recompute(monkeypatch, dict(base, tool_n_calls=2), trace_p=tp)
    assert gap is None and answer == ANSWER
    # no calls: nothing bought, nothing to merge
    answer, gap = _recompute(monkeypatch, dict(base, tool_n_calls=0, tests_from_queries=0))
    assert gap is None and answer == ANSWER


def test_source_answer_is_not_mutated(monkeypatch):
    row, _ = _live_row(monkeypatch)
    patch_bound(monkeypatch, "gated_menu", lambda raw, T: list(MENU))
    RJ._MENUS.clear()
    src = copy.deepcopy(ANSWER)
    RJ.merge_gated_purchases(row, src, object(), pathlib.Path("/nonexistent"))
    RJ.merge_gated_purchases(row, src, object(), pathlib.Path("/nonexistent"))
    assert src == ANSWER
