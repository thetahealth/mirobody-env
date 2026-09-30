"""A gated `ABORT(no_answer:budget)` row with no stored response is re-derived from its own fields.

Such a row has nothing for a judge to read: live never saves a response for it, it stays out of
every denominator, and its verdict depends only on `tool_spent`, `tool_budget` and the absence of
a committed answer. The recompute re-derives that verdict with the predicate the live path uses
(`evaluate.budget_abort`); when it agrees the row takes the current judging stamp and
`restamp_basis = "abort_rederived_from_row"`, when it disagrees the recompute fails and the row
is left as it was. Other `ABORT(...)` kinds rest on facts the row does not record independently
of the verdict (raw text, leak probe, iron-law check) and are left unchanged.

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

FP_NOW = "fp-now-0000000000"
BUDGET_ROW = {"case": "C1", "solver": "m1", "T": 84, "overall": "ABORT(no_answer:budget)",
              "no_response_reason": "budget_exhausted_before_answer", "tool_rounds_used": 3,
              "tool_spent": 128.0, "tool_budget": 128.0, "tool_truncated": 0,
              "judging_sha16": "fp-old"}


def _run(tmp_path, monkeypatch, rows, responses=()):
    import haenv.anchor as _anchor
    monkeypatch.setattr(_anchor, "judging_fp_cached", lambda: FP_NOW)
    d = tmp_path / "batch-synthetic-a"
    d.mkdir()
    (d / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses),
                                       encoding="utf-8")
    rc = RJ.main([str(d)])
    out = [json.loads(x) for x in (d / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return rc, out


def test_consistent_budget_abort_takes_the_current_stamp(tmp_path, monkeypatch):
    leak = {"case": "C2", "solver": "m1", "overall": "ABORT(leak)", "leak": ["x"],
            "judging_sha16": "fp-old"}
    scored = {"case": "C3", "solver": "m1", "overall": "SCORED", "judging_sha16": "fp-old"}
    rc, (a, b, c) = _run(tmp_path, monkeypatch, [BUDGET_ROW, leak, scored],
                         [{"case": "C3", "solver": "m1", "slice_t": None,
                           "raw": '{"differential": [{"diagnosis": "X"}]}'}])
    assert rc == 0
    assert a["judging_sha16"] == FP_NOW and a["restamp_basis"] == "abort_rederived_from_row"
    assert {k: v for k, v in a.items() if k not in ("judging_sha16", "restamp_basis")} == \
        {k: v for k, v in BUDGET_ROW.items() if k != "judging_sha16"}
    # an ABORT kind the row fields cannot reproduce is left exactly as it was
    assert b == leak
    # a row with a stored response goes through the normal recompute
    assert c["judging_sha16"] == FP_NOW and "restamp_basis" not in c


def test_budget_abort_with_budget_left_is_an_error(tmp_path, monkeypatch, capsys):
    bad = dict(BUDGET_ROW, tool_spent=90.0)
    rc, (a,) = _run(tmp_path, monkeypatch, [bad])
    assert rc != 0
    assert a == bad, "an inconsistent row must not be touched"
    assert "C1|m1" in capsys.readouterr().out


def test_budget_abort_without_the_recorded_reason_is_an_error(tmp_path, monkeypatch):
    bad = dict(BUDGET_ROW, no_response_reason=None)
    rc, (a,) = _run(tmp_path, monkeypatch, [bad])
    assert rc != 0 and a == bad


def test_live_gated_path_uses_the_same_predicate(monkeypatch):
    from haenv import gated as _g

    class _Tr:
        leak = None
        raw_empty = raw_unparseable = False
        committed = False
        spent, budget, rounds_used, truncated = 128.0, 128.0, 3, 0

    class _Out:
        _raw = {}

    class _Solver:
        prompt_mode = None
        probe_id = None

    class _VP:
        adjudication = {"ddx": {"diagnosis": "X"}}

    monkeypatch.setattr(EV, "build_instance", lambda raw, T: (types.SimpleNamespace(), _VP()))
    monkeypatch.setattr(_g, "run_gated", lambda raw, T, solver, trace=None: (_Out(), _Tr()))
    monkeypatch.setattr(EV, "RESP_PATH", [None])
    monkeypatch.setattr(EV, "iron_law_precheck", lambda raw, sp, vp, solver: (lambda: (None, [])))
    row = EV._row_gated("C1", "m1", object(), 84, _Solver())
    assert row["overall"] == "ABORT(no_answer:budget)"
    rederived = {k: row[k] for k in ("tool_spent", "tool_budget", "no_response_reason")}
    assert RJ.rederive_abort(dict(row, **rederived))[1] is None
    # the live verdict comes from that predicate: forcing it on a committed, in-budget cell
    # turns the cell into the budget ABORT
    _Tr.committed, _Tr.spent = True, 1.0
    monkeypatch.setattr(EV, "budget_abort", lambda committed, spent, budget: True)
    assert EV._row_gated("C1", "m1", object(), 84, _Solver())["overall"] == "ABORT(no_answer:budget)"


def _rerun(d, *flags):
    rc = RJ.main([str(d), *flags])
    return rc, [json.loads(x) for x in (d / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x]


def test_recompute_conflicts_reflects_the_latest_restamp(tmp_path, monkeypatch):
    """`recompute_conflicts` names this run's dropped fields; a later conflict-free restamp clears it."""
    import haenv.anchor as _anchor
    monkeypatch.setattr(_anchor, "judging_fp_cached", lambda: FP_NOW)
    d = tmp_path / "batch-synthetic-b"
    d.mkdir()
    row = dict(BUDGET_ROW, restamp_basis="hand_set")
    (d / "eval.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "responses.jsonl").write_text("", encoding="utf-8")
    rc, (a,) = _rerun(d)                                 # no authorisation: the change is dropped and named
    assert rc == 0 and a["recompute_conflicts"] == "restamp_basis"
    assert a["restamp_basis"] == "hand_set"
    rc, (b,) = _rerun(d, "--allow-change-all")           # now nothing conflicts
    assert rc == 0 and b["restamp_basis"] == "abort_rederived_from_row"
    assert not b.get("recompute_conflicts"), b.get("recompute_conflicts")
    rc, (c,) = _rerun(d)                                 # a repeat run stays clean
    assert rc == 0 and not c.get("recompute_conflicts")
