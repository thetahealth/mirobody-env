"""`run` / `report` read the recorded build audit back, instead of printing dashes.

On the documented path (`build` -> `verify` -> `run` -> `report`) the report is
rendered from cases read back out of the batch. An audit rebuilt from
`cases.jsonl` and `batch.json` alone makes section 0 print `—` for rounds,
injected noise and solver-visible counts on every emitted case, and "not
recorded" for a blocked case's leak verdict, although `build` wrote all of
those to `audit.jsonl` in the same batch directory.

`store.merge_recorded_audits` lays the recorded rows over the stubs. Both
sides are pinned:

* a matching record fills the columns (and drops the stub's "never recorded"
  premise note when the record carries the premise verdict);
* a record that disagrees with the stub on `emitted` is ignored, and a batch
  with no audit file keeps the stubs unchanged;
* `cli.py` actually calls the merge on the read-back path.

SYNTHETIC, evaluation only, not medical advice.
"""
from __future__ import annotations

import ast
import pathlib

from haenv.store import AUDIT_FILENAME, merge_recorded_audits

ROOT = pathlib.Path(__file__).resolve().parent.parent

_EMITTED_STUB = {"case_id": "C1", "emitted": True, "from_batch": True,
                 "premise_ok": True, "leak_ok": True}
_BLOCKED_STUB = {"case_id": "C2", "emitted": False, "from_batch": True,
                 "post_noise_conflicts": ["event_density_mismatch"],
                 "premise_error": "(never recorded)"}
_REC_EMITTED = {"case_id": "C1", "emitted": True, "premise_ok": True, "leak_ok": True,
                "synth_rounds": 1, "noise_applied": ["a", "b"],
                "solver_visible_signals": 22, "solver_visible_evidence": 7,
                "_verify_report": {"big": "detail"}}
_REC_BLOCKED = {"case_id": "C2", "emitted": False, "premise_ok": True, "leak_ok": True,
                "post_noise_conflicts": ["event_density_mismatch"]}


def test_recorded_rows_fill_the_columns():
    out = merge_recorded_audits([_EMITTED_STUB, _BLOCKED_STUB],
                                [_REC_BLOCKED, _REC_EMITTED], ["C1", "C2"])
    e, b = out
    assert e["solver_visible_signals"] == 22 and e["noise_applied"] == ["a", "b"]
    assert e["synth_rounds"] == 1 and e["audit_source"] == AUDIT_FILENAME
    assert "_verify_report" not in e, "per-item detail must stay out of the report audit"
    assert b["leak_ok"] is True, "the blocked case's recorded leak verdict was not read back"
    assert "premise_error" not in b, "the stub's 'never recorded' note survived a recorded verdict"


def test_no_file_or_disagreeing_record_keeps_the_stubs():
    assert merge_recorded_audits([_EMITTED_STUB], [], ["C1"]) == [_EMITTED_STUB]
    flipped = {**_REC_EMITTED, "emitted": False}
    assert merge_recorded_audits([_EMITTED_STUB], [flipped], ["C1"]) == [_EMITTED_STUB], (
        "a record that disagrees on `emitted` belongs to another build of the case")


def test_rows_follow_the_job_order():
    out = merge_recorded_audits([_BLOCKED_STUB, _EMITTED_STUB], [], ["C1", "C2"])
    assert [a["case_id"] for a in out] == ["C1", "C2"]


def test_cli_calls_the_merge():
    tree = ast.parse((ROOT / "haenv" / "cli.py").read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "merge_recorded_audits" in called, "cli.py no longer merges the recorded audit"
