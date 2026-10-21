"""`tools/calibration/build_solver_timing.py`: a call's duration is its service time, without the wait
for a paid lane."""
from __future__ import annotations
from _paths import ROOT as _TEST_ROOT

import json
import sys
from pathlib import Path

ROOT = _TEST_ROOT
sys.path.insert(0, str(ROOT / "tools"))

import tools.calibration.build_solver_timing as bst  # noqa: E402


def _trace(batch: Path, calls: list[dict]) -> Path:
    batch.mkdir(parents=True)
    lines = []
    for k, data in enumerate(calls):
        base = {"solver": "m", "geometry": "single", "case": f"c{k}", "turn": 0, "step": 1,
                "ts": "2026-01-01T00:00:00+00:00"}
        lines += [{**base, "type": "case/start"}, {**base, "type": "step/start"},
                  {**base, "type": "assistant/message", "data": data},
                  {**base, "type": "case/end", "ts": "2026-01-01T00:01:00+00:00"}]
    (batch / "trace.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return batch


def test_slot_wait_is_subtracted_from_the_call(tmp_path):
    b = _trace(tmp_path / "b", [{"latency_s": 50.0, "slot_wait_s": 40.0}] * 3)
    row = bst.collect([b])["models"]["m"]["single"]
    assert row["median_call_s"] == 10.0 and row["max_success_call_s"] == 10.0
    assert row["calls_without_slot_wait"] == 0


def test_a_trace_without_slot_wait_is_counted(tmp_path):
    b = _trace(tmp_path / "b", [{"latency_s": 50.0}] * 2)
    row = bst.collect([b])["models"]["m"]["single"]
    assert row["median_call_s"] == 50.0 and row["calls_without_slot_wait"] == 2
