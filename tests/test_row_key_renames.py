"""Stored batches are read-only, so renamed row keys are mapped where rows are read.

`evaluate.load_rows` is the one entry point reports and recompute tools use; a batch
written before a rename must come back under the new key, and a batch written after it
must come back unchanged.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json

from haenv.evaluate import RENAMED_ROW_KEYS, load_rows


def test_old_key_comes_back_under_the_new_name(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = []
    for i, (old, new) in enumerate(RENAMED_ROW_KEYS.items()):
        rows.append({"case": f"old{i}", "solver": "m", old: 0.5})
        rows.append({"case": f"new{i}", "solver": "m", new: 1.0})
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = {r["case"]: r for r in load_rows(p)}
    for i, (old, new) in enumerate(RENAMED_ROW_KEYS.items()):
        assert got[f"old{i}"] == {"case": f"old{i}", "solver": "m", new: 0.5}
        assert got[f"new{i}"] == {"case": f"new{i}", "solver": "m", new: 1.0}
