"""`haenv build` persists its per-case audit next to `cases.jsonl`.

The audit file answers "why is this stream, event or case not in the pack": it
keeps one row per input case, launched or not, with the item-level verdicts and
drop reasons.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haenv import job as J                                          # noqa: E402
from haenv.store import AUDIT_FILENAME, load_audits                 # noqa: E402

JOB = ROOT / "inputs" / "example-ew.job.yaml"


def test_build_writes_one_audit_row_per_input_case(tmp_path):
    env = {**os.environ, "HAENV_OUTPUT_ROOT": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "haenv", "build", str(JOB),
                        "--gen", "deterministic", "--fresh"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    audits = list(tmp_path.glob(f"results/*/*/*/{AUDIT_FILENAME}"))
    assert len(audits) == 1, audits
    rows = load_audits(audits[0])
    cases = [json.loads(line)["case_id"]
             for line in (audits[0].parent / "cases.jsonl").read_text(encoding="utf-8").splitlines()]

    assert [a["case_id"] for a in rows] == [cs.case_id for cs in J.load_job(JOB).cases]
    assert sorted(a["case_id"] for a in rows if a.get("emitted")) == sorted(cases)
    # This job holds a case the launch gate blocks; its row is the one a
    # debugging session needs, so it must be there with its reason.
    blocked = [a for a in rows if not a.get("emitted")]
    assert blocked, "scan surface: expected at least one blocked case in this job"
    assert all(a.get("gate_warnings") or a.get("gen_error") or a.get("premise_error")
               for a in blocked), blocked
    assert all("drop_reasons" in (a.get("_verify_report") or {})
               for a in rows if a.get("emitted"))
