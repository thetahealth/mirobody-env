"""The idempotency gate refuses a batch that does not reproduce its predecessor.

A rebuild with the same generation code, world knobs, job file and generator, all
cases rebuilt and no live model call, must match the previous batch case for case.
The v5 pack and the gated pack rebuild identically with 8 workers and zero model
calls.

SYNTHETIC data, evaluation use only, not medical advice.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow          # two builds per test, about 7 s per test

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOB = ROOT / "inputs" / "example-ew.job.yaml"


def _build(out: pathlib.Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HAENV_OUTPUT_ROOT": str(out)}
    return subprocess.run([sys.executable, "-m", "haenv", "build", str(JOB),
                           "--gen", "deterministic", "--fresh"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def _batches(out: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p.parent for p in out.glob("results/*/*/*/batch.json"))


def _first_batch_tampered(out: pathlib.Path, **meta_overrides) -> None:
    first = _batches(out)[0] / "batch.json"
    meta = json.loads(first.read_text(encoding="utf-8"))
    cid = sorted(meta["cases_sha256"])[0]
    meta["cases_sha256"][cid] = "0" * 16
    meta.update(meta_overrides)
    first.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _second_batch_after(out: pathlib.Path) -> None:
    # Batch stamps have one-second resolution.
    import time
    time.sleep(1.1)


def test_reproducible_rebuild_passes(tmp_path):
    """Positive control: an honest rebuild is not refused."""
    assert _build(tmp_path).returncode == 0
    _second_batch_after(tmp_path)
    r = _build(tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    assert len(_batches(tmp_path)) == 2


def test_drift_under_same_inputs_is_refused(tmp_path):
    """Negative control: one case differs while every input matches."""
    assert _build(tmp_path).returncode == 0
    _first_batch_tampered(tmp_path)
    _second_batch_after(tmp_path)
    r = _build(tmp_path)
    assert r.returncode == 4, r.stdout[-3000:]
    meta = json.loads((_batches(tmp_path)[1] / "batch.json").read_text(encoding="utf-8"))
    gates = meta["emission_gates"]
    assert "idempotency_drift" in gates["blocked_by_batch_gate"]
    assert gates["idempotency_drift"]["n_drifted"] == 1


def test_drift_after_a_job_change_is_reported_only(tmp_path):
    """Positive control: a changed job file legitimately changes cases."""
    assert _build(tmp_path).returncode == 0
    _first_batch_tampered(tmp_path, job_sha256="changed")
    _second_batch_after(tmp_path)
    r = _build(tmp_path)
    assert r.returncode == 0, r.stdout[-3000:]
    assert "report only" in r.stdout
