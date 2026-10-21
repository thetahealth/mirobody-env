"""Synthetic twin families: regular gold, nonempty evidence, and bounded edits."""
from _paths import ROOT as _TEST_ROOT

import copy

import json

from dataclasses import asdict

from pathlib import Path

import pytest

import haenv.world.generation.twins as twins

ROOT = _TEST_ROOT

MANIFESTS = {}

FAMILY_CASES = json.loads((_TEST_ROOT / 'tests/fixtures/harness/family_trials.json').read_text())

def remember_question(raw):
    import haenv.world.wq as wq
    MANIFESTS[twins.sha16(asdict(raw))] = copy.deepcopy(wq.injected_manifest(raw.case_id))

def restore_question(raw):
    import haenv.world.wq as wq
    from haenv.world.generation.opaque_ids import register_manifest_alias
    from haenv.world.generation.opaque_ids import stored_solver_id
    wq.register_injection(raw.case_id, MANIFESTS[twins.sha16(asdict(raw))])
    register_manifest_alias(raw.case_id, stored_solver_id(raw))
    return None

def trials_for(raw, pack):
    restore_question(raw)
    return twins.family_candidates(raw, pack)

@pytest.fixture(scope="module")
def real_cases(artifact_cache):
    from haenv.app.job import load_job
    from _family_artifacts import family_records
    for name in ("p3-meds", "p4-followup"):
        load_job(ROOT / f"benchmarks/inputs/{name}.job.yaml")
    from haenv.p2 import plugin
    plugin.register()
    records = family_records(artifact_cache)
    MANIFESTS.update(records[1])
    return records[0]

@pytest.fixture(scope="module")
def prepared_families(real_cases, artifact_cache):
    from _family_artifacts import family_records
    return family_records(artifact_cache)

@pytest.fixture
def replay_source(tmp_path, monkeypatch):
    import importlib.util
    import hashlib
    import json
    import yaml
    from haenv.world.generation.opaque_ids import solver_case_id
    spec = importlib.util.spec_from_file_location("replay_helper", ROOT / 'tools/generation/prepare_twin_families.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    job = tmp_path / "source.job.yaml"
    job.write_text(yaml.safe_dump({"job_id": "synthetic", "cases": [{"case_id": "synthetic"}]}))
    source = tmp_path / "source"
    source.mkdir()
    row = {"case_id": "synthetic", "case": {"latent_premise": {"meta": {
        "solver_case_id": solver_case_id("synthetic", "")}}}}
    (source / "cases.jsonl").write_text(json.dumps(row) + "\n")
    (source / "payloads.jsonl").write_text(json.dumps({"case_id": "synthetic", "sp": {}, "vp": {}}) + "\n")
    digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:16]
    fingerprints = {"world_sha": "world-v1", "kernel_sha256": "kernel-v1", "world_knobs": "physio=1"}
    meta = {**fingerprints, "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest()[:16],
            "generator": "deterministic", "cases_sha256": {"synthetic": digest}}
    (source / "batch.json").write_text(json.dumps(meta))
    monkeypatch.setenv("HAENV_PACK_SEED", "")
    import haenv.app.job
    monkeypatch.setattr(haenv.app.job, "load_job", lambda *a, **k: None)
    monkeypatch.setattr(mod, "replay_fingerprints", lambda: fingerprints, raising=False)
    import haenv.app.artifacts.store, haenv.app.artifacts.payloads
    monkeypatch.setattr(haenv.app.artifacts.store, "load_cases", lambda p: {"synthetic": object()})
    monkeypatch.setattr(haenv.app.artifacts.payloads, "_payload_line", lambda item: ("unused", (source / "payloads.jsonl").read_text()))
    return mod, job, source, tmp_path / "output", meta


__all__ = [name for name in globals() if not name.startswith("__")]
