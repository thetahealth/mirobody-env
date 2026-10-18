"""make_pack refuses a pack when a fail-closed invariant gate reports a violation on the audited pool or its
published items: non-zero exit, no publishable audit manifest, the step recorded; a clean pool passes and keeps it.
Built on real pool rows (`tests/fixtures/invariants/p3.jsonl`). SYNTHETIC data, evaluation only."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mp():
    spec = importlib.util.spec_from_file_location("_make_pack_inv", ROOT / 'tools/generation/make_pack.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(tmp: pathlib.Path, inject: bool) -> tuple[pathlib.Path, list[str]]:
    rows = [json.loads(line) for line in open(ROOT / "tests/fixtures/invariants/p3.jsonl", encoding="utf-8")]
    if inject:
        rows[0]["sp"]["user_profile"]["known_conditions"] = list(rows[0]["sp"]["user_profile"].get("known_conditions")
                                                                   or ()) + ["CKD_never_in_world"]
    b = tmp / "batch"
    b.mkdir()
    with open(b / "cases.jsonl", "w", encoding="utf-8") as fc, open(b / "payloads.jsonl", "w", encoding="utf-8") as fp:
        for r in rows:
            fc.write(json.dumps({"case_id": r["case_id"], "case": r["case"]}, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"case_id": r["case_id"], "sp": r["sp"]}, ensure_ascii=False) + "\n")
    out = tmp / "out"
    (out / "audit").mkdir(parents=True)
    (out / "audit" / "pack.json").write_text(json.dumps({"items": [r["case_id"] for r in rows[:3]]}), encoding="utf-8")
    (tmp / "job.yaml").write_text("cases: []\n", encoding="utf-8")
    return b, [r["case_id"] for r in rows]


@pytest.mark.parametrize("inject", [False, True])
def test_fail_closed_gate_refuses_the_pack(tmp_path, monkeypatch, inject):
    import haenv.world.validation.invariants as INV
    monkeypatch.setattr(INV, "FAIL_CLOSED", ("C1",))      # the item gate the injected defect breaks
    mp = _mp()
    batch, _ = _batch(tmp_path, inject)
    rec = {"steps": {}}
    rc = mp.invariants_step("p3", tmp_path / "job.yaml", batch, tmp_path / "out", rec, jobs=1)
    assert rc == int(inject) and rec["steps"]["invariants"] == int(inject)
    assert (tmp_path / "out" / "audit" / "pack.json").exists() is (not inject)
    doc = json.loads((tmp_path / "out" / "invariants" / "invariants-p3.json").read_text(encoding="utf-8"))
    assert bool(doc["refusals"]) is inject


def test_report_only_gate_does_not_refuse(tmp_path, monkeypatch):
    import haenv.world.validation.invariants as INV
    monkeypatch.setattr(INV, "FAIL_CLOSED", ())
    mp = _mp()
    batch, _ = _batch(tmp_path, True)
    rec = {"steps": {}}
    assert mp.invariants_step("p3", tmp_path / "job.yaml", batch, tmp_path / "out", rec, jobs=1) == 0
    assert (tmp_path / "out" / "audit" / "pack.json").exists()


def test_every_fail_closed_id_is_a_gate():
    import haenv.world.validation.invariants as INV
    assert set(INV.FAIL_CLOSED) <= {g.id for g in INV.GATES}


@pytest.mark.parametrize("defect", ["missing_payload", "duplicate_case", "unknown_published", "missing_job"])
def test_incomplete_inputs_refuse_without_success_pack(tmp_path, defect):
    mp = _mp()
    batch, _ = _batch(tmp_path, False)
    if defect == "missing_payload":
        (batch / "payloads.jsonl").write_text("")
    elif defect == "duplicate_case":
        f = batch / "cases.jsonl"
        f.write_text(f.read_text() * 2)
    elif defect == "unknown_published":
        (tmp_path / "out" / "audit" / "pack.json").write_text('{"items": ["missing"]}')
    else:
        (tmp_path / "job.yaml").unlink()
    rec = {"steps": {}}
    assert mp.invariants_step("p3", tmp_path / "job.yaml", batch, tmp_path / "out", rec) == 1
    assert rec["steps"]["invariants"] == 1
    assert rec["invariant_error"]
    assert not (tmp_path / "out" / "audit" / "pack.json").exists()


def test_violation_outside_published_subset_refuses(tmp_path, monkeypatch):
    import haenv.world.validation.invariants as INV
    monkeypatch.setattr(INV, "FAIL_CLOSED", ("C1",))
    mp = _mp()
    batch, ids = _batch(tmp_path, True)
    (tmp_path / "out" / "audit" / "pack.json").write_text(json.dumps({"items": ids[1:3]}))
    rec = {"steps": {}}
    assert mp.invariants_step("p3", tmp_path / "job.yaml", batch, tmp_path / "out", rec) == 1
    assert rec["invariants"]["C1"] == [1, 0]


def test_purchase_error_invalidates_success_artifact(tmp_path, monkeypatch):
    import haenv.world.validation.invariants_purchase as P
    mp = _mp()
    batch, _ = _batch(tmp_path, False)
    def crash(*args, **kwargs):
        raise RuntimeError("purchase failed")
    monkeypatch.setattr(P, "dump_parallel", crash)
    rec = {"steps": {}}
    assert mp.invariants_step("m2", tmp_path / "job.yaml", batch, tmp_path / "out", rec) == 1
    assert not (tmp_path / "out" / "audit" / "pack.json").exists()


@pytest.mark.parametrize("status", [None, 1])
def test_publish_only_requires_completed_invariants(tmp_path, monkeypatch, status):
    mp = _mp()
    steps = dict(gen=0, build=0, audit=0)
    if status is not None:
        steps["invariants"] = status
    (tmp_path / "make_pack.json").write_text(json.dumps({"pack": "p3", "steps": steps}))
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "pack.json").write_text('{"items": ["old-success"]}')
    monkeypatch.setattr(mp, "publish", lambda *a: pytest.fail("published without invariants"))
    with pytest.raises(SystemExit, match="invariants"):
        mp.main(["--pack", "p3", "--out", str(tmp_path), "--publish-only"])
    assert not (tmp_path / "audit" / "pack.json").exists()


def test_m2_purchase_coverage_is_required(tmp_path):
    import haenv.world.validation.invariants as INV
    batch, _ = _batch(tmp_path, False)
    buy = tmp_path / "buy.jsonl"
    buy.write_text("")
    with pytest.raises(ValueError, match="purchase"):
        INV.load(batch, "m2", buy)


def test_empty_purchase_shard_does_not_buy_every_case(tmp_path, monkeypatch):
    import haenv.world.validation.invariants_purchase as P
    monkeypatch.setattr(P, "dump", lambda *a, **kw: pytest.fail("empty shard ran the whole pool"))
    part = tmp_path / "part.jsonl"
    assert P._dump_shard((tmp_path / "job", tmp_path, part, set())) == 0
    assert part.read_text() == ""


def test_publish_only_rechecks_and_revokes_old_marker(tmp_path, monkeypatch):
    import haenv.world.validation.invariants as INV
    import haenv.world.validation.shared_audit as SA
    monkeypatch.setattr(INV, "FAIL_CLOSED", ("C1",))
    mp = _mp()
    batch, _ = _batch(tmp_path, True)
    out = tmp_path / "out"
    pb = out / "pack/results/joint_dx/p3-meds/batch"
    pb.mkdir(parents=True)
    marker = pb / SA.marker_name("p3")
    marker.write_text("old-success")
    rec = {"pack": "p3", "steps": dict(gen=0, build=0, audit=0, invariants=0, pack_build=0),
           "job": str(tmp_path / "job.yaml"), "batch": str(batch), "pack_batch": str(pb)}
    (out / "make_pack.json").write_text(json.dumps(rec))
    monkeypatch.setattr(mp, "publish", lambda *a: pytest.fail("published after pool changed"))
    assert mp.main(["--pack", "p3", "--out", str(out), "--publish-only"]) == 1
    assert not marker.exists()
    assert not (out / "audit" / "pack.json").exists()
    result = json.loads((out / "make_pack.json").read_text())
    assert result["steps"]["invariants"] == 1
    assert "pack_build" not in result["steps"]


def test_job_selection_mismatch_refuses(tmp_path):
    mp = _mp()
    batch, _ = _batch(tmp_path, False)
    job = tmp_path / "job.yaml"
    job.write_text('_provenance: {pack_ids: [other]}\ncases: []\n')
    rec = {"steps": {}}
    assert mp.invariants_step("p3", job, batch, tmp_path / "out", rec) == 1
    assert not (tmp_path / "out" / "audit" / "pack.json").exists()


def test_legacy_pack2_explicitly_outside_four_pack_inventory(tmp_path):
    mp = _mp()
    rec = {"steps": {}}
    assert mp.invariants_step("pack2", tmp_path / "job", tmp_path / "batch", tmp_path, rec) == 0
    assert rec["invariants_status"] == "not_applicable: legacy pack2"


def test_paired_eligibility_restricts_selection_without_deleting_pool(tmp_path, monkeypatch):
    import yaml
    mp = _mp()
    job = tmp_path / "pool.yaml"
    original = {"cases": [{"case_id": cid, "raw": {"symptoms": [{"text": cid}]},
                            "latent": {"p2": {"background": "on"}}} for cid in ("a", "b", "c")],
                "_provenance": {}}
    job.write_text(yaml.safe_dump(original))
    real, control = tmp_path / "real", tmp_path / "control"
    for path, ids in ((real, ("a", "b")), (control, ("b", "c"))):
        path.mkdir()
        (path / "cases.jsonl").write_text("".join(json.dumps({"case_id": cid, "case": {}}) + "\n" for cid in ids))
    monkeypatch.setattr(mp, "build", lambda *a: 4)
    monkeypatch.setattr(mp, "_batch", lambda *a: control)
    mp.WORLD_SET[:] = ["background=true", "world.history=long"]
    rec = {}
    mp.paired_eligibility("p2", "p2-triage", job, real, tmp_path, {}, rec)
    after = yaml.safe_load(job.read_text())
    off = yaml.safe_load((tmp_path / "paired-control.job.yaml").read_text())
    assert after["cases"] == original["cases"]
    assert after["_provenance"]["eligible_pack_ids"] == ["b"]
    assert off["_provenance"]["eligible_pack_ids"] == ["b"]
    assert all(c["latent"]["p2"]["background"] == "off" for c in off["cases"])
    assert [c["raw"] for c in off["cases"]] == [c["raw"] for c in original["cases"]]
    assert mp.WORLD_SET == ["background=true", "world.history=long"]
    assert rec["paired_eligibility"] == {"real": 2, "control": 2, "common": 1, "control_build_exit": 4}


def test_paired_build_error_cannot_silently_select(tmp_path, monkeypatch):
    mp = _mp()
    job = tmp_path / "pool.yaml"
    job.write_text('cases: []\n')
    monkeypatch.setattr(mp, "build", lambda *a: 2)
    mp.WORLD_SET[:] = ["world.history=long"]
    with pytest.raises(ValueError, match="feasibility"):
        mp.paired_eligibility("p3", "p3-meds", job, tmp_path, tmp_path, {}, {})
    assert mp.WORLD_SET == ["world.history=long"]


def test_publish_keeps_one_batch_and_gate_can_select_it(tmp_path, monkeypatch):
    import haenv.scoring.gate as gate
    import haenv.reporting.pack_readout as pack_readout
    from haenv.reporting.pack_readout import ReadoutPack
    import haenv.app.config
    mp = _mp()
    pool, ids = _batch(tmp_path, False)
    out = tmp_path / 'out'
    root = out / 'pack/results/joint_dx/p3-meds'
    old = root / 'old-build'; old.mkdir(parents=True)
    (old / 'cases.jsonl').write_bytes(b'old batch\n')
    current = root / 'release-build'
    wanted = ids[:3]
    def cut(job, selected, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('job_id: p3-meds\n')
    def build(*a):
        current.mkdir()
        for name in ('cases.jsonl', 'payloads.jsonl'):
            rows = mp.rows_by_id(pool, name)
            (current / name).write_text(''.join(rows[cid] + '\n' for cid in wanted))
        return 0
    monkeypatch.setattr(mp, 'cut_job', cut)
    monkeypatch.setattr(mp, 'build', build)
    monkeypatch.setattr(pack_readout, 'declaration', lambda _: ReadoutPack(name='p3', owns=lambda _: True))
    monkeypatch.setattr(haenv.app.config, 'load_cfg', lambda: {'diversity': {'max_category_share': .5}})
    rec = {'steps': {}}
    assert mp.publish('p3', 'p3-meds', tmp_path / 'job.yaml', pool, out, {}, rec) == 0
    assert not old.exists()
    assert current.exists()
    assert gate.find_source(out / 'pack', 'p3', 3)[1] == current
    assert pool.exists()  # The audited pool remains available for provenance.
    for name in ('cases.jsonl', 'payloads.jsonl'):
        assert mp.rows_by_id(current, name) == {cid: mp.rows_by_id(pool, name)[cid] for cid in wanted}


def test_failed_publish_does_not_remove_previous_batch(tmp_path, monkeypatch):
    mp = _mp()
    pool, ids = _batch(tmp_path, False)
    out = tmp_path / 'out'
    old = out / 'pack/results/joint_dx/p3-meds/old-build'; old.mkdir(parents=True)
    (old / 'cases.jsonl').write_bytes(b'previous\n')
    monkeypatch.setattr(mp, 'cut_job', lambda *a: None)
    monkeypatch.setattr(mp, 'build', lambda *a: 1)
    rec = {'steps': {}}
    assert mp.publish('p3', 'p3-meds', tmp_path / 'job.yaml', pool, out, {}, rec) == 1
    assert (old / 'cases.jsonl').read_bytes() == b'previous\n'
