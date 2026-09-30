"""`haenv-rare-verify <job>`: re-hash every attachment a job file points at (any mapping under
`latent.rare_attachments` that has both `path` and `sha256`), before a remote run. Exit 0 only
when every file exists and matches; a job with no pointers at all is an error, not a pass."""
import hashlib

import yaml

from haenv_rare import verify_attachments as V


def _job(tmp_path, att_by_case):
    p = tmp_path / "x.job.yaml"
    p.write_text(yaml.safe_dump({"job_id": "x", "cases": [{"case_id": c, "latent": {"rare_attachments": a}}
                                                         for c, a in att_by_case.items()]}))
    return p


def _file(tmp_path, name, data=b"abc"):
    f = tmp_path / name
    f.write_bytes(data)
    return str(f), hashlib.sha256(data).hexdigest()


def test_all_match(tmp_path):
    a, sa = _file(tmp_path, "a.vcf.gz")
    b, sb = _file(tmp_path, "b.edf", b"edf")
    job = _job(tmp_path, {"JD-1": {"genome": {"files": {"proband": {"path": a, "sha256": sa}}},
                                   "eeg": {"recordings": [{"path": b, "sha256": sb, "annotations": []}]}}})
    rep = V.verify_job(job)
    assert rep["checked"] == 2 and rep["bad"] == []
    assert V.main([str(job)]) == 0


def test_mismatch_and_missing_are_reported(tmp_path):
    a, sa = _file(tmp_path, "a.vcf.gz")
    job = _job(tmp_path, {"JD-1": {"genome": {"files": {"proband": {"path": a, "sha256": "0" * 64}},
                                              "ped": {"path": str(tmp_path / "gone.ped"), "sha256": sa}}}})
    rep = V.verify_job(job)
    assert sorted(k for _, _, k in rep["bad"]) == ["missing", "sha_mismatch"]
    assert V.main([str(job)]) == 1


def test_no_pointers_is_an_error(tmp_path):
    job = _job(tmp_path, {"JD-1": {}})
    assert V.verify_job(job)["checked"] == 0
    assert V.main([str(job)]) == 2


def test_manifest_lists_every_file_once(tmp_path):
    a, sa = _file(tmp_path, "a.vcf.gz")
    job = _job(tmp_path, {"JD-1": {"genome": {"files": {"proband": {"path": a, "sha256": sa}}}},
                          "JD-2": {"genome": {"files": {"proband": {"path": a, "sha256": sa}}}}})
    out = tmp_path / "SHA256SUMS"
    assert V.main([str(job), "--manifest", str(out)]) == 0
    assert out.read_text().splitlines() == [f"{sa}  {a}"]
