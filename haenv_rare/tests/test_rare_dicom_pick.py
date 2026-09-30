"""build_attachments must not hand the launch gate a DICOM patient it will refuse."""
import pathlib

import haenv_rare.gen as R


def test_pick_skips_phi_flagged_patient_and_wraps(monkeypatch, tmp_path):
    for pid in ("A", "B", "C"):
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "s1.zip").write_bytes(b"x")
    monkeypatch.setattr(R, "_dicom_phi", lambda z: "PatientName='X'" if z.parent.name in ("C", "A") else None)
    assert R.pick_clean_dicom_patient(tmp_path, ["A", "B", "C"], "C") == "B"     # C flagged -> wrap -> A flagged -> B
    assert R.pick_clean_dicom_patient(tmp_path, ["A", "B", "C"], "B") == "B"     # clean seed pick kept
    monkeypatch.setattr(R, "_dicom_phi", lambda z: "bad")
    try:
        R.pick_clean_dicom_patient(tmp_path, ["A", "B", "C"], "A")
    except R.RareSpecError:
        pass
    else:
        raise AssertionError("no clean patient must raise, not silently pick one")
