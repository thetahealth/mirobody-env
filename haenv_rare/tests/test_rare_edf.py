"""EDF / EDF+ writer for EEG attachments (`haenv_rare.edf`).

Contract: the output keeps every original signal header field and every data sample byte for
byte; EDF+C output adds one "EDF Annotations" signal whose TALs carry the gold annotations (first
TAL of each record = timekeeping); only the patient / recording identification fields, the
header length, the reserved field and the signal count of the main header change."""
import struct

import pytest

from haenv_rare import edf as E


def _field(v, n):
    return str(v).ljust(n)[:n].encode("ascii")


def make_edf(path, *, patient="Surrogate Name", recording="01", n_records=3, dur=1,
             signals=(("FP1-F7", 4), ("ECG", 2))):
    ns = len(signals)
    head = (_field("0", 8) + _field(patient, 80) + _field(recording, 80) + _field("07.11.76", 8)
            + _field("13.03.24", 8) + _field(256 * (ns + 1), 8) + _field("", 44) + _field(n_records, 8)
            + _field(dur, 8) + _field(ns, 4))
    cols = [[_field(lab, 16) for lab, _ in signals], [_field("AgAgCl", 80)] * ns, [_field("uV", 8)] * ns,
            [_field(-3276.8, 8)] * ns, [_field(3276.7, 8)] * ns, [_field(-32768, 8)] * ns,
            [_field(32767, 8)] * ns, [_field("HP:0.1Hz", 80)] * ns, [_field(n, 8) for _, n in signals],
            [_field("", 32)] * ns]
    for col in cols:
        head += b"".join(col)
    body = b""
    k = 0
    for _ in range(n_records):
        for _, n in signals:
            body += struct.pack(f"<{n}h", *range(k, k + n))
            k += n
    path.write_bytes(head + body)
    return head, body


def test_header_is_read_back(tmp_path):
    src = tmp_path / "a.edf"
    make_edf(src)
    h = E.read_header(src)
    assert (h.n_records, h.record_duration, h.ns) == (3, 1.0, 2)
    assert [s["label"] for s in h.signals] == ["FP1-F7", "ECG"]
    assert [s["samples"] for s in h.signals] == [4, 2]
    assert h.patient == "Surrogate Name" and not h.edfplus


def test_edfplus_keeps_samples_and_signal_headers_and_carries_the_annotations(tmp_path):
    src, dst = tmp_path / "a.edf", tmp_path / "b.edf"
    _, body = make_edf(src)
    ann = [{"onset_sec": 1.0, "duration_sec": 1.5, "text": "Seizure"},
           {"onset_sec": 2.25, "duration_sec": 0.0, "text": "Sleep stage 2"}]
    meta = E.write_edf(src, dst, patient="X X X X", recording="Startdate 07-NOV-2076 X X X",
                       annotations=ann, edfplus=True)
    h0, h1 = E.read_header(src), E.read_header(dst)
    assert h1.edfplus and h1.reserved.startswith("EDF+C")
    assert h1.patient == "X X X X" and h1.recording == "Startdate 07-NOV-2076 X X X"
    assert h1.ns == 3 and h1.signals[-1]["label"] == "EDF Annotations"
    for a, b in zip(h0.signals, h1.signals):                   # original signal headers untouched
        assert a == b
    assert (h1.n_records, h1.record_duration, h1.startdate, h1.starttime) == \
           (h0.n_records, h0.record_duration, h0.startdate, h0.starttime)
    rec0, rec1 = h0.record_bytes, h1.record_bytes
    data1 = dst.read_bytes()[h1.header_bytes:]
    assert len(data1) == rec1 * h1.n_records
    for r in range(h0.n_records):                              # every original sample byte kept
        assert data1[r * rec1:r * rec1 + rec0] == body[r * rec0:(r + 1) * rec0]
    tals = E.read_annotations(dst)
    assert tals == ann
    assert E.read_timekeeping(dst) == [0.0, 1.0, 2.0]
    assert meta == {"format": "EDF+", "duration_sec": 3.0, "channel_count": 2, "sfreq_hz": 4.0,
                    "annotations": ann}
    assert E.recording_meta(dst) == meta


def test_plain_edf_rewrites_only_the_identification_fields(tmp_path):
    src, dst = tmp_path / "a.edf", tmp_path / "c.edf"
    make_edf(src)
    meta = E.write_edf(src, dst, patient="JD104-R2", recording="01", annotations=None, edfplus=False)
    a, b = src.read_bytes(), dst.read_bytes()
    assert len(a) == len(b)
    assert b[8:88].decode().strip() == "JD104-R2"
    assert a[:8] == b[:8] and a[168:] == b[168:]               # everything after the two ID fields
    assert meta["format"] == "EDF" and meta["annotations"] == [] and meta["channel_count"] == 2


def test_edfplus_without_annotations_has_only_timekeeping(tmp_path):
    src, dst = tmp_path / "a.edf", tmp_path / "d.edf"
    make_edf(src, n_records=2)
    meta = E.write_edf(src, dst, patient="X X X X", recording="Startdate 07-NOV-2076 X X X",
                       annotations=[], edfplus=True)
    assert meta["annotations"] == [] and E.read_timekeeping(dst) == [0.0, 1.0]


def test_an_annotation_outside_the_recording_is_refused(tmp_path):
    src = tmp_path / "a.edf"
    make_edf(src)
    with pytest.raises(ValueError):
        E.write_edf(src, tmp_path / "e.edf", patient="X X X X", recording="Startdate 07-NOV-2076 X X X",
                    annotations=[{"onset_sec": 3.5, "duration_sec": 1, "text": "late"}], edfplus=True)


def test_edfplus_fields():
    assert E.edfplus_date("07.11.76") == "07-NOV-2076"
    assert E.edfplus_date("01.01.93") == "01-JAN-1993"
    assert E.edfplus_patient(None) == "X X X X"
    assert E.edfplus_patient({"code": "P123", "sex": "F", "birthdate": "02-MAY-1951", "name": "Test Name"}) \
        == "P123 F 02-MAY-1951 Test_Name"
    assert E.edfplus_recording("07.11.76") == "Startdate 07-NOV-2076 X X X"
    with pytest.raises(ValueError):
        E.edfplus_patient({"code": "P 1", "sex": "F", "birthdate": "X", "name": "N"})


def test_most_common_rate_ignores_the_annotation_signal(tmp_path):
    src, dst = tmp_path / "a.edf", tmp_path / "f.edf"
    make_edf(src, signals=(("C3", 8), ("C4", 8), ("SAO2", 1)), dur=2)
    meta = E.write_edf(src, dst, patient="X X X X", recording="Startdate 07-NOV-2076 X X X",
                       annotations=[], edfplus=True)
    assert meta["sfreq_hz"] == 4.0 and meta["channel_count"] == 3 and meta["duration_sec"] == 6.0


def test_mne_reads_what_we_write(tmp_path):
    mne = pytest.importorskip("mne", reason="mne not installed: cross-check skipped, not passed")
    src, dst = tmp_path / "a.edf", tmp_path / "g.edf"
    make_edf(src, n_records=4, signals=(("C3", 8), ("C4", 8)))
    ann = [{"onset_sec": 1.5, "duration_sec": 1.0, "text": "Seizure"}]
    E.write_edf(src, dst, patient="X X X X", recording="Startdate 07-NOV-2076 X X X", annotations=ann, edfplus=True)
    raw = mne.io.read_raw_edf(dst, preload=False, verbose="error")
    assert raw.info["nchan"] == 2 and raw.info["sfreq"] == 8.0 and raw.n_times == 32
    assert [(a["onset"], a["duration"], a["description"]) for a in raw.annotations] == [(1.5, 1.0, "Seizure")]
