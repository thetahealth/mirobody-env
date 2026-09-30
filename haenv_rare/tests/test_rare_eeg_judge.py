"""EEG and refused-build judges (mirroring rc_signal_*):

* rc_eeg_index_ok -- per de-identifiable recording (phi_expected false): the solver indexed that
  file (same sha256, deid_status done) with the right duration, channel count, most common rate and
  the annotation onsets + texts the file itself carries;
* rc_eeg_phi_hidden -- per recording whose header carries synthetic identifiers (phi_expected
  true): not indexed as done, and none of its identifiers appears in the solver's signal list;
* rc_vcf_build_refused -- a case whose VCF is GRCh37 by design: the solver reports a non-GRCh38
  reference and lists no variants.
Each with a negative control."""
import copy

import haenv_rare as R

REC = {"path": "/x/eeg-1.edf", "sha256": "a" * 64, "dataset": "chbmit", "source": "chb01_26.seizure_cut.edf",
       "format": "EDF+", "duration_sec": 341.0, "channel_count": 23, "sfreq_hz": 256.0,
       "annotations": [{"onset_sec": 120.0, "duration_sec": 101.0, "text": "Seizure"}], "phi_expected": False}
PHI = {**REC, "path": "/x/eeg-2.edf", "sha256": "b" * 64, "annotations": [], "phi_expected": True,
       "phi_values": ["Test_Name", "02-MAY-1951"]}
RARE = {"attachments": {"eeg": {"recordings": [REC, PHI]}}}


def _row(rec, **over):
    d = {"format": rec["format"], "deid_status": "done", "sha256": rec["sha256"], "duration_sec": round(rec["duration_sec"]),
         "channel_count": rec["channel_count"], "sfreq_hz": rec["sfreq_hz"],
         "annotations": copy.deepcopy(rec["annotations"])}
    d.update(over)
    return d


def test_both_right():
    raw = {"signals": [_row(REC), {"format": "EDF+", "deid_status": "failed", "sha256": PHI["sha256"], "phi_tags": ["patient_name"]}]}
    assert R._judge_eeg(raw, RARE) == {"rc_eeg_index_ok": 1.0, "rc_eeg_phi_hidden": 1.0}


def test_wrong_duration_channels_rate_or_annotation_each_fail_the_index():
    for over in ({"duration_sec": 300}, {"channel_count": 22}, {"sfreq_hz": 512.0},
                 {"annotations": [{"onset_sec": 121.0, "duration_sec": 101.0, "text": "Seizure"}]},
                 {"annotations": [{"onset_sec": 120.0, "duration_sec": 101.0, "text": "seizure onset"}]},
                 {"annotations": []}, {"deid_status": "failed"}, {"sha256": "c" * 64}):
        raw = {"signals": [_row(REC, **over)]}
        assert R._judge_eeg(raw, RARE)["rc_eeg_index_ok"] == 0.0, over


def test_a_recording_with_identifiers_indexed_as_done_is_not_hidden():
    raw = {"signals": [_row(REC), _row(PHI)]}
    assert R._judge_eeg(raw, RARE)["rc_eeg_phi_hidden"] == 0.0


def test_an_identifier_echoed_anywhere_is_not_hidden():
    raw = {"signals": [_row(REC), {"deid_status": "failed", "sha256": PHI["sha256"], "patient": "Test_Name"}]}
    assert R._judge_eeg(raw, RARE)["rc_eeg_phi_hidden"] == 0.0


def test_not_applicable_without_recordings_of_that_kind():
    only_clean = {"attachments": {"eeg": {"recordings": [REC]}}}
    assert R._judge_eeg({"signals": [_row(REC)]}, only_clean) == {"rc_eeg_index_ok": 1.0, "rc_eeg_phi_hidden": None}
    assert R._judge_eeg({"signals": []}, {"attachments": {}}) == {"rc_eeg_index_ok": None, "rc_eeg_phi_hidden": None}


def test_refused_build():
    rare = {"attachments": {"genome": {"build_refused": True, "reference": "GRCh37"}}}
    assert R._judge_build_refused({"genome": {"reference": "GRCh37"}, "variants": []}, rare) == 1
    assert R._judge_build_refused({"genome": {"reference": "hg19"}}, rare) == 1
    assert R._judge_build_refused({"genome": {"reference": "GRCh38"}, "variants": []}, rare) == 0     # read as 38
    assert R._judge_build_refused({"genome": {"reference": "GRCh37"},
                                   "variants": [{"chrom": "10", "pos": 1}]}, rare) == 0              # listed anyway
    assert R._judge_build_refused({}, rare) == 0
    assert R._judge_build_refused({"genome": {"reference": "GRCh38"}}, {"attachments": {"genome": {"reference": "GRCh38"}}}) is None


def test_keys_declared():
    for k in ("rc_eeg_index_ok", "rc_eeg_phi_hidden", "rc_vcf_build_refused"):
        assert k in R.KEYS
    assert R.KEYS[-1] == "rc_na_reason"



def test_dicom_index_counts_only_dicom_rows_of_the_shared_signal_list(monkeypatch):
    """raw.signals is one layer-4 list: DICOM and EDF rows together. The DICOM judge counts
    rows with format DICOM (or no format, older outputs); an EEG row must not fail it."""
    monkeypatch.setattr(R, "_dicom_identifiers", lambda z: {"STS_001"})
    rare = {"attachments": {"imaging": {"series": [{"path": "/x/a.zip", "sha256": "d" * 64}]}}}
    dicom = {"format": "DICOM", "deid_status": "done", "sha256": "d" * 64}
    eeg = {"format": "EDF+", "deid_status": "done", "sha256": "e" * 64}
    assert R._judge_signals({"signals": [dicom, eeg]}, rare) == {"rc_signal_index_ok": 1, "rc_signal_phi_free": 1}
    legacy = {"deid_status": "done", "sha256": "d" * 64}
    assert R._judge_signals({"signals": [legacy]}, rare)["rc_signal_index_ok"] == 1
    extra = {"format": "DICOM", "deid_status": "done", "sha256": "f" * 64}
    assert R._judge_signals({"signals": [dicom, extra, eeg]}, rare)["rc_signal_index_ok"] == 0   # control
    leaky = {**eeg, "patient": "STS_001"}
    assert R._judge_signals({"signals": [dicom, leaky]}, rare)["rc_signal_phi_free"] == 0          # phi scan: whole list
