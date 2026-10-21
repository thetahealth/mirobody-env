"""Critical check reports share actual measurements with screening and simple purchases."""
from types import SimpleNamespace
import pytest
import haenv.interaction.gated as gated
import haenv.resources.registry as registry
from haenv.world.observations.events import _render_findings_if_enabled
import haenv.world.observations.events as events
from haenv.world.clinical.clinical_evidence import clinical_result_series
from haenv.world.clinical.clinical_evidence import ClinicalEvidenceUnavailable

def raw(sid,T=168):
    return SimpleNamespace(case_id="CLINICAL-WIRING", user_profile={"sex":"M"},
        prediction_context={"prediction_time_T":T},latent_premise={},
        adjudication={"ddx":{"spec_id":sid}},longitudinal_data={})

def test_aki_report_is_visible_measurements_not_generic_positive():
    r=raw("HD-UNI-09")
    points=gated.synth_on_demand(r,"AKICreatinineChange",168)
    assert points and "creatinine_history" in points[0].get("details",{})
    current=gated.synth_on_demand(r,"Cr",168)
    assert points[0]["details"]["creatinine_history"][-1]["value"]==current[0]["value"]
    assert points[0]["details"]["delta_48h_umol_l"]>=26.5

def test_ckd_history_shares_current_numeric_reading_with_screening(monkeypatch):
    r=raw("HD-UNI-10")
    monkeypatch.setattr(events,"FINDINGS_ENABLED",[True])
    evs=_render_findings_if_enabled(r,normal_before_day=56)
    hist=gated.synth_on_demand(r,"CKDPersistence",168)[0]["details"]
    entries=[e for e in evs if "LCr" in e["evidence_id"]]
    assert len(entries)==1
    current=gated.synth_on_demand(r,"Cr",168)
    assert str(current[0]["value"]) in entries[0]["symptom"]
    assert hist["creatinine_history"][-1]["value"]==current[0]["value"]
    assert all(e["source_timestamp"]<=168 for e in evs)
    assert not any("112天" in e["symptom"] for e in evs)

def test_insufficient_critical_history_is_unavailable_instead_of_normal():
    r=raw("HD-UNI-09")
    r.adjudication["ddx"]["insufficient"]=True
    with pytest.raises(ClinicalEvidenceUnavailable):
        gated.synth_on_demand(r,"AKICreatinineChange",168)

def test_critical_invalid_assay_does_not_fall_back(monkeypatch):
    original=registry.load_findings
    def bad():
        f=original(); f["Cr"].pop("ref",None); return f
    monkeypatch.setattr(registry,"load_findings",bad)
    with pytest.raises(ValueError):
        gated.synth_on_demand(raw("HD-UNI-09"),"Cr",168)

def test_ecg_returns_observations_not_the_diagnosis_label():
    pt=gated.synth_on_demand(raw("HD-UNI-08"),"ECGAtrialPattern",168)[0]
    assert pt.get("details",{}).get("acquisition") == "12_lead_ecg"
    assert not any(x in str(pt["value"]).lower() for x in ("房颤","atrial fibrillation","阳性"))
