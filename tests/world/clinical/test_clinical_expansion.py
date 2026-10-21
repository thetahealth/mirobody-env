"""Clinical expansion must reach generation and the purchasable evidence catalogue."""
from __future__ import annotations
import pytest
import haenv.resources.registry as R
import haenv.world.overlay as O
NEW = tuple(f"HD-UNI-{i:02d}" for i in range(7, 21))
def test_all_fourteen_conditions_are_in_the_active_registry():
    assert set(NEW) <= set(O.condition_registry(fresh=True))
def test_new_conditions_have_declared_review_and_stable_ids():
    raw = R._load_yaml("conditions_unified.yaml")["conditions"]
    ids = R.load_case_ids()
    for sid in NEW:
        assert raw[sid]["clinical_review"] == "pending"
        assert sid in ids
    assert len({ids[sid] for sid in NEW}) == len(NEW)
def test_new_symptoms_can_be_rendered_without_diagnosis_names():
    from haenv.world.observations.events import symptom_lay
    lay = symptom_lay()
    for sid in NEW:
        spec = O.condition_registry()[sid]
        for _, text, _ in spec["symptoms"]:
            assert text in lay, (sid, text)
            assert len(lay[text]) >= 2
            for sentence in lay[text]:
                assert not any(alias.casefold() in sentence.casefold()
                               for alias in spec["aliases"]), (sid, sentence)
def test_every_new_confirmatory_finding_can_be_purchased():
    from haenv.m2.core import m2_catalogue, m2_menu_findings
    findings = R.load_findings()
    reachable = {fid for t in m2_catalogue() for fid in m2_menu_findings(t, findings)}
    profiles = R.load_condition_findings(findings)
    for sid in NEW:
        assert sid in profiles, sid
        required = {f["id"] for f in profiles[sid]["findings"] if f["role"] == "confirmatory"}
        assert required, sid
        assert required <= reachable, (sid, required - reachable)
def test_new_disease_generation_and_rival_declarations_are_live():
    from haenv.world.clinical.ddx import ddx_case_specs
    rivals = R.load_rivals()
    cases = ddx_case_specs(only=list(NEW), variants=2, density="mixed", insufficient_frac=0)
    assert {c["latent"]["ddx_spec_id"] for c in cases} == set(NEW)
    for sid in NEW:
        assert len(rivals[sid]) >= 2

def test_new_symptoms_declare_physiology_instead_of_leaving_a_role_shortcut():
    from haenv.world.observations.events import symptom_topic
    topics = R._load_yaml("symptom_topics.yaml")["symptoms"]
    measured = set()
    for sid in NEW:
        for _, text, _ in O.condition_registry()[sid]["symptoms"]:
            assert text in topics, (sid, text)
            topic = symptom_topic(text)
            if topic:
                measured.add(sid)
    assert measured == set(NEW)

def test_statin_candidate_has_an_actual_statin_exposure():
    from haenv.world.clinical.ddx import ddx_case_specs
    from haenv.world.clinical.med_course import concurrent_medicines
    for c in ddx_case_specs(only=["HD-UNI-19"],variants=30,density="mixed",insufficient_frac=0):
        raw = c["raw"]
        drugs = {raw["drug"]}
        drugs.update(m["drug"] for m in concurrent_medicines(c["case_id"],raw["disease"],
                     raw["comorbidities"],raw["drug"],c["latent"]["course_end_day"],
                     sex=raw["sex"],age_range=raw["age_range"]))
        assert any("statin" in d for d in drugs), (c["case_id"],drugs)

def test_missing_statin_regimen_cannot_pass_expansion_admission():
    from copy import deepcopy
    import haenv.world.clinical.ddx as ddx
    check = getattr(ddx,"clinical_medication_eligible",None)
    assert callable(check)
    c=ddx.ddx_case_specs(only=["HD-UNI-19"],variants=1,density="mixed",insufficient_frac=0)[0]
    assert check(c)
    bad=deepcopy(c)
    bad["raw"].update(drug="semaglutide",disease="obesity",comorbidities=[])
    assert not check(bad)
