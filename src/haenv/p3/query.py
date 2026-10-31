"""P3 source-oriented retrieval and a visit header without a selected medicine.

The header contains visit type, today, patient words and all current medicines.
Notes, chemistry, prescribing, dispensing and monitoring are retrieved separately.
Legacy section declarations support stored historical payloads; new emissions
contain only source-oriented records. Verifier lookups never reach the solver.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.i18n.text import L, text_of
from haenv.interaction.query.adapter import KeyLookup
from haenv.interaction.query.adapter import QueryAdapter
from haenv.interaction.query.adapter import register

_HEAD = ("condition", "drug", "dose_ladder", "control_axis", "clinic_target", "clinic_protocol", "today", "review_from_day", "note", "documented_course_only")


class P3Adapter(QueryAdapter):
    def audit_block(self, block):
        """Compatibility view for existing record invariants; never served by a tool."""
        from haenv.p3.emr import clinical_view
        return {**clinical_view(block), **block}

    def header(self, block):
        if "current_medications" in block:
            return {k: block[k] for k in ("encounter_type", "today_date", "patient_quote", "current_medications")}
        if block.get("rules_delivery") == "library":
            return {k: block[k] for k in ("instruction", "today_date", "encounter_type")}
        return super().header(block)

    def profile_header(self, profile: dict) -> dict:
        return {k: v for k, v in profile.items() if k != "known_conditions"}

    def lab_label(self, block: dict, section: str) -> str:
        return f"{block.get('control_axis')}" if section == "visits" else section

    def guard_terms(self, vp):
        from haenv.p3.gate import EN_TOKENS
        from haenv.p3.gate import ZH_TOKENS
        from haenv.p3.gate import gold_numbers
        g = (getattr(vp, "adjudication", None) or {}).get("meds_p3") or {}
        return EN_TOKENS + ZH_TOKENS, tuple(gold_numbers(g))

    def key_lookups(self, vp, store):
        g = (getattr(vp, "adjudication", None) or {}).get("meds_p3") or {}
        if "class" not in g:
            return []
        if "lab_results" in store.block:
            sections = ["encounter_record", "lab_results", "prescription_record", "medication_review"]
            if any(s.startswith("adh") for s in (g.get("g3") or {}).get("states") or []):
                sections.append("dispensing_record")
            return [KeyLookup(s, lambda _t, r, s=s: r.get("_from") == s) for s in sections] + self.history_lookups(vp) + self.protocol_lookups(store)
        states = [str(s) for s in (g.get("g3") or {}).get("states") or ()]
        out = [KeyLookup("visits", lambda _t, r: r.get("_from") == "visits")]
        if "encounter_record" in store.block:
            out.append(KeyLookup("encounter_record", lambda _t, r: r.get("_from") == "encounter_record"))
        if any(s.startswith("adh") for s in states):
            out += [KeyLookup("refill_record", lambda _t, r: r.get("_from") == "refill_record"),
                    KeyLookup("pill_count_today", lambda _t, r: r.get("_from") == "pill_count_today")]
        if (g.get("g3") or {}).get("ae") or any(s.startswith("ae") for s in states):
            out.append(KeyLookup("symptom_diary", lambda _t, r: r.get("_from") == "symptom_diary"))
        return out + self.history_lookups(vp) + self.protocol_lookups(store)

    def read_answer(self, sp) -> dict:
        from haenv.p3.features import printed_rule_reader
        d = printed_rule_reader(sp)
        blk = (getattr(sp, "prediction_context", None) or {})[self.block]
        from haenv.p3.emr import clinical_view
        blk = clinical_view(blk)
        change = d in ("uptitrate", "downtitrate", "switch")
        return {"med_decision": d, "target_drug": blk.get("drug"), "new_dose": None, "rationale_evidence": [],
                "action": {"selected_action_class": "A3", "specific_action": text_of("p3.stub.printed_rules"),
                           "clinician_review_required": change}}

    def blind_answer(self, payload=None) -> dict:
        return {"med_decision": "maintain", "target_drug": None, "new_dose": None, "rationale_evidence": [],
                "action": {"selected_action_class": "A3", "specific_action": text_of("p3.stub.not_read"),
                           "clinician_review_required": False}}


ADAPTER = register(P3Adapter(
    pack="p3", block="meds_p3", header_keys=_HEAD,
    # These are supplied immediately in the question, not record sources.
    not_patient_data=("text_source", "today_date", "encounter_type", "patient_quote", "rules_delivery", "instruction"),
    fhir_types={"encounter_record": "DocumentReference", "lab_results": "Observation",
                "prescription_record": "MedicationRequest", "dispensing_record": "MedicationDispense",
                "current_medications": "MedicationStatement", "medication_review": "MedicationStatement",
                "home_measurements": "Observation", "visits": "Encounter",
                "refill_record": "MedicationDispense", "other_medications": "MedicationStatement"},
    sections={
        "administration_record": ("meds", L("p3.menu.administration_record")),
        "medication_instructions": ("meds", L("p3.menu.medication_instructions")),
        "current_medications": ("meds", L("p3.menu.current_medications")),
        "lab_results": ("labs", L("p3.menu.lab_results")),
        "prescription_record": ("meds", L("p3.menu.prescription_record")),
        "dispensing_record": ("meds", L("p3.menu.dispensing_record")),
        "medication_review": ("meds", L("p3.menu.medication_review")),
        "home_measurements": ("series", L("p3.menu.home_measurements")),
        "clinic_documents": ("record", L("p3.menu.clinic_documents")),
        "home_pressure_record": ("series", L("p3.menu.home_pressure_record")),
        "documented_course_only": ("record", L("p3.menu.documented_course_only")),
        "adverse_event_record": ("record", L("p3.menu.adverse_event_record")),
        "review_from_day": ("record", L("p3.menu.review_from_day")),
        "condition": ("record", L("p3.menu.condition")), "drug": ("record", L("p3.menu.drug")),
        "dose_ladder": ("record", L("p3.menu.dose_ladder")), "control_axis": ("record", L("p3.menu.control_axis")),
        "clinic_target": ("record", L("p3.menu.clinic_target")), "clinic_protocol": ("record", L("p3.menu.clinic_protocol")),
        "today": ("record", L("p3.menu.today")), "note": ("record", L("p3.menu.note")),
        "visits": (("labs", "meds"), L("p3.menu.visits")),
        "refill_record": ("meds", L("p3.menu.refill_record")),
        "pill_count_today": ("meds", L("p3.menu.pill_count_today")),
        "other_medications": ("meds", L("p3.menu.other_medications")),
        "symptom_diary": ("notes", L("p3.menu.symptom_diary")),
        "event_history_record": ("notes", L("p3.menu.event_history_record")),
        "encounter_record": ("notes", L("p3.menu.encounter_record")),
        "known_comorbidity": ("profile", L("p3.menu.known_comorbidity")),
    }))
