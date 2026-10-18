"""P3 source-oriented retrieval and a visit header without a selected medicine.

The header contains visit type, today, patient words and all current medicines.
Notes, chemistry, prescribing, dispensing and monitoring are retrieved separately.
Legacy section declarations support stored historical payloads; new emissions
contain only source-oriented records. Verifier lookups never reach the solver.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

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
                "action": {"selected_action_class": "A3", "specific_action": "按本院印出的规则处理",
                           "clinician_review_required": change}}

    def blind_answer(self, payload=None) -> dict:
        return {"med_decision": "maintain", "target_drug": None, "new_dose": None, "rationale_evidence": [],
                "action": {"selected_action_class": "A3", "specific_action": "未查看记录",
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
        "current_medications": ("meds", "复诊时核对的在用药物清单"),
        "lab_results": ("labs", "检验结果，按采样日期"),
        "prescription_record": ("meds", "处方流水，按开具日期"),
        "dispensing_record": ("meds", "取药流水，按发药日期"),
        "medication_review": ("meds", "门诊用药核对记录，按日期"),
        "home_measurements": ("series", "患者提交的家庭测量记录，按日期"),
        "clinic_documents": ("record", "院内诊疗文档"),
        "home_pressure_record": ("series", "复诊时提交的七日家庭收缩压均值，来自本次控制轴的同一份原始记录"),
        "documented_course_only": ("record", "症状经过仅按原始记录登记"),
        "adverse_event_record": ("record", "带日期的既往用药、症状、CK 检验与停药及再用经过"),
        "review_from_day": ("record", "当前复评病程的起始日"),
        "condition": ("record", "被调整的药所治疗的疾病"), "drug": ("record", "被调整的药"),
        "dose_ladder": ("record", "该药的剂量阶梯"), "control_axis": ("record", "控制指标"),
        "clinic_target": ("record", "本院对该患者的目标值"), "clinic_protocol": ("record", "本院复评与调药规程"),
        "today": ("record", "今天的病程日"), "note": ("record", "说明"),
        "visits": (("labs", "meds"), "历次复诊:控制指标读数、处方剂量、起始/加量/续方"),
        "refill_record": ("meds", "本药的取药记录"),
        "pill_count_today": ("meds", "今天的药盒计数"),
        "other_medications": ("meds", "患者其他药的完整带日期用药记录(不在本题调整范围)"),
        "symptom_diary": ("notes", "症状日记:症状、起病日、程度、现状"),
        "event_history_record": ("notes", "患者自述问诊记录，按日期"),
        "encounter_record": ("notes", "门诊病程记录，按就诊日期"),
        "known_comorbidity": ("profile", "已知合并症"),
    }))
