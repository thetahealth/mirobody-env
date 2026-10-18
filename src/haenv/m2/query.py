"""M2 in query mode: purchases from the priced menu inside the common tool loop.

M2 runs in the gated geometry: every stream but weight is behind the Gatekeeper, and tests and
streams are bought from a priced menu within a budget (`haenv/interaction/gated.py`). In query mode the same
loop runs (`haenv/interaction/query/gated.py`); this adapter adds the two purchase tools and says what the
question tells the model about them. The menu, the prices, the budget, the refusal past the budget,
billing by the menu's kind and the tool track are the gated geometry's, unchanged.

Decisive evidence (`key_lookups`): the key signals of the case's gold lines (`core.threads_of`, the
keys of the pack's own `m2_key_acquired`), seen when a purchase or a stream read returned the menu
item that carries them (`score._bought`).

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.interaction.query.adapter import KeyLookup
from haenv.interaction.query.adapter import QueryAdapter
from haenv.interaction.query.adapter import register

PROTOCOL = ("\n· 本题部分数据要**购买**:get_menu 列出可买的项目、价格、预算与余额;order_test(target) 买一项"
            "(可在一轮里买多项)。买到的监测流之后也可以用 get_series 读。"
            "\n  ⚠️ 预算点不全整张菜单 —— 必须挑;超出余额的项目不会执行。菜单里可能有与这位患者无关的项,点了照样计费。"
            "\n  ⚠️ 预算用尽而仍未作答,本题按未作答处理。"
            "\n  ⚠️ 检查项(菜单 tests)买过即计入「你开的检查」 —— `tests_to_order` 里只需再填菜单未列的检查。"
            "\n  · 结果为「无法提供可解释结果」的项目不是正常或阴性,不能据此排除疾病。")


class M2Adapter(QueryAdapter):
    def profile_header(self, profile):
        return {}

    def owns(self, payload) -> bool:
        return False                          # M2 is asked through the gated bridge only

    def key_lookups(self, vp, store):
        from haenv.m2.core import threads_of
        from haenv.m2.score import _bought
        keys = sorted({k for t in threads_of(((getattr(vp, "adjudication", None) or {}).get("ddx")) or {})
                       for k in t.get("key_signals") or []})
        def seen(k, r):
            if k.startswith("archive:"):
                return r.get("_from") == "archived_laboratory_record" and r.get("finding") == k[8:] and bool(r.get("observation")) and r.get("result") == "既往阳性，报告可核对"
            return bool(r.get("_from")) and k in _bought([r["_from"]])
        return [KeyLookup(str(k), lambda _t, r, k=k: seen(k, r)) for k in keys]

    def read_answer(self, sp) -> dict:
        """The constant differential baseline (`baselines.ConstantDdxSolver`): M2's answer is a diagnosis,
        which no rule reads off the record; the offline reader measures the purchase path, not the answer."""
        from haenv.world.generation.baselines import ConstantDdxSolver
        out = ConstantDdxSolver().solve(sp)
        return {"forecast": out.forecast, "drivers": out.drivers, "action": out.action,
                "data_quality": out.data_quality, "cited_evidence": out.cited_evidence, **(out._raw or {}),
                "differential": [{"rank": 1, "diagnosis": "多囊卵巢综合征"}]}     # `gated_minimal`'s constant

    def blind_answer(self, payload=None) -> dict:
        return self.read_answer(payload)


ADAPTER = register(M2Adapter(
    pack="m2", block="m2_frame", header_keys=("encounter_type", "today_date", "patient_quote"),
    not_patient_data=("text_source", "frame", "scene", "followed_indicator", "encounter_type", "today_date", "patient_quote"),
    fhir_types={"archived_laboratory_record": "DiagnosticReport", "followup_record": "Observation",
                "outpatient_laboratory_record": "Observation", "outpatient_encounter_record": "DocumentReference",
                "current_medications": "MedicationStatement", "medication_record": "MedicationRequest",
                "encounter_record": "DocumentReference", "prior_encounter_narrative": "DocumentReference"},
    sections={
        "outpatient_laboratory_record": ("labs", "检验报告，按报告日期"),
        "outpatient_encounter_record": ("notes", "门诊病程，按就诊日期"),
        "archived_laboratory_record": ("labs", "检验报告，按报告日期"),
        "followup_record": ("labs", "检验报告，按报告日期"),
        "prior_encounter_narrative": ("notes", "门诊病程，按就诊日期"),
        "encounter_record": ("notes", "门诊病程，按就诊日期"),
        "current_medications": ("meds", "用药核对，按核对日期"),
        "medication_record": ("meds", "处方记录，按开具日期"),
        "event_history_record": ("notes", "问诊记录，按日期"),
        "laboratory_method_record": ("record", "检验科文档，按生效日期"),
        "clinic_documents": ("record", "院内诊疗文档，按日期"),
    }))
