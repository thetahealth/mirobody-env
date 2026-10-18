"""P2 query sources and verifier-side decisive lookups.

The real-world header carries visit type, date and the patient's statement.
The dated clinical sources and hospital procedures are retrieved through tools.
Legacy section IDs remain stable addresses, with source-only catalog labels.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from haenv.interaction.query.adapter import KeyLookup
from haenv.interaction.query.adapter import QueryAdapter
from haenv.interaction.query.adapter import register
from haenv.p2 import history as _history  # installs the pack-local source question for offline query readers

#: rule id prefix -> the sections whose rows decide it
KEY_SECTIONS = (("R0", ("intake_checklist",)), ("R4", ("intake_checklist",)),
                ("R1", ("home_check_today",)), ("R2", ("home_check_today",)), ("R3", ("home_check_today",)),
                ("R10", ("home_check_today",)), ("R5", ("cgm_nightly",)), ("R11", ("cgm_nightly",)),
                ("R6", ("home_bp_log",)), ("R7", ("home_bp_log",)), ("R12", ("home_bp_log",)),
                ("R8", ("morning_weight", "background")), ("R9", ("wearable_daily",)),
                ("R13", ("intake_checklist",)), ("R14", ("home_check_today", "intake_checklist")))


def _rule_id(rid: str) -> str:
    return str(rid or "").split("_", 1)[0]


class P2Adapter(QueryAdapter):
    def protocol_lookups(self, store):
        from haenv.p2.protocols import scope
        with scope():
            return super().protocol_lookups(store)

    def header(self, block):
        if block.get("rules_delivery") == "library":
            return {k: block[k] for k in ("encounter_type", "today_date", "message")}
        return super().header(block)

    def profile_header(self, profile: dict) -> dict:
        return {k: v for k, v in profile.items() if k != "known_conditions"}

    def guard_terms(self, vp):
        from haenv.p2.gate import EN_TOKENS
        from haenv.p2.gate import ZH_TOKENS
        return EN_TOKENS + ZH_TOKENS, ()

    def key_lookups(self, vp, store):
        g = (getattr(vp, "adjudication", None) or {}).get("triage_p2") or {}
        rid = _rule_id(g.get("rule"))
        secs = next((s for p, s in KEY_SECTIONS if rid == p or (p == "R3" and rid.startswith("R3"))
                     or (p == "R7" and rid.startswith("R7"))), ())
        if "device_record" in store.block:
            secs = (*secs, "device_record")
        if g.get("rule") in ("R0a_head_anticoag", "R0b_chemo_unwell", "R10a_stable_rate"):
            secs = (*secs, "clinical_history_record", "home_check_today", "intake_checklist")
        if (g.get("readings") or {}).get("level2_nights") == 1:
            secs = (*secs, "cgm_nightly")
        return [KeyLookup(s, lambda _t, r, s=s: r.get("_from") == s) for s in secs] + self.history_lookups(vp) + self.protocol_lookups(store)

    def read_answer(self, sp) -> dict:
        from haenv.p2.rules import assess
        block = (getattr(sp, "prediction_context", None) or {})[self.block]
        if not block.get("triage_protocol") and not block.get("protocol_library"):
            raise ValueError("the procedure body has not been retrieved")
        # The date is in the visit header, outside the source catalog. Query
        # receipts rebuild the clinical sources; the shared clock supplies T.
        if "today_day" not in block:
            block["today_day"] = int(sp.prediction_context["prediction_time_T"])
        if "patient_statement" in block.get("intake_checklist", {}):
            block["message"] = block["intake_checklist"]["patient_statement"]
        a = assess(block)
        return {"disposition": a["set"][0], "rule": _rule_id(a["rule"]), "evidence": [],
                "reason": "按本院分诊规程逐条核对"}

    def blind_answer(self, payload=None) -> dict:
        return {"disposition": "routine_followup", "rule": "R14", "evidence": [], "reason": "未查看记录"}


ADAPTER = register(P2Adapter(
    pack="p2", block="triage_p2", header_keys=("today_day", "triage_protocol"),
    not_patient_data=("text_source", "instruction", "today_date", "encounter_type", "today_day", "rules_delivery", "data_note", "message"),
    sections={
        "clinical_history_record": ("record", "门诊病程，按日期"),
        "triage_protocol": ("record", "规程文本，按日期"),
        "intake_checklist": ("record", "问诊记录，按日期"),
        "background": (("profile", "meds"), "健康档案，按日期"),
        "home_check_today": ("record", "家庭测量，按日期"),
        "wearable_daily": ("series", "手表数据，按日期"),
        "event_history_record": ("record", "症状事件，按日期"),
        "device_record": ("record", "设备记录，按日期"),
        "home_bp_log": ("series", "家庭血压，按日期"),
        "morning_weight": ("series", "体重记录，按日期"),
        "cgm_nightly": ("series", "动态血糖，按日期"),
    }))
