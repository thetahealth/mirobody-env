"""M2 visit headers and source records, projected from existing patient facts."""
from __future__ import annotations

from copy import deepcopy


def frame_probe(meta):
    # Register this projection's source in the external world manifest too.
    from haenv.m2.core import frame_probe as original
    return original(meta)


def finalize(raw):
    from haenv.m2.core import is_m2_case
    from haenv.m2.core import f3_followup
    if not is_m2_case(raw):
        return
    pc = raw.prediction_context
    T = int(pc["prediction_time_T"])
    frame = dict(pc.get("m2_frame") or {})
    frame.setdefault("frame", "diagnostic_visit")
    medicines = list(pc["current_medications"])
    # The root list remains the original audit input; tools read a dated review.
    frame["current_medications"] = [{"day": T, "source": "门诊用药核对", "medicines": medicines}]
    if "scene" not in frame:
        frame["scene"] = ("患者目前在用 " + "、".join(medicines) if medicines else "患者目前未使用药物") + "。"
    utterances = sorted((e for e in raw.evidence_ledger or []
                         if e.get("source_type") == "patient_reported_symptom"
                         and int(e.get("source_timestamp", 0)) <= T),
                        key=lambda e: int(e.get("source_timestamp", 0)))
    if not utterances:
        raise ValueError("M2 requires a recorded patient utterance")
    quote = utterances[-1]["symptom"]
    plan = ((raw.latent_premise.get("meta") or {}).get("_external") or {}).get("m2") or {}
    acute = plan.get("acute_event")
    if frame["frame"] == "acute_triage" and acute:
        quote = acute["text"]
        record = {"day": T, "source": "门诊病程", "text": acute["text"]}
        if acute.get("vitals"):
            record["vitals"] = deepcopy(acute["vitals"])
        frame["encounter_record"] = [record]
    frame.update(encounter_type={"acute_triage": "急性来诊", "chronic_med_adjustment": "慢病复诊",
                                 "followup_interpretation": "随访复查", "diagnostic_visit": "门诊就诊"}[frame["frame"]],
                 today_date=f"病程第 {T} 天", patient_quote=quote)
    if "record_policy" in pc:
        frame["clinic_documents"] = [{"day": T, "source": "院内诊疗文档", "text": pc.pop("record_policy")}]
    elif (pc.get("m2_frame") or {}).get("clinic_documents"):
        frame["clinic_documents"] = deepcopy(pc["m2_frame"]["clinic_documents"])
    gold = (raw.adjudication or {}).get("m2_frame") or {}
    if gold.get("frame") == "F3":
        followup = f3_followup(raw.longitudinal_data, gold.get("f3_plan"), T)
        if followup is None:
            raise ValueError("M2 follow-up requires its recorded result")
        frame["followup_record"] = [{"day": T, "source": "检验报告", "indicator": followup["indicator"],
                                     "unit": followup["unit"], "value": followup["today"]["value"]}]
    pc["m2_frame"] = frame


def generate(next_call, *args, **kwargs):
    raw, audit = next_call(*args, **kwargs)
    if raw is not None:
        finalize(raw)
    return raw, audit


def install():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.CASE, "m2:emr", generate)
