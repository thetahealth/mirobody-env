"""M2 visit headers and source records, projected from existing patient facts."""
from __future__ import annotations

from copy import deepcopy

from haenv.i18n.text import J, L, T as tpl


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
    frame["current_medications"] = [{"day": T, "source": L("m2.source.med_reconciliation"), "medicines": medicines}]
    if "scene" not in frame:
        frame["scene"] = tpl("m2.scene.meds_only", meds=med_sentence(medicines))
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
        record = {"day": T, "source": L("m2.source.progress_note"), "text": acute["text"]}
        if acute.get("vitals"):
            record["vitals"] = deepcopy(acute["vitals"])
        frame["encounter_record"] = [record]
    frame.update(encounter_type=L(f"m2.encounter.{frame['frame']}"),
                 today_date=tpl("m2.today_date", day=T), patient_quote=quote)
    if "record_policy" in pc:
        frame["clinic_documents"] = [{"day": T, "source": L("m2.source.clinic_documents"), "text": pc.pop("record_policy")}]
    elif (pc.get("m2_frame") or {}).get("clinic_documents"):
        frame["clinic_documents"] = deepcopy(pc["m2_frame"]["clinic_documents"])
    gold = (raw.adjudication or {}).get("m2_frame") or {}
    if gold.get("frame") == "F3":
        followup = f3_followup(raw.longitudinal_data, gold.get("f3_plan"), T)
        if followup is None:
            raise ValueError("M2 follow-up requires its recorded result")
        frame["followup_record"] = [{"day": T, "source": L("m2.source.lab_report"), "indicator": followup["indicator"],
                                     "unit": followup["unit"], "value": followup["today"]["value"]}]
    pc["m2_frame"] = frame
    from haenv.m2.surface import vary
    raw.prediction_context = vary(pc, raw.case_id)


def med_sentence(medicines) -> object:
    """「患者目前在用 A、B」 or 「患者目前未使用药物」 (no closing stop)."""
    return tpl("m2.scene.on_meds", meds=J("enum", medicines)) if medicines else L("m2.scene.no_meds")


def generate(next_call, *args, **kwargs):
    raw, audit = next_call(*args, **kwargs)
    if raw is not None:
        finalize(raw)
    return raw, audit


def install():
    from haenv.app.pack_hooks import Stage
    from haenv.app.pack_hooks import register_hook
    register_hook(Stage.CASE, "m2:emr", generate)


def render_metadata(raw):
    """Pack rendering boundary after shared management facts have been assembled."""
    from haenv.m2.core import is_m2_case
    if not is_m2_case(raw):
        return
    for protocol in (raw.adjudication.get("open_management") or {}).get("protocols") or ():
        if protocol.get("protocol_id") == "initial-workup":
            protocol.update(applies=L("m2.workup.applies"), excludes=L("m2.workup.excludes"))
