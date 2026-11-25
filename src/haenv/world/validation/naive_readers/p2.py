"""Triage readers and input ablations of the production assessment.

SYNTHETIC data, evaluation only, not medical advice.
"""
from copy import deepcopy
import json

from haenv.p2 import rules as R, audit as A
from haenv.world.protocols.scoring import FIELDS, trusted_assessment, scale_of
from .core import Reader
from .decisive import FactOutcome

NAME = "p2"
FACTS = ("history", "known_conditions", "series", "checklist")
SERIES = ("wearable_daily", "home_bp_log", "morning_weight", "cgm_nightly")
KNOWN_VOCAB = ()
FACT_NOTE = "Note groups include checklist and series; per-fact totals exclude those two aggregate ablations."


def block(sp):
    return sp["prediction_context"]["triage_p2"]


def full(sp):
    return R.assess(block(sp))["set"][0]


def today_only(sp):
    return A.last_day_reader(block(sp))


def trust_system_score(sp):
    b = block(sp)
    return trusted_assessment(b, b["home_check_today"]["system_news2"])["set"][0]


def checklist_only(sp):
    b = deepcopy(block(sp))
    b["message"] = b["data_note"] = ""
    b["intake_checklist"]["patient_statement"] = ""
    b['device_record'] = []  # A checklist-only reader does not read device narratives.
    return R.assess(b)["set"][0]


def printed_value_only(sp):
    return A.vitals_threshold(block(sp))


READERS = (
    Reader("full", "full", full, "Production assessment of all visible evidence."),
    Reader("today_only", "naive", today_only, "Production last-day reader."),
    Reader("trust_system_score", "naive", trust_system_score, "Trusts the machine score's effective input observations."),
    Reader("checklist_only", "naive", checklist_only, "Structured symptoms only; equals full on the measured release, a shortcut defect."),
    Reader("printed_value_only", "stub", printed_value_only, "Production vital-threshold pseudo-model; no protocol."),
)
NOT_APPLICABLE = {
    "latest_row_only": "One home check per day; same as today_only.",
    "summary_sentence_only": "No summary sentence carries the decision.",
    "latest_dispense_pillbox": "No dispensing or pill-count reader.",
    "keyword_only": "No lexical triage reader in production; modality stubs cover it.",
    "anomaly": "No production anomaly reader for triage.",
}


def gold(case):
    return case["adjudication"]["triage_p2"]


def ok(answer, g):
    return answer in g["set"]


def gold_label(g):
    return g["floor"]


def constant_answers(g):
    return set(g["set"])


def answer_from_label(label):
    return label


def group_key(sp):
    return None


def configure(items):
    global KNOWN_VOCAB
    KNOWN_VOCAB = tuple(sorted({k for item in items for k in
                               block(item.sp).get("background", {}).get("known_conditions", [])}))
    return KNOWN_VOCAB


def features(sp):
    b = block(sp)
    check, ck = b["home_check_today"], R.checklist_of(b)
    naive = {key: float(check.get(key) or 0) for key in
             (*FIELDS.values(), "diastolic_bp_mmHg", "on_oxygen")}
    naive.update({f"checklist:{key}": int(bool(ck.get(key))) for key in R.CHECKLIST})
    naive["n_urgent"] = len(ck.get("urgent_symptoms") or [])
    naive["receipt_total"] = (check.get("system_news2") or {}).get("total", -1)
    known = b.get("background", {}).get("known_conditions", [])
    naive.update({f"known:{k}": int(k in known) for k in KNOWN_VOCAB})
    noise = {f"n_{key}": len(b.get(key) or []) for key in
             ("clinical_history_record", *SERIES, "device_record")}
    noise["payload_chars"] = len(json.dumps(sp, ensure_ascii=False))
    return naive, noise


def metadata(sp):
    b = block(sp)
    rec = b["home_check_today"].get("system_news2")
    kind = None if rec is None else (
        "wrong_scale" if rec["spo2_scale"] != scale_of(b) else
        "missing_zero" if any(v is None for v in rec["input_observations"].values()) else "correct")
    if kind == 'correct' and rec is not None and any(
            rec['input_observations'].get(field) != b['home_check_today'].get(field) for field in FIELDS.values()):
        kind = 'earlier_complete_set'
    return {"receipt_kind": kind}


def knockout(sp, full_answer, accepted=None):
    original = block(sp)
    accepted = R.assess(original)["set"] if accepted is None else accepted
    changes = []
    for name in FACTS:
        b = deepcopy(original)
        if name == "history":
            b["clinical_history_record"] = []
        elif name == "known_conditions":
            b.setdefault("background", {})["known_conditions"] = []
        elif name == "series":
            b = R.last_day_only(b)
        else:
            ck = b["intake_checklist"]
            for key, value in list(ck.items()):
                if isinstance(value, bool) and key != "urgent_screen_complete":
                    ck[key] = False
            ck["urgent_symptoms"] = []
        changes.append((name, name, True, name not in {"series", "checklist"}, b))
    cut = R.last_day_only(original)
    for section in SERIES:
        b = deepcopy(original)
        b[section] = cut[section]
        changes.append((f"series:{section}", "series", True, True, b))
    ck = original["intake_checklist"]
    for key, value in ck.items():
        if value is True and key != "urgent_screen_complete":
            b = deepcopy(original)
            b["intake_checklist"][key] = False
            changes.append((f"checklist:{key}", "checklist", False, True, b))
    for symptom in ck.get("urgent_symptoms") or []:
        b = deepcopy(original)
        b["intake_checklist"]["urgent_symptoms"] = [s for s in ck["urgent_symptoms"] if s != symptom]
        changes.append((f"urgent:{symptom}", "checklist", False, True, b))
    # Narrative-only facts need their own knockout; clearing an already silent
    # tick cannot measure them. Count a symptom once across both text channels.
    # Imported historical excerpts stay in the existing history source group.
    from haenv.p2.narrative import reported, vocabulary
    from haenv.i18n.identity import source
    for symptom in sorted(reported(original)):
        b = deepcopy(original)
        def without_report(value):
            text = source(value)
            for phrase in vocabulary()[symptom]:
                text = text.replace(phrase, '')
            return text
        for field in ('message',):
            b[field] = without_report(b.get(field, ''))
        b['intake_checklist']['patient_statement'] = without_report(ck.get('patient_statement', ''))
        for device in b.get('device_record') or ():
            for entry in device.get('entries') or ():
                entry['text'] = without_report(entry.get('text', ''))
        changes.append((f"report:{symptom}", "narrative", False, True, b))
    check = original["home_check_today"]
    points = R.news2_parts(check, R.checklist_of(original), scale_of(original))
    for key, field in FIELDS.items():
        if points[key] == 0:
            continue
        bands = R.table()["news2"][key]
        if key == "spo2" and scale_of(original) == 2:
            from haenv.world.validation.difficulty_r2 import config
            bands = config("p2")["scale2"]["on_oxygen" if check.get("on_oxygen") else "room_air"]
        candidates = [min(max(check[field], lo), hi) for lo, hi, score in bands if score == 0]
        value = min(candidates, key=lambda v: (abs(v - check[field]), v))
        b = deepcopy(original)
        b["home_check_today"][field] = value
        changes.append((f"vital:{field}", "vitals", False, True, b))
    if check.get("on_oxygen"):
        b = deepcopy(original)
        b["home_check_today"]["on_oxygen"] = False
        changes.append(("vital:on_oxygen", "vitals", False, True, b))
    result = []
    for name, group, outside, per_fact, b in changes:
        try:
            answer = R.assess(b)["set"][0]
            outcome, error = ("flip" if answer not in accepted else "same"), None
        except Exception as exc:
            outcome, error = "error", f"{type(exc).__name__}: {exc}"
        result.append(FactOutcome(name, group, outside, outcome, per_fact=per_fact, error=error))
    return result
