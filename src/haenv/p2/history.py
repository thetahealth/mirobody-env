"""Old ECG confirmation and old HF diagnosis under the printed triage rules."""
from __future__ import annotations
from copy import deepcopy
from haenv.world.history.history_dependency import HistoryDependency
from haenv.world.history.history_dependency import old_day
from haenv.world.history.history_dependency import rate
from haenv.world.history.history_dependency import table
from haenv.world.history.history_dependency import uniform
from haenv.p2.gate import BLOCK
from haenv.p2.gate import plan_of


def diagnosed(rows):
    return {r["condition"] for r in rows or () if r.get("conclusion") == "已记录" and r.get("condition")}


def prepare(raw, cid, plan):
    return None


def finalize(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import enabled
    if enabled(plan):
        return finalize_r2(raw, cid, plan)
    from haenv.p2 import rules as R
    b, g = raw.prediction_context[BLOCK], raw.adjudication[BLOCK]
    T = int(b["today_day"])
    role = uniform("history-role", cid) < rate("p2", plan)
    hf = g["rule"] in ("R8_hf_gain", "R7a_htn_severe_hf")
    af = g["rule"] == "R14_nothing"
    active = role and (hf or af)
    condition = "heart_failure" if hf else "atrial_fibrillation"
    phrase = table()["phrasing"]
    source = phrase["encounter_sources"][int(uniform("history-source", cid) * len(phrase["encounter_sources"]))]
    row = {"day": old_day("p2", cid, T), "source": source, "condition": condition,
           "test": "超声心动图与专科评估" if hf else "十二导联心电图",
           "result": ("心功能减退，结合病史符合心力衰竭。" if hf else "心房颤动，心室率控制良好，随访中症状无变化。") if active else
                     ("心功能正常，未见心衰证据。" if hf else "窦性心律，未见房颤；原不规则脉搏为房性早搏。"),
           "conclusion": "已记录" if active else "未记录"}
    b["clinical_history_record"] = [row]
    b["triage_protocol"] = table()["packs"]["p2"]["protocol"]
    if active and hf:
        b["background"]["known_conditions"] = [x for x in b["background"]["known_conditions"] if x != "heart_failure"]
        raw.user_profile["known_conditions"] = [x for x in raw.user_profile["known_conditions"] if x != "heart_failure"]
    if active and af:
        for x in b["wearable_daily"][-2:]:
            x["irregular_rhythm_alerts"] = 1
        # A documented known diagnosis need not be repeated on today's problem list.
        pb = raw.latent_premise.get("patient_basics") or {}
        pb["comorbidities"] = list(pb.get("comorbidities") or ()) + ["atrial_fibrillation"]
    a = R.assess(b)
    if R.floor_of(a["set"]) != g["floor"]:
        raise ValueError("history injection changed the planned triage quota")
    g.update(rule=a["rule"], set=a["set"], readings=a["readings"])
    g["history_dependency"] = {"active": active, "recent_lower": T - 10,
                               "records": [{"section": "clinical_history_record", "day": row["day"]}],
                               "source": table()["packs"]["p2"]["source"]}


def answer(raw):
    from haenv.p2.rules import assess
    from haenv.p2.rules import floor_of
    return floor_of(assess(raw["prediction_context"][BLOCK])["set"])


def twin(raw, day=None):
    out = {**raw, "prediction_context": {**raw["prediction_context"], BLOCK: deepcopy(raw["prediction_context"][BLOCK])}}
    b = out["prediction_context"][BLOCK]
    b["clinical_history_record"] = [r for r in b.get("clinical_history_record", ()) if day is not None and int(r["day"]) != day]
    return out


POLICY = HistoryDependency(BLOCK, plan_of, prepare, finalize, answer, twin)


def scale2(rows, *, age=None, pregnant=False):
    if (age is not None and age < 16) or pregnant:
        return False
    return any(r.get("condition") == "hypercapnic_respiratory_failure"
               and r.get("blood_gas_confirmed") is True and r.get("target_spo2") == [88, 92]
               and r.get("clinical_scale") == 2 for r in rows or ())


def finalize_r2(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import config
    from haenv.world.validation.difficulty_r2 import register
    from haenv.p2 import rules as R
    if plan.get("tier") in R.table()["tiers"] and R.table()["tiers"][plan["tier"]].get("context"):
        from haenv.p2.context import finalize
        return finalize(raw, cid, plan)
    b, g = raw.prediction_context[BLOCK], raw.adjudication[BLOCK]
    align_weight_ledger_source(raw)
    T, cfg = int(b["today_day"]), config("p2")
    hf = g["rule"] in ("R7a_htn_severe_hf", "R8_hf_gain") or "heart_failure" in b["background"]["known_conditions"]
    role = uniform("history-role", cid) < rate("p2", plan)
    family = "heart_failure" if hf else "atrial_fibrillation"
    if not hf and g["rule"] in ("R2_news2_medium", "R10_news2_low", "R13_news2_zero_symptom", "R14_nothing"):
        choices = R.table()["clinical_context"]["native_history_families"]["weights"]
        if plan.get("history_family"):
            family = plan["history_family"]
        else:
            draw = uniform("p2-native-history-family", cid) * sum(choices.values())
            cumulative = 0
            for candidate, weight in choices.items():
                cumulative += weight
                if draw < cumulative:
                    family = candidate
                    break
    source = table()["phrasing"]["encounter_sources"]
    row = {"day": old_day("p2", cid, T), "source": source[int(uniform("history-source", cid) * len(source))],
           "condition": family, "conclusion": "未记录"}
    if family == "heart_failure":
        row.update(test="超声心动图与专科评估", result="心功能减退，结合症状符合心力衰竭。", conclusion="已记录")
    elif family == "hypercapnic_respiratory_failure":
        row.update(test="住院动脉血气及出院氧疗评估", blood_gas_confirmed=role,
                   target_spo2=[88, 92] if role else None, clinical_scale=2 if role else 1,
                   result="血气确认高碳酸呼衰，临床人员登记血氧目标。" if role else "疑似通气不足；复核血气无高碳酸呼衰，临床人员维持常规血氧评分。")
        if role:
            row["conclusion"] = "已记录"
            check = b["home_check_today"]
            non_spo2 = sum(v for k, v in R.news2_parts(check, R.checklist_of(b)).items() if k != "spo2")
            low_band = g["rule"] == "R10_news2_low" and non_spo2 == 0
            check["spo2_pct"] = 86 + int(uniform("history-spo2", cid) * 2) if low_band else 88 + int(uniform("history-spo2", cid) * 5)
            if g["rule"] == "R10_news2_low" and non_spo2:
                # Low aggregate NEWS2 at the prescribed oxygen target. A
                # modest respiratory-rate rise stays in the existing low
                # aggregate scenario; Scale 1 would escalate this patient.
                if non_spo2 == 1:
                    check["respiratory_rate_per_min"] = 22 + int(uniform("history-low-rr", cid) * 3)
                check["spo2_pct"] = 88 + int(uniform("history-spo2", cid) * 4)
            if g["rule"] == "R2_news2_medium":
                # Above the prescribed oxygen target: the archived blood gas
                # selects Scale 2, while the current oxygen contributes to both scales.
                check.update(on_oxygen=True, spo2_pct=95 + int(uniform("history-oxygen-spo2", cid) * 2),
                             respiratory_rate_per_min=16 + int(uniform("history-oxygen-rr", cid) * 5),
                             heart_rate_bpm=70 + int(uniform("history-oxygen-hr", cid) * 20),
                             temperature_c=round(38.2 + uniform("history-oxygen-temp", cid) * .6, 1),
                             systolic_bp_mmHg=112 + int(uniform("history-oxygen-sbp", cid) * 14),
                             diastolic_bp_mmHg=70 + int(uniform("history-oxygen-dbp", cid) * 11))
                today_bp = [x for x in b["home_bp_log"] if int(x["day"]) == T]
                today_bp[-1].update(systolic_mmHg=check["systolic_bp_mmHg"],
                                    diastolic_mmHg=check["diastolic_bp_mmHg"])
            pb = raw.latent_premise.get("patient_basics") or {}
            pb["comorbidities"] = list(dict.fromkeys([*pb.get("comorbidities", ()), family]))
    else:
        row.update(test="十二导联心电图", result="窦性心律伴房性早搏，未见房颤。")
        if role and g["rule"] != "R9_af_alerts":
            row.update(result="心房颤动，心室率控制良好；当时随访无病情变化。", conclusion="已记录")
            for x in b["wearable_daily"][-2:]:
                x["irregular_rhythm_alerts"] = 1
            pb = raw.latent_premise.get("patient_basics") or {}
            pb["comorbidities"] = list(dict.fromkeys([*pb.get("comorbidities", ()), family]))
    # The same clinical source-title pool serves acted and inert notes across
    # both render paths. Titles describe the assessment, never its result.
    titles = R.table()["clinical_context"]["procedure_notes"][
        "volume" if family == "heart_failure" else "oxygen" if family == "hypercapnic_respiratory_failure" else "rhythm"]
    row["test"] = titles[int(uniform("p2-context-title", cid) * len(titles))]
    b["clinical_history_record"] = [row]
    apply_history_positions(raw, cid, cfg["history_positions"])
    b["triage_protocol"] = table()["packs"]["p2"]["protocol"] + cfg["protocol"]
    a = R.assess(b)
    if a["rule"] != g["rule"] or R.floor_of(a["set"]) != g["floor"]:
        raise ValueError(f"historical evidence changed the planned triage scenario: {g['rule']} -> {a['rule']}")
    g.update(rule=a["rule"], set=a["set"], readings=a["readings"])
    alternate = R.floor_of(R.assess({**b, "clinical_history_record": []})["set"])
    register(g, "clinical_history_record", row["day"], alternate != g["floor"], recent_lower=T - 10)
    from haenv.p2.hooks import measurement_series
    for stream, points in measurement_series(b).items():
        for q in (raw.longitudinal_data or {}).get(stream) or ():
            if int(q["ts"]) in points:
                q["value"] = points[int(q["ts"])]


def apply_history_positions(raw, cid, cfg):
    """Place existing diagnoses; sample writing location without reading the answer.

    Every known high-risk diagnosis has an ordinary archived note whether or not
    it is listed today. The core acting/inert note retains its source and date.
    This changes neither medical eligibility nor the NEWS2 authorization test.
    """
    b = raw.prediction_context[BLOCK]
    rows = b["clinical_history_record"]
    current = list(dict.fromkeys([*b["background"]["known_conditions"],
                                 *raw.user_profile.get("known_conditions", ())]))
    confirmed = diagnosed(rows)
    conditions = list(dict.fromkeys([*current, *sorted(confirmed)]))
    core = rows[0]
    placement = []
    for condition in conditions:
        family = cfg["aliases"].get(condition, condition)
        if family not in cfg["conditions"]:
            continue
        spec = cfg["conditions"][family]
        if condition not in confirmed:
            rows.append({"day": core["day"], "source": core["source"], "condition": condition,
                         "test": "既往门诊诊断记录", "result": spec["display"] + "，当时已登记。",
                         "conclusion": "已记录"})
        silent = uniform("history-position-silence", cid, family) < spec["silence"]
        omitted = (not silent and uniform("history-position-diagnosis", cid, family)
                   < spec["diagnosis_omission_given_recent"])
        listed = not (silent or omitted)
        if listed and condition not in current:
            current.append(condition)
        if not listed:
            current = [c for c in current if c != condition]
        placement.append({"condition": condition, "documentation_family": family,
                          "listed": listed, "recent_silent": silent,
                          "diagnosis_omitted_given_recent": omitted})
    b["background"]["known_conditions"] = current
    raw.user_profile["known_conditions"] = list(current)
    # Truth-side provenance only; source titles and empty bundles stay unchanged.
    raw.adjudication[BLOCK]["history_positions"] = placement


def align_weight_ledger_source(raw):
    """Keep scale readings on the scale source instead of the generic wrist source."""
    if "smart_scale" not in (raw.user_profile or {}).get("device_inventory", ()):
        return
    weights = {(int(q["ts"]), float(q["value"])) for q in
               (raw.longitudinal_data or {}).get("weight", ()) if q.get("value") is not None}
    for row in raw.evidence_ledger or ():
        value, day = row.get("measured_value"), row.get("source_timestamp")
        if (row.get("source_type") == "wearable" and isinstance(value, (int, float))
                and not isinstance(value, bool) and isinstance(day, (int, float))
                and (int(day), float(value)) in weights):
            row["source_type"] = "smart_scale"


# Install only the P2 generation/view extension; shared protocol files and
# pack judging hooks retain their identities.
from haenv.p2.sources import install as _install_sources

_install_sources()

from haenv.p2.complaints import install as _install_complaints

_install_complaints()
