"""Reconstruct P4 sources from dated clinical facts and registered rules only."""
from __future__ import annotations

import math

from haenv.i18n.text import text_of
from haenv.p4.textmatch import is_text, same_field
from haenv.p4.features import method_family


def applies(block):
    return any("platform" in r for r in block.get("report") or [])


def comparison(block):
    an, T = block["analyte"], int(block.get("comparison_draw_day", block["today"]))
    rows = [r for r in block.get("report") or [] if r.get("analyte") == an]
    now = [r for r in rows if int(r["day"]) == T]
    past = [r for r in rows if int(r["day"]) < T]
    if len(now) != 1 or not past:
        raise ValueError("same-analyte comparison reports are incomplete")
    day = max(int(r["day"]) for r in past)
    previous = [r for r in past if int(r["day"]) == day]
    if len(previous) != 1:
        raise ValueError("ambiguous prior report")
    return previous[0], now[0]


def _matching_notice(block, report, exposure):
    docs = [n for n in block.get("laboratory_method_record") or []
            if n.get("scope") == block["analyte"] and is_text(n.get("comparison"), exposure) and
            int(n["day"]) <= int(report["day"]) and int(n["valid_from"]) <= int(report["day"]) and
            ("valid_until" not in n or int(report["day"]) <= int(n["valid_until"])) and
            all(same_field(n.get(k), report.get(k)) for k in ("lab", "method", "platform", "reagent_lot"))]
    if not docs:
        return None
    return max(docs, key=lambda n: (int(n["valid_from"]), int(n["day"])))


def variation_evidence(block):
    """Variation assessment from printed reports and dated numeric comparisons.

    Exposure or adherence does not supply a patient-specific quantitative
    response. Only applicable patient-sample method differences correct a
    printed value. No latent truth, cause, or case identifier is consulted.
    """
    from haenv.p4.diversity import table
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    r0, r1 = comparison(block)
    method = lambda r: "jaffe" if method_family(r["method"]) == "jaffe" else None
    R = rcv95(cv_total(block["analyte"], method(r0)), cv_total(block["analyte"], method(r1)))
    corrections, refs = [], []
    for report in (r0, r1):
        docs = [_matching_notice(block, report, table()["rules"][key]["exposure"])
                for key in ("platform_calibration", "reagent_lot", "jaffe_enzymatic_switch")]
        docs = [doc for doc in docs if doc is not None]
        corrections.append(sum(math.log1p(float(doc["patient_sample_difference_pct"]) / 100) for doc in docs))
        refs.extend(docs)
    observed = math.log(float(r1["value"]) / float(r0["value"]))
    corrected = observed - corrections[1] + corrections[0]
    checks = []
    for report in (r0, r1):
        matches = [doc for doc in block.get("laboratory_method_record") or []
                   if doc.get("scope") == block["analyte"] and
                   doc.get("verification_for_day") == report["day"] and
                   doc.get("unit") == report["unit"] and "value" in doc]
        if len(matches) > 1:
            raise ValueError("ambiguous standardized verification")
        if matches:
            checks.append(matches[0])
    from haenv.world.labworld import analyte
    limits = analyte(block["analyte"])["range"]
    if any(not float(limits[0]) <= float(doc["value"]) <= float(limits[1]) for doc in checks):
        raise ValueError("standardized verification is outside the registered measurement range")
    normalized = [float(doc["value"]) / (1 + float(doc["patient_sample_difference_pct"]) / 100) for doc in checks]
    verified = math.log(normalized[1] / normalized[0]) if len(checks) == 2 else None
    verification_R = rcv95(cv_total(block["analyte"]), cv_total(block["analyte"]))
    return {"observed_log_change": observed, "corrected_log_change": corrected, "rcv_log": R,
            "correction_records": refs, "current_records": [r1],
            "verification_records": checks,
            "verified_log_change": verified, "verification_rcv_log": verification_R}


def significance_violations(block, cls, cause):
    """Prospective admission constraints, rechecked by the emission audit."""
    v = variation_evidence(block)
    if len(v["verification_records"]) != 2:
        return ["two independently measured standardized checks are required"]
    if cls == "analytic_biological_noise":
        return []  # Board retains its original noise admission distribution.
    bad = []
    if cls == "true_change":
        if (v["corrected_log_change"] * v["observed_log_change"] <= 0 or
                v["verified_log_change"] * v["observed_log_change"] <= 0):
            bad.append("true change does not persist in visible correction and independent verification")
    else:
        if abs(v["verified_log_change"]) > v["verification_rcv_log"]:
            bad.append("independent standardized observations do not return within paired RCV")
        if cause in ("platform_calibration", "reagent_lot", "jaffe_enzymatic_switch") and abs(v["corrected_log_change"]) > v["rcv_log"]:
            bad.append("applicable numeric method correction does not return within paired RCV")
    return bad


def read_class(block):
    from haenv.p4.diversity import table
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import rcv95
    cfg = table()
    params = cfg["evidence_parameters"]
    an = block["analyte"]
    r0, r1 = comparison(block)
    t0, T = int(r0["day"]), int(r1["day"])
    observed = math.log(float(r1["value"]) / float(r0["value"]))
    sign = 1 if observed > 0 else -1
    candidates = []

    def add(key, side, direction=None):
        rule = cfg["rules"][key]
        if direction is None:
            direction = int(rule["sign"]) * side
        if direction == sign:
            candidates.append((rule["class"], side, key))

    for key in cfg["analytes"][an]["pre"]:
        rule = cfg["rules"][key]
        for side, r in ((-1, r0), (1, r1)):
            day, s = int(r["day"]), r.get("specimen") or {}
            if key == "infusion_sampling" and s.get("infusion_running") and same_field(s.get("sampling_arm"), s.get("infusion_arm")) and is_text(s.get("collection_site"), "p4.site.proximal_to_infusion"):
                add(key, side)
            elif key == "hemolysis":
                docs = [p for p in block.get("specimen_procedure") or [] if p.get("scope") == an and int(p["valid_from"]) <= day]
                if not docs:
                    raise ValueError("hemolysis limit requires applicable specimen document")
                limit = float(max(docs, key=lambda d: int(d["valid_from"]))["free_hemoglobin_limit_g_L"])
                if float(s["free_hemoglobin_g_L"]) > limit:
                    add(key, side)
            elif key == "edta_contamination" and is_text(s.get("blood_transfer_from"), "p4.transfer.k_edta_tube") and float(s["calcium_mmol_L"]) < params["edta_calcium_support_max_mmol_L"]:
                add(key, side)
            elif key == "cold_delayed_separation" and not s.get("cells_separated") and float(s["storage_temperature_C"]) <= params["cold_max_C"] and float(s["separation_hours"]) >= params["cold_separation_min_h"]:
                add(key, side)
            elif key == "glucose_glycolysis" and not s.get("cells_separated") and not is_text(s.get("tube"), "p4.tube.citrate_inhibitor") and float(s["separation_hours"]) >= params["glycolysis_min_h"]:
                add(key, side)
            elif key in ("meat_meal", "nonfasting"):
                for row in block.get("diet_record") or []:
                    if int(row["draw_day"]) != day:
                        continue
                    meat = any(is_text(e.get("food"), "p4.food.cooked_pork") for e in row.get("entries") or [])
                    if key == "meat_meal" and (not row.get("cooked_meat") or not meat):
                        continue
                    lo, hi = rule["window_hours"]
                    if lo <= float(row["last_calories_hours"]) <= hi:
                        add(key, side)
                        break
            elif key == "recent_exercise":
                for row in block.get(rule["section"]) or []:
                    if is_text(row.get("exposure"), rule["exposure"]) and int(row.get("draw_day", -99999)) == day and not row.get("persistent_injury"):
                        lo, hi = rule["window_days"]
                        if lo <= day - int(row["day"]) <= hi:
                            add(key, side)
            elif key == "sampling_time" and params["tsh_afternoon_hours"][0] <= int(r["time"][:2]) <= params["tsh_afternoon_hours"][1]:
                other = r1 if side == -1 else r0
                if params["tsh_morning_hours"][0] <= int(other["time"][:2]) <= params["tsh_morning_hours"][1]:
                    add(key, side)
    for key in cfg["analytes"][an]["method"]:
        rule = cfg["rules"][key]
        if key == "biotin_tsh":
            for side, r in ((-1, r0), (1, r1)):
                if not is_text(r.get("assay_architecture"), "p4.assay.streptavidin_biotin"):
                    continue
                for m in block.get("medication_record") or []:
                    if is_text(m.get("drug"), rule["exposure"]) and int(m.get("draw_day", -99999)) == int(r["day"]) and float(m["dose_mg"]) == params["biotin_dose_mg"]:
                        lo, hi = rule["window_hours"]
                        if lo <= float(m["last_dose_hours"]) <= hi:
                            add(key, side)
            continue
        fields = {"platform_calibration": ("platform", "lab"), "reagent_lot": ("reagent_lot",), "jaffe_enzymatic_switch": ("method",)}[key]
        if all(same_field(r0.get(k), r1.get(k)) for k in fields):
            continue
        for side, r in ((-1, r0), (1, r1)):
            notice = _matching_notice(block, r, rule["exposure"])
            if notice is None:
                continue
            reference = {"platform": notice["reference_platform"], "lab": None,
                         "reagent_lot": notice["reference_lot"], "method": notice["reference_method"]}
            if all(is_text(r.get(k), "p4.lab.outpatient") if k == "lab" else same_field(r.get(k), reference[k]) for k in fields):
                continue
            pct = float(notice["patient_sample_difference_pct"])
            if pct and (1 if pct > 0 else -1) * side == sign:
                add(key, side, sign)
    for key in cfg["analytes"][an]["true_causes"]:
        rule = cfg["rules"][key]
        for row in block.get(rule["section"]) or []:
            day = int(row.get("onset_day", row["day"]))
            if not is_text(row.get("exposure"), rule["exposure"]) or not t0 < day <= T:
                continue
            # Supporting facts are required. A prescription alone is not a
            # diagnosis of statin myopathy, liver injury or thyroiditis.
            supports = [n for n in block.get("outpatient_record") or [] if is_text(n.get("medication"), rule["exposure"])
                        and int(n.get("onset_day", -99999)) == day and same_field(n.get("course_action"), row.get("course_action"))
                        and int(n["day"]) <= T and n.get("text")]
            if not row.get("supporting_record") and not supports:
                continue
            if key in ("thyroxine_adherence", "fibrate_adherence") and T - day < params["treatment_settling_days"]:
                continue
            action = row.get("course_action")
            more = any(is_text(action, k) for k in ("p4.action.start", "p4.action.start_regular", "p4.action.new_onset"))
            less = any(is_text(action, k) for k in ("p4.action.stop_regular", "p4.action.stopped_by_doctor", "p4.action.recovered"))
            if not (more or less):
                continue
            direction = int(rule["sign"]) * (1 if more else -1)
            add(key, direction, direction)
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1:
        return unique[0][0], unique
    if len(unique) > 1:
        return "ambiguous", unique
    method = lambda r: "jaffe" if method_family(r["method"]) == "jaffe" else None
    R = rcv95(cv_total(an, method(r0)), cv_total(an, method(r1)))
    return ("analytic_biological_noise" if abs(observed) < R else "unresolved"), []


def answer(block):
    from haenv.p4.gold import next_step
    cls, candidates = read_class(block)
    if cls in ("ambiguous", "unresolved"):
        raise ValueError("visible records do not determine one source")
    r0, r1 = comparison(block)
    fac = candidates[0][2] if candidates else None
    side = candidates[0][1] if candidates else 0
    rows = [{"day": r["day"], "printed": float(r["value"]), "perturbations":
             [{"acts": True}] if cls in ("preanalytical", "method_difference") and i == (0 if side < 0 else 1) else []}
            for i, r in enumerate((r0, r1))]
    gold = {"class": cls, "cause": fac, "t0": r0["day"], "t1": r1["day"], "records": rows, "acei": {"events": []}}
    from haenv.p4.gold import ingested_interference
    if ingested_interference(fac):
        from haenv.p4.diversity import table
        exposure = table()["rules"][fac]["exposure"]
        status = [r for r in block.get("medication_record", []) if is_text(r.get("drug"), exposure) and
                  "clean_followup" in r and "course_end_day" in r]
        if not status:
            raise ValueError("ingested interference requires a dated follow-up exposure record")
        latest = max(status, key=lambda r: int(r["day"]))
        gold.update(review_day=latest["day"], ingested_exposure={"end_day": latest["course_end_day"],
                                                               "clean_followup": latest["clean_followup"]})
    observed = 100 * (float(r1["value"]) / float(r0["value"]) - 1)
    return {"change_source": cls, "factor": fac, "true_change_pct": round(observed, 1) if cls == "true_change" else 0.0,
            "next_step": next_step(gold)[0], "reason": text_of("p4.reader.reason")}
