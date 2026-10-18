"""Sourced P4 laboratory mechanisms; patient draws precede surface rendering.

Parameters are synthetic where a paper does not provide a response law. Neither
effects nor errors are fitted to a desired amplitude, audit score or patient ID.
"""
from __future__ import annotations

import math
from copy import deepcopy

from haenv.resources.regpath import load_registry
from haenv.resources.regpath import registry_cached
from haenv.p4.plan import h01


@registry_cached("p4_diversity.yaml")
def table():
    return load_registry("p4_diversity.yaml")


def method_bias_limit(an):
    spec = table()["method_bias"]["analytes"].get(an)
    if spec is None:
        return None
    B = float(spec["max_bias_pct"]) / 100
    return 2 * B / (1 - B)


def reference_calibration(an, cid):
    spec = table()["method_bias"]["analytes"].get(an)
    if spec is None:
        return 0.0
    B = float(spec["max_bias_pct"]) / 100
    return B * (2 * h01("p4-reference-calibration", cid, an) - 1)


def sampled_method_bias(an, cid, tag, *, reference_bias=None):
    spec = table()["method_bias"]["analytes"].get(an)
    if spec is None:
        raise ValueError("no sourced method-bias specification")
    B = float(spec["max_bias_pct"]) / 100
    other = B * (2 * h01(tag, cid, an) - 1)
    reference = reference_calibration(an, cid) if reference_bias is None else reference_bias
    return math.log1p(other) - math.log1p(reference)


def eligible(case, spec, streams):
    from haenv.world.labworld import analyte
    from haenv.p4.plan import marker_text
    from haenv.p4.plan import hidden
    from haenv.p4.plan import acei_indication
    from haenv.p4.plan import ACEI_DRUG
    from haenv.world.generation.demographics import avoid_for
    txt = (marker_text(case, spec) if hidden(case.get("latent") or {}) else
           str((case.get("raw") or {}).get("symptoms") or "")).lower()
    raw = case.get("raw") or {}
    out = []
    for an in table()["design"]["analytes"]:
        aliases = {"creatinine": {"creatinine", "Cr"}, "potassium": {"potassium", "K"},
                   "glucose": {"glucose", "fasting_glucose", "FBG"}, "alt": {"alt", "ALT"},
                   "ck": {"ck", "CK"}, "tsh": {"tsh", "TSH"}, "triglycerides": {"triglycerides", "TG"}}[an]
        if aliases & streams or any(str(t).lower() in txt for t in analyte(an)["marker_tokens"]):
            continue
        if raw.get("drug") in table()["analytes"][an].get("avoid_primary_drugs", []):
            continue
        if an == "creatinine" and (acei_indication(raw.get("disease"), raw.get("comorbidities")) is None or
                avoid_for(ACEI_DRUG, str(raw.get("sex") or ""), str(raw.get("age_range") or ""))):
            continue
        out.append(an)
    return out


def causes(an, cls, case=None):
    if cls == "analytic_biological_noise":
        return [None]
    key = {"true_change": "true_causes", "preanalytical": "pre", "method_difference": "method"}[cls]
    options = list(table()["analytes"][an][key])
    if method_bias_limit(an) is None:
        options = [c for c in options if c not in ("platform_calibration", "reagent_lot")]
    if case is not None:
        raw = case.get("raw") or {}
        have = {raw.get("disease")} | set(raw.get("comorbidities") or ())
        options = [c for c in options if not table()["rules"][c].get("indication") or
                   table()["rules"][c]["indication"] in have]
    return options


def existing_medicines(raw):
    wm = (raw.latent_premise or {}).get("world_medication") or {}
    return {m["drug"] for m in [wm.get("primary"), *(wm.get("concurrent") or [])]
            if m and m.get("drug") and not m.get("overlay")}


def renal_medication_ok(raw, key, ckd):
    spec = table()["medication_world"].get(key) or {}
    if "minimum_egfr" not in spec:
        return True
    if ckd:
        return float(ckd["egfr0"]) >= float(spec["minimum_egfr"])
    patient = (raw.latent_premise or {}).get("patient_basics") or {}
    return "CKD" not in {patient.get("disease"), *(patient.get("comorbidities") or [])}


def plans(cases, specs, streams, feasible=None):
    from haenv.p4.plan import CLASSES
    from haenv.p4.plan import alloc_balanced
    from haenv.p4.plan import hidden
    from haenv.p4.plan import comorbidity_of
    ids = [str(c["case_id"]) for c in cases]
    by = {str(c["case_id"]): c for c in cases}
    ok = {i: eligible(by[i], specs.get(str((by[i].get("latent") or {}).get("ddx_spec_id"))), streams.get(i, set())) for i in ids}
    ids = [i for i in ids if ok[i]]
    hs = {i: hidden(by[i].get("latent") or {}) for i in ids}
    cls = alloc_balanced(ids, [hs[i] for i in ids], block={i: str(by[i]["raw"].get("sex")) for i in ids})
    cm = comorbidity_of(ids, hs, cls)
    out = {}
    for c in CLASSES:
        counts, fs, signs = {}, {}, {}
        for cid in sorted((i for i in ids if cls[i] == c), key=lambda i: h01("p4an-diverse", i)):
            opts = [a for a in ok[cid] if causes(a, c, by[cid])]
            if feasible is not None:
                opts = [a for a in opts if any(x[0] == a for x in feasible.get(cid, []))]
            if not opts:
                continue
            an = min(opts, key=lambda a: (counts.get(a, 0), h01("p4an-choice", cid, a)))
            factors = causes(an, c, by[cid])
            fac = min(factors, key=lambda f: (fs.get((an, f), 0), h01("p4cause", cid, f)))
            sign = min((1, -1), key=lambda s: (signs.get((an, fac, s), 0), h01("p4direction", cid, s)))
            counts[an] = counts.get(an, 0) + 1
            fs[(an, fac)] = fs.get((an, fac), 0) + 1
            signs[(an, fac, sign)] = signs.get((an, fac, sign), 0) + 1
            out[cid] = {"version": 2, "difficulty_r2": True, "class": c, "analyte": an,
                        "sign": sign, "u_ratio": round(h01("p4ratio", cid), 6),
                        "factor": {a: min(causes(a, c, by[cid]), key=lambda f: h01("p4cause", cid, a, f)) for a in ok[cid] if causes(a, c, by[cid])},
                        "cause": fac, "analytes_ok": [an], "decoys": {an: []}, "H": hs[cid],
                        **({"comorbidity": cm[cid]} if cid in cm else {})}
    return out


def _day_range(cid, tag, span):
    lo, hi = map(int, span)
    return lo + int(h01(tag, cid) * (hi - lo + 1))


def verification_consistent(item):
    """P4-only measurement/time checks; shared protocol gold stays unchanged."""
    if "verification_records" not in item:
        return []
    from haenv.world.labworld import analyte
    from haenv.world.labworld.observe import record_consistent
    a, checks = analyte(item["analyte"]), item["verification_records"]
    recs = {r["day"]: r for r in item["records"]}
    bad = []
    if len(checks) != 2 or {r["verification_for_day"] for r in checks} != {item["t0"], item["t1"]}:
        bad.append("verification records do not match the two original draws")
    for r in checks:
        why = record_consistent(r, int(a["ndigits"]), tuple(a["range"]))
        if why:
            bad.append("verification " + why)
        if not int(r["verification_for_day"]) <= int(r["day"]) <= int(item["review_day"]):
            bad.append("verification sample is outside its actual observation interval")
        if r["day"] == r["verification_for_day"] and r["time"] <= recs[r["day"]]["time"]:
            bad.append("verification sample precedes the original collection time")
    return bad


def realize(raw, cid, plan, ctx):
    from haenv.world.labworld import analyte
    from haenv.world.labworld import cv_total
    from haenv.world.labworld import z
    from haenv.world.labworld.observe import make_record
    from haenv.p4.gold import g4
    from haenv.p4.gold import next_step
    from haenv.p4.gold import pair_rcv
    from haenv.p4.gold import true_change_pct
    from haenv.p4.gold import gold_block_consistent
    from haenv.p4.realize import Unrealizable
    from haenv.p4.realize import ckd_record
    from haenv.p4.realize import _clock
    from haenv.p4.realize import _drug
    from haenv.p4.diversity_reader import read_class
    cfg, an = table(), plan["analyte"]
    a = analyte(an)
    cls, sign, fac = plan["class"], int(plan["sign"]), plan.get("cause")
    T = int(raw.prediction_context["prediction_time_T"]) - _day_range(cid, "p4-verification-review-lag", cfg["design"]["verification"]["review_lag_days"])
    cal = cfg["design"]["calendar"]
    remote = h01("p4-diverse-remote", cid) < float(cal["remote_share"])
    archive_gap = _day_range(cid, "p4-diverse-archive-gap", cal["remote_gap_days"])
    gap = archive_gap if remote else _day_range(cid, "p4-diverse-gap", cal["recent_gap_days"])
    t0 = T - gap
    n = _day_range(cid, "p4-diverse-draws", cal["history_draws"])
    archive = T - archive_gap
    days = [archive - (n - k) * _day_range(cid, f"p4-diverse-cadence-{k}", [45, 180]) for k in range(n)]
    days = sorted(set(days + ([archive] if not remote else []) + [t0, T]))
    episode_day = max(d for d in days if d < archive)
    episode_factor = "platform_calibration" if method_bias_limit(an) is not None and h01("p4-diverse-episode-kind", cid) < 0.5 else "infusion_sampling"
    episode_rule = cfg["rules"][episode_factor]
    ep_lo, ep_hi = episode_rule["effect_log"]
    episode_response = ep_lo + (ep_hi - ep_lo) * h01("p4-diverse-episode-response", cid)
    if episode_factor == "platform_calibration":
        episode_response = sampled_method_bias(an, cid, "p4-diverse-episode-response",
                                               reference_bias=0.0 if fac == "jaffe_enzymatic_switch" else None)
    ckd = ckd_record(raw, cid) if plan.get("comorbidity") == "CKD" else None
    if fac and not renal_medication_ok(raw, fac, ckd):
        raise Unrealizable("recorded renal function does not permit the proposed medication course")
    lo, hi = cfg["analytes"][an]["baseline"]
    base = float(ckd["cr0"]) if an == "creatinine" and ckd else math.exp(math.log(lo) + h01("p4-diverse-base", cid, an) * math.log(hi / lo))
    rule = cfg["rules"][fac] if fac else None
    if rule and sign not in rule.get("allowed_signs", (1, -1)):
        raise Unrealizable("course direction is not supported by the actual comparison interval")
    if fac in cfg["medication_world"] and fac != "acei_start_or_stop" and cfg["medication_world"][fac]["drug"] in existing_medicines(raw):
        raise Unrealizable("existing medication course cannot be replaced by a second synthetic course")
    response = 0.0
    if rule:
        lo, hi = map(float, rule["effect_log"])
        response = lo + (hi - lo) * h01("p4-diverse-response", cid, an, fac)
        if fac in ("platform_calibration", "reagent_lot"):
            response = sampled_method_bias(an, cid, "p4-diverse-response")
    intrinsic = int(rule["sign"]) if rule else 1
    if fac in ("platform_calibration", "reagent_lot"):
        intrinsic, response = (1 if response > 0 else -1), abs(response)
    if fac == "meat_meal":
        from haenv.p4.realize import clinical_sources
        lo, hi = clinical_sources()["cooked_meat"]["increment_umol_L"]
        response = math.log1p((lo + (hi - lo) * h01("p4-diverse-meat", cid)) / base)
    if fac == "jaffe_enzymatic_switch":
        from haenv.p4.realize import clinical_sources
        from haenv.p4.realize import paired_method_scatter
        comparison = clinical_sources()["method_comparison"]
        delta = ((float(comparison["slope"]) - 1) * base +
                 float(comparison["umol_L_per_mg_dL"]) * float(comparison["intercept_mg_dL"]) +
                 paired_method_scatter(base) * z("p4-diverse-method-residual", cid))
        signed = math.log1p(delta / base)
        intrinsic, response = (1 if signed > 0 else -1), abs(signed)
    side = 1 if sign == intrinsic else 0
    event = T - _day_range(cid, "p4-diverse-clinical-event", rule["event_lead_days"]) if cls == "true_change" else t0 + max(1, gap // 3)
    if cls == "true_change" and event <= t0:
        raise Unrealizable("clinical event does not fall between the actual comparison reports")
    standard = cv_total(an)
    records = []
    for day in days:
        truth = base
        if cls == "true_change":
            if fac in cfg["medication_world"]:
                prior_start = t0 - cfg["medication_world_design"]["prior_established_days"]
                taking = day >= event if sign == intrinsic else prior_start <= day < event
                truth = base * math.exp(intrinsic * response) if taking else base
            else:
                truth = base * math.exp(sign * response) if day > event else base
        pert = []
        if cls in ("preanalytical", "method_difference") and day == (T if side else t0):
            pert = [{"factor": fac, "kind": "method" if cls == "method_difference" else "preanalytical",
                     "role": "cause", "acts": True, "bias_log": intrinsic * response,
                     "additive": None, "params": {}}]
        if day == episode_day:
            pert.append({"factor": episode_factor, "kind": "method" if episode_factor == "platform_calibration" else "preanalytical",
                         "role": "cause", "acts": True, "bias_log": int(episode_rule["sign"]) * episode_response,
                         "additive": None, "params": {}})
        cv = cv_total(an, "jaffe") if fac == "jaffe_enzymatic_switch" and pert else standard
        e = cv * z("p4-diverse-measurement", cid, an, day)
        clock = _clock(cid, day)
        if an == "tsh":
            clock = "15:00:00" if fac == "sampling_time" and day == (T if side else t0) else "08:00:00"
        records.append(make_record(day, clock, truth, e, pert, int(a["ndigits"]), tuple(a["range"])))
    r0, r1 = (next(r for r in records if r["day"] == d) for d in (t0, T))
    R = pair_rcv(an, r0, r1)
    g = g4(r0, r1, R, (1.3, 2.0), source_policy="clinical")
    if g["class"] != cls or g["dlog_obs"] * sign <= 0:
        raise Unrealizable("independent observations do not yield the planned single source")
    if cls == "analytic_biological_noise" and abs(g["dlog_obs"]) >= R:
        raise Unrealizable("unexplained observation outside RCV")
    if cls != "analytic_biological_noise" and abs(math.log(r1["printed"] / r0["printed"])) <= R:
        raise Unrealizable("non-noise apparent change does not exceed paired RCV")
    if remote and cls != "true_change":
        prior = next(r for r in records if r["day"] == episode_day)
        alternative = g4(prior, r1, pair_rcv(an, prior, r1), (1.3, 2.0), source_policy="clinical")["class"]
        if alternative is None or alternative == cls:
            raise Unrealizable("archived comparison does not distinguish the earlier clinical episode")
    item = {"diversity_version": 1, "analyte": an, "class": cls, "sign": sign, "t0": t0, "t1": T,
            "review_day": int(raw.prediction_context["prediction_time_T"]),
            "source_policy": "clinical", "rcv": round(R, 6), "band": [1.3, 2.0],
            "cv_total": [cv_total(an, "jaffe") if fac == "jaffe_enzymatic_switch" and side == i else standard for i in (0, 1)],
            "records": records, "cause": fac, "factor": fac, "cause_params": {"response": response, "intrinsic": intrinsic, "side": side, "event_day": event},
            "true_change_pct": true_change_pct(r0, r1), "base": round(base, 4), "drug": _drug(raw),
            "acei": {}, "glp1_class": False, "dlog_weight": 0.0, "decoys": [], "decoy_substitutions": [],
            "gap_index": 0, "attempt": 0, "g4": g, "option_rank": 0,
            **({"comorbidity": ckd} if ckd else {})}
    if an == "creatinine":
        from haenv.p4.realize import _indication
        acei_day = event if fac == "acei_start_or_stop" else t0 - 100 - _day_range(cid, "p4-inactive-event-acei_start_or_stop", [0, 60])
        stopped = fac == "acei_start_or_stop" and sign < 0
        item["acei"] = {"initial_on": False, "indication": _indication(raw),
                        "events": ([{"day": t0 - cfg["medication_world_design"]["prior_established_days"], "action": "start"}] if stopped else []) +
                                  [{"day": acei_day, "action": "stop" if stopped else "start"}] +
                                  ([{"day": acei_day + cfg["medication_world_design"]["historical_course_days"], "action": "stop"}]
                                   if fac != "acei_start_or_stop" else [])}
    item["next_step"] = next_step(item)[0]
    # Independent newly collected samples, not the original measurements with
    # the sampled perturbation subtracted. Acute clinical changes are checked
    # immediately; transient sampling conditions are checked after washout.
    checks = []
    washout = int(cfg["design"]["verification"]["washout_days"])
    delayed = fac in cfg["design"]["verification"]["washout_causes"]
    for original in (r0, r1):
        day = original["day"] + (washout if delayed else 0)
        truth = base
        if cls == "true_change":
            if fac in cfg["medication_world"]:
                prior_start = t0 - cfg["medication_world_design"]["prior_established_days"]
                taking = day >= event if sign == intrinsic else prior_start <= day < event
                truth = base * math.exp(intrinsic * response) if taking else base
            else:
                truth = base * math.exp(sign * response) if day > event else base
        hour, minute, second = map(int, original["time"].split(":"))
        check_time = "08:00:00" if delayed else f"{hour + 4:02d}:{minute:02d}:{second:02d}"
        verification_bias = 0.0
        if an == "tsh":
            verification_bias = sampled_method_bias(an, cid, "p4-verification-platform")
        check = make_record(day, check_time, truth,
                            standard * z("p4-standardized-verification", cid, an, original["day"]),
                            ([{"factor": "verification_platform", "kind": "method", "acts": True,
                               "bias_log": verification_bias, "additive": None}] if an == "tsh" else []),
                            int(a["ndigits"]), tuple(a["range"]))
        check["verification_for_day"] = original["day"]
        checks.append(check)
    item["verification_records"] = checks
    block = render(item, raw, cid)
    from haenv.p4.diversity_reader import significance_violations
    violations = significance_violations(block, cls, fac)
    if violations:
        raise Unrealizable("; ".join(violations))
    if gold_block_consistent(item) or verification_consistent(item) or read_class(block)[0] != cls:
        raise Unrealizable("visible records do not reconstruct the planned source")
    return item


def _draw_rows(item, raw, cid):
    from haenv.world.labworld import analyte
    from haenv.p4.render import reference_range
    cfg = analyte(item["analyte"])
    out = []
    for r in sorted(item["records"], key=lambda r: r["day"]):
        row = {"day": r["day"], "time": r["time"], "analyte": item["analyte"], "name": cfg["display"],
               "value": f"{r['printed']:.{int(cfg['ndigits'])}f}", "unit": cfg["unit"],
               "reference_range": reference_range(cfg, raw), "fasting": "未登记", "lab": "门诊检验科",
               "method": cfg["default_method"], "platform": "常规平台", "reagent_lot": "在用批号",
               "specimen": {"tube": "肝素锂血浆管", "processing": "采血后30分钟内分离", "separation_hours": 0.3,
                            "storage_temperature_C": 22, "cells_separated": True, "hemolysis_index": "未见溶血",
                            "free_hemoglobin_g_L": 0.05, "sampling_arm": "右", "infusion_arm": "左",
                            "infusion_running": False, "infusion_solution": "0.9%氯化钠", "collection_site": "肘静脉",
                            "blood_transfer_from": "原管", "calcium_mmol_L": 2.3}}
        out.append(row)
    return out


def render(item, raw, cid):
    from haenv.world.labworld import analyte
    from haenv.p4.render import reference_range
    from haenv.p4.render import medication_rows
    cfg, an = table(), item["analyte"]
    a = analyte(an)
    t0, T = int(item["t0"]), int(item["t1"])
    fac = item.get("cause")
    param = item["cause_params"]
    side_day = T if param["side"] else t0
    reports = _draw_rows(item, raw, cid)
    by = {r["day"]: r for r in reports}
    block = {"analyte": an, "name": a["display"], "unit": a["unit"], "reference_range": reference_range(a, raw),
             "today": T, "report": reports, "diet_record": [], "outpatient_record": [],
             "medication_record": medication_rows({**item, "acei": {}}, raw, T), "laboratory_method_record": [],
             "specimen_procedure": [], "patient_remarks": [], "text_source": "template"}
    # Records describe the actual collection, assay and exposure. No cause IDs,
    # relevance roles or effect flags are rendered on the patient side.
    for key in cfg["analytes"][an]["pre"] + cfg["analytes"][an]["method"] + cfg["analytes"][an]["true_causes"]:
        if key in ("platform_calibration", "reagent_lot") and method_bias_limit(an) is None:
            continue
        rule = cfg["rules"][key]
        active = key == fac
        if not renal_medication_ok(raw, key, item.get("comorbidity")):
            continue
        if key in cfg["medication_world"] and key != "acei_start_or_stop" and cfg["medication_world"][key]["drug"] in existing_medicines(raw):
            continue
        if rule.get("indication"):
            patient = (raw.latent_premise or {}).get("patient_basics") or {}
            if rule["indication"] not in {patient.get("disease")} | set(patient.get("comorbidities") or ()):
                continue
        if rule["section"] == "diet_record":
            for day in (t0, T):
                hours = (2 if active and day == side_day else rule["resting_hours"])
                hhmm = by[day]["time"][:5]
                minutes = int(hhmm[:2]) * 60 + int(hhmm[3:]) - int(hours * 60)
                meal_day, minute = day + minutes // 1440, minutes % 1440
                block["diet_record"].append({"day": day, "draw_day": day, "draw_time": hhmm,
                    "entries": [{"day": meal_day, "meal": "正餐", "time": f"{minute//60:02d}:{minute%60:02d}",
                                 "food": "一份熟猪肉" if key == "meat_meal" else "米饭和菜"}],
                    "last_calories_hours": hours, "cooked_meat": key == "meat_meal"})
        elif key in ("platform_calibration", "reagent_lot", "jaffe_enzymatic_switch"):
            for day in (t0, T):
                row = by[day]
                different = active and day == side_day
                if different:
                    row["platform"] = "比对平台" if key != "reagent_lot" else "常规平台"
                    row["reagent_lot"] = "新批号" if key == "reagent_lot" else "在用批号"
                    if key == "jaffe_enzymatic_switch":
                        row["method"] = "Jaffe 苦味酸速率法"
                    row["lab"] = "协作检验室" if key == "platform_calibration" else "门诊检验科"
                # Batch comparisons include inactive earlier notices too. Their
                # scope/date/platform must match a patient's report to apply.
                block["laboratory_method_record"].append({"day": day - 1, "valid_from": day, "scope": an,
                    "source": "检验科方法与试剂使用记录", "lab": row["lab"], "method": row["method"],
                    "platform": row["platform"], "reagent_lot": row["reagent_lot"], "comparison": rule["exposure"],
                    "patient_sample_difference_pct": round(100 * math.expm1(param["intrinsic"] * param["response"]), 1) if different else 0.0,
                    "valid_until": day,
                    "reference_platform": "常规平台", "reference_lot": "在用批号", "reference_method": a["default_method"],
                    "text": "可交换患者标本并行比对；按报告平台、批号和启用日期核对适用范围。"})
            block["laboratory_method_record"].append({"day": t0 - 190, "valid_from": t0 - 190, "valid_until": t0 - 160,
                "scope": an, "source": "检验科历史比对通知", "lab": "协作检验室", "method": a["default_method"],
                "platform": "旧平台", "reagent_lot": "旧批号", "comparison": rule["exposure"],
                "patient_sample_difference_pct": 100 * math.expm1(sampled_method_bias(an, cid, "p4-old-comparison-" + key)),
                "text": "所列文件属于既往检测批次，报告按当时实际使用的平台和试剂登记。"})
        elif key == "biotin_tsh":
            for day in (t0, T):
                by[day]["assay_architecture"] = "链霉亲和素-生物素夹心法" if active or h01("p4-biotin-platform", cid) < 0.5 else "不采用生物素结合体系"
                block["medication_record"].append({"day": day, "draw_day": day, "drug": "生物素", "event": "服用10 mg补充剂",
                    "dose_mg": 10, "last_dose_hours": 2 if active and day == side_day else 72,
                    "reason": "患者自述用于头发和指甲保健"})
        elif rule["section"] in ("medication_record", "outpatient_record"):
            day = param["event_day"] if active else t0 - 100 - _day_range(cid, "p4-inactive-event-" + key, [0, 60])
            direction = item["sign"] if active else rule["sign"]
            more = direction == rule["sign"]
            if rule["section"] == "medication_record":
                action = ("开始规律服用" if more else "中断规律服用") if key in ("thyroxine_adherence", "fibrate_adherence") else ("开始服用" if more else "由医生停用")
            else:
                action = "新发" if more else "恢复"
            row = {"day": day, "exposure": rule["exposure"], "course_action": action,
                   "event": rule["exposure"] + "，" + action,
                   "supporting_record": _support(key, action)}
            if rule["section"] == "medication_record":
                row["drug"] = rule["exposure"]
                row["reason"] = _indication_text(key)
                course = cfg["medication_world"][key]
                row.update(dose=course["dose"], dose_unit=course["unit"])
                if active and not more:
                    prior_start = t0 - cfg["medication_world_design"]["prior_established_days"]
                    row["prior_course_start_day"] = prior_start
                    block["medication_record"].append({"day": prior_start, "drug": rule["exposure"],
                        "dose": course["dose"], "dose_unit": course["unit"],
                        "event": "长期处方开始", "reason": _indication_text(key)})
                if not active:
                    row["course_end_day"] = day + cfg["medication_world_design"]["historical_course_days"]
                # The prescription states the course; the current clinic note
                # states observed response and adherence. Do not backdate a
                # six-week response into the first prescription.
                support = row.pop("supporting_record")
                block["outpatient_record"].append({"day": T, "onset_day": day,
                    "medication": rule["exposure"], "course_action": action,
                    "text": support if active else "患者交代既往处方和配药经过，所列供药期已结束；目前未再使用此药。"})
            elif key != "recent_exercise":
                row.update(day=T, onset_day=day)
            if key == "recent_exercise":
                row.update(day=side_day - 2 if active else side_day - 21, draw_day=side_day,
                           exercise="不习惯的高强度抗阻训练", exercise_duration_minutes=60,
                           persistent_injury=False)
            block[rule["section"]].append(row)
        elif rule["section"] == "report":
            row = by[side_day]
            s = row["specimen"]
            if key == "infusion_sampling":
                for day in (t0, T):
                    by[day]["specimen"].update(infusion_running=True, collection_site="输液点近心侧静脉" if active and day == side_day else "对侧肘静脉",
                        sampling_arm="左" if active and day == side_day else "右")
            elif key == "hemolysis" and active:
                s.update(free_hemoglobin_g_L=1.2, hemolysis_index="明显溶血")
            elif key == "edta_contamination":
                s.update(blood_transfer_from="K-EDTA管" if active else "原管", calcium_mmol_L=1.1 if active else 2.3,
                         tube_order="EDTA管后生化管", collection_system="跨管转移" if active else "闭合真空系统")
            elif key == "cold_delayed_separation":
                s.update(storage_temperature_C=4, separation_hours=6 if active else 0.3,
                         cells_separated=not active, processing="未分离全血冷藏6小时后离心" if active else "分离血浆后冷藏")
            elif key == "glucose_glycolysis":
                s.update(tube="普通肝素锂管" if active else "柠檬酸缓冲抑制管", cells_separated=not active,
                         separation_hours=3 if active else 0.3, processing="室温放置3小时后分离" if active else "采血后30分钟内分离")
            elif key == "sampling_time":
                for day in (t0, T):
                    by[day]["time"] = "15:00:00" if active and day == side_day else "08:00:00"
    issued = min(r["day"] for r in reports) - 30
    block["specimen_procedure"] = [{"day": issued, "valid_from": issued, "scope": an,
        "source": "检验科标本处理规程", "free_hemoglobin_limit_g_L": cfg["evidence_parameters"]["free_hemoglobin_limit_g_L"],
        "text": "报告登记实际采血位置、采样时段、试管、分离时间、存放温度与溶血指数；按项目方法文件核对。"}]
    block["patient_remarks"] = [{"draw_day": d, "text": "这次来复查，想知道前后为什么有差别。" if h01("p4-diverse-quote", cid, d) < 0.5 else "最近的吃药和生活情况都记在随访记录里。"} for d in (t0, T)]
    for record in item["records"]:
        if record["day"] >= t0:
            continue
        for p in record["perturbations"]:
            row = by[record["day"]]
            if p["factor"] == "infusion_sampling":
                row["specimen"].update(infusion_running=True, sampling_arm="左", infusion_arm="左", collection_site="输液点近心侧静脉")
            elif p["factor"] == "platform_calibration":
                row.update(platform="比对平台", lab="协作检验室")
                block["laboratory_method_record"].append({"day": row["day"] - 1, "valid_from": row["day"], "scope": an,
                    "source": "检验科方法与试剂使用记录", "lab": row["lab"], "method": row["method"],
                    "platform": row["platform"], "reagent_lot": row["reagent_lot"], "comparison": "平台比对",
                    "patient_sample_difference_pct": round(100 * math.expm1(p["bias_log"]), 1),
                    "reference_platform": "常规平台", "reference_lot": "在用批号", "reference_method": a["default_method"],
                    "text": "可交换患者标本并行比对；按报告平台、批号和启用日期核对适用范围。"})
    # Every actual draw has its own method-use record, including unchanged
    # batches. Retain the dated records for every assay.
    for row in reports:
        block["laboratory_method_record"].append({"day": row["day"], "valid_from": row["day"], "scope": an,
            "source": "检验科批次方法使用记录", **{k: row[k] for k in ("lab", "method", "platform", "reagent_lot")},
            "text": "本批检测按所列方法文件执行，报告保留所用平台及试剂批号。"})
    for check in item.get("verification_records", []):
        block["laboratory_method_record"].append({
            "day": check["day"], "scope": an,
            "source": "检验科标准化复核记录", "verification_for_day": check["verification_for_day"],
            "value": f"{check['printed']:.{int(a['ndigits'])}f}", "unit": a["unit"],
            "method": "不使用生物素结合体系的TSH检测" if an == "tsh" else a["default_method"],
            "platform": "非生物素复核平台" if an == "tsh" else "常规平台", "reagent_lot": "复核批号" if an == "tsh" else "在用批号",
            "lab": "门诊检验科", "time": check["time"],
            "text": cfg["design"]["verification"]["conditions"],
            "assay_architecture": "不采用生物素结合体系" if an == "tsh" else "常规检测体系"})
        block["laboratory_method_record"][-1].update(
            reference_method=a["default_method"], reference_platform="常规平台", reference_lot="在用批号",
            patient_sample_difference_pct=round(100 * math.expm1(sum(p["bias_log"] for p in check["perturbations"])), 1))
    block["laboratory_method_record"].sort(key=lambda d: (int(d.get("valid_from", d["day"])), int(d["day"])))
    return block


def _support(key, action):
    row = table()["narratives"][key]
    more = action in ("开始服用", "开始规律服用", "新发")
    return row["more" if more else "less"]


def _indication_text(key):
    return table()["narratives"][key]["indication_text"]


def finalize(raw, cid, plan):
    from haenv.world.validation.difficulty_r2 import register
    g = raw.adjudication["followup_p4"]
    if "class" not in g:
        return
    register_medicines(raw)
    T, t0 = int(g["t1"]), int(g["t0"])
    active = t0 < T - 182
    old = max(r["day"] for r in g["records"] if r["day"] < T - 182)
    register(g, "report", t0 if active else old, active, recent_lower=T - 182,
             deletion="indeterminate" if active and g["class"] == "true_change" else "different")
    g["look_alike"]["records"] = [{"section": "report", "day": old, "acts": active}]


def register_medicines(raw):
    """The world's P4 medication courses are the actual dated source courses.

    P4 owns their effects on its followed laboratory stream. Existing primary
    and concurrent courses are preserved; no other pack calls this function.
    """
    cfg = table()
    wm = (raw.latent_premise or {}).get("world_medication")
    if wm is None:
        raise ValueError("P4 medication record has no world medication container")
    block = raw.prediction_context["followup_p4"]
    by_exposure = {cfg["rules"][k].get("exposure"): (k, v) for k, v in cfg["medication_world"].items()}
    present = existing_medicines(raw)
    courses = {}
    for record in block["medication_record"]:
        matched = by_exposure.get(record.get("exposure", record.get("drug")))
        if not matched:
            continue
        key, spec = matched
        if spec["drug"] in present:
            continue
        action = record.get("course_action")
        if key != "biotin_tsh" and action not in ("开始服用", "开始规律服用", "由医生停用", "中断规律服用"):
            continue
        points = courses.setdefault(spec["drug"], [])
        day = int(record["day"])
        if key == "biotin_tsh":
            points.extend([{"ts": day - int(record["last_dose_hours"]) // 24, "value": spec["dose"]},
                           {"ts": day - int(record["last_dose_hours"]) // 24 + 1, "value": 0.0}])
        elif action in ("开始服用", "开始规律服用"):
            points.append({"ts": day, "value": spec["dose"]})
        else:
            points.extend([{"ts": int(record["prior_course_start_day"]), "value": spec["dose"]}, {"ts": day, "value": 0.0}])
        if "course_end_day" in record:
            points.append({"ts": int(record["course_end_day"]), "value": 0.0})
    for drug, points in courses.items():
        spec = next(v for v in cfg["medication_world"].values() if v["drug"] == drug)
        wm["concurrent"] = [m for m in wm.get("concurrent") or [] if not (m.get("overlay") and m.get("drug") == drug)]
        wm.setdefault("concurrent", []).append({"drug": drug, "dose_timeline": sorted(points, key=lambda q: q["ts"]),
            "dose_unit": spec["unit"], "source": "P4 dated medication record", "effect_scope": [block["analyte"]]})
