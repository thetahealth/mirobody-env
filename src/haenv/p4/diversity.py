"""Sourced P4 laboratory mechanisms; patient draws precede surface rendering.

Parameters are synthetic where a paper does not provide a response law. Neither
effects nor errors are fitted to a desired amplitude, audit score or patient ID.
"""
from __future__ import annotations

import math
from copy import deepcopy

from haenv.i18n.text import L, T as text
from haenv.p4.textmatch import is_text

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
    active = table()["question_update_14"]["active_causes"][cls][an]
    options = [c for c in options if c in active]
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
            an = min(opts, key=lambda a: (min(fs.get(f, 0) for f in causes(a, c, by[cid])),
                                         counts.get(a, 0), h01("p4an-choice", cid, a)))
            factors = causes(an, c, by[cid])
            fac = min(factors, key=lambda f: (fs.get(f, 0), h01("p4cause", cid, f)))
            sign = min((1, -1), key=lambda s: (signs.get((an, fac, s), 0), h01("p4direction", cid, s)))
            counts[an] = counts.get(an, 0) + 1
            fs[fac] = fs.get(fac, 0) + 1
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
    exposure = item.get("ingested_exposure")
    if exposure and exposure.get("end_day") is not None:
        clean = [r for r in checks if int(exposure["end_day"]) <= int(r["day"]) <= int(item["review_day"])]
        if not exposure.get("clean_followup") or not clean:
            bad.append("ended ingestion has no completed clean follow-up after cessation")
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
    # Independent newly collected samples, not the original measurements with
    # the sampled perturbation subtracted. Acute clinical changes are checked
    # immediately; transient sampling conditions are checked after washout.
    checks = []
    washout = int(cfg["design"]["verification"]["washout_days"])
    from haenv.p4.gold import ingested_interference, contract_table as contract
    ingested = next((key for key in cfg["analytes"][an]["method"] if ingested_interference(key)), None)
    delayed = fac in cfg["design"]["verification"]["washout_causes"] or ingested is not None
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
    if ingested is not None:
        ongoing = h01("p4-ingested-ongoing", cid) < contract()["exposure_calendar"]["ongoing_share"]
        # The specimen can be old while intake has resumed at the review visit.
        item["ingested_exposure"] = {"factor": ingested, "end_day": None if ongoing else T + 1,
                                      "clean_followup": True}
    item["next_step"] = next_step(item)[0]
    block = render(item, raw, cid)
    from haenv.p4.diversity_reader import significance_violations
    violations = significance_violations(block, cls, fac)
    if violations:
        raise Unrealizable("; ".join(violations))
    if gold_block_consistent(item) or verification_consistent(item) or read_class(block)[0] != cls:
        raise Unrealizable("visible records do not reconstruct the planned source")
    return item


def report_value(item, record):
    """The printed reading as the report shows it (also used when a twin edits a record)."""
    from haenv.world.labworld import analyte
    return f"{record['printed']:.{int(analyte(item['analyte'])['ndigits'])}f}"


def _draw_rows(item, raw, cid):
    from haenv.world.labworld import analyte
    from haenv.p4.render import reference_range
    cfg = analyte(item["analyte"])
    out = []
    for r in sorted(item["records"], key=lambda r: r["day"]):
        row = {"day": r["day"], "time": r["time"], "analyte": item["analyte"], "name": cfg["display"],
               "value": report_value(item, r), "unit": cfg["unit"],
               "reference_range": reference_range(cfg, raw), "fasting": L("p4.fasting.unrecorded"), "lab": L("p4.lab.outpatient"),
               "method": cfg["default_method"], "platform": L("p4.platform.routine"), "reagent_lot": L("p4.lot.current"),
               "specimen": {"tube": L("p4.tube.li_heparin_plasma"), "processing": L("p4.processing.within_30min"), "separation_hours": 0.3,
                            "storage_temperature_C": 22, "cells_separated": True, "hemolysis_index": L("p4.hemolysis.none_seen"),
                            "free_hemoglobin_g_L": 0.05, "sampling_arm": L("p4.arm.right"), "infusion_arm": L("p4.arm.left"),
                            "infusion_running": False, "infusion_solution": L("p4.infusion.saline"), "collection_site": L("p4.site.antecubital"),
                            "blood_transfer_from": L("p4.transfer.original_tube"), "calcium_mmol_L": 2.3}}
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
             "today": int(item["review_day"]), "comparison_draw_day": T, "report": reports, "diet_record": [], "outpatient_record": [],
             "medication_record": medication_rows({**item, "acei": {}}, raw, T), "laboratory_method_record": [],
             "specimen_procedure": [], "patient_remarks": [], "text_source": "template"}
    def wording(kind, day):
        choices = cfg["question_update_14"]["wording"][kind]
        return L(choices[int(h01("p4-method-wording", cid, kind) * len(choices))])
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
                    "entries": [{"day": meal_day, "meal": L("p4.meal.main"), "time": f"{minute//60:02d}:{minute%60:02d}",
                                 "food": L("p4.food.cooked_pork") if key == "meat_meal" else L("p4.food.rice_vegetables")}],
                    "last_calories_hours": hours, "cooked_meat": key == "meat_meal"})
        elif key in ("platform_calibration", "reagent_lot", "jaffe_enzymatic_switch"):
            for day in (t0, T):
                row = by[day]
                inactive_key = ("platform_calibration", "reagent_lot")[int(h01("p4-inactive-method-kind", cid) * 2)]
                different = day == side_day and (active or (item["class"] != "method_difference" and key == inactive_key and
                            h01("p4-inactive-method-present", cid) < 0.5))
                if different:
                    row["platform"] = L("p4.platform.comparison") if key != "reagent_lot" else L("p4.platform.routine")
                    row["reagent_lot"] = L("p4.lot.new") if key == "reagent_lot" else L("p4.lot.current")
                    if key == "jaffe_enzymatic_switch":
                        row["method"] = L("p4.method.jaffe_picrate")
                    row["lab"] = L("p4.lab.partner") if key == "platform_calibration" else L("p4.lab.outpatient")
                # Batch comparisons include inactive earlier notices too. Their
                # scope/date/platform must match a patient's report to apply.
                block["laboratory_method_record"].append({"day": day - 1, "valid_from": day, "scope": an,
                    "source": L("p4.source.method_reagent_use"), "lab": row["lab"], "method": row["method"],
                    "platform": row["platform"], "reagent_lot": row["reagent_lot"], "comparison": rule["exposure"],
                    "patient_sample_difference_pct": round(100 * math.expm1(param["intrinsic"] * param["response"]), 1) if different and active else 0.0,
                    "valid_until": day,
                    "reference_platform": L("p4.platform.routine"), "reference_lot": L("p4.lot.current"), "reference_method": a["default_method"],
                    "text": wording("commutable", day)})
            block["laboratory_method_record"].append({"day": t0 - 190, "valid_from": t0 - 190, "valid_until": t0 - 160,
                "scope": an, "source": L("p4.source.historical_comparison"), "lab": L("p4.lab.partner"), "method": a["default_method"],
                "platform": L("p4.platform.old"), "reagent_lot": L("p4.lot.old"), "comparison": rule["exposure"],
                "patient_sample_difference_pct": 100 * math.expm1(sampled_method_bias(an, cid, "p4-old-comparison-" + key)),
                "text": wording("historical", t0)})
        elif key == "biotin_tsh":
            for day in (t0, T):
                recent_inactive = not active and h01("p4-biotin-platform", cid) < 0.5
                by[day]["assay_architecture"] = L("p4.assay.non_biotin") if recent_inactive else L("p4.assay.streptavidin_biotin")
                block["medication_record"].append({"day": day, "draw_day": day, "drug": rule["exposure"], "event": L("p4.event.biotin_10mg"),
                    "dose_mg": 10, "last_dose_hours": 2 if recent_inactive or active and day == side_day else 72,
                    "reason": L("p4.reason.biotin_hair_nails")})
        elif rule["section"] in ("medication_record", "outpatient_record"):
            day = param["event_day"] if active else t0 - 100 - _day_range(cid, "p4-inactive-event-" + key, [0, 60])
            direction = item["sign"] if active else rule["sign"]
            more = direction == rule["sign"]
            if rule["section"] == "medication_record":
                action = (L("p4.action.start_regular") if more else L("p4.action.stop_regular")) if key in ("thyroxine_adherence", "fibrate_adherence") else (L("p4.action.start") if more else L("p4.action.stopped_by_doctor"))
            else:
                action = L("p4.action.new_onset") if more else L("p4.action.recovered")
            row = {"day": day, "exposure": rule["exposure"], "course_action": action,
                   "event": text("p4.event.exposure_action", exposure=rule["exposure"], action=action),
                   "supporting_record": _support(key, more)}
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
                        "event": L("p4.event.long_term_rx_start"), "reason": _indication_text(key)})
                if not active:
                    row["course_end_day"] = day + cfg["medication_world_design"]["historical_course_days"]
                # The prescription states the course; the current clinic note
                # states observed response and adherence. Do not backdate a
                # six-week response into the first prescription.
                support = row.pop("supporting_record")
                block["outpatient_record"].append({"day": T, "onset_day": day,
                    "medication": rule["exposure"], "course_action": action,
                    "text": support if active else L("p4.outpatient.course_ended")})
            elif key != "recent_exercise":
                row.update(day=T, onset_day=day)
            if key == "recent_exercise":
                row.update(day=side_day - 2 if active else side_day - 21, draw_day=side_day,
                           exercise=L("p4.exercise.unaccustomed_resistance"), exercise_duration_minutes=60,
                           persistent_injury=False)
            block[rule["section"]].append(row)
        elif rule["section"] == "report":
            row = by[side_day]
            s = row["specimen"]
            if key == "infusion_sampling":
                for day in (t0, T):
                    by[day]["specimen"].update(infusion_running=True, collection_site=L("p4.site.proximal_to_infusion") if active and day == side_day else L("p4.site.contralateral_antecubital"),
                        sampling_arm=L("p4.arm.left") if active and day == side_day else L("p4.arm.right"))
            elif key == "hemolysis" and active:
                s.update(free_hemoglobin_g_L=1.2, hemolysis_index=L("p4.hemolysis.marked"))
            elif key == "edta_contamination":
                s.update(blood_transfer_from=L("p4.transfer.k_edta_tube") if active else L("p4.transfer.original_tube"), calcium_mmol_L=1.1 if active else 2.3,
                         tube_order=L("p4.tube_order.edta_before_chemistry"), collection_system=L("p4.collection.cross_tube_transfer") if active else L("p4.collection.closed_vacuum"))
            elif key == "cold_delayed_separation":
                s.update(storage_temperature_C=4, separation_hours=6 if active else 0.3,
                         cells_separated=not active, processing=L("p4.processing.cold_whole_blood_6h") if active else L("p4.processing.plasma_then_cold"))
            elif key == "glucose_glycolysis":
                s.update(tube=L("p4.tube.li_heparin_plain") if active else L("p4.tube.citrate_inhibitor"), cells_separated=not active,
                         separation_hours=3 if active else 0.3, processing=L("p4.processing.room_temp_3h") if active else L("p4.processing.within_30min"))
            elif key == "sampling_time":
                for day in (t0, T):
                    by[day]["time"] = "15:00:00" if active and day == side_day else "08:00:00"
    issued = min(r["day"] for r in reports) - 30
    block["specimen_procedure"] = [{"day": issued, "valid_from": issued, "scope": an,
        "source": L("p4.source.specimen_procedure"), "free_hemoglobin_limit_g_L": cfg["evidence_parameters"]["free_hemoglobin_limit_g_L"],
        "text": L("p4.procedure.report_fields")}]
    block["patient_remarks"] = [{"draw_day": d, "text": L("p4.remark.why_different") if h01("p4-diverse-quote", cid, d) < 0.5 else L("p4.remark.see_followup")} for d in (t0, T)]
    for record in item["records"]:
        if record["day"] >= t0:
            continue
        for p in record["perturbations"]:
            row = by[record["day"]]
            if p["factor"] == "infusion_sampling":
                row["specimen"].update(infusion_running=True, sampling_arm=L("p4.arm.left"), infusion_arm=L("p4.arm.left"), collection_site=L("p4.site.proximal_to_infusion"))
            elif p["factor"] == "platform_calibration":
                row.update(platform=L("p4.platform.comparison"), lab=L("p4.lab.partner"))
                block["laboratory_method_record"].append({"day": row["day"] - 1, "valid_from": row["day"], "scope": an,
                    "source": L("p4.source.method_reagent_use"), "lab": row["lab"], "method": row["method"],
                    "platform": row["platform"], "reagent_lot": row["reagent_lot"], "comparison": cfg["rules"]["platform_calibration"]["exposure"],
                    "patient_sample_difference_pct": round(100 * math.expm1(p["bias_log"]), 1),
                    "reference_platform": L("p4.platform.routine"), "reference_lot": L("p4.lot.current"), "reference_method": a["default_method"],
                    "text": wording("commutable", row["day"])})
    # Every actual draw has its own method-use record, including unchanged
    # batches. Retain the dated records for every assay.
    for row in reports:
        block["laboratory_method_record"].append({"day": row["day"], "valid_from": row["day"], "scope": an,
            "source": L("p4.source.batch_method_use"), **{k: row[k] for k in ("lab", "method", "platform", "reagent_lot")},
            "text": L("p4.notice.batch_method")})
    for check in item.get("verification_records", []):
        block["laboratory_method_record"].append({
            "day": check["day"], "scope": an,
            "source": L("p4.source.standardized_verification"), "verification_for_day": check["verification_for_day"],
            "value": f"{check['printed']:.{int(a['ndigits'])}f}", "unit": a["unit"],
            "method": L("p4.method.tsh_non_biotin") if an == "tsh" else a["default_method"],
            "platform": L("p4.platform.non_biotin_verification") if an == "tsh" else L("p4.platform.routine"), "reagent_lot": L("p4.lot.verification") if an == "tsh" else L("p4.lot.current"),
            "lab": L("p4.lab.outpatient"), "time": check["time"],
            "text": cfg["design"]["verification"]["conditions"],
            "assay_architecture": L("p4.assay.non_biotin") if an == "tsh" else L("p4.assay.routine")})
        block["laboratory_method_record"][-1].update(
            reference_method=a["default_method"], reference_platform=L("p4.platform.routine"), reference_lot=L("p4.lot.current"),
            patient_sample_difference_pct=round(100 * math.expm1(sum(p["bias_log"] for p in check["perturbations"])), 1))
    block["laboratory_method_record"].sort(key=lambda d: (int(d.get("valid_from", d["day"])), int(d["day"])))
    if "ingested_exposure" in item:
        exposure = item["ingested_exposure"]
        block["medication_record"].append({"day": item["review_day"], "drug": cfg["rules"][exposure["factor"]]["exposure"],
            "course_end_day": exposure["end_day"], "clean_followup": exposure["clean_followup"],
            "event": L("p4.event.biotin_ongoing" if exposure["end_day"] is None else "p4.event.biotin_ended")})
    return block


def _support(key, more):
    row = table()["narratives"][key]
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
        matched = next((spec for exposure, spec in by_exposure.items()
                        if is_text(record.get("exposure", record.get("drug")), exposure)), None)
        if not matched:
            continue
        key, spec = matched
        if spec["drug"] in present:
            continue
        action = record.get("course_action")
        if key != "biotin_tsh" and not any(is_text(action, k) for k in ("p4.action.start", "p4.action.start_regular",
                                                                        "p4.action.stopped_by_doctor", "p4.action.stop_regular")):
            continue
        points = courses.setdefault(spec["drug"], [])
        day = int(record["day"])
        if key == "biotin_tsh":
            if "last_dose_hours" in record:
                points.extend([{"ts": day - int(record["last_dose_hours"]) // 24, "value": spec["dose"]},
                               {"ts": day - int(record["last_dose_hours"]) // 24 + 1, "value": 0.0}])
            else:
                points.append({"ts": record["course_end_day"] if record["course_end_day"] is not None else day,
                               "value": 0.0 if record["course_end_day"] is not None else spec["dose"]})
        elif any(is_text(action, k) for k in ("p4.action.start", "p4.action.start_regular")):
            points.append({"ts": day, "value": spec["dose"]})
        else:
            points.extend([{"ts": int(record["prior_course_start_day"]), "value": spec["dose"]}, {"ts": day, "value": 0.0}])
        if record.get("course_end_day") is not None:
            points.append({"ts": int(record["course_end_day"]), "value": 0.0})
    for drug, points in courses.items():
        spec = next(v for v in cfg["medication_world"].values() if v["drug"] == drug)
        wm["concurrent"] = [m for m in wm.get("concurrent") or [] if not (m.get("overlay") and m.get("drug") == drug)]
        wm.setdefault("concurrent", []).append({"drug": drug, "dose_timeline": sorted(points, key=lambda q: q["ts"]),
            "dose_unit": spec["unit"], "source": "P4 dated medication record", "effect_scope": [block["analyte"]]})
