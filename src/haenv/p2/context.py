"""Dated clinical episodes, before gold. All clinical constants are sourced registry data.

Episode titles, authors and dates use the same samplers for acting and inert cases.
No episode role is placed on the solver view. The clinical record carries the
prescription dates, actual administration or the assessed baseline instead.
"""
from __future__ import annotations

from haenv.p2 import rules as R


def decorate(block, cid, plan, rng):
    spec = R.table()["tiers"][plan["tier"]]
    family = spec.get("context")
    if not family:
        return block
    from haenv.world.history.history_dependency import uniform
    from haenv.world.history.history_dependency import table
    from haenv.world.history.history_dependency import old_day
    cfg = R.table()["clinical_context"]
    T = block["today_day"]
    lo, hi = cfg["note_age_days"]
    course_day = T - lo - int(uniform("p2-context-date", cid) * (hi - lo + 1))
    day = course_day if family == "chemo" else old_day("p2", cid, T)
    sources = table()["phrasing"]["encounter_sources"]
    source = sources[int(uniform("history-source", cid) * len(sources))]
    titles = cfg["procedure_notes"][family]
    row = dict(day=day, source=source, test=titles[int(uniform("p2-context-title", cid) * len(titles))])
    acts = spec["acts"]
    check, screen = block["home_check_today"], block["intake_checklist"]
    if family == "head":
        # A low-energy head contact has no other head-injury risk factor.
        screen.update(head_injury_today=True, mild_new_symptom=True,
                      current_prescriptions_reconciled=True, following_current_prescriptions=True,
                      head_injury_screen=dict(loss_of_consciousness=False, amnesia=False,
                                              vomiting=False, persistent_headache=False))
        row.update(medication_name=rng.pick(cfg["anticoagulants"]), medication_class="anticoagulant",
                   treatment_start_day=day - 180, treatment_end_day=None if acts else day,
                   result="处方延续，按原方案服用。" if acts else "既往处方已结束，后续不再服用。")
        block["message"] += rng.pick(cfg["head_statement"])
    elif family == "chemo":
        screen.update(systemically_unwell_today=True, mild_new_symptom=True)
        check["temperature_c"] = rng.unif(*cfg["chemo_temp_c"])
        row.update(condition="cancer", conclusion="已记录", treatment_class="systemic_anticancer",
                   treatment_given_day=day if acts else T - rng.unif(*cfg["chemo_recovered_age_days"], dp=0),
                   treatment_administered=True, course_ongoing=acts,
                   result="本周期系统治疗已给药，仍在治疗期。" if acts else "系统治疗疗程已结束，血象复查恢复；目前未再接受治疗。")
        block["message"] += rng.pick(cfg["unwell_statement"])
    elif family == "oxygen":
        row.update(condition="hypercapnic_respiratory_failure", conclusion="已记录" if acts else "未记录",
                   blood_gas_confirmed=acts, clinical_scale=2 if acts else 1,
                   target_spo2=[88, 92] if acts else None,
                   result="血气确认高碳酸呼衰，临床人员登记血氧目标；平时指夹读数在目标范围。" if acts else
                          "慢阻肺随访，血气未证实高碳酸呼衰，无量表二授权。")
        check["spo2_pct"] = rng.unif(*cfg["oxygen_pct"], dp=0)
    elif family == "rate":
        screen["rate_medication_changed_recently"] = False
        row.update(condition="sinus_bradycardia", conclusion="已记录", baseline_assessed=True,
                   baseline_rate_bpm=cfg["assessed_rate_baseline_bpm"] if acts else cfg["recovered_rate_baseline_bpm"],
                   baseline_asymptomatic=True, rate_medication_unchanged=True,
                   result="心电图为窦性慢心率，临床评估长期稳定，无相关症状，原随访计划延续。" if acts else
                          "既往用药期间窦性慢心率，停用减慢心率的药后已恢复，随后多次心率在复核范围内。")
        check["heart_rate_bpm"] = rng.unif(*cfg["sinus_rate_bpm"], dp=0)
    elif family == "rhythm":
        row.update(condition="atrial_fibrillation", conclusion="已记录" if acts else "未记录",
                   result="心电图证实房颤，心室率已控制，持续随访；症状和活动能力无变化。" if acts else
                          "心电图为窦性心律伴房性早搏，未记录房颤。")
        for r in block["wearable_daily"][-2:]:
            r["irregular_rhythm_alerts"] = 1
    elif family == "volume":
        row.update(condition="heart_failure", conclusion="已记录",
                   result="超声心动图结合专科评估，记录心力衰竭，持续随访。")
        course = dict(day=course_day, source=source, test="用药核对", discharge_day=course_day,
                   medication_name="呋塞米", previous_dose_mg=cfg["hf_diuretic_before_mg"],
                   current_dose_mg=cfg["hf_diuretic_current_mg"], dose_change_day=course_day,
                   result="心衰出院后容量复评，原利尿剂按医嘱减量，继续每日晨重记录。")
    else:
        raise ValueError(f"unknown P2 clinical context {family}")
    block["clinical_history_record"] = [row]
    if family == "volume":
        block["clinical_history_record"].append(course)
    return block


def facts(block):
    """Only patient-visible dates, medicines and assessments may trigger an exception."""
    T = int(block["today_day"])
    rows = block.get("clinical_history_record") or ()
    current_anticoag = any(r.get("medication_class") == "anticoagulant"
                          and r.get("treatment_start_day", T + 1) <= T
                          and (r.get("treatment_end_day") is None or T <= r["treatment_end_day"])
                          for r in rows)
    window = R.table()["thresholds"]["anticancer_recent_days"]
    recent_chemo = any(r.get("treatment_class") == "systemic_anticancer"
                       and r.get("treatment_administered") is True
                       and T - window <= r.get("treatment_given_day", T + 1) <= T for r in rows)
    hr = block["home_check_today"]["heart_rate_bpm"]
    baseline_rate = any(r.get("condition") == "sinus_bradycardia" and r.get("baseline_assessed") is True
                        and r.get("baseline_asymptomatic") is True and r.get("rate_medication_unchanged") is True
                        and isinstance(r.get("baseline_rate_bpm"), list)
                        and r["baseline_rate_bpm"][0] <= hr <= r["baseline_rate_bpm"][1] for r in rows)
    ck = R.checklist_of(block)
    return dict(head_anticoag=bool(ck.get("head_injury_today") and current_anticoag
                                 and ck.get("current_prescriptions_reconciled") is True
                                 and ck.get("following_current_prescriptions") is True),
                chemo_unwell=bool(ck.get("systemically_unwell_today") and ck[R.MILD] and recent_chemo),
                stable_rate=baseline_rate and ck.get("rate_medication_changed_recently") is False
                and not (ck.get(R.DIZZY) or any(ck[k] for k in R.R0_KEYS)))


def finalize(raw, cid, plan):
    """Keep the episodes that produced gold and annotate actual deletion effects."""
    from haenv.p2.history import apply_history_positions
    from haenv.p2.history import align_weight_ledger_source
    from haenv.world.validation.difficulty_r2 import config
    from haenv.world.validation.difficulty_r2 import register
    from haenv.world.history.history_dependency import table
    b, g = raw.prediction_context["triage_p2"], raw.adjudication["triage_p2"]
    align_weight_ledger_source(raw)
    apply_history_positions(raw, cid, config("p2")["history_positions"])
    b["triage_protocol"] = table()["packs"]["p2"]["protocol"] + config("p2")["protocol"]
    a = R.assess(b)
    if a["rule"] != g["rule"] or R.floor_of(a["set"]) != g["floor"]:
        raise ValueError("clinical history placement changed planned P2 decision")
    g.update(rule=a["rule"], set=a["set"], readings=a["readings"])
    cut = {**b, "clinical_history_record": []}
    active = R.floor_of(R.assess(cut)["set"]) != g["floor"]
    register(g, "clinical_history_record", b["clinical_history_record"][0]["day"], active,
             recent_lower=int(b["today_day"]) - 10)


def install_invariants():
    """Replace only P2's entries in the existing gate inventory, preserving other packs."""
    import haenv.world.validation.invariants as I
    from dataclasses import replace
    from collections import Counter
    if getattr(I.history_timing, "_p2_context", False):
        return
    old_floors, old_r1, old_timing = I._p2_rule_floors, I.r1, I.history_timing

    def floors(n_items=100):
        from haenv.resources.regpath import load_registry
        if "tiers" not in load_registry("pack2_triage_scenarios.yaml"):
            return old_floors(n_items)  # supported rule-only legacy registry schema
        from haenv.p2.audit import planned_cells
        counts = Counter()
        for tier, count in planned_cells(n_items).items():
            counts[R.table()["tiers"][tier]["rule"]] += count
        return {rid: (n if n_items <= 100 else 1) for rid, n in counts.items() if n > 0}

    def coverage(items):
        if not items or items[0]["pack"] != "p2":
            return old_r1(items)
        # Saved legacy rows with no planned tier retain their original contract.
        minimum = floors(len(items)) if any(I.gold(it).get("tier") for it in items) else old_floors(len(items))
        counts = Counter(I.gold(it).get("rule") for it in items)
        return [f"{rid}: {counts[rid]} < {n}" for rid, n in minimum.items() if counts[rid] < n]

    def timing(it):
        spec = R.table()["clinical_context"]["recent_history_rules"].get(I.gold(it).get("rule")) if it["pack"] == "p2" else None
        if not spec:
            return old_timing(it)
        from haenv.world.history.history_dependency import metadata
        m, b, T = metadata(it["case"], "p2"), I.block(it), I.T_of(it)
        if not m.get("active"):
            return []
        bad = []
        for ref in m.get("records") or ():
            if int(ref["day"]) > int(m["recent_lower"]) - spec["min_separation_days"]:
                bad.append("decisive treatment record is inside the recent window")
            if int(ref["day"]) < T - spec["max_age_days"]:
                bad.append("decisive treatment record is outside the registered current-course window")
            if not any(x.get("day") == ref["day"] for x in b.get(ref["section"]) or ()):
                bad.append("decisive dated source missing from solver records")
        return bad or ([] if m.get("records") else ["no decisive history reference"])

    timing._p2_context = True
    I._p2_rule_floors, I.r1, I.history_timing = floors, coverage, timing
    for g in list(I.GATES):
        if g.id not in ("R1", "history_timing") or "p2" not in g.packs:
            continue
        I.GATES.remove(g)
        I.GATES.append(replace(g, packs=tuple(p for p in g.packs if p != "p2")))
        I.GATES.append(replace(g, packs=("p2",), check=coverage if g.id == "R1" else timing,
                              statement="P2 positive published rule quotas and pool rule coverage." if g.id == "R1" else
                                        "P2 archival separation or registered recent-treatment window, with retrievable decisive sources."))

