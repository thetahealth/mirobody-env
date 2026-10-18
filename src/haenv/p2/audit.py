"""Rebuilt pack 2's declaration for the shared batch audit (`haenv/world/validation/shared_audit.py`, `tools/validation/pack_audit.py`).

Composition (SA-3): the published N follow the rule quotas (`planned_cells`), per tier the
first emitted items by the seeded sha256("p2pick", id). Class for SA-4/5/6 = the floor (least
urgent acceptable disposition). Content-free variables (SA-5/SA-6): age, sex, index time, ledger size,
ledger characters, wording, beta-blocker. Pack gates (design §6, zero calls):

  P-S6  rule reader: the printed rules on the shown block reproduce every gold set (gate: 100%)
  P-S5  last-day reader: the same rules on the block cut to today (report; design target: cost_cc
        at least 0.3 below the rule reader)
  P-PM  pseudo-models (no calls): always-ed_now / -within_24h / -routine / -home, a vitals-threshold
        policy, the last-day reader and the rule reader, by cost_cc / cost_skill / under / over
        (gate: rule reader 1.00, always-home < 0, vitals-threshold below the rule reader)
  P-ST  modality stubs: vitals-only, symptom-only (checklist + message), demographics-only, trend-only,
        surface-only (filled fields, drug names, diagnoses, checklist items no rule reads alone)
        GBMs (GroupKFold by condition spec), floor-class cc and cost_cc (gate: demographics-only cc
        below the 0.20 shortcut line; the others are reported)
  P-NE  answer neutrality: wording and beta-blocker crossed with the floor class (report), the message
        mentions no checklist symptom the checklist denies (gate, also SA-1)
  P-IN  insufficient tier: every item carries what its rule reads (gate: 0 items short)
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

import collections

import haenv.world.validation.pack_size as PS
from haenv.p2 import rules as R
from haenv.p2 import history as _history  # register the same P2 source vocabulary as generation
from haenv.p2.score import DISPOSITIONS
from haenv.p2.score import cc_cost
from haenv.p2.score import composite
from haenv.p2.score import atoms

SURFACE = ("age", "sex", "T", "n_ledger", "ledger_chars", "wording", "bb", "cohort")
UNIVARIATE = SURFACE + ("n_known_world",)
WORDING = {"calm": 0, "neutral": 1, "alarming": 2}


def h01(tag: str, cid: str) -> float:
    from haenv.p2.render import u01
    return u01(tag, cid)


def _blk(it) -> dict:
    return ((it.get("sp") or it["case"]).get("prediction_context") or {}).get("triage_p2") or {}


def tier_of(it) -> str:
    return it["gold"].get("tier")


def _floor_tiers(n_items: int) -> dict:
    """{floor: {tier: n}}: sourced RPM disposition shares, each floor over its rules by
    the table's `rule_weights`, a rule's count in equal shares over the tiers built to fire it."""
    tab = R.table()
    weights = tab["rule_weights"]
    out = {}
    for floor, k in PS.apportion(n_items, tab["disposition_prior"]["weights"]).items():
        out[floor] = {}
        for rule, m in PS.apportion(k, weights[floor]).items():
            ts = [t for t, v in tab["tiers"].items() if v["rule"] == rule and v["floor"] == floor]
            out[floor].update(PS.apportion(m, dict.fromkeys(ts, 1)))
    return out


def planned_cells(n_items: int) -> dict:
    """{tier: n} of a published pack of n items (`_floor_tiers`)."""
    return {t: k for tiers in _floor_tiers(n_items).values() for t, k in tiers.items()}


def quota_cells(n_items: int) -> dict:
    """Explicit tier x cohort allocation: each floor's atypical-eligible share (`cohort_weights`) over
    the floor's tiers in proportion to their counts, so the rule quotas and the cohort quota hold together."""
    tab = R.table()
    out = {}
    for floor, tiers in _floor_tiers(n_items).items():
        cohort = PS.apportion(sum(tiers.values()), tab["cohort_weights"])
        reserve = PS.apportion(cohort["atypical_eligible"], tiers)
        for t, k in tiers.items():
            out[t, "atypical_eligible"] = reserve[t]
            out[t, "ordinary"] = k - reserve[t]
    return out


def assemble(items: list[dict], n_items: int, seed: str) -> list[dict]:
    by = collections.defaultdict(list)
    for it in items:
        by[tier_of(it), it["gold"].get("cohort", "ordinary")].append(it)
    tag = PS.seeded("p2pick", seed)
    out = []
    from haenv.world.validation.shared_audit import take_patient
    counts = collections.Counter()
    for size in PS.nest_chain(n_items):
        need = quota_cells(size)
        PS.short(need, {k: len(v) for k, v in by.items()}, "p2 emitted items")
        used = {it["case_id"] for it in out}
        have = collections.Counter((tier_of(it), it["gold"].get("cohort", "ordinary")) for it in out)
        for t, n in sorted(need.items()):
            left = [it for it in by[t] if it["case_id"] not in used]
            for _ in range(n - have[t]):
                out.append(take_patient(left, counts, lambda it: h01(tag, it["case_id"])))
    return out


def rows(items: list[dict], ctx=None) -> list[dict]:
    out = []
    for it in items:
        sp = it.get("sp") or it["case"]
        up = sp.get("user_profile") or {}
        led = sp.get("evidence_ledger") or ()
        from haenv.p2.render import needs
        g = it["gold"]
        pl = g.get("plan") or {}
        added = set(needs(g["tier"], g["stratum"], bool(g.get("beta_blocker")), pl.get("comorbid") or (),
                          pl.get("dm_agent"))["add_conditions"])
        known = [k for k in up.get("known_conditions") or () if k not in added]
        out.append({"case_id": it["case_id"], "class": it["class"], "spec": it["spec"],
                    "age": int(str(up.get("age_range") or "0").split("-")[0].rstrip("+") or 0),
                    "sex": int(up.get("sex") == "F"), "T": int(_blk(it).get("today_day") or 0),
                    "n_ledger": len(led), "ledger_chars": sum(len(str(e.get("symptom") or e.get("note") or "")) for e in led),
                    "wording": WORDING.get(it["gold"].get("wording"), 1), "bb": int(bool(it["gold"].get("beta_blocker"))),
                    "cohort": int(it["gold"].get("cohort") == "atypical_eligible"),
                    "n_known_world": len(known)})
    return out


# ------------------------------------------------------------------------------ readers and stubs

def rule_reader(blk: dict) -> str:
    """The pseudo-model that applies the printed rules and answers the most urgent acceptable member
    (any member scores 0)."""
    return R.assess(blk)["set"][0]


def last_day_reader(blk: dict) -> str:
    return R.assess(R.last_day_only(blk))["set"][0]


def offline_oracle(block: dict) -> dict:
    """Independent interpreter of the printed policy (never calls assess/readings/news2).

    It uses the same sourced registry, but computes the bands, windows and first-match
    ordering independently of generation's reader. This checks implementation agreement,
    not clinical validation of the authored home-disposition policy.
    """
    tab = R.table()
    th, T = tab["thresholds"], int(block["today_day"])
    screen = block["intake_checklist"]
    urgent = screen.get("urgent_symptoms")
    if urgent is None:
        urgent = [k for k in R.R0_KEYS if screen.get(k)]
    home = block["home_check_today"]
    prior_rows = block.get("clinical_history_record") or ()
    scale2 = any(r.get("blood_gas_confirmed") is True and r.get("clinical_scale") == 2
                 and r.get("target_spo2") == [88, 92] for r in prior_rows)
    parts = []
    named = {}
    for key, name in (("rr", "respiratory_rate_per_min"), ("spo2", "spo2_pct"),
                      ("sbp", "systolic_bp_mmHg"), ("hr", "heart_rate_bpm"), ("temp", "temperature_c")):
        spans = tab["news2"][key]
        if key == "spo2" and scale2:
            from haenv.world.validation.difficulty_r2 import config
            spans = config("p2")["scale2"]["on_oxygen" if home.get("on_oxygen") else "room_air"]
        bands = [points for lo, hi, points in spans if lo <= home[name] <= hi]
        if len(bands) != 1:
            raise ValueError(f"oracle: ambiguous/missing band for {name}")
        parts.extend(bands)
        named[key] = bands[0]
    parts.append(tab["news2"]["new_confusion_points"] if "new_confusion" in urgent else 0)
    bp = [x for x in block["home_bp_log"] if x["day"] == T]
    severe = sum(x["systolic_mmHg"] >= th["htn_sys_ge"] or x["diastolic_mmHg"] >= th["htn_dia_ge"] for x in bp)
    raised = sum(x["systolic_mmHg"] >= 140 or x["diastolic_mmHg"] >= 90 for x in bp)
    weights = {x["day"]: x["kg"] for x in block["morning_weight"]}
    gain = weights[T] - weights[T - th["hf_gain_days"]] if T - th["hf_gain_days"] in weights and T in weights else None
    nights = [x for x in block["cgm_nightly"] if T - 7 <= x["night_of_day"] < T]
    l2 = sum(x["min_mmol_L"] < th["hypo_level2_mmol"] for x in nights)
    l1 = sum(x["min_mmol_L"] < th["hypo_level1_mmol"] for x in nights)
    alerts = sum(x["irregular_rhythm_alerts"] for x in block["wearable_daily"] if T - th["af_window_days"] < x["day"] <= T)
    slow = named["hr"] == 3 and home["heart_rate_bpm"] <= 40
    vital3 = (named["rr"] == 3 or (named["hr"] == 3 and not slow)
              or (named["sbp"] == 3 and home["systolic_bp_mmHg"] <= 90)
              or home["temperature_c"] < th["hypothermia_temp_lt"])
    prior = {r.get("condition") for r in block.get("clinical_history_record") or () if r.get("conclusion") == "已记录"}
    hf = "heart_failure" in block["background"]["known_conditions"] or "heart_failure" in prior
    headache = screen.get(R.HTN_SYMPTOM)
    if headache is None:
        headache = screen.get(R.LEGACY_HTN_SYMPTOM)
    total = sum(parts) + (2 if home.get("on_oxygen") else 0)
    anticoag = any(x.get("medication_class") == "anticoagulant"
                   and x.get("treatment_start_day", T + 1) <= T
                   and (x.get("treatment_end_day") is None or x["treatment_end_day"] >= T) for x in prior_rows)
    chemo = any(x.get("treatment_class") == "systemic_anticancer" and x.get("treatment_administered") is True
                and T - th["anticancer_recent_days"] <= x.get("treatment_given_day", T + 1) <= T for x in prior_rows)
    baseline = any(x.get("condition") == "sinus_bradycardia" and x.get("baseline_assessed") is True
                   and x.get("baseline_asymptomatic") is True and x.get("rate_medication_unchanged") is True
                   and isinstance(x.get("baseline_rate_bpm"), list)
                   and x["baseline_rate_bpm"][0] <= home["heart_rate_bpm"] <= x["baseline_rate_bpm"][1] for x in prior_rows)
    triggers = [bool(urgent), bool(screen.get("head_injury_today") and anticoag
                                  and screen.get("current_prescriptions_reconciled") is True
                                  and screen.get("following_current_prescriptions") is True),
                bool(screen.get("systemically_unwell_today") and screen.get(R.MILD) and chemo),
                total >= 7, total >= 5, vital3, named["spo2"] == 3, slow, max(parts) >= 3,
                bool(screen.get(R.HYPO_HELP)), l2 >= th["hypo_level2_nights_ge"],
                severe >= th["htn_readings_ge"] and bool(headache),
                severe >= th["htn_readings_ge"] and hf,
                severe >= th["htn_readings_ge"],
                hf and gain is not None and round(gain, 2) > th["hf_gain_kg_gt"],
                alerts >= th["af_alerts_ge"] and "atrial_fibrillation" not in prior and "atrial_fibrillation" not in block["background"]["known_conditions"],
                baseline and total == 1 and named["hr"] == 1 and home["heart_rate_bpm"] <= 50
                and screen.get("rate_medication_changed_recently") is False
                and not screen.get(R.DIZZY) and not urgent,
                sum(parts) >= 1, bool(nights) and l1 >= 1 and l2 == 0,
                raised >= th["htn_readings_ge"], bool(screen.get(R.MILD)), True]
    rule = tab["rules"][triggers.index(True)]
    accepted = list(rule["set"])
    single = tab["acceptance"]["single_level2"]
    least = max(accepted, key=DISPOSITIONS.index)
    if l2 == 1 and least in single["when_floor"]:
        accepted = list(single["set"])
    return {"rule": rule["id"], "set": accepted}


def vitals_threshold(blk: dict) -> str:
    """A plausible no-rules policy on today's home check only."""
    c = blk["home_check_today"]
    bad = sum([c["temperature_c"] >= 38.5, c["heart_rate_bpm"] >= 110, c["spo2_pct"] <= 93,
               c["systolic_bp_mmHg"] >= 180 or c["diastolic_bp_mmHg"] >= 110, c["respiratory_rate_per_min"] >= 22])
    mild = sum([c["temperature_c"] >= 37.8, c["heart_rate_bpm"] >= 100, c["spo2_pct"] <= 95,
                c["systolic_bp_mmHg"] >= 150])
    return "ed_now" if bad >= 2 else "within_24h" if bad == 1 else "routine_followup" if mild else "watchful_waiting"


def _readers() -> dict:
    out = {f"always_{d}": (lambda blk, d=d: d) for d in DISPOSITIONS}
    out.update({"vitals_threshold": vitals_threshold, "last_day_reader": last_day_reader, "rule_reader": rule_reader})
    return out


def pseudo_models(items: list[dict]) -> dict:
    sets = [it["gold"]["set"] for it in items]
    out = {}
    for name, fn in _readers().items():
        pred = [fn(_blk(it)) for it in items]
        c = composite([atoms({"disposition": p}, {"set": s}) for s, p in zip(sets, pred)])
        out[name] = {k: (round(c[k], 4) if isinstance(c[k], float) else c[k])
                     for k in ("cost_cc", "cost_skill", "under_rate", "over_rate", "in_set_rate")}
    return out


def _features(blk: dict, kind: str) -> list[float]:
    c = blk["home_check_today"]
    ck = R.checklist_of(blk)
    if kind == "vitals":
        return [c["temperature_c"], c["heart_rate_bpm"], c["spo2_pct"], c["respiratory_rate_per_min"],
                c["systolic_bp_mmHg"], c["diastolic_bp_mmHg"]]
    if kind == "symptom":
        msg = str(blk.get("message") or "")
        kw = ("胸口", "晕", "气不够", "没劲", "模糊", "低血糖", "嗓子", "咳", "发冷", "心律不齐", "鞋子", "血压", "手表", "害怕")
        return [float(bool(ck.get(k))) for k in R.CHECKLIST] + [float(w in msg) for w in kw]
    if kind == "trend":
        wd = blk.get("wearable_daily") or []
        f = []
        for key in ("resting_hr_bpm", "hrv_ms", "night_spo2_min_pct", "skin_temp_delta_c", "steps", "sleep_h"):
            xs = [float(r[key]) for r in wd if r[key] is not None] or [0.0]
            f += [xs[-1] - xs[0], max(xs) - min(xs)]
        f.append(float(sum(r["irregular_rhythm_alerts"] for r in wd)))
        w = blk.get("morning_weight") or []
        f.append(w[-1]["kg"] - w[-4]["kg"] if len(w) >= 4 else 0.0)
        cg = blk.get("cgm_nightly") or []
        f.append(min([r["min_mmol_L"] for r in cg], default=6.0))
        return f
    if kind == "surface":
        # no value read: which fields are filled, the checklist items no rule reads alone, drug names, diagnoses
        wd = blk.get("wearable_daily") or []
        meds = " ".join(blk["background"].get("medications") or ())
        known = set(blk["background"].get("known_conditions") or ())
        return ([float(any(r.get("workout_max_hr_bpm") is not None for r in wd)), float(any(r.get("fit_note") for r in wd)),
                 float(bool(blk.get("cgm_nightly"))), float(bool(ck.get(R.HTN_SYMPTOM))), float(bool(ck.get(R.DIZZY))),
                 float(bool(ck.get(R.MILD)))]
                + [float(w in meds) for w in SURFACE_DRUGS] + [float(c in known) for c in SURFACE_CONDITIONS])
    raise ValueError(kind)


SURFACE_DRUGS = ("比索洛尔", "美托洛尔", "胰岛素", "格列美脲", "呋塞米", "达格列净", "氨氯地平", "二甲双胍",
                 "阿托伐他汀", "左甲状腺素", "依那普利", "氯沙坦")
SURFACE_CONDITIONS = ("heart_failure", "hypertension", "T2D", "T1D", "dyslipidemia", "hypothyroidism", "CAD", "CKD")


def modality_stubs(items: list[dict], rws: list[dict]) -> dict:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from haenv.world.validation.shared_audit import balanced_acc
    from haenv.world.validation.shared_audit import chance_corrected
    y = np.asarray([it["class"] for it in items])
    g = np.asarray([it["spec"] for it in items])
    sets = [it["gold"]["set"] for it in items]
    out = {}
    for kind in ("vitals", "symptom", "trend", "surface", "demographics"):
        if kind == "demographics":
            X = np.asarray([[r["age"], r["sex"], r["T"], r["n_known_world"]] for r in rws], float)
        else:
            X = np.asarray([_features(_blk(it), kind) for it in items], float)
        pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pred[te] = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        cc = chance_corrected(balanced_acc(y.tolist(), pred.tolist()), len(set(y.tolist())))
        out[kind] = {"class_cc": round(cc, 4), "cost_cc": round(cc_cost(sets, pred.tolist()), 4)}
    return out


# ------------------------------------------------------------------------------ pack gates

def p_s6(items, rws, ctx=None) -> dict:
    bad = [it["case_id"] for it in items if offline_oracle(_blk(it)) != {k: it["gold"][k] for k in ("rule", "set")}]
    single = [it for it in items if len(it["gold"]["set"]) == 1]
    cc = cc_cost([it["gold"]["set"] for it in items], [rule_reader(_blk(it)) for it in items])
    return {"id": "P-S6", "n": len(items), "n_single_gold": len(single), "mismatch": bad[:10],
            "rule_reader_cost_cc": round(cc, 4), "independent_oracle": True,
            "pass": not bad and abs(cc - 1.0) < 1e-9}


def p_s5(items, rws, ctx=None) -> dict:
    sets = [it["gold"]["set"] for it in items]
    full = cc_cost(sets, [rule_reader(_blk(it)) for it in items])
    last = cc_cost(sets, [last_day_reader(_blk(it)) for it in items])
    wrong = sum(1 for it in items if last_day_reader(_blk(it)) not in it["gold"]["set"])
    return {"id": "P-S5", "report_only": True, "rule_reader_cost_cc": round(full, 4), "last_day_cost_cc": round(last, 4),
            "gap": round(full - last, 4), "target_gap": 0.3, "meets_target": bool(full - last >= 0.3),
            "last_day_wrong_items": wrong, "pass": True}


def p_pm(items, rws, ctx=None) -> dict:
    pm = pseudo_models(items)
    ok = (abs(pm["rule_reader"]["cost_cc"] - 1.0) < 1e-9 and pm["always_watchful_waiting"]["cost_cc"] < 0
          and pm["vitals_threshold"]["cost_cc"] < pm["rule_reader"]["cost_cc"])
    return {"id": "P-PM", "policies": pm, "pass": bool(ok)}


def p_st(items, rws, ctx=None) -> dict:
    from haenv.world.validation.shared_audit import SHORTCUT_CC
    st = modality_stubs(items, rws)
    return {"id": "P-ST", "stubs": st, "line": SHORTCUT_CC, "pass": st["demographics"]["class_cc"] < SHORTCUT_CC}


def p_ne(items, rws, ctx=None) -> dict:
    from haenv.p2.gate import leak_hits
    wx = collections.Counter((it["gold"]["wording"], it["class"]) for it in items)
    bx = collections.Counter((int(bool(it["gold"]["beta_blocker"])), it["class"]) for it in items)
    msg = [it["case_id"] for it in items if any("checklist says no" in h for h in leak_hits(_blk(it)))]
    return {"id": "P-NE", "wording_x_floor": {f"{a}|{b}": n for (a, b), n in sorted(wx.items())},
            "beta_blocker_x_floor": {f"{a}|{b}": n for (a, b), n in sorted(bx.items())},
            "message_checklist_conflicts": msg, "pass": not msg}


def p_in(items, rws, ctx=None) -> dict:
    short = []
    for it in items:
        b, t = _blk(it), it["gold"]["tier"]
        sc = R.table()["tiers"][t]["scenario"]
        need = {"hypoglycaemia": bool(b.get("cgm_nightly")), "heart_failure": len(b.get("morning_weight") or ()) >= 4,
                "hypertension": len([r for r in b.get("home_bp_log") or () if r["day"] == b["today_day"]]) >= 2,
                "af": len(b.get("wearable_daily") or ()) >= 3}.get(sc, True)
        if not need or not b.get("home_check_today"):
            short.append(it["case_id"])
    return {"id": "P-IN", "short": short, "pass": not short}


def _strata_report(items) -> dict:
    n = len(items)
    st = collections.Counter(it["gold"]["stratum"] for it in items)
    border = sum(1 for it in items if len(it["gold"]["set"]) > 1)
    contra = sum(1 for it in items if (it["gold"]["wording"] == "calm" and it["class"] in ("ed_now", "within_24h"))
                 or (it["gold"]["wording"] == "alarming" and it["class"] in ("routine_followup", "watchful_waiting")))
    lastday_ok = sum(1 for it in items if last_day_reader(_blk(it)) in it["gold"]["set"])
    return {"n": n, "strata": dict(st), "borderline_share": round(border / n, 3) if n else None,
            "contradiction_share": round(contra / n, 3) if n else None,
            "today_decidable_share": round(lastday_ok / n, 3) if n else None,
            "floor": dict(collections.Counter(it["class"] for it in items)),
            "rules": dict(collections.Counter(it["gold"]["rule"] for it in items))}


def quota_report(items, n_items: int, seed: str) -> dict:
    """Machine-authored report of selection quotas and evidence validity (not model ability)."""
    from haenv.p2.gate import measurement_conflicts
    from haenv.p2.gate import world_in_window
    from haenv.p2.hooks import prompt_payload
    from types import SimpleNamespace
    chosen = assemble(items, n_items, seed)
    expected = quota_cells(n_items)
    observed = collections.Counter((it["gold"]["tier"], it["gold"].get("cohort", "ordinary")) for it in chosen)
    mismatch = {f"{t}|{c}": [observed[t, c], n] for (t, c), n in expected.items() if observed[t, c] != n}
    invalid, measurements, ids = [], [], []
    by_floor = collections.Counter()
    history, dependent, false_trend = [], [], []
    for it in chosen:
        g, b, raw = it["gold"], _blk(it), it["case"]
        up = raw["user_profile"]
        age = int(str(up["age_range"]).split("-")[0].rstrip("+"))
        eligible = age >= 65 or bool(set(b["background"]["known_conditions"]) & {"T2D", "T1D", "type2_diabetes"})
        if g.get("cohort") == "atypical_eligible":
            by_floor[it["class"]] += 1
            if not eligible:
                invalid.append(it["case_id"])
        problems = measurement_conflicts(b) + [d for _, d in world_in_window(g, SimpleNamespace(**raw))]
        if problems:
            measurements.append({"case_id": it["case_id"], "conflicts": problems})
        sp = it.get("sp")
        try:
            if not sp:
                raise ValueError("missing saved solver payload")
            prompt_payload(SimpleNamespace(**sp))
            if it["case_id"] in str(sp):
                raise ValueError("real id in saved solver payload")
        except ValueError as exc:
            ids.append({"case_id": it["case_id"], "error": str(exc)})
        needs_history = g["rule"] in R.table()["history_rules"]
        last_correct = last_day_reader(b) in g["set"]
        if needs_history:
            history.append(it["case_id"])
        if not last_correct:
            dependent.append(it["case_id"])
        if g["stratum"] == "trend" and last_correct:
            false_trend.append(it["case_id"])
    report = {"id": "P-QUOTA", "n": len(chosen), "requested_n": n_items,
              "expected_cells": {f"{t}|{c}": n for (t, c), n in expected.items()},
              "observed_cells": {f"{t}|{c}": observed[t, c] for t, c in expected},
              "mismatch": mismatch, "atypical_cohort_x_floor": dict(by_floor), "ineligible_cohort": invalid,
              "measurement_conflicts": measurements, "opaque_id_failures": ids,
              "history_rule_n": len(history), "actual_history_dependency_n": len(dependent),
              "today_sufficient_n": len(chosen) - len(dependent), "false_trend_label_ids": false_trend,
              "history_rules": R.table()["history_rules"], "mix": _strata_report(chosen),
              "offline_policies": pseudo_models(chosen)}
    report["pass"] = (len(chosen) == n_items and not any((mismatch, invalid, measurements, ids, false_trend)))
    return report


def world_stream_in_window(it) -> list[str]:
    """A world stream in the solver view with a point in the block's window that differs from the
    block's reading of that day (the weight stream is kept by the ledger and mirrors the block)."""
    from haenv.p2.gate import world_in_window
    from types import SimpleNamespace
    sp = it.get("sp") or it["case"]
    return [detail for _, detail in world_in_window(it["gold"], SimpleNamespace(**sp))]


def audit(n_items: int = 50, seed: str = ""):
    from haenv.world.validation.shared_audit import PackAudit
    from haenv.world.validation.shared_audit import balanced_acc
    from haenv.world.validation.shared_audit import chance_corrected

    def cc(gold, pred):
        return chance_corrected(balanced_acc(gold, pred), len(set(gold)))

    def p_mix(items, rws, ctx=None):
        return {"id": "P-MIX", "report_only": True, "pool": _strata_report(items),
                "pack": _strata_report(assemble(items, n_items, seed)), "pass": True}

    return PackAudit(
        name="p2", block="triage_p2", classes=DISPOSITIONS,
        class_of=lambda g: g.get("floor") if g.get("set") else None,
        rows=rows, univariate=UNIVARIATE, surface=SURFACE, cc=cc,
        cell_of=lambda it, ctx: (tier_of(it), it["gold"].get("cohort", "ordinary")),
        planned_cells=lambda items, ctx: quota_cells(n_items),
        select=lambda items: assemble(items, n_items, seed),
        realism={"world_stream_in_window": world_stream_in_window},
        pack_gates={"P-S6": p_s6, "P-S5": p_s5, "P-PM": p_pm, "P-ST": p_st, "P-NE": p_ne, "P-IN": p_in, "P-MIX": p_mix,
                    "P-QUOTA": lambda items, rws, ctx=None: quota_report(items, n_items, seed)})


if __name__ == "__main__":
    import argparse
    import json
    import pathlib
    from haenv.world.validation.shared_audit import load_items
    ap = argparse.ArgumentParser(description="Offline p2 quota, oracle and evidence report; zero calls")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = load_items(pathlib.Path(a.batch), "triage_p2", lambda g: g.get("floor"))
    report = quota_report(its, a.n, PS.cli_seed(None))
    report["oracle"] = p_s6(its, [])
    pathlib.Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] and report["oracle"]["pass"] else 1)
