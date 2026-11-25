"""P2 visible-block interventions, bounded by the production rule tables."""
from copy import deepcopy
from functools import lru_cache, partial

import haenv.p2.render  # The production renderer installs clinical rule overlays.
from haenv.p2 import clinical_rules, rules as R
from haenv.p2.history import scale2
from haenv.world.protocols.scoring import FIELDS, attach

from .batch import InvalidTwin, Operator, load_batch
from .trace import RuleTrace

_NEWS = ("news2_ge7", "news2_ge5", "news2_ge1", "news2_eq1", "news2_single3",
        "single3_vital", "single3_spo2", "single3_low_hr")
_ORDER = ("rr", "spo2", "sbp", "hr", "temp")


@lru_cache(maxsize=1)
def trace():
    base = RuleTrace(clinical_rules._base_assess, label="p2.base_assess")
    clinical = RuleTrace(clinical_rules._assess, label="p2.clinical_assess",
                         replace_globals={"_base_assess": base.instrumented})
    return RuleTrace(clinical_rules.assess, label="p2.assess", replace_globals={"_assess": clinical.instrumented})


def _observed(b, target):
    from .registry import atom_map, premise_id
    _, events = trace().run(b)
    mapping = atom_map("p2")
    values = {event[3] for event in events if premise_id(mapping[event[1]]) == target}
    return next(iter(values)) if len(values) == 1 else None


def _decisive(b, target):
    from .receipt import roles
    return roles(P2(), b)[target]["role"] in ("acting", "guard")


def _bands(b, parameter):
    if parameter == "spo2" and scale2(b.get("clinical_history_record")):
        from haenv.world.validation.difficulty_r2 import config
        return config("p2")["scale2"]["on_oxygen" if b["home_check_today"].get("on_oxygen") else "room_air"]
    return R.table()["news2"][parameter]


def _typed(value, original, parameter):
    if isinstance(original, int):
        return int(value)
    return round(float(value), 1) if parameter == "temp" else float(value)


def news2_reduce(b, target):
    if not _decisive(b, target):
        return None
    original = _observed(b, target)
    h = deepcopy(b)
    for _ in range(6):
        check = h["home_check_today"]
        scale = 2 if scale2(h.get("clinical_history_record")) else 1
        parts = R.news2_parts(check, R.checklist_of(h), scale)
        suffix = target.split(".")[1]
        parameters = (R.readings(h)["single3_vital"] if suffix == "single3_vital" else
                      ["spo2"] if suffix == "single3_spo2" else ["hr"] if suffix == "single3_low_hr" else
                      [k for k in _ORDER if parts[k] == 3] if suffix == "news2_single3" else _ORDER)
        candidates = []
        for key in _ORDER:
            if key not in parameters or parts[key] == 0:
                continue
            old = check[FIELDS[key]]
            zeros = [max(lo, min(hi, old)) for lo, hi, score in _bands(h, key) if score == 0]
            if zeros:
                value = min(zeros, key=lambda x: (abs(x - old), x))
                candidates.append((key, _typed(value, old, key)))
        if not candidates:
            return None
        key, value = max(candidates, key=lambda pair: parts[pair[0]])
        check[FIELDS[key]] = value
        if _observed(h, target) is not None and _observed(h, target) != original:
            return h
    return None


def _edit(b, name):
    h = deepcopy(b)
    th, T = R.table()["thresholds"], int(b["today_day"])
    ck = h["intake_checklist"]
    if name == "urgent_clear":
        if "urgent_symptoms" in ck:
            ck["urgent_symptoms"] = []
        else:
            for key in R.R0_KEYS:
                ck[key] = False
    elif name in ("hypo_help_clear", "htn_symptom_clear", "mild_clear", "head_injury_clear", "unwell_clear",
                  "infection_clear", "rate_change_recent"):
        key = {"hypo_help_clear": R.HYPO_HELP, "htn_symptom_clear": R.HTN_SYMPTOM,
               "mild_clear": R.MILD, "head_injury_clear": "head_injury_today", "unwell_clear": "systemically_unwell_today",
               "infection_clear": "suspected_infection_today", "rate_change_recent": "rate_medication_changed_recently"}[name]
        if key == R.HTN_SYMPTOM and key not in ck and R.LEGACY_HTN_SYMPTOM in ck:
            key = R.LEGACY_HTN_SYMPTOM
        if key not in ck:
            return None
        ck[key] = name == "rate_change_recent"
    elif name == "new_symptom_clear":
        # The printed AF exemption reads exactly these ticks (`rules.NEW_SYMPTOM_KEYS`).
        if not any(ck.get(key) for key in R.NEW_SYMPTOM_KEYS):
            return None
        for key in R.NEW_SYMPTOM_KEYS:
            if key in ck:
                ck[key] = False
    elif name in ("hf_remove", "known_af_remove"):
        condition = "heart_failure" if name == "hf_remove" else "atrial_fibrillation"
        background = h.get("background") or {}
        if "known_conditions" in background:
            background["known_conditions"] = [c for c in background["known_conditions"] if c != condition]
        if "clinical_history_record" in h:
            h["clinical_history_record"] = [r for r in h["clinical_history_record"] if r.get("condition") != condition]
    elif name in ("cgm_level2_lift", "cgm_level1_lift"):
        threshold = th["hypo_level2_mmol" if name == "cgm_level2_lift" else "hypo_level1_mmol"]
        for row in h.get("cgm_nightly") or ():
            if T - 7 < int(row["night_of_day"]) + 1 <= T and row["min_mmol_L"] < threshold:
                row["min_mmol_L"] = _typed(threshold, row["min_mmol_L"], "cgm")
                if name == "cgm_level1_lift":
                    row["minutes_below_3_9"] = 0
    elif name in ("bp_severe_lower", "bp_raised_lower"):
        thresholds = (th["htn_sys_ge"], th["htn_dia_ge"]) if name == "bp_severe_lower" else (140, 90)
        for row in h.get("home_bp_log") or ():
            if int(row["day"]) == T:
                for key, boundary in zip(("systolic_mmHg", "diastolic_mmHg"), thresholds):
                    if row[key] >= boundary:
                        row[key] = _typed(boundary - 1, row[key], key)
    elif name == "weight_gain_cap":
        weights = {int(row["day"]): row for row in h.get("morning_weight") or ()}
        if T not in weights or T - th["hf_gain_days"] not in weights:
            return None
        if R.readings(b)["weight_gain_3d"] <= th["hf_gain_kg_gt"]:
            return None
        value = round(weights[T - th["hf_gain_days"]]["kg"] + th["hf_gain_kg_gt"], 1)
        weights[T]["kg"] = _typed(value, weights[T]["kg"], "kg")
    elif name == "af_alerts_cap":
        rows = sorted((r for r in h.get("wearable_daily") or () if T - th["af_window_days"] < int(r["day"]) <= T),
                      key=lambda r: int(r["day"]), reverse=True)
        excess = sum(int(r.get("irregular_rhythm_alerts") or 0) for r in rows) - (th["af_alerts_ge"] - 1)
        for row in rows:
            decrement = min(max(0, excess), int(row.get("irregular_rhythm_alerts") or 0))
            if decrement:
                row["irregular_rhythm_alerts"] -= decrement
                excess -= decrement
    elif name == "null_swap_conditions":
        conditions = (h.get("background") or {}).get("known_conditions") or []
        if len(conditions) < 2:
            return None
        other = next((i for i in range(1, len(conditions)) if conditions[i] != conditions[0]), None)
        if other is None:
            return None
        conditions[0], conditions[other] = conditions[other], conditions[0]
    elif name == "vital_in_band":
        check = h["home_check_today"]
        for key in ("temp", "rr", "spo2", "sbp", "hr"):
            field = FIELDS[key]
            old, step = check[field], 0.1 if key == "temp" else 1
            for direction in (1, -1):
                value = _typed(old + direction * step, old, key)
                bands = _bands(h, key)
                try:
                    same = R._points(bands, value) == R._points(bands, old)
                except ValueError:
                    same = False
                if same and value != old:
                    check[field] = value
                    return h
        return None
    elif name.startswith("old_stream_point:"):
        weight = name.endswith("weight")
        rows = [r for r in h.get("morning_weight" if weight else "home_bp_log") or ()
                if int(r["day"]) < T - th["hf_gain_days"]] if weight else [
                    r for r in h.get("home_bp_log") or () if int(r["day"]) != T]
        if not rows:
            return None
        row = min(rows, key=lambda r: int(r["day"]))
        key = "kg" if weight else "systolic_mmHg"
        row[key] = round(row[key] + (0.5 if weight else 5), 1) if weight else row[key] + 5
    else:
        raise ValueError(f"unknown operator: {name}")
    return h if h != b else None


def _infection(b, target):
    return _edit(b, "infection_clear") if _decisive(b, target) else None


def _cgm_target(b):
    return "p2.cgm_level2_single" if R.readings(b)["level2_nights"] == 1 else "p2.cgm_level2_repeat"


class P2:
    pack = "p2"

    def items(self, batch):
        return load_batch(batch)

    def rule(self):
        return trace()

    def block(self, raw):
        return raw if isinstance(raw, dict) else raw.prediction_context["triage_p2"]

    def rule_args(self, item):
        return (self.block(item.raw),)

    def decision(self, result):
        return tuple(result["set"])

    def scored(self, raw_like):
        return sorted(R.assess(self.block(raw_like))["set"])

    def visible_answer(self, sp):
        return tuple(sorted(clinical_rules.assess_visible(sp["prediction_context"]["triage_p2"])["set"]))

    def operators(self):
        targets = dict(urgent_clear="red_flag", hypo_help_clear="hypo_help", cgm_level2_lift=_cgm_target,
            cgm_level1_lift="cgm_level1", bp_severe_lower="bp_severe", bp_raised_lower="bp_raised",
            htn_symptom_clear="htn_symptom", mild_clear="mild", head_injury_clear="head_anticoag",
            unwell_clear="chemo_unwell", rate_change_recent="stable_rate", hf_remove="heart_failure",
            known_af_remove="known_af", weight_gain_cap="weight_gain", af_alerts_cap="af_alerts",
            new_symptom_clear="new_symptom")
        ops = {name: Operator(name, target if callable(target) else "p2." + target, "R", partial(_edit, name=name))
               for name, target in targets.items()}
        for suffix in _NEWS:
            target = "p2." + suffix
            name = "news2_reduce:" + target
            ops[name] = Operator(name, target, "R", partial(news2_reduce, target=target))
        for suffix in ("infection_medium_risk", "relative_hypotension"):
            target = "p2." + suffix
            name = "infection_clear:" + target
            ops[name] = Operator(name, target, "R", partial(_infection, target=target))
        for name in ("null_swap_conditions", "vital_in_band", "old_stream_point:weight", "old_stream_point:bp"):
            ops[name] = Operator(name, None, "R", partial(_edit, name=name))
        return ops

    def twin_payload(self, item, twin):
        sp = deepcopy(item.sp)
        original = self.block(item.raw)
        if twin["home_check_today"] != original["home_check_today"] and "system_news2" in original["home_check_today"]:
            try:
                attach(twin, item.case_id)
            except Exception as exc:
                raise InvalidTwin(str(exc)) from exc
        sp["prediction_context"]["triage_p2"] = deepcopy(twin)
        return sp

    def baseline_payload(self, item):
        return item.sp

    def visible_blocks(self, sp):
        return sp["prediction_context"]["triage_p2"]

    def metadata(self, item, result):
        if isinstance(item.raw, dict):
            return {"rule": result["rule"]}
        gold = item.raw.adjudication["triage_p2"]
        return {"rule": result["rule"], "stored_gold_equal": result["rule"] == gold["rule"]
                and sorted(result["set"]) == sorted(gold["set"]),
                "stored_block_equal": self.block(item.raw) == item.sp["prediction_context"]["triage_p2"]}


Adapter = P2
