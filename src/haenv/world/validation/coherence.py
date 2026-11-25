"""Sourced, report-only coherence invariants over saved solver-visible batches.

No generation, purchases, model calls or batch writes occur here.
SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from haenv.resources.regpath import load_registry, registry_cached, registry_path
from haenv.world.validation import coherence_packs as P


@registry_cached("coherence_invariants.yaml")
def registry():
    doc = load_registry("coherence_invariants.yaml")
    validate_registry(doc)
    return doc


def validate_registry(doc):
    """Reject unsourced numeric decisions, including nested tables and formula constants."""
    sources = doc.get("sources") or {}
    def walk(node, path, source=None):
        if isinstance(node, dict):
            source = node.get("source", source)
            if source is not None and source not in sources:
                raise ValueError(f"unresolved source at {path}: {source}")
            for k, v in node.items():
                if k not in ("source", "ref_low", "ref_high"):
                    walk(v, f"{path}.{k}", source)
        elif isinstance(node, list):
            for n, v in enumerate(node):
                walk(v, f"{path}[{n}]", source)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and source not in sources:
            raise ValueError(f"numeric parameter lacks source at {path}")
    for key, node in doc.items():
        if key not in ("version", "sources", "checks"):
            walk(node, key)
    seen = set()
    for row in doc.get("checks", []):
        if row["id"] in seen or row["class"] != row["id"].split(".")[0]:
            raise ValueError(f"invalid or duplicate check id: {row['id']}")
        seen.add(row["id"])
        if row.get("source") not in sources or row.get("level") not in ("report", "gate"):
            raise ValueError(f"invalid check source/level: {row['id']}")
        for key, param in row.get("params", {}).items():
            if not isinstance(param, dict) or param.get("source") not in sources or "value" not in param:
                raise ValueError(f"parameter lacks source: {row['id']}.{key}")
            walk(param, f"{row['id']}.{key}")
    return doc


@registry_cached("condition_findings.yaml", "rivals.yaml", "drug_indications.yaml", "p4_diversity.yaml")
def _tables():
    return {name: load_registry(name + ".yaml") for name in
            ("condition_findings", "rivals", "drug_indications", "p4_diversity")}


def param(check, key):
    return check["params"][key]["value"]


def _check(check_id):
    return next(row for row in registry()["checks"] if row["id"] == check_id)


def _values(name):
    return set(registry()[name]["values"])


def _limit(fid):
    return registry()["decision_limits"][fid]["diabetes"]["value"]


def match_words(text, lexicon, lang="zh"):
    """Try preferred language first, retaining matches from both lexicons."""
    text = str(text or "").casefold()
    def hit(word, language):
        word = word.casefold()
        if language != "en":
            return word in text
        # English words match at a word start ("hot" must not match "photo"; "start" matches "started").
        return re.search(r"(?<![a-z])" + re.escape(word), text) is not None
    return {language: [w for w in lexicon.get(language, []) if hit(w, language)]
            for language in dict.fromkeys((lang, "zh", "en"))}


def _matched(matches):
    return any(matches.values())


def classify(finding_id, value, ref_low=None, ref_high=None, context=None, flag=None):
    value = P.number(value)
    if value is None:
        return {"positive": "positive", "negative": "negative", "+": "positive", "-": "negative"}.get(str(flag).casefold())
    limit = registry()["decision_limits"].get(finding_id, {}).get(context)
    if limit:
        return "high" if value >= limit["value"] else "normal"
    low, high = P.number(ref_low), P.number(ref_high)
    if low is None or high is None:
        return None
    return "low" if value < low else "high" if value > high else "normal"


def egfr(creatinine_umol, age, sex):
    cfg = registry()["egfr_equation"]
    if sex not in ("M", "F") or P.number(creatinine_umol) is None or float(creatinine_umol) <= 0:
        return None
    female = sex == "F"
    ratio = float(creatinine_umol) / cfg["scr_umol_per_mg_dl"] / cfg["k_female" if female else "k_male"]
    return (cfg["coefficient"] * min(ratio, 1) ** cfg["alpha_female" if female else "alpha_male"]
            * max(ratio, 1) ** cfg["upper_power"] * cfg["age_base"] ** age
            * (cfg["female_factor"] if female else 1))


def egfr_interval(value, age_range, sex, unit=None):
    match = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", str(age_range or ""))
    if not match or sex not in ("M", "F") or P.number(value) is None:
        return None
    low, high = map(int, match.groups())
    if low > high or float(value) <= 0:
        return None
    unit = str(unit or "").casefold().replace("μ", "u").replace("µ", "u").replace(" ", "")
    if unit == "mg/dl":
        value = float(value) * registry()["egfr_equation"]["scr_umol_per_mg_dl"]
    elif unit not in ("", "umol/l"):
        return None
    return {"lo": egfr(value, high, sex), "mid": egfr(value, (low + high) / 2, sex), "hi": egfr(value, low, sex)}


def clopper_pearson(k, n, confidence=None):
    """Exact two-sided interval; scipy is loaded only by the pool-level check."""
    from scipy.special import betaincinv
    confidence = param(_check("K10.penetrance"), "confidence") if confidence is None else confidence
    if n <= 0 or not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval inputs")
    tail = (1 - confidence) / 2
    return [0.0 if k == 0 else float(betaincinv(k, n - k + 1, tail)),
            1.0 if k == n else float(betaincinv(k + 1, n - k, 1 - tail))]


def adag_upper(a1c):
    table = param(_check("K9.adag_fasting"), "table")
    if not table[0][0] <= a1c <= table[-1][0]:
        return None
    for left, right in zip(table, table[1:]):
        if left[0] <= a1c <= right[0]:
            return left[2] + (right[2] - left[2]) * (a1c - left[0]) / (right[0] - left[0])
    return table[-1][2]


@dataclass
class Result:
    applicable: bool = False
    testable: bool = False
    violation: bool = False
    facts: dict = field(default_factory=dict)
    reason: str = ""
    groups: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _result(facts, violation=False, *, testable=True, groups=None, extra=None):
    return Result(True, testable, bool(violation) if testable else False, facts,
                  json.dumps(facts, ensure_ascii=False, separators=(",", ":"))[:260] if violation else "",
                  groups or [], extra or {})


def _units(rows, facts, extra=None):
    """Missing subcriteria cannot silently become a coherent pass."""
    if not rows:
        return Result(facts=facts, extra=extra or {})
    bad = any(row["testable"] and row["violation"] for row in rows)
    tested = bad or all(row["testable"] for row in rows)
    return _result(facts, bad, testable=tested, groups=rows, extra=extra)


@dataclass
class View:
    item: dict
    lang: str = "zh"

    def __post_init__(self):
        self.pack = self.item["pack"]
        self.g = P.gold(self.item)
        self.b = P.block(self.item)
        self.T = P.prediction_time(self.item)
        self.sid = P.spec_or_tier(self.item)
        self.comps = P.components(self.g.get("spec_id")) if self.pack == "m2" else []
        self.known = P.known_conditions(self.item)
        self.meds = P.current_medications(self.item)
        self.drugs = P.drug_ids(self.meds)
        self.profile = self.item["sp"].get("user_profile") or {}
        self.labs = P.m2_view(self.item)
        self.dayT = [r for r in self.labs if r["channel"] == "purchase" and r["ts"] == self.T]
        self.hbT = self.at("HbA1c")
        self.fbgT = self.at("FBG")
        self.dm_readings = [r for r in self.labs if r["finding"] in ("HbA1c", "FBG") and P.number(r["value"]) is not None]
        dm_keys = {(r["finding"], r["ts"]) for r in self.dm_readings if float(r["value"]) >= _limit(r["finding"])}
        self.n_dm = len(dm_keys)
        self.same_dm = any(v >= _limit("HbA1c") for v in self.hbT) and any(v >= _limit("FBG") for v in self.fbgT)
        self.ada = self.same_dm or self.n_dm >= param(_check("K1.diabetes_criterion"), "repeat_readings")
        self.renal = []
        for r in self.labs:
            if r["finding"] == "Cr":
                interval = egfr_interval(r["value"], self.profile.get("age_range"), self.profile.get("sex"), r.get("unit"))
                if interval:
                    self.renal.append({"ts": r["ts"], "creatinine": r["value"], "unit": r.get("unit"), **interval})

    def at(self, fid):
        return [float(r["value"]) for r in self.dayT if r["finding"] == fid and P.number(r["value"]) is not None]

    def stream(self, name):
        return sorted([p for p in (self.item["sp"].get("longitudinal_data") or {}).get(name, [])
                       if self.T is not None and P.number(p.get("ts")) is not None and p["ts"] <= self.T
                       and P.number(p.get("value")) is not None], key=lambda p: p["ts"])

    def diabetes_facts(self):
        return {"hbT": self.hbT, "fbgT": self.fbgT, "n_dm_range_readings": self.n_dm,
                "ada_sameday_T": self.same_dm, "ada_met": self.ada, "readings": self.dm_readings}

    def background(self):
        return {"spec": self.sid, "components": self.comps, "known": self.known,
                "meds": self.meds, "drug_ids": sorted(self.drugs)}

    def has_class(self, name):
        return bool(self.drugs & set(registry()["drug_classes"][name]))

    def symptoms(self, check):
        return [(r, matched) for r in self.item["sp"].get("evidence_ledger") or []
                if r.get("source_type") == "patient_reported_symptom"
                and _matched(matched := match_words(r.get("symptom", ""), check["lexicon"], self.lang))]


def _gold_criteria(v, c):
    cid = c["id"]
    if v.g.get("insufficient"):
        return Result()
    if cid in ("K1.diabetes_criterion", "K1.diabetes_contradiction"):
        if not set(v.comps) & _values("diabetes_conditions"):
            return Result()
        if cid.endswith("diabetes_criterion"):
            return _result(v.diabetes_facts(), not v.ada, testable=bool(v.dm_readings))
        return _result(v.diabetes_facts(), any(x < _limit("HbA1c") for x in v.hbT), testable=bool(v.hbT))
    if cid == "K1.hypoglycaemia_contradiction":
        if not set(v.comps) & _values("hypoglycaemia_conditions"):
            return Result()
        facts = {**v.diabetes_facts(), "diabetes_range_at_T": any(x >= _limit("HbA1c") for x in v.hbT)
                 or any(x >= _limit("FBG") for x in v.fbgT),
                 "whipple_glucose_met": any(r["finding"] == "FBG" and float(r["value"]) < param(c, "whipple") for r in v.dm_readings)}
        return _result(facts, any(x >= param(c, "level1") for x in v.fbgT), testable=bool(v.fbgT))
    if cid == "K1.ckd_criterion":
        if not set(v.comps) & _values("ckd_conditions"):
            return Result()
        uacr = [r for r in v.labs if r["finding"] == "UACR" and P.number(r["value"]) is not None]
        valid_uacr, markers = [], [r["ts"] for r in v.renal if r["hi"] < param(c, "egfr")]
        for r in uacr:
            unit = str(r.get("unit") or "").casefold().replace(" ", "")
            limit = param(c, "uacr_mg_g") if unit == "mg/g" else param(c, "uacr_mg_mmol") if unit == "mg/mmol" else None
            if limit is not None:
                valid_uacr.append(r)
                if float(r["value"]) >= limit:
                    markers.append(r["ts"])
        span = max(markers) - min(markers) if markers else None
        return _result({"egfr": v.renal, "uacr": uacr, "marker_days": sorted(set(markers)), "span_days": span},
                       span is None or span < param(c, "days"), testable=bool(v.renal or valid_uacr))
    if cid == "K1.lab_criterion":
        units, rows = [], []
        for row in registry()["criteria"]:
            for cond in set(v.comps) & set(row["condition_ids"]):
                obs = [r for r in v.dayT if r["finding"] == row["finding"] and P.number(r["value"]) is not None]
                usable = [r for r in obs if P.number(r.get("ref_low" if row["criterion"] == "low" else "ref_high")) is not None]
                second = [r for r in v.dayT if r["finding"] == row.get("second_finding") and P.number(r["value"]) is not None and P.number(r.get("ref_low")) is not None]
                kind = row["criterion"]
                testable = bool(usable) and (kind != "phpt" or bool(second))
                met = any(float(r["value"]) < float(r["ref_low"]) if kind == "low" else
                          float(r["value"]) >= row["multiple"]["value"] * float(r["ref_high"]) if kind == "uln" else
                          float(r["value"]) > float(r["ref_high"]) for r in usable)
                if kind == "phpt":
                    met = met and any(float(r["value"]) >= float(r["ref_low"]) for r in second)
                units.append({"condition": cond, "testable": testable, "violation": not met})
                rows.append({"condition": cond, "criterion": kind, "observations": obs, "second_observations": second})
        return _units(units, {"criteria": rows})
    # A rival is compared in its component's diagnostic context.
    units, facts = [], []
    for comp in v.comps:
        for rival in _tables()["rivals"]["rivals"].get(comp, []):
            d = rival.get("discriminator_finding")
            if not d:
                continue
            obs = [r for r in v.dayT if r["finding"] == d["finding"]]
            classes = [classify(d["finding"], r["value"], r.get("ref_low"), r.get("ref_high"),
                                "diabetes" if comp in _values("diabetes_conditions") else None, r.get("flag")) for r in obs]
            usable = [x for x in classes if x is not None]
            bad = any(x == d["rival"] and x != d["gold"] for x in usable)
            units.append({"condition": comp, "testable": bool(usable), "violation": bad})
            facts.append({"component": comp, **d, "observations": obs, "classes": classes})
    return _units(units, {"discriminators": facts})


def _background(v, c):
    cid = c["id"]
    facts = v.background()
    known = set(v.known or [])
    if cid == "K3.m2_unlabelled_diabetes":
        findings = _tables()["condition_findings"]["condition_findings"]
        registered = {f["id"] for comp in v.comps for f in findings.get(comp, {}).get("findings", [])}
        if set(v.comps) & _values("diabetes_conditions") or registered & {"HbA1c", "FBG"} or "T2D" in known:
            return Result()
        return _result({**facts, **v.diabetes_facts()}, any(x >= _limit("HbA1c") for x in v.hbT)
                       or any(x >= _limit("FBG") for x in v.fbgT), testable=v.known is not None and bool(v.hbT or v.fbgT))
    if cid == "K3.drug_indication":
        indications = defaultdict(list)
        for disease, drugs in _tables()["drug_indications"]["pairs"].items():
            for drug, row in drugs.items():
                if row.get("status") in ("approved", "off_label"):
                    indications[drug].append(disease)
        for row in registry().get("indication_extensions") or []:
            indications[row["drug"]].extend(d for d in row["diseases"] if d not in indications[row["drug"]])
        candidates = sorted(v.drugs & indications.keys())
        missing, rows = [], []
        for drug in candidates:
            diseases = indications[drug]
            met = any((set(registry()["disease_codes"].get(d, [d])) & (known | set(v.comps))) for d in diseases)
            rows.append({"drug_id": drug, "indications": diseases, "indication_present": met})
            if not met:
                missing.append(drug)
        facts.update(drug_indications=rows, missing=missing, metformin_without_T2D="metformin" in v.drugs and "T2D" not in known,
                     unresolved=[s for s in v.meds or [] if P.resolve_drug(s) is None])
        if v.meds is not None and not candidates:
            return Result(facts=facts)
        return _result(facts, bool(missing), testable=v.meds is not None and v.known is not None,
                       extra={"per_drug": dict(Counter(missing)), "metformin_without_T2D": int(facts["metformin_without_T2D"])})
    if cid == "K3.standard_of_care":
        units = []
        rates = _soc_rows(v)
        doc = _soc_table()
        rated = set(doc["conditions"]) if doc else set()
        for row in registry()["standard_of_care"]:
            cond = row["condition"]
            applicable = cond in known or (v.pack == "p2" and cond == "atrial_fibrillation" and v.sid.startswith("af_"))
            if applicable and cond not in rated:
                units.append({"condition": cond, "testable": v.meds is not None, "violation": not v.has_class(row["drug_class"])})
        return _units(units, {**facts, "rate_observations": rates})
    if cid == "K3.renal_unlabelled":
        if set(v.comps) & _values("renal_conditions") or "CKD" in known:
            return Result()
        return _result({**facts, "egfr": v.renal}, any(r["hi"] < param(c, "egfr") for r in v.renal), testable=bool(v.renal) and v.known is not None)
    if cid == "K3.ckd_problem_list":
        if not set(v.comps) & set(registry()["ckd_progression_conditions"]):
            return Result()
        return _result(facts, "CKD" not in known, testable=v.known is not None)
    if cid == "K3.p2_chemo_cancer":
        if not v.sid.startswith("chemo"):
            return Result()
        return _result(facts, not known & set(registry()["cancer_codes"]), testable=v.known is not None)
    readings = v.stream("HbA1c")
    recent = [float(p["value"]) for p in readings if p["ts"] >= v.T - param(c, "median_days")]
    last = readings[-1] if readings else None
    facts.update(last_HbA1c=last, median_60d=statistics.median(recent) if recent else None)
    return _result(facts, bool(last and float(last["value"]) >= _limit("HbA1c") and "T2D" not in known),
                   testable=bool(last) and v.known is not None)


def _soc_table():
    if not registry_path("standard_of_care.yaml").exists():
        return None
    from haenv.world.truth import soc
    return soc.table()


def _soc_rows(v):
    """Sourced class observations; eligibility and exclusions use saved case facts."""
    doc = _soc_table()
    if doc is None:
        return []
    from haenv.world.truth import soc
    from haenv.i18n.text import is_text
    lp = v.item["case"].get("latent_premise") or {}
    context = (lp.get("world_medication") or {}).get("soc_context") or {}
    # Eligibility comes from the visible diagnosed/problem list. The saved SoC
    # context also contains hidden indications that were blocked at generation.
    known = set(v.known or ())
    known.update(r.get("condition") for r in v.b.get("clinical_history_record", ())
                 if is_text(r.get("conclusion"), "hist.recorded"))
    age = context.get("age_years", (lp.get("truth") or {}).get("age_years"))
    if age is None:
        try:
            age = soc.age_years(v.item["case_id"], v.profile.get("age_range"))
        except (TypeError, ValueError):
            age = None
    forbidden = set(context.get("forbid_classes", ()))
    blocked = set(context.get("forbidden_indications", ()))
    decisive = set().union(*(soc.drug_classes(r.get("drug", ""))
                            for r in lp.get("decisive_medications", ())))
    rows = []
    for cond, cfg in doc["conditions"].items():
        if cond not in known:
            continue
        classes = cfg.get("classes", {})
        if cond == "atrial_fibrillation":
            # Low-risk AF has no sourced treatment rate to test.
            classes = {"oac": cfg["oac"]["score_ge_2"]} if age is None or soc.cha2ds2_va(known, age) >= doc["score"]["oac_minimum"] else {}
        for cls, row in classes.items():
            if row.get("rate") is None or not row.get("applied", True):
                continue
            if age is not None and age < row.get("minimum_age", 0):
                continue
            members = set(doc["agents"][cls]["weights"]) | set(doc.get("existing_classes", {}).get(cls, ()))
            exclusion = ("decisive_medication" if cls in decisive else "forbidden_class" if cls in forbidden
                         else "gold_indication" if cond in blocked else None)
            rows.append({"condition": cond, "drug_class": cls, "rate": row["rate"],
                         "source": row.get("source", cfg["source"]), "present": bool(v.drugs & members),
                         "testable": v.meds is not None and (age is not None or cond != "atrial_fibrillation"),
                         "exclusion": exclusion})
    return rows


def standard_of_care(items, lang="zh"):
    """Pool-level exact binomial checks, with all and constraint-eligible shares."""
    groups = defaultdict(list)
    for item in items:
        for row in _soc_rows(View(item, lang)):
            groups[row["condition"], row["drug_class"]].append(row)
    out = []
    confidence = param(_check("K3.standard_of_care"), "confidence")
    for (cond, cls), rows in sorted(groups.items()):
        def measure(observations):
            valid = [r for r in observations if r["testable"]]
            n, k = len(valid), sum(r["present"] for r in valid)
            ci = clopper_pearson(k, n, confidence) if n else None
            rate = rows[0]["rate"]
            return {"n": n, "k": k, "share": k / n if n else None, "ci95": ci,
                    "not_testable": len(observations) - n,
                    "violation": bool(n and (k < n if rate == 1 else not ci[0] <= rate <= ci[1]))}
        eligible = measure([r for r in rows if not r["exclusion"]])
        out.append({"condition": cond, "drug_class": cls, "rate": rows[0]["rate"], "source": rows[0]["source"],
                    **eligible, "all": measure(rows), "marginal": {**measure(rows), "level": "report"},
                    "excluded": dict(Counter(r["exclusion"] for r in rows if r["exclusion"]))})
    return out


def _symptom_stream(v, c):
    cid = c["id"]
    if cid == "K4.fever_temperature":
        symptoms = v.symptoms(c)
        units, facts = [], []
        temp = v.stream("body_temp")
        for symptom, matches in symptoms:
            day = P.number(symptom.get("source_timestamp"))
            same = [p for p in temp if day is not None and p["ts"] == day]
            near = [p for p in temp if day is not None and abs(p["ts"] - day) <= param(c, "context_days")]
            units.append({"condition": v.sid, "testable": bool(same), "violation": bool(same) and all(float(p["value"]) <= param(c, "temperature") for p in same)})
            facts.append({"symptom_day": day, "matches": matches, "same_day": same, "within_context": near})
        return _units(units, {"symptoms": facts})
    if cid == "K4.hf_weight":
        if not set(v.comps) & _values("hf_conditions"):
            return Result()
        symptoms = v.symptoms(c)
        if not symptoms:
            return Result()
        days = [P.number(s.get("source_timestamp")) for s, _ in symptoms]
        days = [d for d in days if d is not None]
        weights = [p for p in v.stream("weight") if days and p["ts"] >= min(days) - param(c, "days")]
        gains = [float(b["value"]) - float(a["value"]) for a in weights for b in weights if 0 < b["ts"] - a["ts"] <= param(c, "days")]
        return _result({"symptom_days": days, "matches": [m for _, m in symptoms], "weights": weights, "gains": gains},
                       not any(x > param(c, "gain_kg") for x in gains), testable=len(weights) >= param(c, "minimum_readings"))
    home = v.b.get("home_check_today") or {}
    spo2 = cid == "K4.p2_spo2_channels"
    key = "spo2_pct" if spo2 else "heart_rate_bpm"
    wk = "night_spo2_min_pct" if spo2 else "resting_hr_bpm"
    today = P.number(home.get(key))
    if today is not None and today > param(c, "home_max"):
        return Result()
    wearable = sorted([p for p in v.b.get("wearable_daily") or [] if P.number(p.get("day")) is not None
                       and v.T is not None and p["day"] <= v.T and P.number(p.get(wk)) is not None], key=lambda p: p["day"])
    last = P.number(wearable[-1][wk]) if wearable else None
    bad = last is not None and (last >= param(c, "wearable_min") if spo2 else param(c, "wearable_low") <= last <= param(c, "wearable_high"))
    return _result({"today": today, "last_wearable": wearable[-1] if wearable else None}, bad, testable=today is not None and last is not None)


def _statement(v, c):
    cid = c["id"]
    if cid == "K5.m2_glycaemic_statement":
        symptoms = v.symptoms(c)
        if not symptoms:
            return Result()
        return _result({**v.background(), "matches": [m for _, m in symptoms]}, "T2D" not in (v.known or []) and not v.has_class("glucose_lowering"),
                       testable=v.known is not None and v.meds is not None)
    intake = v.b.get("intake_checklist") or {}
    messages = [v.b.get("message"), intake.get("patient_statement")]
    exists = any(isinstance(s, str) for s in messages)
    text = " ".join(str(s or "") for s in messages)
    matches = match_words(text, c["lexicon"], v.lang)
    temp = P.number((v.b.get("home_check_today") or {}).get("temperature_c"))
    facts = {"temperature": temp, "tier": v.sid, "matches": matches}
    if cid == "K5.p2_tier_marker":
        family = "infection_bp_*/chemo_*" if v.sid.startswith(("infection_bp_", "chemo_")) else "inf_n*" if v.sid.startswith("inf_n") else "other"
        sentences = [s.strip() for s in re.split(r"[。.!?！？]", text) if _matched(match_words(s, c["lexicon"], v.lang))]
        return _result({**facts, "sentences": sentences, "family": family}, testable=exists,
                       extra={"tier_family": {family: {"items": 1, "chills": int(_matched(matches))}}})
    if cid == "K5.p2_fever_words":
        if (v.g.get("plan") or {}).get("scenario") != "infection":
            return Result()
        if temp is not None and temp <= param(c, "temperature"):
            return Result()
        return _result(facts, not _matched(matches), testable=temp is not None and exists)
    if temp is not None and temp < param(c, "temperature"):
        return Result()
    return _result(facts, _matched(matches), testable=temp is not None and exists)


def _event_matches(event, category, lang):
    return _matched(match_words(event, registry()["medication_events"][category], lang))


def _last_state(record, T, lang):
    state = {}
    for r in sorted(record, key=lambda r: P.number(r.get("day")) if P.number(r.get("day")) is not None else float("-inf")):
        day = P.number(r.get("day"))
        drug = P.resolve_drug(r.get("drug"))
        if drug is None or day is None or T is None or day > T:
            continue
        event = str(r.get("event") or "")
        status = "paused" if _event_matches(event, "pause", lang) else "stopped" if _event_matches(event, "stop", lang) else "on"
        state[drug] = {"status": status, "day": day, "event": event}
    return state


def _exposure_visibility(v, c):
    cid = c["id"]
    if cid == "K6.p3_coadministration":
        if v.g.get("subtype") != "absorption":
            return Result()
        adm = v.b.get("administration_record")
        rows = [adm] if isinstance(adm, dict) else adm or []
        supplements = [P.resolve_drug(r.get("supplement")) for r in rows if r.get("supplement")]
        return _result({"supplements": supplements, "current": v.meds}, any(s not in v.drugs for s in supplements),
                       testable=bool(supplements) and all(supplements) and v.meds is not None)
    record = P.medication_record(v.item)
    if cid == "K6.p2_rule_exposure":
        rows = [r for r in registry()["rule_exposures"] if v.sid in r["rule_or_tier"] or v.g.get("rule") in r["rule_or_tier"]]
        if not rows:
            return Result()
        ac = set(registry()["drug_classes"][rows[0]["drug_class"]])
        history = v.b.get("clinical_history_record") or []
        facts = {"current": sorted(v.drugs & ac), "clinical_history_record": [r for r in history if r.get("medication_class") == rows[0]["drug_class"] or P.resolve_drug(json.dumps(r, ensure_ascii=False)) in ac],
                 "medication_record": [r for r in record if P.resolve_drug(r.get("drug")) in ac],
                 "current_prescriptions_reconciled": (v.b.get("intake_checklist") or {}).get("current_prescriptions_reconciled")}
        recorded = set(_last_state(record, v.T, v.lang))
        return _result(facts, not v.drugs & ac or not v.drugs & ac & recorded, testable=v.meds is not None)
    state = _last_state(record, v.T, v.lang)
    missing = sorted(v.drugs - state.keys())
    stopped = sorted(d for d in v.drugs & state.keys() if state[d]["status"] == "stopped")
    facts = {"current": v.meds, "last_state": state, "cur_not_in_record": missing, "stopped_but_current": stopped,
             "record_on_not_in_cur": sorted(d for d in state.keys() - v.drugs if state[d]["status"] == "on"),
             "unresolved": [s for s in v.meds or [] if P.resolve_drug(s) is None]}
    present = "medication_record" in (v.b.get("background") or {})
    return _result(facts, bool(missing), testable=v.meds is not None and present and not facts["unresolved"], extra={"per_drug": dict(Counter(missing))})


def _decoys(v, c):
    cid = c["id"]
    ex = [r for r in P.exposures(v.item) if r["cause"] != v.g.get("cause")]
    if cid == "K7.retired_decoy":
        retired = _tables()["p4_diversity"]["question_update_14"]["retired_from_allocation"]
        bad = [r for r in ex if r["cause"] in retired]
        units = [{"condition": f"{v.sid}/{r['cause']}", "testable": True, "violation": r["cause"] in retired} for r in ex if r["cause"]]
        return _units(units, {"exposures": ex, "retired": [r["cause"] for r in bad]},
                      extra={"nonretired_exposures": dict(Counter(r["cause"] for r in ex if r["cause"] and r["cause"] not in retired))})
    if cid == "K7.new_label_age":
        ages = defaultdict(list)
        for r in ex:
            if r["cause"] and _matched(match_words(r.get("course_action"), c["lexicon"], v.lang)):
                day, onset = P.number(r.get("day")), P.number(r.get("onset_day"))
                if day is not None and onset is not None:
                    ages[r["cause"]].append(day - onset)
        if not ages:
            return Result()
        return _result({"ages": dict(ages)}, extra={"ages_by_cause": dict(ages)})
    if cid == "K7.thyroiditis_window":
        thyroid = [r for r in ex if r["cause"] == "thyroiditis"]
        if v.sid != "tsh" or v.g.get("class") == "true_change" or not thyroid:
            return Result()
        t0, t1 = P.number(v.g.get("t0")), P.number(v.g.get("t1"))
        rows = []
        for r in thyroid:
            onset = P.number(r.get("onset_day", r.get("day")))
            rows.append({"onset_day": onset, "t0_minus_onset": t0 - onset if t0 is not None and onset is not None else None,
                         "t1_minus_onset": t1 - onset if t1 is not None and onset is not None else None})
        truths = {r.get("day"): P.number(r.get("truth")) for r in v.g.get("records") or []}
        x, y = truths.get(t0), truths.get(t1)
        flat = abs(y - x) / abs(x) < param(c, "flat_fraction") if x not in (None, 0) and y is not None else None
        bad = any(g is not None and 0 <= g <= param(c, "days") for r in rows for g in (r["t0_minus_onset"], r["t1_minus_onset"]))
        return _result({"draws": [t0, t1], "exposures": rows, "truth_flat": flat, "truth_t0": x, "truth_t1": y}, bad,
                       testable=t0 is not None and t1 is not None and all(r["onset_day"] is not None for r in rows))
    if v.sid != "ck" or v.g.get("cause") != "statin_myopathy":
        return Result()
    text = json.dumps(v.b.get("outpatient_record") or [], ensure_ascii=False)
    matches = match_words(text, c["lexicon"], v.lang)
    if not _matched(matches):
        return Result()
    exercise = [r for r in ex if r["cause"] == "recent_exercise"]
    gaps = [P.number(r.get("draw_day", v.g.get("t1"))) - P.number(r.get("day")) for r in exercise
            if P.number(r.get("draw_day", v.g.get("t1"))) is not None and P.number(r.get("day")) is not None]
    low, high = _tables()["p4_diversity"]["rules"]["recent_exercise"]["window_days"]
    return _result({"matches": matches, "gaps": gaps, "exercise_records": exercise}, any(low <= g <= high for g in gaps),
                   testable=not exercise or len(gaps) == len(exercise), extra={"cooccurrence": int(bool(exercise))})


def _concurrent(v, c):
    if v.g.get("class") == "true_change":
        return Result()
    t0, t1 = P.number(v.g.get("t0")), P.number(v.g.get("t1"))
    effects = [r for r in registry()["drug_effects_on_analytes"] if v.sid in r["analytes"]]
    if not effects:
        return Result()
    unmodelled, modelled = [], []
    for r in P.medication_record(v.item):
        day = P.number(r.get("day"))
        drug = P.resolve_drug(r.get("drug"))
        event = str(r.get("event") or "")
        if t0 is None or t1 is None or day is None or not t0 < day <= t1:
            continue
        if _event_matches(event, "unchanged", v.lang) or not _event_matches(event, "change", v.lang):
            continue
        for effect in effects:
            target = effect["drug_class_or_id"]
            ids = registry()["drug_classes"].get(target, [target])
            if drug in ids:
                fact = {"drug_id": drug, "day": day, "event": event, "analyte": v.sid, "modelled_by": effect.get("modelled_by", {}).get(v.sid)}
                (modelled if fact["modelled_by"] else unmodelled).append(fact)
    return _result({"t0": t0, "t1": t1, "unmodelled": unmodelled, "modelled": modelled}, bool(unmodelled),
                   testable=t0 is not None and t1 is not None and "medication_record" in v.b,
                   extra={"modelled_events": len(modelled), "unmodelled_events": len(unmodelled)})


def _derived(v, c):
    cid = c["id"]
    if cid == "K9.adag_fasting":
        if v.pack == "m2":
            pairs = [(v.T, a, b) for a in v.hbT for b in v.fbgT]
        else:
            pairs = [(a["ts"], float(a["value"]), float(b["value"])) for a in v.stream("HbA1c") for b in v.stream("fasting_glucose") if a["ts"] == b["ts"]]
        facts = []
        for day, hb, fbg in pairs:
            upper = adag_upper(hb)
            eag = param(c, "slope") * hb + param(c, "intercept")
            facts.append({"day": day, "HbA1c": hb, "FBG": fbg, "adag_upper": upper, "ratio": fbg / eag if eag else None})
        units = [{"condition": v.sid, "testable": r["adag_upper"] is not None,
                  "violation": r["adag_upper"] is not None and r["FBG"] > r["adag_upper"]} for r in facts]
        if not units:
            return _result({"pairs": facts}, testable=False)
        return _units(units, {"pairs": facts})
    labs = P.visible_labs(v.item)
    if cid == "K9.egfr_printed":
        printed = [r for r in labs if r["finding"].casefold() == "egfr" and P.number(r["value"]) is not None]
        if not printed:
            return Result(extra={"printed_egfr_found": 0})
        pairs, units = [], []
        for r in printed:
            cr = [x for x in labs if x["finding"] == "Cr" and x["ts"] == r["ts"]]
            intervals = [egfr_interval(x["value"], v.profile.get("age_range"), v.profile.get("sex"), x.get("unit")) for x in cr]
            intervals = [x for x in intervals if x]
            digits = max(0, -Decimal(str(r["value"])).as_tuple().exponent)
            inside = any(round(x["lo"], digits) <= float(r["value"]) <= round(x["hi"], digits) for x in intervals)
            units.append({"condition": v.sid, "testable": bool(intervals), "violation": not inside})
            pairs.append({"printed": r, "creatinine": cr, "intervals": intervals})
        return _units(units, {"pairs": pairs}, extra={"printed_egfr_found": len(printed)})
    if cid == "K9.lipid_identity":
        by_fid = {f: [r for r in v.dayT if r["finding"] == f and P.number(r["value"]) is not None] for f in ("TC", "LDL", "HDL", "TG")}
        if not all(by_fid.values()):
            return _result({"printed": by_fid}, testable=False)
        # The identity binds only where the visible laboratory method record names the calculation.
        methods = [r for r in (v.item["sp"].get("prediction_context") or {}).get("laboratory_method_record") or []
                   if v.T is not None and P.number(r.get("day")) is not None and r["day"] <= v.T
                   and _matched(match_words(json.dumps(r, ensure_ascii=False), c["lexicon"], v.lang))]
        if not methods:
            return _result({"printed": by_fid, "calculation_declared": False}, testable=False)
        from itertools import product
        pairs = []
        for tc, ldl, hdl, tg in product(*(by_fid[f] for f in ("TC", "LDL", "HDL", "TG"))):
            if float(tg["value"]) >= param(c, "tg_max"):
                continue
            expected = float(ldl["value"]) + float(hdl["value"]) + float(tg["value"]) / param(c, "tg_divisor")
            tolerance = sum(P.half_decimal(r["value"]) for r in (tc, ldl, hdl, tg))
            pairs.append({"TC": tc["value"], "LDL": ldl["value"], "HDL": hdl["value"], "TG": tg["value"], "expected": expected, "rounding_tolerance": tolerance})
        if not pairs:
            return Result(facts={"printed": by_fid})
        return _result({"pairs": pairs}, any(abs(float(r["TC"]) - r["expected"]) > r["rounding_tolerance"] for r in pairs))
    bmi = [r for r in labs if r["finding"] == "BMI" and P.number(r["value"]) is not None]
    if not bmi:
        return Result(extra={"printed_bmi_found": 0})
    pairs, units = [], []
    for r in bmi:
        weight = [x for x in labs if x["finding"] == "weight" and x["ts"] == r["ts"] and P.number(x["value"]) is not None]
        height = [x for x in labs if x["finding"] == "height" and x["ts"] == r["ts"] and P.number(x["value"]) is not None]
        candidates = []
        for w in weight:
            for h in height:
                unit = str(h.get("unit") or "").casefold()
                if unit not in ("cm", "m") or float(h["value"]) <= 0:
                    continue
                scale = param(c, "cm_per_m") if unit == "cm" else 1
                height_m = float(h["value"]) / scale
                expected = float(w["value"]) / height_m ** param(c, "height_power")
                dh = P.half_decimal(h["value"]) / scale
                dw = P.half_decimal(w["value"])
                if height_m <= dh:
                    continue
                lo = (float(w["value"]) - dw) / (height_m + dh) ** param(c, "height_power")
                hi = (float(w["value"]) + dw) / (height_m - dh) ** param(c, "height_power")
                delta = P.half_decimal(r["value"])
                candidates.append({"weight": w, "height": h, "expected": expected, "rounded_interval": [lo - delta, hi + delta]})
        bad = not any(x["rounded_interval"][0] <= float(r["value"]) <= x["rounded_interval"][1] for x in candidates)
        units.append({"condition": v.sid, "testable": bool(candidates), "violation": bad})
        pairs.append({"BMI": r, "candidates": candidates})
    return _units(units, {"pairs": pairs}, extra={"printed_bmi_found": len(bmi)})


def _two_path(v, c):
    if v.g.get("insufficient"):
        return Result()
    rows = []
    for comp in v.comps:
        for f in _tables()["condition_findings"]["condition_findings"].get(comp, {}).get("findings", []):
            if f["id"] not in registry()["two_path_findings"] or f.get("direction") not in ("high", "low"):
                continue
            obs = [r for r in v.dayT if r["finding"] == f["id"] and classify(f["id"], r["value"], r.get("ref_low"), r.get("ref_high")) is not None]
            stream = next((s for s, fid in P.stream_findings().items() if fid == f["id"]), None)
            for r in obs[:1]:
                rows.append({"component": comp, "finding": f["id"], "direction": f["direction"], "observation": r,
                             "has_stream": bool(v.stream(stream)), "abnormal": classify(f["id"], r["value"], r.get("ref_low"), r.get("ref_high")) == f["direction"]})
    if not rows:
        return Result()
    extra = {"by_path": {name: {"n": sum(r["has_stream"] == has for r in rows),
                                     "abnormal": sum(r["has_stream"] == has and r["abnormal"] for r in rows)} for name, has in (("with_stream", True), ("no_stream", False))}}
    return _result({"pairs": rows}, extra=extra)


def evaluate(check_id, item, lang="zh", *, view=None):
    """Evaluate one registered check; missing evidence is distinct from a pass."""
    c = _check(check_id)
    v = view or View(item, lang)
    if v.pack not in c["packs"]:
        return Result()
    if check_id.startswith(("K1.", "K2.")):
        return _gold_criteria(v, c)
    if check_id.startswith("K3."):
        return _background(v, c)
    if check_id.startswith("K4."):
        return _symptom_stream(v, c)
    if check_id.startswith("K5."):
        return _statement(v, c)
    if check_id.startswith("K6."):
        return _exposure_visibility(v, c)
    if check_id.startswith("K7."):
        return _decoys(v, c)
    if check_id.startswith("K8."):
        return _concurrent(v, c)
    if check_id.startswith("K9."):
        return _derived(v, c)
    if check_id == "K10.two_path":
        return _two_path(v, c)
    raise ValueError(f"pool-level check needs penetrance(): {check_id}")


def penetrance(items):
    """Pool-level unit is a (component, finding) pair, not an individual case."""
    counts = {}
    cf = _tables()["condition_findings"]["condition_findings"]
    for item in items:
        v = View(item)
        for comp in v.comps:
            for f in cf.get(comp, {}).get("findings", []):
                if "penetrance" not in f or f.get("direction") not in ("high", "low"):
                    continue
                key = (comp, f["id"])
                row = counts.setdefault(key, {"component": comp, "finding": f["id"], "direction": f["direction"],
                                               "registered": f["penetrance"], "k": 0, "n": 0, "missing": 0,
                                               "registered_source": f.get("penetrance_source")})
                obs = [r for r in v.dayT if r["finding"] == f["id"] and classify(f["id"], r["value"], r.get("ref_low"), r.get("ref_high")) is not None]
                if not obs:
                    row["missing"] += 1
                    continue
                row["n"] += 1
                row["k"] += classify(f["id"], obs[0]["value"], obs[0].get("ref_low"), obs[0].get("ref_high")) == f["direction"]
    rows = []
    for key in sorted(counts):
        row = counts[key]
        ci = clopper_pearson(row["k"], row["n"]) if row["n"] else None
        rows.append({**row, "CI": ci, "violation": ci is not None and not ci[0] <= row["registered"] <= ci[1]})
    return rows


def _merge(target, extra):
    for key, value in extra.items():
        if isinstance(value, dict):
            _merge(target.setdefault(key, {}), value)
        elif isinstance(value, list):
            target.setdefault(key, []).extend(value)
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            target[key] = value


def _counts():
    return {"tested": 0, "violations": 0, "not_testable": 0}


def _count(target, result):
    if result.applicable:
        target["tested" if result.testable else "not_testable"] += 1
        target["violations"] += int(result.testable and result.violation)


def _sources(check):
    sources = {check["source"]}
    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("source"):
                sources.add(obj["source"])
            for val in obj.values():
                walk(val)
        elif isinstance(obj, list):
            for val in obj:
                walk(val)
    walk(check)
    dependencies = []
    cid = check["id"]
    if cid in ("K1.diabetes_criterion", "K1.diabetes_contradiction", "K1.hypoglycaemia_contradiction",
               "K2.rival_discriminator", "K3.m2_unlabelled_diabetes", "K3.p3_unlabelled_diabetes"):
        dependencies.append("decision_limits")
    if cid in ("K1.ckd_criterion", "K3.renal_unlabelled", "K9.egfr_printed"):
        dependencies.append("egfr_equation")
    dependencies.extend({"K1.lab_criterion": ["criteria"], "K3.standard_of_care": ["standard_of_care"],
                         "K3.drug_indication": ["indication_extensions"],
                         "K8.concurrent_drug": ["drug_effects_on_analytes"]}.get(cid, []))
    for name in dependencies:
        walk(registry()[name])
    if cid == "K3.standard_of_care" and (doc := _soc_table()):
        walk(doc["conditions"])
    return sorted(sources)


def _instrument():
    root = Path(__file__).resolve()
    while root.parent != root and not (root / "pyproject.toml").exists():
        root = root.parent
    files = {"coherence.py": Path(__file__), "coherence_packs.py": Path(P.__file__)}
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {"name": "coherence", "version": registry()["version"],
            "standard_of_care_sha256": hashlib.sha256(registry_path("standard_of_care.yaml").read_bytes()).hexdigest() if _soc_table() else None,
            "registry_sha256": hashlib.sha256(registry_path("coherence_invariants.yaml").read_bytes()).hexdigest(),
            "code_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()},
            "git_head": proc.stdout.strip() if proc.returncode == 0 else None}


def build_report(items, pack, lang, batch, published=None, published_batch=None):
    if pack not in P.BLOCKS or lang not in ("zh", "en"):
        raise ValueError("unsupported pack/language")
    ids = [it["case_id"] for it in items]
    if len(set(ids)) != len(ids) or any(it["pack"] != pack for it in items):
        raise ValueError("report requires unique cases of the selected pack")
    pub = set() if published is None else set(published)
    if not pub <= set(ids):
        raise ValueError("published IDs must be a subset of pool IDs")
    checks = [c for c in registry()["checks"] if pack in c["packs"]]
    reports = {c["id"]: {**{k: c[k] for k in ("id", "class", "title", "level")}, "sources": _sources(c),
                          "pool": _counts(), "published": _counts(), "by_condition": {}, "examples": [], "extra": {}} for c in checks}
    item_rows = []
    for item in items:
        v = View(item, lang)
        row = {"case_id": item["case_id"], "published": item["case_id"] in pub, "spec_or_tier": v.sid,
               "facts": {**v.background(), "prediction_time_T": v.T, "dayT_observations": v.dayT}, "violations": []}
        for c in checks:
            cid = c["id"]
            if cid == "K10.penetrance":
                continue
            result = evaluate(cid, item, lang, view=v)
            row["facts"][cid] = {**result.facts, "applicable": result.applicable, "testable": result.testable}
            report = reports[cid]
            _count(report["pool"], result)
            if row["published"]:
                _count(report["published"], result)
            groups = result.groups or [{"condition": v.sid, "testable": result.testable, "violation": result.violation}]
            grouped = defaultdict(list)
            for g in groups:
                grouped[g["condition"]].append(g)
            if result.applicable:
                for name, units in grouped.items():
                    _count(report["by_condition"].setdefault(name, _counts()), _units(units, {}))
            _merge(report["extra"], result.extra)
            if result.testable and result.violation:
                row["violations"].append(cid)
                if len(report["examples"]) < 8:
                    report["examples"].append({"case_id": item["case_id"], "published": row["published"], "reason": result.reason})
        item_rows.append(row)
    if "K10.penetrance" in reports:
        report = reports["K10.penetrance"]
        report["extra"] = {"pool_pairs": penetrance(items), "published_pairs": penetrance([it for it in items if it["case_id"] in pub])}
        for scope in ("pool", "published"):
            for r in report["extra"][scope + "_pairs"]:
                result = _result(r, r["violation"], testable=r["n"] > 0)
                _count(report[scope], result)
                if scope == "pool":
                    _count(report["by_condition"].setdefault(r["component"], _counts()), result)
        report["extra"]["unit"] = "component/finding pair"
    if "K3.standard_of_care" in reports and _soc_table():
        report = reports["K3.standard_of_care"]
        report["extra"].update(unit="condition/class pair (unrated rows remain per patient)",
                               sources=_soc_table()["sources"])
        for scope, selected in (("pool", items), ("published", [it for it in items if it["case_id"] in pub])):
            pairs = standard_of_care(selected, lang)
            report["extra"][scope + "_pairs"] = pairs
            for pair in pairs:
                result = _result(pair, pair["violation"], testable=pair["n"] > 0)
                _count(report[scope], result)
                if scope == "pool":
                    _count(report["by_condition"].setdefault(pair["condition"] + "/" + pair["drug_class"], _counts()), result)
    return {"instrument": _instrument(), "scope": {"pack": pack, "lang": lang, "batch": Path(batch).name,
            "published": Path(published_batch).name if published_batch is not None else None,
            "n_pool": len(items), "n_published": len(pub)}, "checks": list(reports.values()), "items": item_rows}


def markdown(report):
    scope = report["scope"]
    lines = [f"# Coherence · {scope['pack']} · {scope['lang']}", "", "SYNTHETIC data, evaluation only, not medical advice.", "",
             f"Pool: {scope['n_pool']}; published: {scope['n_published']}.", "",
             "| Check | Class | Pool tested / violations | Published tested / violations | Examples |",
             "|---|---|---:|---:|---|"]
    for r in report["checks"]:
        examples = "; ".join(f"{e['case_id']}: {e['reason']}" for e in r["examples"][:2])
        examples = examples.replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['id']} | {r['class']} | {r['pool']['tested']} / {r['pool']['violations']} | "
                     f"{r['published']['tested']} / {r['published']['violations']} | {examples} |")
    for check in report["checks"]:
        if check["id"] == "K3.standard_of_care" and "pool_pairs" in check["extra"]:
            lines.extend(["", "| SoC condition / class | Source | Target | Background k / n | Exact 95% CI | Marginal k / n (report) | Exclusions | Pass |",
                          "|---|---|---:|---:|---|---:|---|---|"])
            for row in check["extra"]["pool_pairs"]:
                lines.append(f"| {row['condition']} / {row['drug_class']} | {row['source']} | {row['rate']} | {row['k']} / {row['n']} | "
                             f"{row['ci95']} | {row['all']['k']} / {row['all']['n']} | {row['excluded']} | {not row['violation'] if row['n'] else 'not testable'} |")
    lines.extend(["", "Missing applicable evidence is counted as not_testable in JSON. K3 rated rows and K10.penetrance count pairs.", ""])
    return "\n".join(lines)
