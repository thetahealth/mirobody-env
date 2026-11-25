"""Sourced background therapy for coupled synthetic worlds (evaluation only).

Assignments are case-owned, language independent and appended after legacy draws.
No effect sizes live here: the shared drug-effect readers own those numbers.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from haenv.resources.regpath import load_registry, registry_cached
from haenv.world.generation.rng import unit


@registry_cached("standard_of_care.yaml")
def table():
    doc = load_registry("standard_of_care.yaml")
    validate(doc)
    return doc


def validate(doc):
    """Refuse unsourced numeric leaves, invalid rates and unavailable dose rows."""
    sources = doc.get("sources") or {}

    def walk(obj, inherited=None, path=""):
        if isinstance(obj, dict):
            source = obj.get("source", inherited)
            if source is not None and (source not in sources or sources[source].get("status") != "verified"):
                if obj.get("applied", True):
                    raise ValueError(f"unverified source at {path}: {source}")
            if obj.get("applied", True) is False:
                return
            for key, value in obj.items():
                if key == "rate" and value is not None and not 0 <= value <= 1:
                    raise ValueError(f"invalid rate at {path}")
                walk(value, source, f"{path}/{key}")
        elif isinstance(obj, list):
            for item in obj:
                walk(item, inherited, path)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if inherited is None or not math.isfinite(obj):
                raise ValueError(f"unsourced/nonfinite number at {path}")

    for section in ("conditions", "agents", "doses", "acute", "score", "tcm"):
        walk(doc.get(section, {}), path=section)
    if set(doc.get("tcm", {})) - {"hypertension_add_on"}:
        raise ValueError("unsupported TCM indication")
    for cfg in doc.get("tcm", {}).values():
        if set(cfg) - {"applies_to", "rate", "source", "effect", "note",
                       "agents_by_prescription_share", "products", "dropped_agents"}:
            raise ValueError("unsupported TCM field")
        if cfg.get("effect") != "none":
            raise ValueError("TCM must be record-only")
        if cfg.get("rate") is None or not cfg.get("source"):
            raise ValueError("TCM requires a sourced rate")
        weights, products = cfg.get("agents_by_prescription_share", {}), cfg.get("products", {})
        if not weights or any(v <= 0 for v in weights.values()) or set(weights) != set(products):
            raise ValueError("TCM requires positive weights and verified products")
        if set(weights) & set(cfg.get("dropped_agents", {})):
            raise ValueError("dropped TCM agent cannot be assigned")
        from haenv.i18n.text import lexicon
        for drug, product in products.items():
            if set(product) != {"paper_name", "name_key", "dose_key", "source"}:
                raise ValueError("TCM requires a sourced product name and dose")
            if not all(isinstance(v, str) and v for v in product.values()):
                raise ValueError("TCM product fields must be nonempty text")
            for field in ("name_key", "dose_key"):
                if not all(product[field] in lexicon(lang) for lang in ("zh", "en")):
                    raise ValueError("TCM requires paired name and dose text")
    for cls, row in doc["agents"].items():
        if not row.get("weights") or any(v < 0 for v in row["weights"].values()) or not sum(row["weights"].values()):
            raise ValueError(f"{cls}: invalid agent weights")
        for drug in row["weights"]:
            dose = doc["doses"].get(drug)
            if not dose:
                raise ValueError(f"{cls}: no verified dose for {drug}")


def age_years(case_id, age_range):
    lo, hi = map(float, str(age_range).split("-"))
    return lo + unit(case_id, "truth", "age") * (hi + 1 - lo)


def cha2ds2_va(conditions, age, sex=None):
    """ESC CHA2DS2-VA; sex is deliberately not scored."""
    conds = set(conditions)
    cfg = table()["score"]
    score = sum(row["points"] for row in cfg["conditions"] if conds.intersection(row["codes"]))
    score += next((row["points"] for row in cfg["age"] if age >= row["minimum"]), 0)
    return score


def drug_classes(drug):
    return {c for c, row in table()["agents"].items() if drug in row["weights"]} | {
        c for c, members in table().get("existing_classes", {}).items() if drug in members}


def _choice(weights, draw):
    total = sum(weights.values())
    if not total:
        return None
    target = draw * total
    for key, value in weights.items():
        target -= value
        if target < 0:
            return key
    return next(reversed(weights))


def dose_ladder(drug):
    """Read existing ladders, including the shared analyte-effect dose keys."""
    row = table()["doses"][drug]
    if not row.get("applied", True):
        return None
    if row.get("reader") == "statin":
        from haenv.world.truth.drugs import analyte_effects
        return tuple(sorted(map(float, analyte_effects()[drug, "LDL"]["pct_by_dose_mg"])))
    if row.get("reader") == "concurrent":
        from haenv.world.clinical.med_course import CONCURRENT
        return next(ladder for options in CONCURRENT.values() for name, ladder in options if name == drug)
    if row.get("reader") == "p2_existing":
        from haenv.p2.render import WORLD_DRUG
        return (next(dose for name, dose in WORLD_DRUG.values() if name == drug),)
    if row.get("reader") == "indication":
        from haenv.resources.regpath import load_registry
        return tuple(load_registry("drug_indications.yaml")["indications"]["T2D"][drug]["dose_steps"])
    return tuple(row.get("values") or ())


def _dose(drug, case_id, age, constraints):
    row = table()["doses"][drug]
    if not row.get("applied", True):
        return None
    if drug == "apixaban":
        red = row["reduction"]
        criteria = (age >= red["age_ge"],
                    constraints.get("weight_kg", math.inf) <= red["weight_le"],
                    constraints.get("creatinine_mg_dl", -math.inf) >= red["creatinine_ge"])
        return red["dose"] if sum(criteria) >= red["criteria_ge"] else row["values"][0]
    if drug == "rivaroxaban":
        crcl = constraints.get("crcl_ml_min")
        if crcl is None or constraints.get("dialysis") or crcl < row["automatic_crcl_min"]:
            return None
        return row["standard"] if crcl > row["standard_crcl_gt"] else row["renal"]
    if drug == "warfarin":
        # Synthetic individual maintenance value within the label's typical range;
        # uniform sampling is a design convention, not an empirical dose distribution.
        lo, hi = row["typical_maintenance_range"]
        return round(lo + (hi-lo) * unit(case_id, "truth-soc", drug, "individual-dose"), 2)
    ladder = dose_ladder(drug)
    return ladder[int(unit(case_id, "truth-soc", drug, "dose") * len(ladder))] if ladder else None


def assign(case_id, conditions, age, sex, existing, constraints=None):
    """New therapy/reason rows only; existing classes and pack premises take priority.

    Constraints accept forbid_classes, require_classes, forbidden_indications,
    onset_days, history_lo, T, course_end and optional measured renal/dose covariates.
    Reasons have an empty drug and dose line, so they can never act on a stream.
    """
    constraints = constraints or {}
    doc = table()
    conds = set(conditions)
    present = set().union(*(drug_classes(m.get("drug", "")) for m in existing)) if existing else set()
    forbidden = set(constraints.get("forbid_classes", ()))
    required = set(constraints.get("require_classes", ()))
    blocked = set(constraints.get("forbidden_indications", ()))
    score = cha2ds2_va(conds, age, sex)
    out = []

    def reason(indication, cls, kind):
        weights = doc["conditions"]["atrial_fibrillation"]["no_oac_reason"]["weights"]
        why = _choice(weights, unit(case_id, "truth-soc", indication, cls, "reason")) if cls == "oac" else kind
        out.append({"drug": "", "class": cls, "indication": indication, "soc": True,
                    "dose_timeline": [], "reason": why, "exclusion": kind,
                    "source": "GARDENIA_2026" if cls == "oac" else doc["conditions"][indication]["source"]})

    for indication, cfg in doc["conditions"].items():
        if indication not in conds:
            continue
        classes = dict(cfg.get("classes", {}))
        if indication == "atrial_fibrillation":
            classes["oac"] = cfg["oac"]["score_ge_2" if score >= doc["score"]["oac_minimum"] else "score_1" if score else "score_0"]
        for cls, row in classes.items():
            if not row.get("applied", True):
                continue
            if age < row.get("minimum_age", 0):
                continue
            if cls in present:
                continue
            if cls == "oac" and score < doc["score"]["oac_minimum"] and cls not in required:
                continue
            rate = row.get("rate")
            selected = cls in required or (rate is not None and unit(case_id, "truth-soc", indication, cls) < rate)
            if not selected:
                if cls == "oac" and score >= doc["score"]["oac_minimum"]:
                    reason(indication, cls, "not_selected")
                continue
            if indication in blocked or cls in forbidden:
                reason(indication, cls, "gold_indication" if indication in blocked else "forbidden_class")
                continue
            weights = dict(doc["agents"][cls]["weights"])
            # The aspirin source explicitly excludes an OAC indication.
            if "oac" in present or ("atrial_fibrillation" in conds and score >= doc["score"]["oac_minimum"]):
                weights.pop("aspirin", None)
            doses = {drug: _dose(drug, case_id, age, constraints) for drug in weights}
            unavailable = [drug for drug, dose in doses.items() if dose is None]
            weights = {drug: weight for drug, weight in weights.items() if doses[drug] is not None}
            drug = _choice(weights, unit(case_id, "truth-soc", indication, cls, "agent"))
            if drug is None:
                reason(indication, cls, "no_verified_dose")
                continue
            start = int((constraints.get("onset_days") or {}).get(indication, int(constraints.get("history_lo", 0))-1))
            dose = float(doses[drug])
            points = _course(case_id, drug, dose, start, cfg, constraints)
            out.append({"drug": drug, "class": cls, "indication": indication, "dose": dose,
                        "start_day": start, "dose_timeline": points, "soc": True,
                        "source": row.get("source", cfg["source"]),
                        "dose_source": doc["doses"][drug].get("source", "B1_existing_dose"),
                        **({"excluded_agents": unavailable} if unavailable else {}),
                        **({"dose_method": "individualized_by_INR", "target_inr": doc["doses"][drug]["target_inr"]}
                           if drug == "warfarin" else {})})
            present.update(drug_classes(drug))
    return out


def _course(cid, drug, dose, start, cfg, constraints):
    from haenv.world.clinical import med_course as mc
    lo, end = int(constraints.get("history_lo", 0)), int(constraints.get("course_end", 1))
    if not constraints.get("dynamics"):
        return [{"ts": start, "value": dose}]
    ladder = dose_ladder(drug)
    # Renal/individualized prescriptions must not be replaced by a generic ladder.
    if drug in {"apixaban", "rivaroxaban", "warfarin"}:
        ladder = (dose,)
    pts = mc.practice_course(cid, drug, ladder or (dose,), end, lo, int(constraints.get("T", end)),
                             mc._quiet_from(drug, constraints.get("quiet")))
    pts = [{"ts": start, "value": dose}] + [dict(p) for p in pts if p["ts"] > start]
    if cfg.get("permanent_stop") == "none":
        pts = [p for p in pts if p["value"] > 0]
    return pts


def premise_kwargs(p, cid):
    """Carry GEN24b exclusions into every reconstruction of the same medicine course."""
    from haenv.world.validation.gates_case import check_gold_line_in_known_conditions
    pb, meta = p.patient_basics, p.meta or {}
    ddx = meta.get("ddx") or {}
    conds = [pb.get("disease")] + list(pb.get("comorbidities") or ())
    cs = SimpleNamespace(latent={"ddx_spec_id": ddx.get("spec_id")})
    blocked = [c for c in conds if c and check_gold_line_in_known_conditions(
        SimpleNamespace(user_profile={"known_conditions": [c]}), cs)]
    covariates = {k: pb[k] for k in ("weight_kg", "creatinine_mg_dl", "crcl_ml_min", "dialysis", "onset_days") if k in pb}
    if "weight_kg" not in covariates and meta.get("start") is not None:
        covariates["weight_kg"] = float(meta["start"])
    from haenv.contracts.external_gold import SLOT
    p2 = (meta.get(SLOT) or {}).get("p2")
    if p2:
        from haenv.p2.planned import planned_conditions
        from haenv.p2.render import soc_constraints
        covariates.update(soc_constraints(p2))
        covariates["conditions"] = planned_conditions({**p2, "context_cid": cid, "known_conditions": conds})
    return {"age_range": str(pb.get("age_range") or ""), "sex": str(pb.get("sex") or ""),
            "constraints": {"forbidden_indications": blocked, **covariates}}


def known_conditions(basics, concurrent, day, known=(), blocked=()):
    """Problem list for active indicated background therapy, in stable record order."""
    from haenv.world.clinical.med_course import dose_at
    conditions = [basics.get("disease"), *basics.get("comorbidities", ()), *known] + [m["indication"] for m in concurrent
        if m.get("drug") and m.get("indication") and dose_at(m.get("dose_timeline") or [], day) > 0]
    return list(dict.fromkeys(c for c in conditions if c and c not in blocked))


def tcm_assignment(case_id, world_medication, day):
    """One outpatient record at the visit; never an exposure or a dose timeline.

    Called after clinical generation. Only case-owned, active western therapy and
    known adult hypertension qualify; hidden indications remain excluded.
    """
    from haenv.world.clinical.med_course import active_medicines
    cfg = table().get("tcm", {}).get("hypertension_add_on")
    context = world_medication.get("soc_context") or {}
    if (not cfg or context.get("age_years", 0) < 18
            or "hypertension" not in context.get("conditions", ())
            or "hypertension" in context.get("forbidden_indications", ())):
        return []
    if not any("antihypertensive" in drug_classes(drug)
               for drug, _, _ in active_medicines(world_medication, day)):
        return []
    if unit(case_id, "truth-soc", "tcm", "hypertension_add_on") >= cfg["rate"]:
        return []
    drug = _choice(cfg["agents_by_prescription_share"], unit(case_id, "truth-soc", "tcm", "agent"))
    product = cfg["products"][drug]
    return [{"drug": drug, "indication": "hypertension", "day": int(day),
             "name_key": product["name_key"], "dose_key": product["dose_key"],
             "source": cfg["source"], "dose_source": product["source"], "effect": "none"}]


def _tcm_records(raw, block, wm, day):
    """Persist the assignment, then project its paired text into medication sources."""
    from haenv.i18n.text import J, L
    assigned = tcm_assignment(str(raw.case_id), wm, day)
    if not assigned:
        return []
    wm["tcm_record_only"] = assigned
    pc = raw.prediction_context
    for row in assigned:
        name, dose = L(row["name_key"]), L(row["dose_key"])
        shown = J("space", [name, dose])
        list_key = "medications" if isinstance(pc.get("triage_p2"), dict) else "current_medications"
        # p4 has a dated medication source but no separate current list.
        if not isinstance(pc.get("followup_p4"), dict):
            block.setdefault(list_key, []).append(shown)
        if block is pc:
            for review in (pc.get("m2_frame") or {}).get("current_medications", []):
                review["medicines"].append(shown)
        record_key = "prescription_record" if isinstance(pc.get("meds_p3"), dict) else "medication_record"
        block.setdefault(record_key, []).append({"day": day, "drug": name, "event": dose})
    return assigned


def record_notes(raw, audit):
    """Add reason/acute-use records after pack rendering, without assigning an exposure."""
    from haenv.i18n.text import L
    wm = raw.latent_premise.get("world_medication") or {}
    meds = wm.get("concurrent") or []
    T = int(raw.prediction_context["prediction_time_T"])
    rows = [{"day": int(wm.get("history_lo", 0)), "drug": L("soc.oac"), "event": L("soc.no_oac."+m["reason"])}
            for m in meds if m.get("class") == "oac" and not m.get("drug") and m.get("reason")]
    acute = []
    for cls, cfg in table()["acute"].items():
        if unit(str(raw.case_id), "truth-soc", "acute", cls) < cfg["rate"]:
            acute.append(cls)
            rows.append({"day": T, "drug": L("soc.acute."+cls), "event": L("soc.recent_use")})
    pc = raw.prediction_context
    block = next((pc[k] for k in ("meds_p3", "followup_p4", "triage_p2") if isinstance(pc.get(k), dict)), pc)
    if isinstance(pc.get("triage_p2"), dict):
        block = block.setdefault("background", {})
    if rows:
        block.setdefault("medication_record", []).extend(rows)
    tcm = _tcm_records(raw, block, wm, T)
    audit["standard_of_care"] = {"assigned": sum(bool(m.get("drug")) for m in meds if m.get("soc")),
        "exclusions": [m for m in meds if m.get("soc") and m.get("exclusion")],
        "excluded_agents": [m["excluded_agents"] for m in meds if m.get("excluded_agents")],
        "acute_record_only": acute, "tcm_record_only": tcm}
