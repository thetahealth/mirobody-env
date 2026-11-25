"""Deterministic, language-independent case truth and background constraints."""
from __future__ import annotations

import math
from statistics import NormalDist

from haenv.resources.regpath import load_registry
from haenv.resources.registry import condition_findings_for_case, load_findings
from haenv.world.generation import rng
from haenv.world.observations import findings_render as FR
from haenv.world.truth.derived import adag_band, approach, cr_at_egfr, reflect_below
from haenv.world.truth.registry import effects


ESTABLISHED_ONSET = -100000


def _visible_spec(raw, panel_vocab, fid):
    from haenv.m2 import core
    spec = panel_vocab[fid]
    finding = load_findings()[fid]
    purchase = core._sex_spec(raw, fid, finding) or finding
    ref = {}
    # A declared direction must hold on every channel's printed interval.
    for side, combine in (("low", min), ("high", max)):
        values = [s.get("ref", {}).get(side) for s in (spec, purchase)]
        values = [v for v in values if v is not None]
        if values:
            ref[side] = combine(values)
    return {**spec, "ref": ref}


def _rounded(node):
    if isinstance(node, float):
        return round(node, 6)
    if isinstance(node, dict):
        return {k: _rounded(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_rounded(v) for v in node]
    return node


def prepare_penetrance(raw, declared=None):
    """Declare expression metadata before the event verifier snapshots adjudication."""
    ddx = (getattr(raw, "adjudication", None) or {}).get("ddx") or {}
    sid = ddx.get("spec_id")
    if not sid:
        return set(), []
    if declared is None:
        profile = condition_findings_for_case(load_findings()).get(sid) or {}
        declared = {r["id"]: r for r in profile.get("findings") or ()}
    eligible = {fid for fid, row in declared.items()
                if row.get("direction") in ("high", "low", "positive", "negative")}
    if ddx.get("insufficient"):
        expressed, unexpressed = set(), []
    elif "unexpressed" in ddx:
        unexpressed = sorted(ddx["unexpressed"])
        expressed = eligible - set(unexpressed)
    else:
        expressed = set()
        keys = {"glycaemia" if fid in ("FBG", "HbA1c") else fid for fid in eligible}
        draws = {key: rng.unit(str(raw.case_id), "truth-penetrance", key) for key in sorted(keys)}
        for fid in sorted(eligible):
            key = "glycaemia" if fid in ("FBG", "HbA1c") else fid
            p = declared[fid].get("penetrance")
            if draws[key] < (1.0 if p is None else p):
                expressed.add(fid)
        unexpressed = sorted(eligible - expressed)
    raw.adjudication["ddx"]["unexpressed"] = unexpressed
    return expressed, unexpressed


def build_plan(raw, *, first_symptom_day):
    """Create and store one plan; callers opt in before entering this function."""
    ddx = (getattr(raw, "adjudication", None) or {}).get("ddx") or {}
    sid = ddx.get("spec_id")
    if not sid:
        return None
    from haenv.m2.core import components
    cid = str(raw.case_id)
    doc = effects()
    user = raw.user_profile or {}
    sex = str(user["sex"]).upper()[:1]
    a, b = (float(x) for x in str(user["age_range"]).split("-"))
    age = a + rng.unit(cid, "truth", "age") * (b + 1 - a)
    vocab = FR.with_sex_ref(load_findings(), sex)
    profile = condition_findings_for_case(load_findings()).get(sid) or {}
    declared = {r["id"]: r for r in profile.get("findings") or ()}
    insufficient = bool(ddx.get("insufficient"))
    expressed, unexpressed = prepare_penetrance(raw, declared)
    onset = int(first_symptom_day) if first_symptom_day is not None else 0
    plan = {"version": 1, "age_years": age, "sex": sex,
            "onset": {"first_symptom": first_symptom_day, "course_start": 0},
            "unexpressed": unexpressed, "observables": {}, "constraints": {}, "sources": {}}
    obs = plan["observables"]
    ld = raw.longitudinal_data or {}
    rows = {c: doc["conditions"][c] for c in components(sid) if c in doc["conditions"]}
    glycs = [(c, row["glycaemia"]) for c, row in rows.items() if "glycaemia" in row]
    glyc = glycs[0][1] if glycs else {}
    glyc_ids = {fid for fid in ("FBG", "HbA1c") if fid in expressed}
    direction = (declared.get("FBG") or declared.get("HbA1c") or {}).get("direction")
    # The declaration supplies the default spectrum; effects rows override its course.
    if not insufficient and glyc_ids and direction in ("high", "low"):
        established = glyc.get("onset") == "established"
        start = ESTABLISHED_ONSET if established else 0 if glyc.get("onset") == "course_start" else onset
        visible = {fid: _visible_spec(raw, vocab, fid) for fid in ("HbA1c", "FBG")}
        bases = {fid: FR._value(vocab[fid], "normal", None, cid, fid, 0) for fid in ("HbA1c", "FBG")}
        streams = {fid: (FR._PANEL_TO_TS.get(fid) if ld.get(FR._PANEL_TO_TS.get(fid)) else None)
                   for fid in ("HbA1c", "FBG")}
        clinical = ((getattr(raw, "latent_premise", None) or {}).get("meta") or {}).get("clinical") or {}
        t2d_cohort = any((clinical.get(s) or {}).get("cohort") == "T2D" for s in ("HbA1c", "fasting_glucose"))
        system = doc["glycaemia"]
        margin = system["defining_margin_cv"]["value"]
        defining = glyc.get("defining")
        sd = {fid: _sd(fid, bool(streams[fid]), t2d_cohort) for fid in ("HbA1c", "FBG")}
        _, fbg_hi = FR._band(visible["FBG"])
        members = {}
        if direction == "low":
            members["HbA1c"] = {"target": min(bases["HbA1c"], system["adag_table"]["value"][0][0])}
            fbg = {"target": FR._value(visible["FBG"], "low", declared.get("FBG", {}).get("magnitude"), cid, "FBG", 0)}
            if defining == "hypoglycaemia":
                fbg["ceiling_at_T"] = system["hypoglycaemia_limit"]["FBG"] * math.exp(-margin * sd["FBG"])
            members["FBG"] = fbg
        elif "HbA1c" in expressed:
            band = glyc.get("hba1c_target_range")
            if band:
                lo, hi = band["value"]
                target = lo + (hi-lo)*rng.unit(cid, "truth-glyc", "a1c")
            else:
                target = FR._value(visible["HbA1c"], "high", declared["HbA1c"].get("magnitude"), cid, "HbA1c", 0)
            a1c = {"target": target}
            fbg = {"target": None, "from_a1c": rng.unit(cid, "truth-glyc", "fbg"), "ref_high": fbg_hi,
                   "step": 10.0 ** -(FR._ndigits("FBG") or 1),
                   # a declared FBG reads its own penetrance draw; an undeclared one follows HbA1c freely
                   "clamp": ("abnormal" if "FBG" in expressed else "normal") if "FBG" in declared else None}
            if defining == "diabetes":
                a1c["floor_at_T"] = system["diabetes_limits"]["HbA1c"] * math.exp(margin * sd["HbA1c"])
                fbg["floor_at_T"] = system["diabetes_limits"]["FBG"] * math.exp(margin * sd["FBG"])
            members["HbA1c"], members["FBG"] = a1c, fbg
        elif "FBG" in expressed:
            members["FBG"] = {"target": None, "cap_at_a1c_T": True,
                              "declared_target": FR._value(visible["FBG"], "high", declared["FBG"].get("magnitude"), cid, "FBG", 0),
                              "a1c_base_T": None if streams["HbA1c"] else bases["HbA1c"]}
        if defining == "discriminator":
            rivals = load_registry("rivals.yaml")["rivals"]
            for comp, rule in glycs:
                if rule.get("defining") != "discriminator":
                    continue
                for rival in rivals.get(comp, ()):
                    disc = rival.get("discriminator_finding") or {}
                    fid = disc.get("finding")
                    if (fid not in members or fid not in glyc_ids
                            or disc.get("gold") != declared[fid].get("direction")):
                        continue
                    lo, hi = FR._band(visible[fid])
                    # A printed reading equal to the limit reads normal, so the limit is moved by half a
                    # printed step before the reading margin is applied.
                    half = 0.5 * 10.0 ** -(FR._ndigits(fid) or 0)
                    if direction == "high":
                        members[fid]["floor_at_T"] = (hi + half) * math.exp(margin * sd[fid])
                    else:
                        members[fid]["ceiling_at_T"] = (lo - half) * math.exp(-margin * sd[fid])
        for fid, extra in members.items():
            obs[fid] = {"kind": "approach", "onset": start, "base": None if streams[fid] else bases[fid],
                        "half_time": doc["kinetics"][fid]["half_time_days"], "stream": streams[fid],
                        "direction": direction, "source": [doc["kinetics"][fid]["source"], "CF_DECLARATION"],
                        **extra}
            if established:
                obs[fid]["established"] = True
            if glyc.get("ceiling") == "non_diabetic":
                obs[fid]["cap_at_T"] = system["non_diabetic_ceiling"][fid]
        T_q = int((raw.prediction_context or {}).get("prediction_time_T") or 84)
        resolve_glycaemia(plan, T_q)
        # A glycaemic member with a truth entry is read from it in every channel; only a member
        # without one stays an undeclared reading.
        plan["sources"]["glycaemia"] = [f"conditions.{c}.glycaemia" for c, _ in glycs] or ["CF_DECLARATION"]
    if not insufficient:
        from haenv.world.truth.renal import course_points
        T = int((raw.prediction_context or {}).get("prediction_time_T") or 84)
        for comp, row in rows.items():
            if "renal_course" in row:
                obs.update(course_points(cid, comp, T, vocab, profile, user, row["renal_course"]["template"]))
                plan["sources"]["renal_course"] = f"conditions.{comp}.renal_course"
        for fid in sorted(expressed - set(obs)):
            row, spec = declared[fid], vocab[fid]
            if row["direction"] not in ("high", "low") or spec.get("qualitative"):
                continue
            spec = _visible_spec(raw, vocab, fid)
            stream = FR._PANEL_TO_TS.get(fid)
            obs[fid] = {"kind": "step", "onset": onset,
                        "target": FR._value(spec, row["direction"], row.get("magnitude"), cid, fid, 0),
                        "stream": stream if ld.get(stream) else None, "base": None,
                        "direction": row["direction"], "source": ["CF_DECLARATION"]}
    pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {}
    conditions = " ".join([str(pb.get("disease") or ""), *map(str, pb.get("comorbidities") or ())]).lower()
    markers = load_registry("clinical_baselines.yaml")["diabetes_markers"]
    for i, row in enumerate(doc["background_constraints"]):
        fid = row["observable"]
        unless = markers if row["unless_world_condition"].lower() == "diabetes" else ["ckd"]
        raises = (direction == "high" and bool(glyc_ids) if fid in ("HbA1c", "FBG") else
                  (obs.get("Cr", {}).get("kind") == "course" or
                   "Cr" in expressed and declared["Cr"]["direction"] == "high"))
        if not any(str(m).lower() in conditions for m in unless) and not raises:
            limit = cr_at_egfr(row["egfr_limit"], age, sex) if "egfr_limit" in row else row["limit"]
            plan["constraints"][fid] = {"limit": limit}
            plan["sources"][f"constraints.{fid}"] = f"background_constraints[{i}]"
    plan = _rounded(plan)
    _plan_drugs(raw, plan, vocab)
    if getattr(raw, "latent_premise", None) is None:
        raw.latent_premise = {}
    raw.latent_premise["truth"] = plan
    raw.adjudication["ddx"]["unexpressed"] = unexpressed
    return plan


def _plan_drugs(raw, plan, vocab):
    """Persist analyte-only channels, so panels and purchases read the same dated course.

    build_plan is entered only by the coupled generation path; query-time readers need
    neither the build switch nor any newly generated medication assignments.
    """
    from haenv.world.truth import drugs
    medication = (getattr(raw, "latent_premise", None) or {}).get("world_medication") or {}
    medication = {**medication, "case_id": str(raw.case_id)}
    ld = raw.longitudinal_data or {}
    planned = []
    for med in drugs.medicines(medication):
        for (name, fid), row in drugs.analyte_effects().items():
            if name != med["drug"] or not row.get("applied", True) or ld.get(FR._PANEL_TO_TS.get(fid)):
                continue
            planned.append({"drug": name, "observable": fid, "row_key": f"{name}/{fid}",
                            "dose_timeline": med.get("dose_timeline") or [],
                            "adherence_ref": "world_medication.adherence_true" if med is medication.get("primary") else None})
            entry = plan["observables"].get(fid)
            undeclared = entry is None
            if entry is None:
                base = FR._value(vocab[fid], "normal", None, str(raw.case_id), fid, 0)
                entry = {"kind": "approach", "onset": 0, "half_time": None, "stream": None,
                         "base": base, "target": base, "source": []}
                plan["observables"][fid] = entry
            if entry.get("base") is None:
                entry["base"] = FR._value(vocab[fid], "normal", None, str(raw.case_id), fid, 0)
            if entry["kind"] == "step":
                entry["kind"] = "approach"
                entry["half_time"] = None
            spec = {"base": entry["base"], "cohort": _ldl_cohort(raw)}
            entry["base"] = drugs.untreated_base(fid, spec, medication)
            if undeclared:
                entry["target"] = entry["base"]
            entry["drug_effect"] = {"observable": fid, "world_medication": medication}
    plan["drugs"] = planned


def _ldl_cohort(raw):
    pb = (getattr(raw, "latent_premise", None) or {}).get("patient_basics") or {}
    from haenv.world.clinical.indicators import cohort_of
    return cohort_of("LDL", pb.get("disease"), pb.get("comorbidities") or [])


def _sd(fid, on_stream, t2d_cohort):
    """Per-reading SD (log scale) of a glycaemic member: the registered measurement CV and, on a lab
    stream, the registered visit-to-visit state variation (cohort-scaled as `observe_labs` does)."""
    reg = load_registry("physio_streams.yaml")
    name = FR._PANEL_TO_TS.get(fid, fid)
    cv = float(reg["clinical_measurement"]["cv"].get(name) or 0)
    if not on_stream:
        return cv
    sv = reg["clinical_observation"]["state_variation"]
    state = float((sv.get("cv") or {}).get(name) or 0)
    if not t2d_cohort:
        state *= float((sv.get("non_glycaemic_scale") or {}).get(name, 1.0))
    return math.sqrt(cv * cv + state * state)


def _fraction(entry, T):
    if T < entry["onset"]:
        return 0.0
    h = entry.get("half_time")
    return 1.0 - 2.0 ** (-(T - entry["onset"]) / h) if h else 1.0


def _reach(entry, T, bound, high):
    """The smallest (high) / largest (low) target whose approach meets `bound` on day T."""
    f = _fraction(entry, T)
    if f <= 0 or entry.get("base") is None:
        return entry["target"]
    need = entry["base"] + (bound - entry["base"]) / f
    return max(entry["target"], need) if high else min(entry["target"], need)


def resolve_glycaemia(plan, T):
    """Fix the glycaemic targets once the bases are known (plan time without a stream, `observe_labs`
    with one). A defining limit is met on day T by the truth, not only by the target; the FBG target lies
    inside the ADAG 95 % band of the HbA1c truth on day T. Returns False while a base is missing."""
    obs = plan["observables"]
    a1c, fbg = obs.get("HbA1c"), obs.get("FBG")
    if a1c is not None and a1c.get("base") is None:
        return False
    if a1c is not None and a1c.get("floor_at_T") is not None:
        a1c["target"] = min(_reach(a1c, T, a1c["floor_at_T"], True), effects()["glycaemia"]["adag_table"]["value"][-1][0])
    if a1c is not None:
        if a1c.get("ceiling_at_T") is not None:
            a1c["target"] = _reach(a1c, T, a1c["ceiling_at_T"], False)
        _cap_target(a1c)
    if fbg is None:
        return True
    if fbg.get("from_a1c") is not None:
        lo, hi = adag_band(value_at(a1c, T))
        if fbg.get("clamp") == "abnormal":
            lo = max(lo, fbg["ref_high"] + fbg["step"])
            hi = max(hi, lo)
        elif fbg.get("clamp") == "normal":
            hi = min(hi, fbg["ref_high"])
            lo = min(lo, hi)
        fbg["target"] = lo + (hi - lo) * fbg["from_a1c"]
    elif fbg.get("cap_at_a1c_T"):
        if fbg.get("a1c_base_T") is None:
            return False
        fbg["target"] = min(fbg["declared_target"], adag_band(fbg["a1c_base_T"])[1])
    if fbg.get("base") is None:
        return False
    if fbg.get("floor_at_T") is not None:
        fbg["target"] = _reach(fbg, T, fbg["floor_at_T"], True)
    if fbg.get("ceiling_at_T") is not None:
        fbg["target"] = _reach(fbg, T, fbg["ceiling_at_T"], False)
    _cap_target(fbg)
    return True


def _cap_target(entry):
    cap = entry.get("cap_at_T")
    if cap is not None:
        entry["target"] = min(entry["target"], cap)
        if entry.get("floor_at_T", cap) > cap:
            entry["floor_capped"] = True


def effective_target(entry):
    target, base = entry["target"], entry["base"]
    return min(target, base) if entry.get("direction") == "low" else max(target, base)


def delta(entry, day):
    if (not entry or entry.get("kind") not in ("approach", "step") or entry.get("base") is None
            or entry.get("target") is None):
        return 0.0
    return approach(entry["base"], effective_target(entry), day, entry["onset"], entry.get("half_time")) - entry["base"]


def value_at(entry, day):
    if entry["kind"] == "course":
        points = entry["points"]
        if day <= points[0][0]:
            return points[0][1]
        for a, b in zip(points, points[1:]):
            if day <= b[0]:
                return a[1] + (b[1]-a[1])*(day-a[0])/(b[0]-a[0])
        return points[-1][1]
    value = entry["base"] + delta(entry, day)
    if entry.get("drug_effect"):
        from haenv.world.truth import drugs
        from haenv.world.clinical.med_course import dose_at
        from haenv.world.clinical.drug_effects import floor_term
        saved = entry["drug_effect"]
        med = saved["world_medication"]
        by_drug = {m["drug"]: m for m in drugs.medicines(med)}
        primary = (med.get("primary") or {}).get("drug")
        def adherence(name, d):
            pts = [p for p in med.get("adherence_true") or [] if int(p["ts"]) <= d]
            return float(pts[-1]["value"]) if name == primary and pts else 1.0
        effect = drugs.term(saved["observable"], day, drugs=by_drug,
                            dose_at=lambda name, d: dose_at(by_drug[name].get("dose_timeline") or [], d),
                            adh_at=adherence, base=lambda d: entry["base"] + delta(entry, d),
                            world_medication=med)
        sig = FR._PANEL_TO_TS.get(saved["observable"], saved["observable"])
        value += floor_term(sig, value, effect)
    return value


def observed(cid, fid, entry, day):
    value = value_at(entry, day)
    if entry["kind"] == "course":
        return value
    stream = FR._PANEL_TO_TS.get(fid, fid)
    cv = load_registry("physio_streams.yaml")["clinical_measurement"]["cv"].get(stream) or 0
    u = max(1e-9, min(1-1e-9, rng.unit(cid, "truth-obs", fid, int(day))))
    z = max(-3, min(3, NormalDist().inv_cdf(u)))
    return value * math.exp(cv * z)


def constrain(plan, fid, value, ndigits, ref_low):
    row = plan.get("constraints", {}).get(fid)
    if not row:
        return value
    ndigits = 2 if ndigits is None else ndigits
    step = 10**-ndigits
    value = reflect_below(value, row["limit"], step, ref_low)
    rounded = round(value, ndigits)
    # A fractional creatinine limit can lie between two printable values.
    if rounded >= row["limit"]:
        rounded = round(reflect_below(rounded, row["limit"], step, ref_low), ndigits)
    return rounded
