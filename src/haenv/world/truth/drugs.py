"""Sourced drug-to-analyte effects for coupled worlds. SYNTHETIC, evaluation only."""
from __future__ import annotations

import math

from haenv.resources.regpath import load_registry, registry_cached
from haenv.world.clinical import drug_effects as legacy
from haenv.world.clinical.med_course import dose_at as recorded_dose
from haenv.world.generation import rng
from haenv.world.truth.derived import approach


_BACKGROUNDS = {"monotherapy", "add_on_oral", "add_on_insulin"}
_KINDS = {"delta_additive", "delta_log", "titrated_target"}
_FIELDS = {"drug", "observable", "kind", "applied", "source", "value_by_background",
           "ci95_by_background", "kinetics", "fbg_from_hba1c", "target", "responder_share",
           "horizon_days", "pct_by_dose_mg", "value_by_dose_mg", "ci95", "required_dose_ug",
           "t4_half_life_days", "dose_policy", "full_replacement_ug_per_kg", "steady_state_days",
           "on_treatment_tsh", "nonadherence_share"}
_ORAL = {"metformin", "gliclazide", "glimepiride", "glipizide", "glibenclamide",
         "sitagliptin", "linagliptin", "vildagliptin", "empagliflozin", "dapagliflozin",
         "canagliflozin", "acarbose", "pioglitazone"}
_GLP = {"semaglutide", "liraglutide", "dulaglutide", "tirzepatide", "exenatide"}


def _many(value):
    return [value] if isinstance(value, str) else value


def observable(sig):
    from haenv.world.observations.findings_render import _TS_TO_PANEL
    return _TS_TO_PANEL.get(sig, sig)


def _pairs(row):
    for drug in _many(row["drug"]):
        for fid in _many(row["observable"]):
            yield drug, fid
        if row.get("fbg_from_hba1c"):
            yield drug, "FBG"


def validate(doc):
    """Validate the appended table, including sources, numeric fields and exclusive producers."""
    from haenv.resources.registry import load_findings
    from haenv.world.clinical.med_course import BP_STANDARD_DOSE, BP_SIGNALS
    known = set(load_findings()) | {observable(s) for s in BP_SIGNALS}
    sources = doc.get("analyte_sources") or {}
    if set(doc) - {"version", "response", "forbidden_signals", "applies_to_cohorts_default",
                   "baseline_scaling", "drugs", "upstream_gap", "analyte_sources",
                   "analyte_effects", "untreated_base"}:
        raise ValueError("unknown drug effects section")

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("expected a finite analyte number")

    def interval(value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("expected a numeric interval")
        for v in value:
            number(v)
        if value[0] > value[1]:
            raise ValueError("reversed numeric interval")

    def source(node, applied=True):
        ids = _many(node.get("source") or [])
        if not ids or any(s not in sources for s in ids):
            raise ValueError("missing or unresolved analyte source")
        if applied and any(sources[s].get("status") != "verified" for s in ids):
            raise ValueError("applied analyte source is not verified")

    def numbers(node, path=()):
        if isinstance(node, dict):
            for k, v in node.items():
                numbers(v, (*path, k))
        elif isinstance(node, list):
            for v in node:
                numbers(v, path)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if not math.isfinite(node):
                raise ValueError("non-finite analyte number")
            if path[-1] in ("note", "derivation", "source", "drug", "observable", "kind", "dose_policy", "applied"):
                raise ValueError("numeric leaf outside a registered numeric field")

    produced = set()
    # Legacy metformin has no decomposable world term (trial_weight_kg is null).
    for drug, row in (doc.get("drugs") or {}).items():
        for sig, field in legacy.EFFECT_FIELDS.items():
            if any(r.get("trial_weight_kg") is not None and r.get(field) is not None
                   for r in [row, *(row.get("dose_ladder") or {}).values()]):
                produced.add((drug, observable(sig)))
    produced |= {(drug, observable(sig)) for drug in BP_STANDARD_DOSE for sig in BP_SIGNALS}
    seen = set()
    for row in doc.get("analyte_effects") or []:
        if set(row) - _FIELDS:
            raise ValueError("unknown analyte field (unsourced numeric leaves are forbidden)")
        applied = row.get("applied", True)
        source(row, applied)
        if row.get("kind") not in _KINDS and (applied or "kind" in row):
            raise ValueError("unknown analyte effect kind")
        if row.get("dose_policy", "nearest_lower") not in ("nearest_lower", "registered_only"):
            raise ValueError("unknown analyte dose policy")
        if not _many(row.get("drug") or []) or not _many(row.get("observable") or []):
            raise ValueError("analyte drug and observable are required")
        for pair in _pairs(row):
            if pair[1] not in known or pair[1] in legacy.FORBIDDEN:
                raise ValueError(f"unknown or forbidden analyte: {pair[1]}")
            if pair in seen:
                raise ValueError(f"duplicate drug/analyte pair: {pair}")
            seen.add(pair)
            if applied and pair in produced:
                raise ValueError(f"drug/analyte pair already produced: {pair}")
        for key in ("value_by_background", "ci95_by_background"):
            if key in row and set(row[key]) != _BACKGROUNDS:
                raise ValueError("unknown or missing treatment background")
            for value in row.get(key, {}).values():
                (interval if key.startswith("ci95") else number)(value)
        for key, allowed in {
            "kinetics": {"half_time_days", "source", "derivation"},
            "fbg_from_hba1c": {"slope_mmol_per_pp", "source", "derivation"},
            "responder_share": {"value", "note"},
            "required_dose_ug": {"weight_kg", "tsh0", "intercept", "source"},
            "t4_half_life_days": {"euthyroid", "hypothyroid", "source"},
            "full_replacement_ug_per_kg": {"value", "source"},
            "steady_state_days": {"value", "source"},
            "nonadherence_share": {"value", "source"},
        }.items():
            if key in row:
                if set(row[key]) - allowed:
                    raise ValueError(f"unknown numeric field in {key}")
                if key != "responder_share":
                    source(row[key], applied)
        if "ci95" in row:
            interval(row["ci95"])
        for key, fields in {"kinetics": ("half_time_days",),
                            "fbg_from_hba1c": ("slope_mmol_per_pp",),
                            "required_dose_ug": ("weight_kg", "tsh0", "intercept"),
                            "responder_share": ("value",), "full_replacement_ug_per_kg": ("value",),
                            "steady_state_days": ("value",), "nonadherence_share": ("value",)}.items():
            if key in row:
                for field in fields:
                    number(row[key].get(field))
        for field in ("euthyroid", "hypothyroid"):
            if "t4_half_life_days" in row:
                interval(row["t4_half_life_days"].get(field))
        if "horizon_days" in row:
            number(row["horizon_days"])
        for value in row.get("target", {}).values():
            number(value)
        numbers(row)
        kind = row.get("kind")
        if applied and kind == "delta_additive" and not any(k in row for k in ("value_by_background", "value_by_dose_mg")):
            raise ValueError("additive effect has no value")
        if applied and kind == "delta_log" and not row.get("pct_by_dose_mg"):
            raise ValueError("log effect has no dose table")
        for pct in row.get("pct_by_dose_mg", {}).values():
            number(pct)
            if not -100 < pct:
                raise ValueError("invalid percentage effect")
        for key in ("pct_by_dose_mg", "value_by_dose_mg"):
            for dose, value in row.get(key, {}).items():
                number(dose)
                number(value)
                if not isinstance(dose, (int, float)) or dose <= 0 or not isinstance(value, (int, float)):
                    raise ValueError("invalid analyte dose table")
        if "target" in row and set(row["target"]) != set(_many(row["observable"])):
            raise ValueError("unknown or missing target observable")
        if applied and kind == "titrated_target":
            if not row.get("target") or row.get("horizon_days", 0) <= 0 or not 0 <= row.get("responder_share", {}).get("value", -1) <= 1:
                raise ValueError("invalid titrated target")
        if row.get("kinetics", {}).get("half_time_days", 1) <= 0:
            raise ValueError("invalid half time")
        if "on_treatment_tsh" in row:
            # B4: steady-state TSH bands of established levothyroxine users, each sourced, shares <= 1
            bands = row["on_treatment_tsh"]
            if set(bands) - {"suppressed", "low", "under"} or not bands:
                raise ValueError("unknown on-treatment TSH state")
            for state, band in bands.items():
                if set(band) - {"band_miu_l", "share", "source"}:
                    raise ValueError(f"unknown field in on-treatment TSH state {state}")
                source(band, applied)
                interval(band.get("band_miu_l"))
                number(band.get("share"))
                if not band["band_miu_l"][0] > 0 or not 0 <= band["share"] <= 1:
                    raise ValueError(f"invalid on-treatment TSH state {state}")
            if sum(b["share"] for b in bands.values()) > 1:
                raise ValueError("on-treatment TSH shares exceed 1")
        if not 0 <= row.get("nonadherence_share", {}).get("value", 0) <= 1:
            raise ValueError("invalid nonadherence share")
    for fid, cohorts in (doc.get("untreated_base") or {}).items():
        if fid != "LDL":
            raise ValueError("unknown untreated observable")
        for cohort, row in cohorts.items():
            if cohort not in ("CAD", "hypothyroidism") or set(row) != {"quantiles", "source"}:
                raise ValueError("unknown untreated base field")
            source(row)
            if set(row["quantiles"]) != {0.1, 0.5, 0.9} or sorted(row["quantiles"].values()) != list(row["quantiles"].values()):
                raise ValueError("invalid untreated quantiles")
            numbers(row)
    return doc


@registry_cached("drug_effects.yaml")
def _doc():
    return validate(load_registry("drug_effects.yaml"))


@registry_cached("drug_effects.yaml")
def analyte_effects():
    return {pair: row for row in _doc()["analyte_effects"] for pair in _pairs(row)}


def medicines(world_medication):
    primary = world_medication.get("primary") or {}
    return ([primary] if primary.get("drug") else []) + list(world_medication.get("concurrent") or [])


def background_of(world_medication, drug, day):
    others = {m["drug"] for m in medicines(world_medication)
              if m["drug"] != drug and recorded_dose(m.get("dose_timeline") or [], day) > 0}
    if any(d.startswith("insulin") for d in others):
        return "add_on_insulin"
    return "add_on_oral" if others & (_ORAL | _GLP) else "monotherapy"


def _at_dose(table, dose, policy="nearest_lower"):
    if policy == "registered_only" and dose not in table:
        return 0.0
    keys = sorted(table)
    lower = [k for k in keys if dose is not None and k <= dose]
    return float(table[lower[-1] if lower else keys[0]])


def steady_state(row, dose, *, fid, background="monotherapy", base=0.0, responder=True):
    """Registered plateau; dose zero is an inactive course, None an unknown active dose."""
    if not row.get("applied", True) or dose == 0:
        return 0.0
    if row["kind"] == "titrated_target":
        return min(0.0, row["target"][fid] - base) if responder else 0.0
    if row["kind"] == "delta_log":
        pct = _at_dose(row["pct_by_dose_mg"], dose)
        return base * math.expm1(math.log1p(pct / 100))
    if "value_by_background" in row:
        value = row["value_by_background"][background]
    else:
        value = _at_dose(row["value_by_dose_mg"], dose, row.get("dose_policy", "nearest_lower"))
    return value * row["fbg_from_hba1c"]["slope_mmol_per_pp"] if fid == "FBG" and row.get("fbg_from_hba1c") else value


def term(sig, day, *, drugs, dose_at, adh_at, base, world_medication):
    """Sum the table's effects along each dated dose/adherence line, including prehistory.

    HbA1c-derived FBG retains the same exposure (the ADAG derivation). Other first-order
    rows use their registered half-time. Glargine uses a linear approach over its trial
    horizon; rows without a sourced time course are steps. These forms add no time constants.
    """
    fid = observable(sig)
    if sig in legacy.FORBIDDEN:
        raise ValueError("drug effects may not move a gold signal")
    rows = analyte_effects()
    total = 0.0
    for drug in sorted(set(drugs)):
        row = rows.get((drug, fid))
        if not row or not row.get("applied", True):
            continue
        pts = [p for m in medicines(world_medication) if m["drug"] == drug
               for p in m.get("dose_timeline") or [] if p.get("value", 0) > 0]
        start = min((int(p["ts"]) for p in pts), default=0)
        if day < start:
            continue
        baseline = float(base(start - 1) if callable(base) else base or 0.0)
        responder = (rng.unit(str(world_medication.get("case_id") or "HAENV"), "truth-drug-responder", drug)
                     < row.get("responder_share", {}).get("value", 1))

        def drive(d):
            return steady_state(row, dose_at(drug, d), fid=fid,
                                background=background_of(world_medication, drug, d),
                                base=baseline, responder=responder) * max(0.0, min(1.0, adh_at(drug, d)))

        if row.get("fbg_from_hba1c"):
            # Reuse the shared two-stage integrator rather than copying its recurrence.
            value = legacy._kinetic_fraction("HbA1c", int(day) - start, lambda d: drive(d + start),
                                              hba1c_half_time=row["kinetics"]["half_time_days"])
        elif row.get("kinetics"):
            value = 0.0
            for d in range(start + 1, int(day) + 1):
                value = approach(value, drive(d), 1, 0, row["kinetics"]["half_time_days"])
        elif row["kind"] == "titrated_target":
            value = drive(day) * min(1.0, (day - start) / row["horizon_days"])
        else:
            value = drive(day)
        total += value
    return total


def untreated_base(sig, spec, world_medication):
    """Replace an on-treatment LDL cohort only for cases carrying atorvastatin.

    Piecewise interpolation between sourced p10/p50/p90, clipped at the outer quantiles;
    the interpolation and tail clipping are modelling choices, not extra fitted numbers.
    """
    if observable(sig) != "LDL" or not any(m["drug"] == "atorvastatin" for m in medicines(world_medication)):
        return spec.get("base")
    cohort = spec.get("cohort")
    row = (_doc().get("untreated_base", {}).get("LDL") or {}).get(cohort)
    if not row:
        return spec.get("base")
    u = rng.unit(str(world_medication.get("case_id") or "HAENV"), "truth", "untreated-LDL", cohort)
    knots = sorted(row["quantiles"].items())
    u = max(knots[0][0], min(knots[-1][0], u))
    for (a, x), (b, y) in zip(knots, knots[1:]):
        if u <= b:
            return x + (y - x) * (u - a) / (b - a)
    return knots[-1][1]
