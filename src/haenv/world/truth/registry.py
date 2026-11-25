"""Sourced condition effects, read only by the opted-in truth layer."""
from __future__ import annotations

from haenv.resources.regpath import load_registry, registry_cached

_KEYS = {"version", "sources", "kinetics", "glycaemia", "conditions", "renal_course",
         "background_constraints", "egfr_equation", "events", "drug_vitals"}


def _validate_events(doc):
    """New event/vital schemas are closed, and applied sources must be verified."""
    import math

    def keys(row, allowed):
        if set(row) - set(allowed):
            raise ValueError(f"unknown event/vital fields: {sorted(set(row) - set(allowed))}")

    def verified(row, active=True):
        active = active and row.get("applied", True)
        source = row.get("source")
        if source and active and doc["sources"][source].get("status") != "verified":
            raise ValueError(f"event/vital source not verified: {source}")
        for value in row.values():
            if isinstance(value, dict):
                verified(value, active)

    events = doc.get("events") or {}
    keys(events, ("fever", "hf_decompensation"))
    for name, row in events.items():
        keys(row, ("trigger", "body_temp", "resting_hr", "replaces_kernels", "applied") if name == "fever"
             else ("trigger", "weight", "applied"))
        if row.get("trigger") != "symptom_episode":
            raise ValueError("unknown event trigger")
        for sig, allowed in (("body_temp", ("level_c", "source", "note", "applied")),
                             ("resting_hr", ("per_degree_c", "source", "derived_from", "note", "applied")),
                             ("weight", ("gain_kg_range", "onset_days_before_episode", "source", "note", "applied"))):
            if sig in row:
                keys(row[sig], allowed)
        if name == "hf_decompensation" and "weight" in row:
            w = row["weight"]
            lo, hi = w["gain_kg_range"]
            if not 0 < lo <= hi or not w["onset_days_before_episode"] > 0:
                raise ValueError("invalid HF range or onset")
        verified(row)
    for row in (doc.get("drug_vitals") or {}).values():
        keys(row, ("resting_hr", "applied"))
        for sig, vital in row.items():
            if sig == "applied":
                continue
            keys(vital, ("drug_effect_bpm_by_dose_mg", "source", "note", "applied"))
            table = vital.get("drug_effect_bpm_by_dose_mg") or {}
            if any(not math.isfinite(float(k)) or float(k) <= 0 or not math.isfinite(float(v)) for k, v in table.items()):
                raise ValueError("invalid vital dose/effect")
        verified(row)


def validate(doc):
    """Reject unregistered conditions/templates/onsets and unsourced numeric leaves."""
    unknown = set(doc) - _KEYS
    if unknown:
        raise ValueError(f"unknown effects keys: {sorted(unknown)}")
    sources = doc.get("sources") or {}

    def walk(node, path, inherited=()):
        if isinstance(node, dict):
            source = node.get("source", inherited)
            source = [source] if isinstance(source, str) else source
            if not isinstance(source, (list, tuple)):
                raise ValueError(f"invalid source at {path}: {source}")
            if any(s not in sources for s in source):
                raise ValueError(f"unresolved source at {path}: {source}")
            for key, value in node.items():
                if key == "onset" and value not in ("first_symptom", "course_start", "established"):
                    raise ValueError(f"unknown onset kind: {value}")
                if key == "defining" and value not in ("diabetes", "hypoglycaemia", "discriminator"):
                    raise ValueError(f"unknown defining kind: {value}")
                if key == "ceiling" and value != "non_diabetic":
                    raise ValueError(f"unknown ceiling kind: {value}")
                if key != "source":
                    walk(value, f"{path}.{key}", source)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]", inherited)
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and not inherited:
            raise ValueError(f"number without source at {path}")

    for key, value in doc.items():
        if key not in ("version", "sources"):
            walk(value, key)
    known = load_registry("condition_findings.yaml")["condition_findings"]
    for cid, row in (doc.get("conditions") or {}).items():
        if cid not in known:
            raise ValueError(f"unknown condition id: {cid}")
        onset = (row.get("glycaemia") or {}).get("onset", "first_symptom")
        if onset not in ("first_symptom", "course_start", "established"):
            raise ValueError(f"unknown onset kind: {onset}")
        renal = row.get("renal_course")
        if renal and (renal.get("template") not in ("chronic", "acute")
                      or renal["template"] not in doc.get("renal_course", {})):
            raise ValueError(f"unknown renal template: {renal.get('template')}")
    if set(doc.get("renal_course") or {}) - {"chronic", "acute"}:
        raise ValueError("unknown renal template")
    _validate_events(doc)
    return doc


@registry_cached("effects.yaml", "condition_findings.yaml")
def effects():
    return validate(load_registry("effects.yaml"))
