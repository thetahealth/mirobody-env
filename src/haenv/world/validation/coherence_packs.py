"""Visible-value and adjudication adapters for the read-only coherence instrument.

SYNTHETIC data, evaluation only, not medical advice.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from pathlib import PurePosixPath
import re

from haenv.resources.regpath import load_registry, registry_cached

BLOCKS = {"m2": "m2_frame", "p2": "triage_p2", "p3": "meds_p3", "p4": "followup_p4"}
# Resource names are relative to the active registry directory.
_LEXICON_ROOT = PurePosixPath("i18n")
_ZH_DRUGS = str(_LEXICON_ROOT / "zh" / "lexicon" / "drugs.yaml")
_EN_CLINICAL = str(_LEXICON_ROOT / "en" / "lexicon" / "clinical.yaml")


def number(value):
    """Finite numeric scalar, including a printed number, without guessing units."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def half_decimal(value):
    """Half the last printed decimal; JSON numbers retain their available precision."""
    try:
        return float(Decimal(10) ** Decimal(str(value)).as_tuple().exponent / 2)
    except (InvalidOperation, TypeError, ValueError):
        return None


@registry_cached("composition_comorbid.yaml")
def _composites():
    return load_registry("composition_comorbid.yaml")["pairs"]


def components(spec_id):
    return list((_composites().get(spec_id) or {}).get("pair") or [spec_id]) if spec_id else []


def block(item):
    return (item["sp"].get("prediction_context") or {}).get(BLOCKS[item["pack"]]) or {}


def gold(item):
    key = "ddx" if item["pack"] == "m2" else BLOCKS[item["pack"]]
    return (item["case"].get("adjudication") or {}).get(key) or {}


def prediction_time(item):
    return number((item["sp"].get("prediction_context") or {}).get("prediction_time_T"))


def spec_or_tier(item):
    g = gold(item)
    if item["pack"] == "m2":
        return str(g.get("spec_id") or "unknown")
    if item["pack"] == "p2":
        return str((g.get("plan") or {}).get("tier") or "unknown")
    if item["pack"] == "p3":
        return f"{g.get('axis', 'unknown')}/{g.get('subtype') or 'none'}"
    return str(g.get("analyte") or block(item).get("analyte") or "unknown")


def medication_record(item):
    b = block(item)
    rows = (b.get("background") or {}).get("medication_record") if item["pack"] == "p2" else b.get("medication_record")
    return rows if isinstance(rows, list) else []


def current_medications(item):
    """None denotes a missing list; an explicitly empty list is visible evidence."""
    pc = item["sp"].get("prediction_context") or {}
    b = block(item)
    if item["pack"] == "m2":
        return pc.get("current_medications")
    if item["pack"] == "p2":
        return (b.get("background") or {}).get("medications")
    if item["pack"] == "p3":
        return b.get("current_medications")
    return [r.get("drug", "") for r in medication_record(item)] if "medication_record" in b else None


def known_conditions(item):
    profile = item["sp"].get("user_profile") or {}
    if item["pack"] == "p2":
        return (block(item).get("background") or {}).get("known_conditions")
    return profile.get("known_conditions")


@registry_cached("coherence_invariants.yaml", "labworld.yaml", _ZH_DRUGS, _EN_CLINICAL)
def _drug_names():
    cfg = load_registry("coherence_invariants.yaml")
    names = {}
    def add(drug, name):
        if name:
            key = str(name).strip().casefold()
            if key in names and names[key] != drug:
                raise ValueError(f"ambiguous drug alias: {name}")
            names[key] = drug
        names[drug.casefold()] = drug
    en = load_registry(_EN_CLINICAL).get("entries", {})
    for path in (_ZH_DRUGS, _EN_CLINICAL):
        for key, name in load_registry(path).get("entries", {}).items():
            if str(key).startswith("drug."):
                add(str(key)[5:], name)
    def walk(obj):
        if isinstance(obj, dict):
            for drug, name in (obj.get("drugs") or {}).items():
                if isinstance(name, str):
                    add(drug, name)
                    add(drug, re.split(r"[（(]", name)[0])
                    add(drug, en.get(name))
            for child in obj.values():
                walk(child)
        elif isinstance(obj, list):
            for child in obj:
                walk(child)
    walk(load_registry("labworld.yaml"))
    for drug, aliases in cfg.get("drug_names", {}).items():
        for name in aliases:
            add(drug, name)
    return sorted(names.items(), key=lambda p: (-len(p[0]), p[0]))


def resolve_drug(text):
    """Case-insensitive longest alias match across both language registries."""
    text = str(text or "").casefold()
    return next((drug for name, drug in _drug_names() if name in text), None)


def drug_ids(texts):
    return {drug for text in texts or [] if (drug := resolve_drug(text)) is not None}


def drug_strings(item):
    """Enumerate current, dated, prescription and administration drug fields for audits."""
    texts = list(current_medications(item) or [])
    b = block(item)
    if b.get("drug"):
        texts.append(b["drug"])
    pc = item["sp"].get("prediction_context") or {}
    for container in (pc, b, b.get("background") or {}):
        for key in ("medication_record", "prescription_record", "administration_record",
                    "clinical_history_record", "outpatient_record"):
            records = container.get(key) or []
            for row in [records] if isinstance(records, dict) else records:
                if isinstance(row, dict):
                    texts.extend(row.get(k, "") for k in ("drug", "medication", "prescription", "supplement"))
    return sorted({str(t) for t in texts if t})


@registry_cached("p4_diversity.yaml", _EN_CLINICAL)
def exposure_labels():
    table = load_registry("p4_diversity.yaml")
    en = load_registry(_EN_CLINICAL).get("entries", {})
    result = {}
    for cause, row in table["rules"].items():
        label = row.get("exposure")
        if label:
            result[str(label).casefold()] = cause
            if en.get(label):
                result[str(en[label]).casefold()] = cause
    return result


def exposures(item):
    return [{**r, "cause": exposure_labels().get(str(r.get("exposure", "")).casefold())}
            for r in block(item).get("outpatient_record") or [] if r.get("exposure")]


@registry_cached("coherence_invariants.yaml")
def stream_findings():
    # streams.yaml declares stream existence, but contains no finding-id mapping.
    return load_registry("coherence_invariants.yaml")["m2_stream_findings"]


def m2_view(item):
    """All solver-visible observations at/before T; purchase duplicates removed."""
    T = prediction_time(item)
    out = []
    def append(fid, day, raw, channel):
        if not fid or T is None or number(day) is None or float(day) > T:
            return
        out.append({"finding": str(fid), "ts": number(day), "value": raw.get("value"),
                    "channel": channel, **{k: raw.get(k) for k in ("ref_low", "ref_high", "unit", "flag")}})
    for stream, points in (item["sp"].get("longitudinal_data") or {}).items():
        if stream not in stream_findings():
            continue
        for p in points:
            if number(p.get("value")) is not None:
                append(stream_findings()[stream], p.get("ts"), p, "stream")
    pc = item["sp"].get("prediction_context") or {}
    for r in pc.get("outpatient_laboratory_record") or []:
        append(r.get("finding"), r.get("day"), r, "outpatient")
    for r in pc.get("archived_laboratory_record") or []:
        for o in r.get("observation") or []:
            append(o.get("finding_id"), o.get("ts"), o, "archived")
    seen = set()
    for result in ((item.get("buy") or {}).get("results") or {}).values():
        if result.get("kind") == "vitals":
            continue
        for o in result.get("observation") or []:
            if not isinstance(o, dict):
                continue
            key = (o.get("finding_id"), o.get("ts"), str(o.get("value")))
            if key in seen:
                continue
            seen.add(key)
            append(o.get("finding_id"), o.get("ts"), o, "purchase")
            for row in history_rows(o):
                hkey = (row.get("finding_id"), row.get("ts"), str(row.get("value")))
                if hkey not in seen:
                    seen.add(hkey)
                    append(row.get("finding_id"), row.get("ts"), row, "history_record")
    return sorted(out, key=lambda r: (r["ts"], r["finding"], r["channel"]))


def history_rows(observation):
    """Dated readings of a longitudinal record, kept only where the printed text shows the value."""
    details = observation.get("details")
    text = observation.get("value")
    if not isinstance(details, dict) or not isinstance(text, str):
        return []
    rows = []
    for key in ("creatinine_history", "uacr_history"):
        for row in details.get(key) or []:
            if isinstance(row, dict) and number(row.get("value")) is not None and re.search(
                    r"(?<![\d.])" + re.escape(str(row["value"])) + r"(?![\d])", text):
                rows.append(row)
    return rows


def visible_labs(item):
    """Structured labs plus explicitly labelled numeric lab text in every pack."""
    rows = m2_view(item)
    pc = item["sp"].get("prediction_context") or {}
    b = block(item)
    T = prediction_time(item)
    records = []
    for container in (pc, b, b.get("background") or {}):
        for key in ("lab_results", "laboratory_record", "outpatient_laboratory_record", "report"):
            records.extend(container.get(key) or [])
    aliases = {"egfr": "eGFR", "estimated glomerular filtration rate": "eGFR", "肾小球滤过率": "eGFR",
               "creatinine": "Cr", "肌酐": "Cr", "bmi": "BMI", "body mass index": "BMI", "体重指数": "BMI",
               "height": "height", "身高": "height", "weight": "weight", "体重": "weight"}
    for r in records:
        day = number(r.get("day", r.get("ts")))
        if day is None or T is None or day > T:
            continue
        fid = r.get("finding_id") or r.get("finding") or r.get("analyte")
        if fid and number(r.get("value")) is not None:
            fid = aliases.get(str(fid).casefold(), fid)
            rows.append({"finding": fid, "ts": day, "value": r["value"], "channel": "lab_record",
                         "unit": r.get("unit"), "ref_low": r.get("ref_low"), "ref_high": r.get("ref_high")})
        text = " ".join(str(r.get(k) or "") for k in ("test", "result", "text"))
        for label, finding in aliases.items():
            for match in re.finditer(re.escape(label) + r"\s*[:：=]?\s*(\d+(?:\.\d+)?)\s*([^,;，；\s]*)", text, re.I):
                rows.append({"finding": finding, "ts": day, "value": match[1], "unit": match[2], "channel": "lab_text"})
    # Vitals may be purchased separately and must not enter m2 laboratory checks.
    for result in ((item.get("buy") or {}).get("results") or {}).values():
        if result.get("kind") == "vitals":
            for r in result.get("observation") or []:
                if isinstance(r, dict) and number(r.get("ts")) is not None and T is not None and r["ts"] <= T:
                    fid = str(r.get("finding_id") or "")
                    rows.append({**r, "finding": aliases.get(fid.casefold(), fid), "channel": "vitals"})
    return rows
