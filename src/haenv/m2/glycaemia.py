"""Disease-coupled M2 glycaemia and visible specimen provenance for purchases."""
from __future__ import annotations

import math

from haenv.i18n.text import J, L, is_text, lexicon
from haenv.m2.evidence import registry
from haenv.resources.regpath import load_registry
from haenv.world.generation import rng


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def visible_readings(raw, fid, T):
    """Dated observations only, including stream, follow-up and outside reports."""
    from haenv.interaction.gated import _resolve_target
    from haenv.m2.core import _findings
    from haenv.world.observations.findings_render import _TS_TO_PANEL
    stream = {v: k for k, v in _TS_TO_PANEL.items()}.get(fid)
    out = []
    for point in (getattr(raw, "longitudinal_data", None) or {}).get(stream) or ():
        if _numeric(point.get("value")) and int(point.get("ts", T + 1)) <= T:
            out.append({**point, "ts": int(point["ts"])})

    def visit(obj):
        if isinstance(obj, list):
            for value in obj:
                visit(value)
        elif isinstance(obj, dict):
            identity = next((obj[k] for k in ("finding_id", "finding", "indicator", "name") if obj.get(k)), None)
            day = obj.get("ts", obj.get("day", obj.get("source_timestamp")))
            if (identity and day is not None and int(day) <= T and _numeric(obj.get("value"))
                    and (identity in (fid, stream) or _resolve_target(str(identity), _findings()) == fid)):
                out.append({**obj, "ts": int(day)})
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    visit(value)
    visit(getattr(raw, "prediction_context", None) or {})
    visit(getattr(raw, "evidence_ledger", None) or [])
    # EMR finalization writes F3's follow-up after the inner emission gates.
    # Read that same declared observation here so a low follow-up cannot pass
    # the gate and then appear on the finished solver record.
    gold = (getattr(raw, "adjudication", None) or {}).get("m2_frame") or {}
    if gold.get("frame") == "F3" and (gold.get("f3_plan") or {}).get("stream") == stream:
        from haenv.m2.core import f3_followup
        followup = f3_followup(raw.longitudinal_data, gold["f3_plan"], T)
        if followup:
            out.append({"ts": T, "value": followup["today"]["value"]})
    return sorted(out, key=lambda p: p["ts"])


def _canonical(text):
    """The registry (zh) source of a symptom, whichever language the record was rendered in.

    A built English record stores the rendered sentence; ownership must not depend on the
    display language, or zh and en would draw different day-T values for one world.
    """
    from haenv.i18n.identity import source
    value = source(text)
    return _reverse_en().get(str(value), value)


def _reverse_en():
    global _REVERSE_EN
    if _REVERSE_EN is None:
        _REVERSE_EN = {}
        for key, text in lexicon("en").items():
            _REVERSE_EN.setdefault(text, key)
    return _REVERSE_EN


_REVERSE_EN = None


def _onset(raw, conditions, T):
    """Use component symptom provenance; fall back to the earliest visible symptom."""
    from haenv.m2.core import component_templates, _norm
    from haenv.world.observations.events import symptom_lay_every
    ledger = [e for e in getattr(raw, "evidence_ledger", ()) if e.get("symptom")
              and int(e.get("source_timestamp", T + 1)) <= T]
    ways = symptom_lay_every()
    texts = {text for comp in conditions for text in component_templates(comp)}
    # Diversity replaces component symptoms with its registered complaint bank.
    # That bank still establishes ownership when only one of its components is
    # present; do not lose the onset merely because wording was dispersed.
    from haenv.m2.core import components
    from haenv.m2.diversity import rules as diversity_rules
    present = set(components(str((raw.adjudication.get("ddx") or {}).get("spec_id") or "")))
    for group in diversity_rules()["presentations"]["groups"].values():
        owners = present & set(group["components"])
        if owners and owners <= conditions:
            texts.update(text for pair in group["complaints"] for text in pair)
    variants = {_norm(v) for text in texts for v in [text, *ways.get(text, ())]}
    days = [int(e["source_timestamp"]) for e in ledger if _norm(_canonical(e["symptom"])) in variants]
    ext = ((getattr(raw, "latent_premise", None) or {}).get("meta") or {}).get("_external") or {}
    ea = (ext.get("m2") or {}).get("explain_away") or {}
    for line in ea.get("lines") or ():
        if line["component"] in conditions:
            days += [int(s["day"]) for s in line.get("symptoms") or () if int(s["day"]) <= T]
    days += [int(c["day"]) for c in ea.get("clues") or ()
             if c["component"] in conditions and int(c["day"]) <= T]
    return min(days) if days else min((int(e["source_timestamp"]) for e in ledger), default=T)


def _nonfasting(raw, point, T):
    if point.get("fasting") is False or (point.get("details") or {}).get("fasting") is False:
        return True
    from haenv.i18n.identity import source
    note = str(source(point.get("note") or "")).lower()
    if "非空腹" in note or "non-fasting" in note or "nonfasting" in note:
        return True
    for block in (getattr(raw, "adjudication", None) or {}).values():
        if not isinstance(block, dict):
            continue
        diversity = block.get("diversity") or {}
        # meal_glucose's first draw is non-fasting only when its exposure acts;
        # its repeat, and a new day-T acquisition, follow fasting instructions.
        if (diversity.get("factor") in ("meal_glucose", "fasting_active") and diversity.get("acting")
                and int(diversity.get("draw_day", T + 1)) == int(T)):
            return True
    return False


def fasting_note_ids():
    zh, en = lexicon("zh"), lexicon("en")
    ids = [k for k, v in zh.items() if k.startswith("m2.specimen_note.") and "空腹" in v]
    if not ids or any("fasting" not in en.get(k, "").lower() for k in ids):
        raise ValueError("M2 fasting specimen notes require matching zh/en messages")
    return sorted(ids)


def has_fasting_statement(point):
    # Notes may combine an existing report comment with a specimen phrase.
    from haenv.i18n.identity import source
    note = str(source(point.get("note") or ""))
    return not _explicit_nonfasting(point) and any(
        is_text(point.get("note"), key) or str(L(key)) in note for key in fasting_note_ids())


def _explicit_nonfasting(point):
    note = str(point.get("note") or "").lower()
    return (point.get("fasting") is False or (point.get("details") or {}).get("fasting") is False
            or "非空腹" in note or "non-fasting" in note or "nonfasting" in note)


def _specimen(raw, point, T):
    if point.get("finding_id") != "FBG" or _nonfasting(raw, point, T):
        return point
    if has_fasting_statement(point):
        return point
    note = L(rng.pick(fasting_note_ids(), str(raw.case_id), "m2-fbg-specimen", int(T)))
    return {**point, "note": J("space", [point["note"], note]) if point.get("note") else note}


def _point(raw, fid, value, T):
    from haenv.interaction.gated import _report_face
    from haenv.m2.core import _findings
    from haenv.world.observations.findings_render import _band
    spec = _findings()[fid]
    lo, hi = _band(spec)
    flag = "H" if value > hi else "L" if value < lo else "NORMAL"
    name, sym = _report_face(str(raw.case_id), fid, flag)
    return {"ts": int(T), "finding_id": fid, "value": value, "unit": spec.get("unit", ""),
            "ref_low": lo, "ref_high": hi, "flag": sym, "name": name or spec["name_cn"]}


def result(raw, fid, T):
    from haenv.m2.core import components
    ddx = (raw.adjudication or {}).get("ddx") or {}
    policy = registry()["glycaemic_course"]
    conditions = set(components(str(ddx.get("spec_id") or ""))) & set(policy["conditions"])
    if fid not in ("FBG", "HbA1c") or ddx.get("insufficient") or not conditions:
        return None
    visible = visible_readings(raw, fid, T)
    today = [p for p in visible if p["ts"] == T]
    if today:
        # Preserve even an abnormal or insufficient value: the premise gate decides.
        return [{**_point(raw, fid, today[-1]["value"], T),
                 **{k: v for k, v in today[-1].items() if k in ("note", "fasting", "details")}}]
    def sample(key):
        lo, hi = policy[key]
        return lo + (hi - lo) * rng.unit(str(raw.case_id), "m2-glycaemic-course", key)
    target = sample("target_hba1c_pct")
    if fid == "HbA1c":
        previous = visible[-1] if visible else None
        onset = _onset(raw, conditions, T)
        t0 = max(previous["ts"], onset) if previous else onset
        h0 = previous["value"] if previous else target
        tau = load_registry("physio_streams.yaml")["clinical_observation"]["state_variation"]["glycation_ema_days"]["HbA1c"]
        value = target + (h0 - target) * math.exp(-(T - t0) / tau)
    else:
        eag = policy["eag"]
        value = max(policy["fpg_floor_mmol_l"], (eag["slope"] * target + eag["intercept"]) * sample("fpg_over_eag"))
    from haenv.world.observations.findings_render import _ndigits
    return [_point(raw, fid, round(value, _ndigits(fid)), T)]


def install_results():
    from haenv.app.pack_hooks import Stage, register_hook
    def synth(next_call, orig, raw, target, T):
        from haenv.m2.core import is_m2_case, _findings
        from haenv.interaction.gated import _resolve_target
        if not is_m2_case(raw):
            return next_call(orig, raw, target, T)
        fid = _resolve_target(str(target).strip(), _findings())
        points = result(raw, fid, int(T))
        if points is None:
            points = next_call(orig, raw, target, T)
        return [_specimen(raw, p, T) for p in points] if points else points
    register_hook(Stage.M2_FINDING, "m2:glycaemia", synth)
