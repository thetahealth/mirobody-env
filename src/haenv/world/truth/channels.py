"""Generation-side panel and noise-free lab integrations for saved truth plans."""
from __future__ import annotations

from haenv.resources.registry import load_findings
from haenv.world.observations import findings_render as FR
from haenv.world.truth.plan import constrain, delta, observed, resolve_glycaemia
from haenv.world.truth.derived import adag_band


def panel(raw, entries):
    """Replace coupled panel measurements, retaining dates and evidence identities."""
    plan = raw.latent_premise["truth"]
    if (raw.adjudication.get("ddx") or {}).get("insufficient"):
        return entries
    vocab = FR.with_sex_ref(load_findings(), plan["sex"])
    prefix = f"EV-{raw.case_id}-L"
    for entry in entries:
        eid = str(entry.get("evidence_id") or "")
        if not eid.startswith(prefix):
            continue
        key = eid[len(prefix):]
        fid = next((f for f in sorted(vocab, key=len, reverse=True)
                    if key.startswith(f) and key[len(f):].isdigit()), None)
        if fid is None:
            continue
        spec = vocab[fid]
        effect = plan["observables"].get(fid) or {}
        if effect.get("kind") in ("approach", "course") and (effect.get("base") is not None or effect["kind"] == "course"):
            value = FR._round(observed(str(raw.case_id), fid, effect, int(entry["source_timestamp"])), fid)
        else:
            # An undeclared reading, and any finding constrained only by the person's background, reads
            # the case's own stream draw of that day when there is one; otherwise it keeps its panel value.
            # This is what makes a free panel, a record stream and a purchase three views of one draw.
            stream = FR._PANEL_TO_TS.get(fid)
            points = (getattr(raw, "longitudinal_data", None) or {}).get(stream) or []
            same = [q for q in points if int(q.get("ts", -1)) == int(entry["source_timestamp"])]
            if same:
                # The background constraint was applied when this observed stream draw was rendered;
                # copy it unchanged rather than reflecting a disease-range draw a second time.
                entry["symptom"] = FR._panel_text(str(raw.case_id), fid, spec, float(same[-1]["value"])).strip()
                continue
            elif fid in plan["constraints"]:
                value = FR._panel_value(entry, spec)
            else:
                continue
        if value is not None:
            value = constrain(plan, fid, value, FR._ndigits(fid), FR._band(spec)[0])
            entry["symptom"] = FR._panel_text(str(raw.case_id), fid, spec, value).strip()
    return entries


def couple_streams(raw, truth):
    """Add the condition delta to each noise-free course from the condition's onset on."""
    plan = raw.latent_premise["truth"]
    obs = plan["observables"]
    T = int((raw.prediction_context or {}).get("prediction_time_T") or 84)
    for entry in obs.values():
        points = truth.get(entry.get("stream")) or []
        if not points or entry["kind"] not in ("approach", "step"):
            continue
        before = [q for q in points if int(q["ts"]) < entry["onset"]]
        entry["base"] = round(float((before[-1] if before else points[0])["value"]), 6)
    fbg = obs.get("FBG") or {}
    if fbg.get("cap_at_a1c_T") and fbg.get("a1c_base_T") is None:
        points = truth.get("HbA1c") or []
        before = [q for q in points if int(q["ts"]) <= T]
        if points:
            fbg["a1c_base_T"] = round(float((before[-1] if before else points[0])["value"]), 6)
    resolve_glycaemia(plan, T)
    for entry in obs.values():
        points = truth.get(entry.get("stream")) or []
        if entry["kind"] not in ("approach", "step"):
            continue
        for q in points:
            day = int(q["ts"])
            if entry["onset"] <= day:
                q["value"] = round(float(q["value"]) + delta(entry, day), 6)


def constrain_stream(plan, sig, value, ndigits):
    fid = FR._TS_TO_PANEL.get(sig, sig)
    if fid not in plan["constraints"]:
        return value
    vocab = FR.with_sex_ref(load_findings(), plan["sex"])
    return constrain(plan, fid, value, ndigits, FR._band(vocab[fid])[0])


def course_terms(raw, longitudinal_data):
    """Net condition increment per lab stream between its first and last point (GEN15 reads the
    stream's net change; a condition course is one of its declared drivers)."""
    plan = (getattr(raw, "latent_premise", None) or {}).get("truth") or {}
    out = {}
    for entry in (plan.get("observables") or {}).values():
        sig = entry.get("stream")
        pts = (longitudinal_data or {}).get(sig) or []
        if sig and len(pts) >= 2 and entry.get("kind") in ("approach", "step"):
            out[sig] = delta(entry, int(pts[-1]["ts"])) - delta(entry, int(pts[0]["ts"]))
    return out
